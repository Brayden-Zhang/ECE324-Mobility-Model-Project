#!/usr/bin/env python3
import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from route_rangers.cli.run_benchmarks import (  # noqa: E402
    forward_backbone,
    load_backbone,
    sample_mask,
)
from route_rangers.eval.length_utils import (  # noqa: E402
    aggregate_length_metrics,
    bin_name_for_length,
    expected_calibration_error,
    gap_decision_from_seed_values,
    parse_bins,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate length sensitivity for a trajectory model."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--local_data", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument("--mask_ratio", type=float, default=0.3)
    parser.add_argument(
        "--dest_mask_last_k",
        type=int,
        default=1,
        help=(
            "Mask last K valid points when computing destination metrics "
            "to avoid trivial copying (0 disables)."
        ),
    )
    parser.add_argument(
        "--length_bins",
        type=str,
        default="",
        help="Comma-separated raw length cutoffs, e.g. 50,200",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=3,
        help="Number of evaluation seeds when --seeds is not provided.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Explicit comma-separated seed list, e.g. 42,43,44",
    )
    parser.add_argument(
        "--ci_method",
        type=str,
        default="seed",
        choices=["seed", "bootstrap"],
    )
    parser.add_argument("--bootstrap_iters", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--variability_k",
        type=float,
        default=1.0,
        help="Decision multiplier for variability-aware gap threshold.",
    )
    parser.add_argument(
        "--gap_tolerance",
        type=float,
        default=0.0,
        help="Absolute minimum tolerance for short-vs-long performance gaps.",
    )
    parser.add_argument(
        "--primary_gap_metric",
        type=str,
        default="dest_top1",
        choices=["recon_acc_l1", "next_step_top1", "dest_top1", "dest_nll"],
    )
    parser.add_argument(
        "--ece_bins",
        type=int,
        default=10,
        help="Number of confidence bins for destination ECE.",
    )
    # Default name matches collect_results.py glob: cache/length_sensitivity_*.json
    parser.add_argument(
        "--output", type=str, default="cache/length_sensitivity_latest.json"
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_local_data(path: str) -> List[dict]:
    import pandas as pd

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_pickle(path)
    return df.to_dict(orient="records")


def parse_times(times) -> np.ndarray:
    if times is None:
        return np.zeros((0,), dtype=np.float32)
    arr = np.asarray(times)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(np.float32, copy=False)
    if np.issubdtype(arr.dtype, np.datetime64):
        ts = arr.astype("datetime64[ns]").astype(np.int64) / 1e9
        return ts.astype(np.float32)
    try:
        import pandas as pd

        dt = pd.to_datetime(arr, errors="coerce", utc=True, format="mixed")
        ts = dt.view("int64").to_numpy(dtype=np.float64) / 1e9
        invalid = ~np.isfinite(ts)
        if invalid.any():
            ts[invalid] = np.arange(ts.shape[0], dtype=np.float64)[invalid]
        return ts.astype(np.float32)
    except Exception:
        out = np.zeros((arr.shape[0],), dtype=np.float32)
        for i, _ in enumerate(arr):
            out[i] = float(i)
        return out


def normalize_lon_lat(points) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("trajectory must be a list/array of [lat, lon] or [lon, lat]")
    lat = arr[:, 0]
    lon = arr[:, 1]
    if np.any(np.abs(lat) > 90) and np.any(np.abs(lon) <= 90):
        lat, lon = lon, lat
    return lat, lon


def deterministic_downsample(length: int, max_len: int) -> np.ndarray:
    if length <= max_len:
        return np.arange(length, dtype=np.int64)
    idx = np.linspace(0, length - 1, num=max_len, endpoint=True)
    idx = np.round(idx).astype(np.int64)
    idx = np.clip(idx, 0, length - 1)
    idx = np.maximum.accumulate(idx)
    return idx[:max_len]


def preprocess_record(record: dict, max_len: int) -> Optional[dict]:
    traj = None
    for key in ("trajectory", "traj", "points"):
        value = record.get(key, None)
        if value is not None:
            traj = value
            break
    if traj is None:
        return None

    times = None
    for key in ("time", "times", "timestamp"):
        value = record.get(key, None)
        if value is not None:
            times = value
            break

    lat, lon = normalize_lon_lat(traj)
    valid = np.isfinite(lat) & np.isfinite(lon)
    if valid.sum() < 2:
        return None
    lat = lat[valid]
    lon = lon[valid]
    if times is None:
        ts = np.arange(lat.shape[0], dtype=np.float32)
    else:
        ts = parse_times(times)
        if ts.shape[0] != valid.shape[0]:
            n = min(ts.shape[0], lat.shape[0])
            lat = lat[:n]
            lon = lon[:n]
            ts = ts[:n]
        else:
            ts = ts[valid]
    if lat.shape[0] < 2:
        return None

    raw_len = int(lat.shape[0])
    idx = deterministic_downsample(raw_len, max_len)
    lat = lat[idx]
    lon = lon[idx]
    ts = ts[idx]

    coords = np.zeros((max_len, 2), dtype=np.float32)
    timestamps = np.zeros((max_len,), dtype=np.float32)
    attention = np.zeros((max_len,), dtype=np.float32)
    vlen = lat.shape[0]
    coords[:vlen, 0] = lat
    coords[:vlen, 1] = lon
    timestamps[:vlen] = ts
    attention[:vlen] = 1.0
    return {
        "coords": torch.from_numpy(coords),
        "timestamps": torch.from_numpy(timestamps),
        "attention_mask": torch.from_numpy(attention),
        "raw_length": raw_len,
        "effective_length": int(vlen),
    }


class FixedTrajectoryDataset(Dataset):
    def __init__(self, records: List[dict], max_len: int, sample_limit: int = 0):
        processed = []
        for rec in records:
            p = preprocess_record(rec, max_len=max_len)
            if p is not None:
                processed.append(p)
                if sample_limit > 0 and len(processed) >= sample_limit:
                    break
        self.samples = processed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_fixed(batch: List[dict]) -> dict:
    return {
        "coords": torch.stack([b["coords"] for b in batch], dim=0),
        "timestamps": torch.stack([b["timestamps"] for b in batch], dim=0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
        "raw_length": torch.tensor([b["raw_length"] for b in batch], dtype=torch.long),
        "effective_length": torch.tensor(
            [b["effective_length"] for b in batch], dtype=torch.long
        ),
    }


def update_bin_metrics(
    store: dict,
    name: str,
    recon_correct: float,
    recon_total: float,
    next_correct: float,
    next_total: float,
    next_nll: float,
    dest_correct: float,
    dest_nll: float,
    dest_entropy: float,
    dest_conf: float,
    dest_is_correct: float,
):
    bucket = store.setdefault(
        name,
        {
            "recon_correct_l1": 0.0,
            "recon_total": 0.0,
            "next_correct": 0.0,
            "next_total": 0.0,
            "next_nll_sum": 0.0,
            "dest_correct": 0.0,
            "dest_nll_sum": 0.0,
            "dest_entropy_sum": 0.0,
            "dest_conf": [],
            "dest_is_correct": [],
            "samples": 0,
        },
    )
    bucket["recon_correct_l1"] += recon_correct
    bucket["recon_total"] += recon_total
    bucket["next_correct"] += next_correct
    bucket["next_total"] += next_total
    bucket["next_nll_sum"] += next_nll
    bucket["dest_correct"] += dest_correct
    bucket["dest_nll_sum"] += dest_nll
    bucket["dest_entropy_sum"] += dest_entropy
    bucket["dest_conf"].append(dest_conf)
    bucket["dest_is_correct"].append(dest_is_correct)
    bucket["samples"] += 1


def finalize_metrics(store: dict, ece_bins: int) -> Dict[str, dict]:
    out = {}
    for name, b in store.items():
        recon_acc = (
            b["recon_correct_l1"] / b["recon_total"] if b["recon_total"] > 0 else 0.0
        )
        next_top1 = b["next_correct"] / b["next_total"] if b["next_total"] > 0 else 0.0
        next_nll = b["next_nll_sum"] / b["samples"] if b["samples"] > 0 else 0.0
        dest_top1 = b["dest_correct"] / b["samples"] if b["samples"] > 0 else 0.0
        dest_nll = b["dest_nll_sum"] / b["samples"] if b["samples"] > 0 else 0.0
        dest_entropy = b["dest_entropy_sum"] / b["samples"] if b["samples"] > 0 else 0.0
        conf = np.asarray(b["dest_conf"], dtype=np.float64)
        corr = np.asarray(b["dest_is_correct"], dtype=np.float64)
        dest_ece = expected_calibration_error(conf, corr, n_bins=ece_bins)
        out[name] = {
            "recon_acc_l1": recon_acc,
            "next_step_top1": next_top1,
            "next_step_nll": next_nll,
            "dest_top1": dest_top1,
            "dest_nll": dest_nll,
            "dest_entropy": dest_entropy,
            "dest_ece": dest_ece,
            "samples": b["samples"],
            "mask_tokens": b["recon_total"],
            "next_tokens": b["next_total"],
        }
    return out


def _ci_from_values(
    values: np.ndarray,
    ci_method: str,
    alpha: float,
    bootstrap_iters: int,
    rng_seed: int,
) -> Dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    if values.size == 1:
        return {
            "mean": mean,
            "std": std,
            "ci_low": mean,
            "ci_high": mean,
            "n": 1,
        }
    if ci_method == "bootstrap":
        rng = np.random.default_rng(rng_seed)
        sample_means = np.empty((bootstrap_iters,), dtype=np.float64)
        for i in range(bootstrap_iters):
            idx = rng.integers(0, values.size, size=values.size)
            sample_means[i] = float(values[idx].mean())
        lo = float(np.quantile(sample_means, alpha / 2.0))
        hi = float(np.quantile(sample_means, 1.0 - alpha / 2.0))
    else:
        se = std / math.sqrt(float(values.size))
        z = 1.959963984540054
        lo = mean - z * se
        hi = mean + z * se
    return {
        "mean": mean,
        "std": std,
        "ci_low": lo,
        "ci_high": hi,
        "n": int(values.size),
    }


def evaluate_seed(args, pack, dataset, bins: np.ndarray, seed: int):
    set_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fixed,
    )
    rng = torch.Generator(device=args.device)
    rng.manual_seed(seed + 19)
    bucket_store: Dict[str, dict] = {}

    for batch in loader:
        attention = batch["attention_mask"].to(args.device)
        raw_len = batch["raw_length"].detach().cpu().numpy()
        recon_mask = sample_mask(attention, args.mask_ratio, generator=rng)

        # Prevent destination triviality by masking the last point(s) for dest metrics.
        dest_mask = torch.zeros_like(recon_mask)
        if args.dest_mask_last_k > 0:
            vlen = attention.sum(dim=1).long().clamp(min=1)
            for i in range(vlen.shape[0]):
                end = int(vlen[i].item())
                start = max(0, end - int(args.dest_mask_last_k))
                if start < end:
                    dest_mask[i, start:end] = True
        recon_mask = recon_mask & ~dest_mask
        mask = recon_mask | dest_mask

        outputs_masked, _, t1, _, _ = forward_backbone(
            batch, pack, device=args.device, max_len=args.max_len, mask=mask
        )
        outputs_plain, _, t1_plain, _, _ = forward_backbone(
            batch, pack, device=args.device, max_len=args.max_len, mask=None
        )

        t1 = t1.to(args.device)
        t1_plain = t1_plain.to(args.device)

        # reconstruction metrics
        p1 = outputs_masked["step_logits"]["l1"].argmax(dim=-1)
        recon_correct = ((p1 == t1) & recon_mask).sum(dim=1).detach().cpu().numpy()
        recon_total = recon_mask.sum(dim=1).detach().cpu().numpy()

        # next-step micro metrics (token-level shift)
        next_correct = np.zeros((attention.shape[0],), dtype=np.float32)
        next_total = np.zeros((attention.shape[0],), dtype=np.float32)
        next_nll = np.zeros((attention.shape[0],), dtype=np.float32)
        logits_next_all = outputs_plain["step_logits"]["l1"]
        for i in range(attention.shape[0]):
            vlen = int(attention[i].sum().item())
            if vlen <= 1:
                continue
            logits_i = logits_next_all[i, : vlen - 1]
            targets_i = t1_plain[i, 1:vlen]
            pred_i = logits_i.argmax(dim=-1)
            corr = (pred_i == targets_i).float()
            next_correct[i] = float(corr.sum().item())
            next_total[i] = float(corr.shape[0])
            next_nll[i] = float(
                F.cross_entropy(logits_i.float(), targets_i, reduction="mean").item()
            )

        # destination metrics
        dest_logits = outputs_masked["dest_logits"]
        last_idx = attention.sum(dim=1).long().clamp(min=1) - 1
        dest_targets = t1.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        dest_pred = dest_logits.argmax(dim=-1)
        probs = torch.softmax(dest_logits.float(), dim=-1)
        dest_conf = probs.max(dim=-1).values.detach().cpu().numpy()
        dest_correct = (
            (dest_pred == dest_targets).detach().cpu().numpy().astype(np.float32)
        )
        dest_nll = (
            F.cross_entropy(dest_logits.float(), dest_targets, reduction="none")
            .detach()
            .cpu()
            .numpy()
        )
        dest_entropy = (
            (-probs * torch.log(probs + 1e-9)).sum(dim=-1).detach().cpu().numpy()
        )

        for i in range(raw_len.shape[0]):
            bucket = bin_name_for_length(int(raw_len[i]), bins)
            update_bin_metrics(
                bucket_store,
                bucket,
                recon_correct=float(recon_correct[i]),
                recon_total=float(recon_total[i]),
                next_correct=float(next_correct[i]),
                next_total=float(next_total[i]),
                next_nll=float(next_nll[i]),
                dest_correct=float(dest_correct[i]),
                dest_nll=float(dest_nll[i]),
                dest_entropy=float(dest_entropy[i]),
                dest_conf=float(dest_conf[i]),
                dest_is_correct=float(dest_correct[i]),
            )

    metrics = finalize_metrics(bucket_store, ece_bins=args.ece_bins)
    gap = {}
    if "short" in metrics and "long" in metrics:
        for k in ("recon_acc_l1", "next_step_top1", "dest_top1", "dest_nll", "dest_ece"):
            gap[k] = float(metrics["long"].get(k, 0.0) - metrics["short"].get(k, 0.0))
    return metrics, gap


def main():
    args = parse_args()
    set_seed(args.seed)
    if not Path(args.local_data).exists():
        raise FileNotFoundError(f"local_data not found: {args.local_data}")

    pack = load_backbone(
        args.checkpoint, device=args.device, override_max_len=args.max_len
    )
    records = load_local_data(args.local_data)
    dataset = FixedTrajectoryDataset(
        records, max_len=args.max_len, sample_limit=args.sample_limit
    )
    if len(dataset) < 10:
        raise RuntimeError(f"not enough samples for length sensitivity: {len(dataset)}")
    lengths = np.array([s["raw_length"] for s in dataset.samples], dtype=np.int64)
    bins, strategy = parse_bins(args.length_bins, lengths)

    if args.seeds.strip():
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed + i for i in range(max(1, args.num_seeds))]

    per_seed_metrics = []
    per_seed_gap = []
    for seed in seeds:
        metrics_seed, gap_seed = evaluate_seed(args, pack, dataset, bins, seed=seed)
        per_seed_metrics.append(metrics_seed)
        per_seed_gap.append(gap_seed)

    metrics_ci = aggregate_length_metrics(
        per_seed_metrics,
        ci_method=args.ci_method,
        alpha=args.alpha,
        bootstrap_iters=args.bootstrap_iters,
        rng_seed=args.seed + 999,
    )
    metrics_mean = {
        b: {m: float(v["mean"]) for m, v in metrics_ci[b].items()} for b in metrics_ci
    }

    gap_metrics = sorted({k for g in per_seed_gap for k in g.keys()})
    gap_mean = {}
    gap_stats = {}
    for metric in gap_metrics:
        vals = np.asarray(
            [float(g[metric]) for g in per_seed_gap if metric in g], dtype=np.float64
        )
        ci = _ci_from_values(
            vals,
            ci_method=args.ci_method,
            alpha=args.alpha,
            bootstrap_iters=args.bootstrap_iters,
            rng_seed=args.seed + 171,
        )
        short_vals = np.asarray(
            [
                float(m.get("short", {}).get(metric, 0.0))
                for m in per_seed_metrics
                if "short" in m and metric in m["short"]
            ],
            dtype=np.float64,
        )
        long_vals = np.asarray(
            [
                float(m.get("long", {}).get(metric, 0.0))
                for m in per_seed_metrics
                if "long" in m and metric in m["long"]
            ],
            dtype=np.float64,
        )
        decision = gap_decision_from_seed_values(
            short_vals,
            long_vals,
            variability_k=args.variability_k,
            tolerance=args.gap_tolerance,
        )
        gap_mean[metric] = float(ci["mean"])
        gap_stats[metric] = {**ci, **decision}

    primary = args.primary_gap_metric
    primary_pass = bool(gap_stats.get(primary, {}).get("pass", True))
    overall_pass = bool(
        gap_stats.get("recon_acc_l1", {}).get("pass", True)
        and gap_stats.get("next_step_top1", {}).get("pass", True)
        and gap_stats.get("dest_top1", {}).get("pass", True)
        and primary_pass
    )

    out = {
        "checkpoint": args.checkpoint,
        "dataset": args.local_data,
        "samples": len(dataset),
        "max_len": args.max_len,
        "mask_ratio": args.mask_ratio,
        "dest_mask_last_k": args.dest_mask_last_k,
        "seed": args.seed,
        "seeds": seeds,
        "bins": bins.tolist(),
        "bin_strategy": strategy,
        "metrics": metrics_mean,
        "metrics_ci": metrics_ci,
        "length_sensitivity_gap": gap_mean,
        "length_sensitivity_gap_stats": gap_stats,
        "decision": {
            "overall_pass": overall_pass,
            "primary_gap_metric": primary,
            "primary_pass": primary_pass,
            "variability_k": args.variability_k,
            "gap_tolerance": args.gap_tolerance,
        },
        "per_seed": [
            {"seed": seed, "metrics": m, "gap": g}
            for seed, m, g in zip(seeds, per_seed_metrics, per_seed_gap)
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved length sensitivity results: {out_path}")


if __name__ == "__main__":
    main()
