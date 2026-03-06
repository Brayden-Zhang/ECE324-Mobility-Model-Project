#!/usr/bin/env python3
import argparse
import json
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


def parse_bins(length_bins: str, lengths: np.ndarray) -> Tuple[np.ndarray, str]:
    if length_bins:
        parts = [int(p.strip()) for p in length_bins.split(",") if p.strip()]
        parts = sorted(set(parts))
        return np.asarray(parts, dtype=np.int64), "fixed"
    if lengths.size == 0:
        return np.asarray([0, 0], dtype=np.int64), "quantile"
    q1, q2 = np.quantile(lengths, [0.33, 0.66])
    return np.asarray([int(q1), int(q2)], dtype=np.int64), "quantile"


def bin_name_for_length(length: int, bins: np.ndarray) -> str:
    if length <= bins[0]:
        return "short"
    if length <= bins[1]:
        return "medium"
    return "long"


def update_bin_metrics(
    store: dict,
    name: str,
    recon_correct: float,
    recon_total: float,
    dest_correct: float,
    dest_nll: float,
    dest_entropy: float,
):
    bucket = store.setdefault(
        name,
        {
            "recon_correct_l1": 0.0,
            "recon_total": 0.0,
            "dest_correct": 0.0,
            "dest_nll_sum": 0.0,
            "dest_entropy_sum": 0.0,
            "samples": 0,
        },
    )
    bucket["recon_correct_l1"] += recon_correct
    bucket["recon_total"] += recon_total
    bucket["dest_correct"] += dest_correct
    bucket["dest_nll_sum"] += dest_nll
    bucket["dest_entropy_sum"] += dest_entropy
    bucket["samples"] += 1


def finalize_metrics(store: dict) -> Dict[str, dict]:
    out = {}
    for name, b in store.items():
        recon_acc = (
            b["recon_correct_l1"] / b["recon_total"] if b["recon_total"] > 0 else 0.0
        )
        dest_top1 = b["dest_correct"] / b["samples"] if b["samples"] > 0 else 0.0
        dest_nll = b["dest_nll_sum"] / b["samples"] if b["samples"] > 0 else 0.0
        dest_entropy = b["dest_entropy_sum"] / b["samples"] if b["samples"] > 0 else 0.0
        out[name] = {
            "recon_acc_l1": recon_acc,
            "dest_top1": dest_top1,
            "dest_nll": dest_nll,
            "dest_entropy": dest_entropy,
            "samples": b["samples"],
            "mask_tokens": b["recon_total"],
        }
    return out


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

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fixed,
    )

    rng = torch.Generator(device=args.device)
    rng.manual_seed(args.seed + 19)

    bucket_store: Dict[str, dict] = {}
    for batch in loader:
        attention = batch["attention_mask"].to(args.device)
        recon_mask = sample_mask(attention, args.mask_ratio, generator=rng)

        # Prevent destination triviality by masking the last point(s)
        # before computing dest logits.
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
        outputs, _, t1, _, _ = forward_backbone(
            batch, pack, device=args.device, max_len=args.max_len, mask=mask
        )

        p1 = outputs["step_logits"]["l1"].argmax(dim=-1)
        correct = (p1 == t1.to(args.device)) & recon_mask
        correct_per = correct.sum(dim=1).detach().cpu().numpy()
        total_per = recon_mask.sum(dim=1).detach().cpu().numpy()

        dest_logits = outputs["dest_logits"]
        last_idx = attention.sum(dim=1).long().clamp(min=1) - 1
        dest_targets = t1.to(args.device).gather(1, last_idx.unsqueeze(1)).squeeze(1)
        dest_pred = dest_logits.argmax(dim=-1)
        dest_correct = (
            (dest_pred == dest_targets).detach().cpu().numpy().astype(np.float32)
        )
        dest_nll = (
            F.cross_entropy(dest_logits.float(), dest_targets, reduction="none")
            .detach()
            .cpu()
            .numpy()
        )
        probs = torch.softmax(dest_logits.float(), dim=-1)
        dest_entropy = (
            (-probs * torch.log(probs + 1e-9)).sum(dim=-1).detach().cpu().numpy()
        )

        raw_len = batch["raw_length"].detach().cpu().numpy()

        for i in range(raw_len.shape[0]):
            bucket = bin_name_for_length(int(raw_len[i]), bins)
            update_bin_metrics(
                bucket_store,
                bucket,
                float(correct_per[i]),
                float(total_per[i]),
                float(dest_correct[i]),
                float(dest_nll[i]),
                float(dest_entropy[i]),
            )

    metrics = finalize_metrics(bucket_store)
    gap = {}
    if "short" in metrics and "long" in metrics:
        gap = {
            "recon_acc_l1": metrics["long"]["recon_acc_l1"]
            - metrics["short"]["recon_acc_l1"],
            "dest_top1": metrics["long"]["dest_top1"] - metrics["short"]["dest_top1"],
            "dest_nll": metrics["long"]["dest_nll"] - metrics["short"]["dest_nll"],
            "dest_entropy": metrics["long"]["dest_entropy"]
            - metrics["short"]["dest_entropy"],
        }

    out = {
        "checkpoint": args.checkpoint,
        "dataset": args.local_data,
        "samples": len(dataset),
        "max_len": args.max_len,
        "mask_ratio": args.mask_ratio,
        "dest_mask_last_k": args.dest_mask_last_k,
        "seed": args.seed,
        "bins": bins.tolist(),
        "bin_strategy": strategy,
        "metrics": metrics,
        "length_sensitivity_gap": gap,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved length sensitivity results: {out_path}")


if __name__ == "__main__":
    main()
