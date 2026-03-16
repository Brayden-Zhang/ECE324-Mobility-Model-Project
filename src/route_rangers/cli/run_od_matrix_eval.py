#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from route_rangers.cli import run_benchmarks as rb
from route_rangers.eval.od_utils import (
    build_time_vocab,
    build_zone_vocab,
    compute_od_tensor,
    metric_mae_rmse_mape,
)

try:
    import h3
except Exception:  # optional
    h3 = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate meso-scale OD matrix forecasting with MAE/RMSE."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--local_data", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split_mode",
        type=str,
        default="both",
        choices=["both", "random", "temporal"],
    )
    parser.add_argument("--h3_res", type=int, default=5)
    parser.add_argument("--max_zones", type=int, default=128)
    parser.add_argument("--bin_seconds", type=int, default=3600)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="cache/od_eval.json")
    return parser.parse_args()


def _h3_cell(lat: float, lon: float, res: int) -> str:
    if h3 is None:
        raise RuntimeError("h3 is required for OD evaluation")
    if hasattr(h3, "latlng_to_cell"):
        return str(h3.latlng_to_cell(lat, lon, res))
    return str(h3.geo_to_h3(lat, lon, res))


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def collect_split_records(
    dataset,
    indices: List[int],
    pack: rb.BackbonePack,
    args,
) -> Dict[str, np.ndarray]:
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=rb.collate_fixed,
    )

    embeddings = []
    origin_cells: List[str] = []
    dest_cells: List[str] = []
    start_ts: List[float] = []

    for batch in loader:
        outputs, _, _, _, attention = rb.forward_backbone(
            batch, pack, device=args.device, max_len=args.max_len, mask=None
        )
        pooled_step = _masked_mean(outputs["step_hidden"], attention)
        pooled_mid = (
            _masked_mean(outputs["mid_hidden"], outputs["mid_mask"])
            if outputs["mid_hidden"].shape[1] > 0
            else torch.zeros_like(pooled_step)
        )
        emb = torch.cat([pooled_step, pooled_mid], dim=-1).detach().cpu().numpy()
        embeddings.append(emb)

        coords = batch["coords"].numpy()
        attn = batch["attention_mask"].numpy()
        ts = batch["start_ts"].numpy()
        for i in range(coords.shape[0]):
            vlen = int(attn[i].sum())
            if vlen <= 0:
                origin_cells.append("")
                dest_cells.append("")
                start_ts.append(float(ts[i]))
                continue
            o = coords[i, 0, :2]
            d = coords[i, vlen - 1, :2]
            origin_cells.append(_h3_cell(float(o[0]), float(o[1]), args.h3_res))
            dest_cells.append(_h3_cell(float(d[0]), float(d[1]), args.h3_res))
            start_ts.append(float(ts[i]))

    return {
        "embeddings": np.concatenate(embeddings, axis=0).astype(np.float32)
        if embeddings
        else np.zeros((0, int(pack.ckpt_args["embed_dim"]) * 2), dtype=np.float32),
        "origin_cells": np.asarray(origin_cells, dtype=object),
        "dest_cells": np.asarray(dest_cells, dtype=object),
        "start_ts": np.asarray(start_ts, dtype=np.float64),
    }


def build_training_rows(
    split_data: Dict[str, np.ndarray],
    zone_vocab: Dict[str, int],
    time_vocab: Dict[int, int],
    bin_seconds: int,
    num_times: int,
    num_zones: int,
) -> Tuple[np.ndarray, np.ndarray]:
    emb = split_data["embeddings"]
    origin_cells = split_data["origin_cells"]
    dest_cells = split_data["dest_cells"]
    ts = split_data["start_ts"]

    time_bins = np.floor(ts / float(bin_seconds)).astype(np.int64)
    time_idx = np.asarray([time_vocab.get(int(t), -1) for t in time_bins], dtype=np.int64)
    origin_idx = np.asarray([zone_vocab.get(str(c), -1) for c in origin_cells], dtype=np.int64)
    dest_idx = np.asarray([zone_vocab.get(str(c), -1) for c in dest_cells], dtype=np.int64)

    od = compute_od_tensor(origin_idx, dest_idx, time_idx, num_times, num_zones)

    # aggregate embeddings per (time, origin)
    agg_sum: Dict[Tuple[int, int], np.ndarray] = {}
    agg_n: Dict[Tuple[int, int], int] = {}
    for i in range(emb.shape[0]):
        t = int(time_idx[i])
        o = int(origin_idx[i])
        if t < 0 or o < 0:
            continue
        key = (t, o)
        if key not in agg_sum:
            agg_sum[key] = emb[i].copy()
            agg_n[key] = 1
        else:
            agg_sum[key] += emb[i]
            agg_n[key] += 1

    xs = []
    ys = []
    for (t, o), v in agg_sum.items():
        t_next = t + 1
        if t_next >= num_times:
            continue
        y = od[t_next, o]
        if float(y.sum()) <= 0.0:
            continue
        xs.append(v / float(max(1, agg_n[(t, o)])))
        ys.append(y)
    if not xs:
        return (
            np.zeros((0, emb.shape[1]), dtype=np.float32),
            np.zeros((0, num_zones), dtype=np.float32),
        )
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def train_od_head(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args,
) -> torch.nn.Module:
    in_dim = train_x.shape[1]
    out_dim = train_y.shape[1]
    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, args.hidden_dim),
        torch.nn.GELU(),
        torch.nn.Linear(args.hidden_dim, out_dim),
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    tx = torch.from_numpy(train_x).to(args.device)
    ty = torch.from_numpy(train_y).to(args.device)
    vx = torch.from_numpy(val_x).to(args.device)
    vy = torch.from_numpy(val_y).to(args.device)

    best = None
    best_val = float("inf")
    for _ in range(max(1, args.epochs)):
        model.train()
        pred = torch.relu(model(tx))
        loss = torch.nn.functional.mse_loss(pred, ty)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = torch.relu(model(vx))
            val_loss = float(torch.nn.functional.mse_loss(val_pred, vy).item())
        if val_loss < best_val:
            best_val = val_loss
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best is not None:
        model.load_state_dict(best)
    return model


def evaluate_od_head(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, args) -> Dict[str, float]:
    if x.shape[0] == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "n_rows": 0}
    with torch.no_grad():
        pred = torch.relu(model(torch.from_numpy(x).to(args.device))).cpu().numpy()
    metrics = metric_mae_rmse_mape(pred, y, eps=1e-3)
    metrics["n_rows"] = int(x.shape[0])
    return metrics


def run_split(dataset, split_idx, pack, args) -> Dict[str, dict]:
    train_idx, val_idx, test_idx = split_idx
    train_data = collect_split_records(dataset, train_idx, pack, args)
    val_data = collect_split_records(dataset, val_idx, pack, args)
    test_data = collect_split_records(dataset, test_idx, pack, args)

    zone_vocab = build_zone_vocab(
        train_data["origin_cells"].tolist(),
        train_data["dest_cells"].tolist(),
        max_zones=args.max_zones,
    )

    all_ts = np.concatenate(
        [train_data["start_ts"], val_data["start_ts"], test_data["start_ts"]], axis=0
    )
    _, time_vocab = build_time_vocab(all_ts, bin_seconds=args.bin_seconds)
    num_times = len(time_vocab)
    num_zones = len(zone_vocab)

    train_x, train_y = build_training_rows(
        train_data,
        zone_vocab,
        time_vocab,
        args.bin_seconds,
        num_times,
        num_zones,
    )
    val_x, val_y = build_training_rows(
        val_data,
        zone_vocab,
        time_vocab,
        args.bin_seconds,
        num_times,
        num_zones,
    )
    test_x, test_y = build_training_rows(
        test_data,
        zone_vocab,
        time_vocab,
        args.bin_seconds,
        num_times,
        num_zones,
    )
    if train_x.shape[0] == 0 or val_x.shape[0] == 0:
        return {
            "train": {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "n_rows": 0},
            "val": {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "n_rows": 0},
            "test": {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "n_rows": 0},
            "meta": {"num_times": num_times, "num_zones": num_zones},
        }

    model = train_od_head(train_x, train_y, val_x, val_y, args)
    return {
        "train": evaluate_od_head(model, train_x, train_y, args),
        "val": evaluate_od_head(model, val_x, val_y, args),
        "test": evaluate_od_head(model, test_x, test_y, args),
        "meta": {
            "num_times": int(num_times),
            "num_zones": int(num_zones),
            "train_rows": int(train_x.shape[0]),
            "val_rows": int(val_x.shape[0]),
            "test_rows": int(test_x.shape[0]),
        },
    }


def main():
    args = parse_args()
    rb.set_seed(args.seed)
    if h3 is None:
        raise RuntimeError("h3 is required. Install `h3` to run OD evaluation.")

    if not Path(args.local_data).exists():
        raise FileNotFoundError(f"local_data not found: {args.local_data}")
    pack = rb.load_backbone(
        args.checkpoint,
        device=args.device,
        override_max_len=args.max_len,
        disable_graph=False,
    )
    records = rb.load_local_data(args.local_data)
    dataset = rb.FixedTrajectoryDataset(
        records, max_len=args.max_len, sample_limit=args.sample_limit
    )
    if len(dataset) < 10:
        raise RuntimeError(f"not enough samples: {len(dataset)}")

    split_modes = (
        ["random", "temporal"] if args.split_mode == "both" else [args.split_mode]
    )
    results = {
        "checkpoint": args.checkpoint,
        "dataset": args.local_data,
        "samples": len(dataset),
        "settings": {
            "h3_res": args.h3_res,
            "max_zones": args.max_zones,
            "bin_seconds": args.bin_seconds,
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
        },
        "splits": {},
    }
    for mode in split_modes:
        split_idx = rb.split_indices(dataset, mode=mode, seed=args.seed)
        results["splits"][mode] = run_split(dataset, split_idx, pack, args)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved OD evaluation results: {out}")


if __name__ == "__main__":
    main()
