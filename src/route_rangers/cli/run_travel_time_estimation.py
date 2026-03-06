#!/usr/bin/env python3
"""
Travel-time estimation downstream task.

Given a trajectory prefix (first K% of points), predict the total travel time
of the full trajectory. Reports MAE, RMSE, and MAPE in seconds.

This is a key NeurIPS downstream eval: the model's step embeddings are probed
to predict a scalar (duration) using a lightweight linear head.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from route_rangers.cli import run_benchmarks as rb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Travel-time estimation downstream probe"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--local_data", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument(
        "--prefix_ratios", type=float, nargs="+", default=[0.25, 0.50, 0.75, 1.0]
    )
    parser.add_argument("--probe_epochs", type=int, default=10)
    parser.add_argument("--probe_lr", type=float, default=2e-3)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def compute_travel_times(dataset: rb.FixedTrajectoryDataset) -> torch.Tensor:
    """Compute total travel time in seconds for each trajectory."""
    times = []
    for sample in dataset.samples:
        ts = sample["timestamps"]
        attn = sample["attention_mask"]
        vlen = int(attn.sum().item())
        if vlen <= 1:
            times.append(0.0)
        else:
            total_time = float(ts[vlen - 1] - ts[0])
            times.append(max(0.0, total_time))
    return torch.tensor(times, dtype=torch.float32)


def collect_prefix_embeddings(
    loader: DataLoader,
    pack: rb.BackbonePack,
    device: str,
    max_len: int,
    prefix_ratio: float,
) -> torch.Tensor:
    """Get pooled embeddings from the first prefix_ratio of each trajectory."""
    embeddings = []
    for batch in loader:
        attention = batch["attention_mask"].to(device)
        # Create prefix attention: only attend to first prefix_ratio of valid tokens
        prefix_attention = attention.clone()
        if prefix_ratio < 1.0:
            bsz = attention.shape[0]
            for b in range(bsz):
                vlen = int(attention[b].sum().item())
                cutoff = max(1, int(vlen * prefix_ratio))
                prefix_attention[b, cutoff:] = 0.0

        batch_in = {
            "coords": batch["coords"].to(device),
            "timestamps": batch["timestamps"].to(device),
            "attention_mask": prefix_attention,
            "start_ts": batch["start_ts"],
        }
        outputs, _, _, _, _ = rb.forward_backbone(
            batch_in,
            pack,
            device=device,
            max_len=max_len,
            mask=None,
        )
        pooled = masked_mean(outputs["step_hidden"], prefix_attention).detach().cpu()
        embeddings.append(pooled)
    return torch.cat(embeddings, dim=0)


def train_tte_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    epochs: int,
    lr: float,
    device: str,
) -> torch.nn.Linear:
    in_dim = train_x.shape[-1]
    head = torch.nn.Linear(in_dim, 1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    train_x = train_x.to(device)
    train_y = train_y.to(device).float()
    val_x = val_x.to(device)
    val_y = val_y.to(device).float()

    # Normalize targets
    y_mean = train_y.mean()
    y_std = train_y.std().clamp(min=1e-6)
    train_yn = (train_y - y_mean) / y_std
    val_yn = (val_y - y_mean) / y_std

    best_state = None
    best_val = float("inf")
    for _ in range(max(1, epochs)):
        perm = torch.randperm(train_x.shape[0], device=device)
        train_x = train_x[perm]
        train_yn = train_yn[perm]
        bs = 1024
        for start in range(0, train_x.shape[0], bs):
            xb = train_x[start : start + bs]
            yb = train_yn[start : start + bs]
            pred = head(xb).squeeze(-1)
            loss = torch.nn.functional.mse_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            val_pred = head(val_x).squeeze(-1)
            val_loss = torch.nn.functional.mse_loss(val_pred, val_yn).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
    if best_state:
        head.load_state_dict(best_state)
    head._y_mean = y_mean.cpu()
    head._y_std = y_std.cpu()
    return head


def evaluate_tte(
    head, x: torch.Tensor, y: torch.Tensor, device: str
) -> Dict[str, float]:
    x = x.to(device)
    y = y.float()
    with torch.no_grad():
        pred_n = head(x).squeeze(-1).cpu()
    pred = pred_n * head._y_std + head._y_mean
    error = pred - y
    abs_error = error.abs()
    mae = float(abs_error.mean().item())
    rmse = float(torch.sqrt((error**2).mean()).item())
    # MAPE: avoid division by zero
    nonzero = y.abs() > 1.0
    if nonzero.sum() > 0:
        mape = float((abs_error[nonzero] / y[nonzero].abs()).mean().item())
    else:
        mape = 0.0
    return {"mae_s": mae, "rmse_s": rmse, "mape": mape, "n": int(x.shape[0])}


def main():
    args = parse_args()
    rb.set_seed(args.seed)
    pack = rb.load_backbone(
        args.checkpoint, device=args.device, override_max_len=args.max_len
    )
    raw_records = rb.load_local_data(args.local_data)
    dataset = rb.FixedTrajectoryDataset(
        raw_records, max_len=args.max_len, sample_limit=args.sample_limit
    )

    travel_times = compute_travel_times(dataset)
    train_idx, val_idx, test_idx = rb.split_indices(
        dataset, mode="random", seed=args.seed
    )

    results = {
        "checkpoint": args.checkpoint,
        "dataset": args.local_data,
        "prefix_ratios": {},
    }

    for ratio in args.prefix_ratios:
        print(f"[TTE] prefix_ratio={ratio}")
        full_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=rb.collate_fixed,
        )
        all_emb = collect_prefix_embeddings(
            full_loader, pack, args.device, args.max_len, ratio
        )

        train_x = all_emb[train_idx]
        val_x = all_emb[val_idx]
        test_x = all_emb[test_idx]
        train_y = travel_times[train_idx]
        val_y = travel_times[val_idx]
        test_y = travel_times[test_idx]

        head = train_tte_probe(
            train_x,
            train_y,
            val_x,
            val_y,
            args.probe_epochs,
            args.probe_lr,
            args.device,
        )

        test_metrics = evaluate_tte(head, test_x, test_y, args.device)
        val_metrics = evaluate_tte(head, val_x, val_y, args.device)
        results["prefix_ratios"][str(ratio)] = {
            "val": val_metrics,
            "test": test_metrics,
        }
        print(f"  test: {test_metrics}")

    print(json.dumps(results, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
