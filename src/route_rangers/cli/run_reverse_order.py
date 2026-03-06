#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from route_rangers.cli.run_benchmarks import (  # noqa: E402
    FixedTrajectoryDataset,
    collate_fixed,
    load_backbone,
    load_local_data,
)
from route_rangers.cli.run_benchmarks import forward_backbone  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Reverse-order stress test.")
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
    # Default name matches scripts/collect_results.py glob: cache/reverse_order_*.json
    parser.add_argument("--output", type=str, default="cache/reverse_order_latest.json")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dest_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    if logits.numel() == 0:
        return {"top1": 0.0, "top5": 0.0, "nll": 0.0}
    loss = F.cross_entropy(logits.float(), targets, reduction="mean").item()
    pred = logits.argmax(dim=-1)
    top1 = (pred == targets).float().mean().item()
    k = min(5, logits.shape[-1])
    topk = logits.topk(k, dim=-1).indices
    top5 = (topk == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return {"top1": top1, "top5": top5, "nll": loss}


def reverse_batch(batch: dict) -> dict:
    coords = batch["coords"].clone()
    timestamps = batch["timestamps"].clone()
    attention = batch["attention_mask"].clone()
    bsz, seq_len = attention.shape
    for b in range(bsz):
        vlen = int(attention[b].sum().item())
        if vlen <= 1:
            continue
        coords[b, :vlen] = torch.flip(coords[b, :vlen], dims=[0])
        timestamps[b, :vlen] = torch.flip(timestamps[b, :vlen], dims=[0])
    return {
        "coords": coords,
        "timestamps": timestamps,
        "attention_mask": attention,
        "start_ts": batch.get("start_ts", torch.zeros((bsz,))).clone()
        if "start_ts" in batch
        else None,
    }


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
        raise RuntimeError(f"not enough samples for reverse-order test: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fixed,
    )

    total = {
        "orig": {"top1": 0.0, "top5": 0.0, "nll": 0.0, "n": 0},
        "rev": {"top1": 0.0, "top5": 0.0, "nll": 0.0, "n": 0},
    }

    for batch in loader:
        outputs, _, t1, _, attention = forward_backbone(
            batch, pack, device=args.device, max_len=args.max_len, mask=None
        )
        last_idx = attention.sum(dim=1).long().clamp(min=1) - 1
        dest_targets = t1.to(args.device).gather(1, last_idx.unsqueeze(1)).squeeze(1)
        metrics = _dest_metrics(outputs["dest_logits"], dest_targets)
        bs = dest_targets.shape[0]
        for k in ("top1", "top5", "nll"):
            total["orig"][k] += metrics[k] * bs
        total["orig"]["n"] += bs

        rev = reverse_batch(batch)
        outputs_r, _, t1_r, _, attention_r = forward_backbone(
            rev, pack, device=args.device, max_len=args.max_len, mask=None
        )
        last_idx_r = attention_r.sum(dim=1).long().clamp(min=1) - 1
        dest_targets_r = (
            t1_r.to(args.device).gather(1, last_idx_r.unsqueeze(1)).squeeze(1)
        )
        metrics_r = _dest_metrics(outputs_r["dest_logits"], dest_targets_r)
        bs_r = dest_targets_r.shape[0]
        for k in ("top1", "top5", "nll"):
            total["rev"][k] += metrics_r[k] * bs_r
        total["rev"]["n"] += bs_r

    def _avg(m):
        if m["n"] == 0:
            return {"top1": 0.0, "top5": 0.0, "nll": 0.0}
        return {k: m[k] / m["n"] for k in ("top1", "top5", "nll")}

    orig = _avg(total["orig"])
    rev = _avg(total["rev"])
    out = {
        "checkpoint": args.checkpoint,
        "dataset": args.local_data,
        "samples": int(total["orig"]["n"]),
        "original": orig,
        "reversed": rev,
        "delta": {k: rev[k] - orig[k] for k in orig},
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved reverse-order results: {out_path}")


if __name__ == "__main__":
    main()
