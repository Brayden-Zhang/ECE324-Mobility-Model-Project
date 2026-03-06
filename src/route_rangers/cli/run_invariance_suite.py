#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from route_rangers.cli.run_benchmarks import FixedTrajectoryDataset, collate_fixed, load_backbone, load_local_data  # noqa: E402
from route_rangers.cli.run_benchmarks import forward_backbone  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run invariance/robustness suite for trajectory models.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--local_data", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument("--prefix_ratios", type=str, default="0.25,0.5,0.75,1.0")
    parser.add_argument("--time_shifts_sec", type=str, default="0,43200,86400")
    parser.add_argument("--downsample_rates", type=str, default="0.5,0.25")
    parser.add_argument("--output", type=str, default="cache/invariance_suite.json")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(float(x.strip())) for x in text.split(",") if x.strip()]


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


def evaluate_prefixes(loader, pack, device, max_len, ratios: List[float]):
    results = {}
    for ratio in ratios:
        ratio_key = f"{ratio:.2f}"
        total = {"top1": 0.0, "top5": 0.0, "nll": 0.0, "n": 0}
        for batch in loader:
            full_attention = batch["attention_mask"].to(device)
            vlen = full_attention.sum(dim=1).long()
            prefix_len = torch.clamp((vlen.float() * ratio).ceil().long(), min=2)
            arange = torch.arange(full_attention.shape[1], device=device).unsqueeze(0)
            prefix_mask = arange < prefix_len.unsqueeze(1)
            prefix_attention = full_attention * prefix_mask.float()

            prefix_batch = dict(batch)
            prefix_batch["attention_mask"] = prefix_attention

            outputs, _, t1, _, _ = forward_backbone(prefix_batch, pack, device=device, max_len=max_len, mask=None)
            last_idx = full_attention.sum(dim=1).long().clamp(min=1) - 1
            dest_targets = t1.to(device).gather(1, last_idx.unsqueeze(1)).squeeze(1)
            metrics = _dest_metrics(outputs["dest_logits"], dest_targets)
            bs = dest_targets.shape[0]
            total["top1"] += metrics["top1"] * bs
            total["top5"] += metrics["top5"] * bs
            total["nll"] += metrics["nll"] * bs
            total["n"] += bs
        if total["n"] > 0:
            results[ratio_key] = {
                "dest_top1": total["top1"] / total["n"],
                "dest_top5": total["top5"] / total["n"],
                "dest_nll": total["nll"] / total["n"],
                "samples": total["n"],
            }
    return results


def evaluate_time_shifts(loader, pack, device, max_len, shifts: List[int]):
    results = {}
    for shift in shifts:
        total = {"top1": 0.0, "top5": 0.0, "nll": 0.0, "n": 0}
        for batch in loader:
            attention = batch["attention_mask"].to(device)
            timestamps = batch["timestamps"].to(device)
            shifted = timestamps + attention * float(shift)
            shift_batch = dict(batch)
            shift_batch["timestamps"] = shifted
            outputs, _, t1, _, _ = forward_backbone(shift_batch, pack, device=device, max_len=max_len, mask=None)
            last_idx = attention.sum(dim=1).long().clamp(min=1) - 1
            dest_targets = t1.to(device).gather(1, last_idx.unsqueeze(1)).squeeze(1)
            metrics = _dest_metrics(outputs["dest_logits"], dest_targets)
            bs = dest_targets.shape[0]
            total["top1"] += metrics["top1"] * bs
            total["top5"] += metrics["top5"] * bs
            total["nll"] += metrics["nll"] * bs
            total["n"] += bs
        if total["n"] > 0:
            results[str(shift)] = {
                "dest_top1": total["top1"] / total["n"],
                "dest_top5": total["top5"] / total["n"],
                "dest_nll": total["nll"] / total["n"],
                "samples": total["n"],
            }
    return results


def evaluate_downsample(loader, pack, device, max_len, rates: List[float]):
    results = {}
    for rate in rates:
        if rate <= 0 or rate > 1:
            continue
        stride = max(1, int(round(1.0 / rate)))
        total = {"top1": 0.0, "top5": 0.0, "nll": 0.0, "n": 0}
        for batch in loader:
            attention = batch["attention_mask"].to(device)
            idx = (torch.arange(attention.shape[1], device=device) % stride) == 0
            down_mask = idx.unsqueeze(0).float()
            down_attention = attention * down_mask
            down_batch = dict(batch)
            down_batch["attention_mask"] = down_attention
            outputs, _, t1, _, _ = forward_backbone(down_batch, pack, device=device, max_len=max_len, mask=None)
            last_idx = attention.sum(dim=1).long().clamp(min=1) - 1
            dest_targets = t1.to(device).gather(1, last_idx.unsqueeze(1)).squeeze(1)
            metrics = _dest_metrics(outputs["dest_logits"], dest_targets)
            bs = dest_targets.shape[0]
            total["top1"] += metrics["top1"] * bs
            total["top5"] += metrics["top5"] * bs
            total["nll"] += metrics["nll"] * bs
            total["n"] += bs
        if total["n"] > 0:
            results[f"{rate:.2f}"] = {
                "dest_top1": total["top1"] / total["n"],
                "dest_top5": total["top5"] / total["n"],
                "dest_nll": total["nll"] / total["n"],
                "samples": total["n"],
                "stride": stride,
            }
    return results


def main():
    args = parse_args()
    set_seed(args.seed)

    if not Path(args.local_data).exists():
        raise FileNotFoundError(f"local_data not found: {args.local_data}")

    pack = load_backbone(args.checkpoint, device=args.device, override_max_len=args.max_len)
    records = load_local_data(args.local_data)
    dataset = FixedTrajectoryDataset(records, max_len=args.max_len, sample_limit=args.sample_limit)
    if len(dataset) < 10:
        raise RuntimeError(f"not enough samples for invariance suite: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fixed,
    )

    prefix_ratios = _parse_floats(args.prefix_ratios)
    time_shifts = _parse_ints(args.time_shifts_sec)
    down_rates = _parse_floats(args.downsample_rates)

    results = {
        "checkpoint": args.checkpoint,
        "dataset": args.local_data,
        "samples": len(dataset),
        "prefix_destination": evaluate_prefixes(loader, pack, args.device, args.max_len, prefix_ratios),
        "time_shift_destination": evaluate_time_shifts(loader, pack, args.device, args.max_len, time_shifts),
        "downsample_destination": evaluate_downsample(loader, pack, args.device, args.max_len, down_rates),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved invariance suite results: {out}")


if __name__ == "__main__":
    main()
