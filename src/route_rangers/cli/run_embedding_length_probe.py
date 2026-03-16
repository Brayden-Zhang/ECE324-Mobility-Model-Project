#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from route_rangers.cli import run_benchmarks as rb
from route_rangers.cli import run_length_sensitivity as ls
from route_rangers.eval.length_utils import bin_name_for_length, parse_bins


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe whether embeddings encode trajectory-length category."
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
        "--split_mode", type=str, default="both", choices=["both", "random", "temporal"]
    )
    parser.add_argument("--length_bins", type=str, default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="cache/embedding_length_probe.json")
    return parser.parse_args()


def _macro_f1(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    pred = logits.argmax(dim=-1)
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (targets == c)).sum().item()
        fp = ((pred == c) & (targets != c)).sum().item()
        fn = ((pred != c) & (targets == c)).sum().item()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1s.append(f1)
    return float(np.mean(f1s))


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def collect_embeddings(dataset, indices: List[int], pack, args):
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ls.collate_fixed,
    )
    xs = []
    for batch in loader:
        batch_bench = {
            "coords": batch["coords"],
            "timestamps": batch["timestamps"],
            "attention_mask": batch["attention_mask"],
            "start_ts": torch.zeros((batch["coords"].shape[0],), dtype=torch.float32),
        }
        outputs, _, _, _, attention = rb.forward_backbone(
            batch_bench, pack, device=args.device, max_len=args.max_len, mask=None
        )
        pooled_step = _masked_mean(outputs["step_hidden"], attention)
        pooled_mid = (
            _masked_mean(outputs["mid_hidden"], outputs["mid_mask"])
            if outputs["mid_hidden"].shape[1] > 0
            else torch.zeros_like(pooled_step)
        )
        x = torch.cat([pooled_step, pooled_mid], dim=-1).detach().cpu()
        xs.append(x)
    return torch.cat(xs, dim=0) if xs else torch.zeros((0, int(pack.ckpt_args["embed_dim"]) * 2))


def train_probe(train_x, train_y, val_x, val_y, args):
    num_classes = int(train_y.max().item() + 1)
    head = torch.nn.Linear(train_x.shape[-1], num_classes).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tx = train_x.to(args.device)
    ty = train_y.to(args.device)
    vx = val_x.to(args.device)
    vy = val_y.to(args.device)

    best = None
    best_val = -1.0
    for _ in range(max(1, args.epochs)):
        head.train()
        logits = head(tx)
        loss = torch.nn.functional.cross_entropy(logits, ty)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(vx)
            val_acc = float((val_logits.argmax(dim=-1) == vy).float().mean().item())
        if val_acc > best_val:
            best_val = val_acc
            best = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    if best is not None:
        head.load_state_dict(best)
    return head


def eval_probe(head, x, y):
    with torch.no_grad():
        logits = head(x)
    acc = float((logits.argmax(dim=-1) == y).float().mean().item())
    num_classes = int(y.max().item() + 1) if y.numel() > 0 else 1
    return {
        "accuracy": acc,
        "macro_f1": _macro_f1(logits, y, num_classes),
        "n": int(y.shape[0]),
    }


def run_split(dataset, split_idx, labels, pack, args):
    train_idx, val_idx, test_idx = split_idx
    train_x = collect_embeddings(dataset, train_idx, pack, args)
    val_x = collect_embeddings(dataset, val_idx, pack, args)
    test_x = collect_embeddings(dataset, test_idx, pack, args)

    train_y = torch.from_numpy(labels[train_idx]).long()
    val_y = torch.from_numpy(labels[val_idx]).long()
    test_y = torch.from_numpy(labels[test_idx]).long()

    head = train_probe(train_x, train_y, val_x, val_y, args).to(args.device)
    return {
        "train": eval_probe(head, train_x.to(args.device), train_y.to(args.device)),
        "val": eval_probe(head, val_x.to(args.device), val_y.to(args.device)),
        "test": eval_probe(head, test_x.to(args.device), test_y.to(args.device)),
    }


def main():
    args = parse_args()
    rb.set_seed(args.seed)
    if not Path(args.local_data).exists():
        raise FileNotFoundError(f"local_data not found: {args.local_data}")

    pack = rb.load_backbone(args.checkpoint, device=args.device, override_max_len=args.max_len)
    records = ls.load_local_data(args.local_data)
    dataset = ls.FixedTrajectoryDataset(records, max_len=args.max_len, sample_limit=args.sample_limit)
    if len(dataset) < 10:
        raise RuntimeError(f"not enough samples: {len(dataset)}")

    lengths = np.asarray([int(s["raw_length"]) for s in dataset.samples], dtype=np.int64)
    bins, strategy = parse_bins(args.length_bins, lengths)
    labels = np.asarray(
        [
            0 if bin_name_for_length(int(s["raw_length"]), bins) == "short"
            else 1 if bin_name_for_length(int(s["raw_length"]), bins) == "medium"
            else 2
            for s in dataset.samples
        ],
        dtype=np.int64,
    )

    split_modes = ["random", "temporal"] if args.split_mode == "both" else [args.split_mode]
    out = {
        "checkpoint": args.checkpoint,
        "dataset": args.local_data,
        "samples": len(dataset),
        "bins": bins.tolist(),
        "bin_strategy": strategy,
        "splits": {},
    }
    for mode in split_modes:
        idx = rb.split_indices(dataset, mode=mode, seed=args.seed)
        out["splits"][mode] = run_split(dataset, idx, labels, pack, args)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved embedding length probe results: {out_path}")


if __name__ == "__main__":
    main()
