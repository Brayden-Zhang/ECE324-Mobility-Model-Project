#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

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
from route_rangers.cli.run_benchmarks import forward_backbone, masked_mean  # noqa: E402


def _derangement(n: int, device: torch.device) -> torch.Tensor:
    if n <= 1:
        return torch.arange(n, device=device)
    perm = torch.randperm(n, device=device)
    fixed = perm == torch.arange(n, device=device)
    # Resolve fixed points via one-step cyclic shift among fixed indices.
    if fixed.any():
        idx = fixed.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 1:
            i = int(idx.item())
            j = 0 if i != 0 else 1
            perm[i], perm[j] = perm[j].clone(), perm[i].clone()
        else:
            rolled = idx.roll(shifts=1)
            perm[idx] = perm[rolled]
    return perm


def parse_args():
    parser = argparse.ArgumentParser(description="Embedding change detection analog.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--local_data", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_limit", type=int, default=1000)
    parser.add_argument(
        "--split_ratio", type=float, default=0.5, help="fraction for prefix vs suffix"
    )
    # Default name matches collect_results.py glob: cache/change_detection_*.json
    parser.add_argument(
        "--output", type=str, default="cache/change_detection_latest.json"
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auc_from_scores(pos: np.ndarray, neg: np.ndarray) -> float:
    # Mann-Whitney U / rank-based AUC
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores)) + 1
    pos_ranks = ranks[labels == 1]
    u = pos_ranks.sum() - len(pos_ranks) * (len(pos_ranks) + 1) / 2
    auc = u / (len(pos) * len(neg))
    return float(auc)


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
        raise RuntimeError(f"not enough samples for change detection: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fixed,
    )

    prefix_embs: List[torch.Tensor] = []
    suffix_embs: List[torch.Tensor] = []

    for batch in loader:
        attention = batch["attention_mask"].to(args.device)
        vlen = attention.sum(dim=1).long()
        split = torch.clamp((vlen.float() * args.split_ratio).ceil().long(), min=2)
        arange = torch.arange(attention.shape[1], device=args.device).unsqueeze(0)
        prefix_mask = arange < split.unsqueeze(1)
        suffix_mask = arange >= (vlen.unsqueeze(1) - split.unsqueeze(1))

        prefix_batch = dict(batch)
        prefix_batch["attention_mask"] = attention * prefix_mask.float()
        outputs_p, _, _, _, att_p = forward_backbone(
            prefix_batch, pack, device=args.device, max_len=args.max_len, mask=None
        )
        p_step = masked_mean(outputs_p["step_hidden"], att_p)
        p_mid = (
            masked_mean(outputs_p["mid_hidden"], outputs_p["mid_mask"])
            if outputs_p["mid_hidden"].shape[1] > 0
            else torch.zeros_like(p_step)
        )
        prefix_embs.append(torch.cat([p_step, p_mid], dim=-1).detach().cpu())

        suffix_batch = dict(batch)
        suffix_batch["attention_mask"] = attention * suffix_mask.float()
        outputs_s, _, _, _, att_s = forward_backbone(
            suffix_batch, pack, device=args.device, max_len=args.max_len, mask=None
        )
        s_step = masked_mean(outputs_s["step_hidden"], att_s)
        s_mid = (
            masked_mean(outputs_s["mid_hidden"], outputs_s["mid_mask"])
            if outputs_s["mid_hidden"].shape[1] > 0
            else torch.zeros_like(s_step)
        )
        suffix_embs.append(torch.cat([s_step, s_mid], dim=-1).detach().cpu())

    prefix = torch.cat(prefix_embs, dim=0)
    suffix = torch.cat(suffix_embs, dim=0)
    n = prefix.shape[0]
    if n < 2:
        raise RuntimeError("not enough samples for change detection")

    prefix = F.normalize(prefix, dim=1)
    suffix = F.normalize(suffix, dim=1)

    # Positive: same-trajectory prefix vs suffix
    pos = 1.0 - (prefix * suffix).sum(dim=1).numpy()

    # Negatives: random derangements (no self-pairs), aggregated across rounds
    neg_list = []
    rounds = 3
    for _ in range(rounds):
        perm = _derangement(n, device=prefix.device)
        neg_round = 1.0 - (prefix * suffix[perm]).sum(dim=1).numpy()
        neg_list.append(neg_round)
    neg = np.concatenate(neg_list)

    pos_sim = 1.0 - pos
    neg_sim = 1.0 - neg
    out = {
        "checkpoint": args.checkpoint,
        "dataset": args.local_data,
        "samples": int(n),
        "pos_mean_dist": float(np.mean(pos)),
        "neg_mean_dist": float(np.mean(neg)),
        "auc": auc_from_scores(pos_sim, neg_sim),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved change detection results: {out_path}")


if __name__ == "__main__":
    main()
