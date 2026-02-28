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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_benchmarks import FixedTrajectoryDataset, collate_fixed, load_backbone  # noqa: E402
from scripts.run_benchmarks import forward_backbone, load_local_data, masked_mean  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Embedding retrieval evaluation (kNN) for trajectory models.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--local_data", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_limit", type=int, default=1000)
    parser.add_argument("--topk", type=int, default=5)
    # Default name matches scripts/collect_results.py glob: cache/embedding_retrieval_*.json
    parser.add_argument("--output", type=str, default="cache/embedding_retrieval_latest.json")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    if not Path(args.local_data).exists():
        raise FileNotFoundError(f"local_data not found: {args.local_data}")

    pack = load_backbone(args.checkpoint, device=args.device, override_max_len=args.max_len)
    records = load_local_data(args.local_data)
    dataset = FixedTrajectoryDataset(records, max_len=args.max_len, sample_limit=args.sample_limit)
    if len(dataset) < 10:
        raise RuntimeError(f"not enough samples for retrieval: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fixed,
    )

    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            outputs, _, t1, _, attention = forward_backbone(batch, pack, device=args.device, max_len=args.max_len, mask=None)
            pooled = masked_mean(outputs["step_hidden"], attention).detach().cpu()
            last_idx = attention.sum(dim=1).long().clamp(min=1) - 1
            dest_targets = t1.gather(1, last_idx.unsqueeze(1)).squeeze(1).detach().cpu()
            embeddings.append(pooled)
            labels.append(dest_targets)

    emb = torch.cat(embeddings, dim=0)
    lab = torch.cat(labels, dim=0)
    n = emb.shape[0]
    if n < 2:
        raise RuntimeError("not enough samples for retrieval evaluation")

    # Normalize for cosine similarity.
    emb = F.normalize(emb, dim=1)
    sim = emb @ emb.t()
    sim.fill_diagonal_(-1e9)
    k = min(args.topk, n - 1)
    topk = sim.topk(k, dim=1).indices
    top1 = (lab[topk[:, 0]] == lab).float().mean().item()
    top5 = (lab[topk] == lab.unsqueeze(1)).any(dim=1).float().mean().item()

    out = {
        "checkpoint": args.checkpoint,
        "dataset": args.local_data,
        "samples": int(n),
        "top1": float(top1),
        "top5": float(top5),
        "k": int(k),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved embedding retrieval results: {out_path}")


if __name__ == "__main__":
    main()
