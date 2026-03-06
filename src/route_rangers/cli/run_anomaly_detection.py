#!/usr/bin/env python3
"""
Anomaly detection downstream task for trajectory foundation models.

Strategy: embeddings of "normal" trajectory prefixes cluster tightly.
"Anomalous" trajectories (simulated via large coordinate noise, reversed order,
or random point swaps) should produce embeddings far from the normal cluster.

Metrics: AUROC and PR-AUC for distinguishing normal vs anomalous trajectories.
"""
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from route_rangers.cli import run_benchmarks as rb


def parse_args():
    parser = argparse.ArgumentParser(description="Anomaly detection downstream task")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--local_data", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_limit", type=int, default=2000)
    parser.add_argument("--anomaly_ratio", type=float, default=0.2, help="Fraction of trajs to corrupt")
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def create_anomalous_batch(batch: dict, anomaly_type: str, rng: np.random.RandomState) -> dict:
    """Create anomalous versions of a trajectory batch."""
    new_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    coords = new_batch["coords"].clone()
    attention = new_batch["attention_mask"]

    if anomaly_type == "noise":
        # Add large coordinate noise (~500m in degrees)
        noise = torch.randn_like(coords) * 0.005  # ~500m
        coords = coords + noise * attention.unsqueeze(-1)
    elif anomaly_type == "reverse":
        # Reverse the trajectory order
        bsz = coords.shape[0]
        for b in range(bsz):
            vlen = int(attention[b].sum().item())
            if vlen > 1:
                coords[b, :vlen] = coords[b, :vlen].flip(0)
                new_batch["timestamps"][b, :vlen] = new_batch["timestamps"][b, :vlen].flip(0)
    elif anomaly_type == "swap":
        # Randomly swap 30% of points
        bsz = coords.shape[0]
        for b in range(bsz):
            vlen = int(attention[b].sum().item())
            if vlen <= 2:
                continue
            n_swap = max(1, int(vlen * 0.3))
            idx = rng.choice(vlen, size=n_swap * 2, replace=True)
            for i in range(0, len(idx) - 1, 2):
                a, b_idx = int(idx[i]), int(idx[i + 1])
                coords[b, a], coords[b, b_idx] = coords[b, b_idx].clone(), coords[b, a].clone()
    elif anomaly_type == "detour":
        # Shift middle 30% of trajectory by ~2km
        bsz = coords.shape[0]
        for b in range(bsz):
            vlen = int(attention[b].sum().item())
            if vlen <= 4:
                continue
            start = vlen // 3
            end = 2 * vlen // 3
            coords[b, start:end, 0] += 0.02  # ~2km lat shift
            coords[b, start:end, 1] += 0.02  # ~2km lon shift

    new_batch["coords"] = coords
    return new_batch


def collect_embeddings(
    loader: DataLoader,
    pack: rb.BackbonePack,
    device: str,
    max_len: int,
) -> torch.Tensor:
    """Get pooled step embeddings for all trajectories."""
    embeddings = []
    for batch in loader:
        outputs, _, _, _, attention = rb.forward_backbone(
            batch, pack, device=device, max_len=max_len, mask=None,
        )
        pooled = masked_mean(outputs["step_hidden"], attention).detach().cpu()
        embeddings.append(pooled)
    return torch.cat(embeddings, dim=0)


def collect_anomalous_embeddings(
    loader: DataLoader,
    pack: rb.BackbonePack,
    device: str,
    max_len: int,
    anomaly_type: str,
    rng: np.random.RandomState,
) -> torch.Tensor:
    """Get embeddings for anomalous versions of trajectories."""
    embeddings = []
    for batch in loader:
        anom_batch = create_anomalous_batch(batch, anomaly_type, rng)
        # Move to device
        anom_batch_dev = {
            "coords": anom_batch["coords"].to(device),
            "timestamps": anom_batch["timestamps"].to(device),
            "attention_mask": anom_batch["attention_mask"].to(device),
            "start_ts": anom_batch["start_ts"],
        }
        outputs, _, _, _, attention = rb.forward_backbone(
            anom_batch_dev, pack, device=device, max_len=max_len, mask=None,
        )
        pooled = masked_mean(outputs["step_hidden"], attention).detach().cpu()
        embeddings.append(pooled)
    return torch.cat(embeddings, dim=0)


def compute_anomaly_scores(normal_emb: torch.Tensor, test_emb: torch.Tensor) -> torch.Tensor:
    """Compute anomaly score as distance from mean normal embedding."""
    centroid = normal_emb.mean(dim=0, keepdim=True)
    dists = torch.sqrt(((test_emb - centroid) ** 2).sum(dim=-1))
    return dists


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute AUROC and PR-AUC."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    auroc = float(roc_auc_score(labels, scores))
    prauc = float(average_precision_score(labels, scores))
    return {"auroc": auroc, "pr_auc": prauc}


def main():
    args = parse_args()
    rb.set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    pack = rb.load_backbone(args.checkpoint, device=args.device, override_max_len=args.max_len)
    raw_records = rb.load_local_data(args.local_data)
    dataset = rb.FixedTrajectoryDataset(raw_records, max_len=args.max_len, sample_limit=args.sample_limit)

    train_idx, val_idx, test_idx = rb.split_indices(dataset, mode="random", seed=args.seed)

    # Get normal embeddings from train set
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=rb.collate_fixed,
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_idx),
        batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=rb.collate_fixed,
    )

    print("[anomaly] Computing normal train embeddings...")
    normal_emb = collect_embeddings(train_loader, pack, args.device, args.max_len)

    print("[anomaly] Computing normal test embeddings...")
    test_emb_normal = collect_embeddings(test_loader, pack, args.device, args.max_len)

    anomaly_types = ["noise", "reverse", "swap", "detour"]
    results = {
        "checkpoint": args.checkpoint,
        "dataset": args.local_data,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "anomaly_types": {},
    }

    for atype in anomaly_types:
        print(f"[anomaly] Computing {atype} anomalous embeddings...")
        test_emb_anom = collect_anomalous_embeddings(
            test_loader, pack, args.device, args.max_len, atype, rng,
        )

        # Normal test gets label 0, anomalous gets label 1
        all_emb = torch.cat([test_emb_normal, test_emb_anom], dim=0)
        labels = np.concatenate([
            np.zeros(test_emb_normal.shape[0]),
            np.ones(test_emb_anom.shape[0]),
        ])

        scores = compute_anomaly_scores(normal_emb, all_emb).numpy()
        metrics = compute_metrics(scores, labels)
        results["anomaly_types"][atype] = metrics
        print(f"  {atype}: {metrics}")

    # Combined: mix all anomaly types
    all_anom_embs = []
    for atype in anomaly_types:
        anom_emb = collect_anomalous_embeddings(
            test_loader, pack, args.device, args.max_len, atype, rng,
        )
        all_anom_embs.append(anom_emb)
    combined_anom = torch.cat(all_anom_embs, dim=0)
    # Subsample to match normal test size
    if combined_anom.shape[0] > test_emb_normal.shape[0]:
        idx = rng.choice(combined_anom.shape[0], size=test_emb_normal.shape[0], replace=False)
        combined_anom = combined_anom[idx]
    all_emb = torch.cat([test_emb_normal, combined_anom], dim=0)
    labels = np.concatenate([
        np.zeros(test_emb_normal.shape[0]),
        np.ones(combined_anom.shape[0]),
    ])
    scores = compute_anomaly_scores(normal_emb, all_emb).numpy()
    combined_metrics = compute_metrics(scores, labels)
    results["combined"] = combined_metrics
    print(f"  combined: {combined_metrics}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
