#!/usr/bin/env python3
"""
Similarity search / retrieval downstream task.

Given a query trajectory, retrieve the most similar trajectories from a database
using the model's learned embedding space. Evaluate using:
1. Geographic consistency: retrieved trajectories should be in similar geographic areas
2. Temporal consistency: similar departure times / durations
3. Self-retrieval: augmented versions of the same trajectory should be top-retrieved

This is a key foundation-model eval: good embeddings cluster similar trajectories.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_benchmarks as rb  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Similarity retrieval downstream task")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--local_data", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_limit", type=int, default=1000)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    return R * c


def get_traj_summaries(dataset: rb.FixedTrajectoryDataset) -> Dict[str, np.ndarray]:
    """Get origin, destination, mean position for each trajectory."""
    origins = []
    destinations = []
    means = []
    durations = []
    for sample in dataset.samples:
        coords = sample["coords"].numpy()
        ts = sample["timestamps"].numpy()
        attn = sample["attention_mask"].numpy()
        vlen = int(attn.sum())
        if vlen <= 0:
            origins.append([0.0, 0.0])
            destinations.append([0.0, 0.0])
            means.append([0.0, 0.0])
            durations.append(0.0)
            continue
        origins.append(coords[0, :2].tolist())
        destinations.append(coords[vlen - 1, :2].tolist())
        means.append(coords[:vlen, :2].mean(axis=0).tolist())
        durations.append(float(ts[vlen - 1] - ts[0]))
    return {
        "origins": np.array(origins),
        "destinations": np.array(destinations),
        "means": np.array(means),
        "durations": np.array(durations),
    }


def collect_embeddings(
    dataset: rb.FixedTrajectoryDataset,
    pack: rb.BackbonePack,
    device: str,
    max_len: int,
    batch_size: int,
) -> torch.Tensor:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=rb.collate_fixed,
    )
    embeddings = []
    for batch in loader:
        outputs, _, _, _, attention = rb.forward_backbone(
            batch, pack, device=device, max_len=max_len, mask=None,
        )
        pooled = masked_mean(outputs["step_hidden"], attention).detach().cpu()
        embeddings.append(pooled)
    return torch.cat(embeddings, dim=0)


def collect_noisy_embeddings(
    dataset: rb.FixedTrajectoryDataset,
    pack: rb.BackbonePack,
    device: str,
    max_len: int,
    batch_size: int,
    noise_std: float = 0.001,
    seed: int = 42,
) -> torch.Tensor:
    """Collect embeddings from lightly noised versions of trajectories for self-retrieval."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=rb.collate_fixed,
    )
    torch.manual_seed(seed + 99)
    embeddings = []
    for batch in loader:
        coords = batch["coords"].to(device)
        noise = torch.randn_like(coords) * noise_std
        noisy_coords = coords + noise * batch["attention_mask"].to(device).unsqueeze(-1)
        noisy_batch = {
            "coords": noisy_coords,
            "timestamps": batch["timestamps"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "start_ts": batch["start_ts"],
        }
        outputs, _, _, _, attention = rb.forward_backbone(
            noisy_batch, pack, device=device, max_len=max_len, mask=None,
        )
        pooled = masked_mean(outputs["step_hidden"], attention).detach().cpu()
        embeddings.append(pooled)
    return torch.cat(embeddings, dim=0)


def knn_retrieval(query_emb: torch.Tensor, db_emb: torch.Tensor, k: int) -> torch.Tensor:
    """Return top-k indices from db for each query (cosine similarity)."""
    q_norm = torch.nn.functional.normalize(query_emb, dim=-1)
    d_norm = torch.nn.functional.normalize(db_emb, dim=-1)
    sim = q_norm @ d_norm.t()
    _, topk = sim.topk(k, dim=-1)
    return topk


def evaluate_geographic_consistency(
    topk_indices: torch.Tensor,
    summaries: Dict[str, np.ndarray],
    top_k: int,
) -> Dict[str, float]:
    """Evaluate if retrieved trajectories are geographically close to query."""
    n = topk_indices.shape[0]
    origin_dists = []
    dest_dists = []
    mean_dists = []

    for i in range(n):
        q_origin = summaries["origins"][i]
        q_dest = summaries["destinations"][i]
        q_mean = summaries["means"][i]
        for j in range(min(top_k, topk_indices.shape[1])):
            idx = int(topk_indices[i, j].item())
            if idx == i:
                continue
            r_orig = summaries["origins"][idx]
            r_dest = summaries["destinations"][idx]
            r_mean = summaries["means"][idx]
            origin_dists.append(haversine_m(q_origin[0], q_origin[1], r_orig[0], r_orig[1]))
            dest_dists.append(haversine_m(q_dest[0], q_dest[1], r_dest[0], r_dest[1]))
            mean_dists.append(haversine_m(q_mean[0], q_mean[1], r_mean[0], r_mean[1]))

    return {
        "origin_mean_dist_km": float(np.mean(origin_dists) / 1000) if origin_dists else 0.0,
        "dest_mean_dist_km": float(np.mean(dest_dists) / 1000) if dest_dists else 0.0,
        "spatial_mean_dist_km": float(np.mean(mean_dists) / 1000) if mean_dists else 0.0,
    }


def evaluate_self_retrieval(
    clean_emb: torch.Tensor,
    noisy_emb: torch.Tensor,
    top_k: int,
) -> Dict[str, float]:
    """Evaluate if noisy version retrieves the clean version as top-1."""
    topk = knn_retrieval(noisy_emb, clean_emb, top_k + 1)
    n = topk.shape[0]
    top1_hits = 0
    topk_hits = 0
    for i in range(n):
        if int(topk[i, 0].item()) == i:
            top1_hits += 1
        for j in range(min(top_k, topk.shape[1])):
            if int(topk[i, j].item()) == i:
                topk_hits += 1
                break
    return {
        "self_retrieval_top1": top1_hits / max(1, n),
        f"self_retrieval_top{top_k}": topk_hits / max(1, n),
    }


def main():
    args = parse_args()
    rb.set_seed(args.seed)

    pack = rb.load_backbone(args.checkpoint, device=args.device, override_max_len=args.max_len)
    raw_records = rb.load_local_data(args.local_data)
    dataset = rb.FixedTrajectoryDataset(raw_records, max_len=args.max_len, sample_limit=args.sample_limit)

    print("[retrieval] Computing trip summaries...")
    summaries = get_traj_summaries(dataset)

    print("[retrieval] Computing clean embeddings...")
    clean_emb = collect_embeddings(dataset, pack, args.device, args.max_len, args.batch_size)

    print("[retrieval] Computing noisy embeddings for self-retrieval...")
    noisy_emb = collect_noisy_embeddings(
        dataset, pack, args.device, args.max_len, args.batch_size,
        noise_std=0.001, seed=args.seed,
    )

    print("[retrieval] Running kNN retrieval...")
    topk = knn_retrieval(clean_emb, clean_emb, args.top_k + 1)
    # Remove self-match (first column is usually self)
    topk_no_self = topk[:, 1:]

    geo_metrics = evaluate_geographic_consistency(topk_no_self, summaries, args.top_k)
    self_metrics = evaluate_self_retrieval(clean_emb, noisy_emb, args.top_k)

    results = {
        "checkpoint": args.checkpoint,
        "dataset": args.local_data,
        "n": len(dataset),
        "top_k": args.top_k,
        "geographic_consistency": geo_metrics,
        "self_retrieval": self_metrics,
    }

    print(json.dumps(results, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
