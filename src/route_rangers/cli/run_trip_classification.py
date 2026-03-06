#!/usr/bin/env python3
"""
Trip purpose / mode classification downstream task.

Since we don't have ground-truth labels, we use unsupervised proxy tasks:
1. Speed-class proxy: classify trajectories by average speed bucket
   (walking < 5 km/h, cycling 5-20, driving 20-80, highway > 80)
2. Duration-class proxy: classify by trip duration bucket
3. Distance-class proxy: classify by total displacement bucket

These test whether the model's embeddings capture semantically meaningful
trajectory properties that correlate with real-world trip attributes.
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
    parser = argparse.ArgumentParser(
        description="Trip classification downstream probes"
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
    parser.add_argument("--probe_epochs", type=int, default=10)
    parser.add_argument("--probe_lr", type=float, default=2e-3)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def haversine_m(lat1, lon1, lat2, lon2):
    """Haversine distance in meters between two points (degrees)."""
    R = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    return R * c


def compute_trip_attributes(
    dataset: rb.FixedTrajectoryDataset,
) -> Dict[str, np.ndarray]:
    """Compute speed, duration, distance for each trajectory."""
    speeds = []
    durations = []
    distances = []

    for sample in dataset.samples:
        coords = sample["coords"].numpy()
        ts = sample["timestamps"].numpy()
        attn = sample["attention_mask"].numpy()
        vlen = int(attn.sum())

        if vlen <= 1:
            speeds.append(0.0)
            durations.append(0.0)
            distances.append(0.0)
            continue

        # Total distance (haversine)
        total_dist = 0.0
        for i in range(1, vlen):
            d = haversine_m(
                coords[i - 1, 0], coords[i - 1, 1], coords[i, 0], coords[i, 1]
            )
            total_dist += d

        # Total duration
        duration = max(0.0, float(ts[vlen - 1] - ts[0]))
        durations.append(duration)
        distances.append(total_dist)

        # Average speed in km/h
        if duration > 0:
            speeds.append((total_dist / 1000.0) / (duration / 3600.0))
        else:
            speeds.append(0.0)

    return {
        "speed_kmh": np.array(speeds, dtype=np.float32),
        "duration_s": np.array(durations, dtype=np.float32),
        "distance_m": np.array(distances, dtype=np.float32),
    }


def bucketize(values: np.ndarray, thresholds: List[float]) -> np.ndarray:
    """Assign each value to a bucket index based on thresholds."""
    labels = np.zeros(len(values), dtype=np.int64)
    for i, t in enumerate(thresholds):
        labels[values > t] = i + 1
    return labels


def collect_embeddings(
    dataset: rb.FixedTrajectoryDataset,
    pack: rb.BackbonePack,
    device: str,
    max_len: int,
    batch_size: int,
) -> torch.Tensor:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=rb.collate_fixed,
    )
    embeddings = []
    for batch in loader:
        outputs, _, _, _, attention = rb.forward_backbone(
            batch,
            pack,
            device=device,
            max_len=max_len,
            mask=None,
        )
        pooled = masked_mean(outputs["step_hidden"], attention).detach().cpu()
        embeddings.append(pooled)
    return torch.cat(embeddings, dim=0)


def run_classification_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    num_classes: int,
    epochs: int,
    lr: float,
    device: str,
) -> Dict[str, float]:
    return rb.train_probe(
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        num_classes=num_classes,
        epochs=epochs,
        lr=lr,
        weight_decay=1e-4,
        batch_size=1024,
        device=device,
    )


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

    print("[trip_class] Computing trip attributes...")
    attrs = compute_trip_attributes(dataset)

    print("[trip_class] Computing embeddings...")
    all_emb = collect_embeddings(
        dataset, pack, args.device, args.max_len, args.batch_size
    )

    train_idx, val_idx, test_idx = rb.split_indices(
        dataset, mode="random", seed=args.seed
    )
    train_x = all_emb[train_idx]
    val_x = all_emb[val_idx]
    test_x = all_emb[test_idx]

    # Define classification tasks
    tasks = {
        "speed_mode": {
            "values": attrs["speed_kmh"],
            "thresholds": [5.0, 20.0, 80.0],  # walk / cycle / drive / highway
            "names": ["walk", "cycle", "drive", "highway"],
        },
        "duration_bucket": {
            "values": attrs["duration_s"],
            "thresholds": [300.0, 1800.0, 7200.0],  # <5min / 5-30min / 30min-2h / >2h
            "names": ["very_short", "short", "medium", "long"],
        },
        "distance_bucket": {
            "values": attrs["distance_m"],
            "thresholds": [1000.0, 5000.0, 20000.0],  # <1km / 1-5km / 5-20km / >20km
            "names": ["local", "neighborhood", "city", "regional"],
        },
    }

    results = {"checkpoint": args.checkpoint, "dataset": args.local_data, "tasks": {}}

    for task_name, task_cfg in tasks.items():
        labels = bucketize(task_cfg["values"], task_cfg["thresholds"])
        num_classes = len(task_cfg["thresholds"]) + 1

        # Check class distribution
        unique, counts = np.unique(labels, return_counts=True)
        dist = {
            task_cfg["names"][int(u)]: int(c)
            for u, c in zip(unique, counts)
            if int(u) < len(task_cfg["names"])
        }
        print(f"\n[trip_class] {task_name}: classes={num_classes}, distribution={dist}")

        if len(unique) < 2:
            print(f"  Skipping {task_name}: only one class present")
            continue

        train_y = torch.from_numpy(labels[train_idx]).long()
        val_y = torch.from_numpy(labels[val_idx]).long()
        test_y = torch.from_numpy(labels[test_idx]).long()

        probe_results = run_classification_probe(
            train_x,
            train_y,
            val_x,
            val_y,
            test_x,
            test_y,
            num_classes,
            args.probe_epochs,
            args.probe_lr,
            args.device,
        )
        results["tasks"][task_name] = {
            "num_classes": num_classes,
            "class_names": task_cfg["names"],
            "class_distribution": dist,
            "train": probe_results["train"],
            "val": probe_results["val"],
            "test": probe_results["test"],
        }
        print(
            f"  test: top1={probe_results['test']['top1']:.3f} "
            f"top5={probe_results['test'].get('top5', 0):.3f}"
        )

    print(json.dumps(results, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
