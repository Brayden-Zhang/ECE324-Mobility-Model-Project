#!/usr/bin/env python3
"""Run leave-one-city-out transfer for next-POI and user-ID probes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from route_rangers.cli import run_benchmarks as rb
from route_rangers.cli.run_next_poi_eval import POIDataset, run_eval


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-city transfer (leave-one-city-out) for mobility probes"
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
    parser.add_argument("--probe_epochs", type=int, default=8)
    parser.add_argument("--probe_lr", type=float, default=2e-3)
    parser.add_argument("--probe_weight_decay", type=float, default=1e-4)
    parser.add_argument("--probe_batch_size", type=int, default=2048)
    parser.add_argument("--max_points", type=int, default=300000)
    parser.add_argument("--min_city_records", type=int, default=50)
    parser.add_argument("--max_holdouts", type=int, default=0)
    parser.add_argument("--skip_user_identification", action="store_true")
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def _safe(d: dict, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def mean_std(vals):
    arr = np.asarray([v for v in vals if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}


def main():
    args = parse_args()
    rb.set_seed(args.seed)

    records = rb.load_local_data(args.local_data)
    dataset = POIDataset(records, max_len=args.max_len, sample_limit=args.sample_limit)
    if len(dataset) < 50:
        raise RuntimeError("Too few valid POI records for cross-city transfer.")

    city_counts = {}
    for s in dataset.samples:
        city_counts[s["city"]] = city_counts.get(s["city"], 0) + 1
    eligible = sorted(
        [c for c, n in city_counts.items() if n >= args.min_city_records],
        key=lambda c: city_counts[c],
        reverse=True,
    )
    if args.max_holdouts > 0:
        eligible = eligible[: args.max_holdouts]

    if len(eligible) < 2:
        raise RuntimeError(
            "Need at least two eligible cities; lower --min_city_records or provide more data."
        )

    pack = rb.load_backbone(args.checkpoint, device=args.device, override_max_len=args.max_len)

    city_results = []
    for heldout in eligible:
        train_cities = [c for c in eligible if c != heldout]
        res = run_eval(
            dataset,
            pack,
            seed=args.seed,
            max_len=args.max_len,
            batch_size=args.batch_size,
            max_points=args.max_points,
            probe_epochs=args.probe_epochs,
            probe_lr=args.probe_lr,
            probe_weight_decay=args.probe_weight_decay,
            probe_batch_size=args.probe_batch_size,
            split_mode="random",
            train_cities=train_cities,
            test_cities=[heldout],
            skip_user_identification=args.skip_user_identification,
            device=args.device,
        )
        city_results.append(
            {
                "heldout_city": heldout,
                "heldout_records": city_counts[heldout],
                "results": res,
            }
        )

    np_top1 = [_safe(r, "results", "next_poi", "test", "top1") for r in city_results]
    np_top5 = [_safe(r, "results", "next_poi", "test", "top5") for r in city_results]
    np_top10 = [_safe(r, "results", "next_poi", "test", "top10") for r in city_results]
    np_ndcg10 = [_safe(r, "results", "next_poi", "test", "ndcg10") for r in city_results]
    np_mrr = [_safe(r, "results", "next_poi", "test", "mrr") for r in city_results]

    uid_top1 = [_safe(r, "results", "user_identification", "test", "top1") for r in city_results]

    out = {
        "checkpoint": args.checkpoint,
        "dataset": args.local_data,
        "settings": {
            "max_len": args.max_len,
            "probe_epochs": args.probe_epochs,
            "probe_lr": args.probe_lr,
            "min_city_records": args.min_city_records,
            "max_holdouts": args.max_holdouts,
        },
        "eligible_cities": [{"city": c, "records": city_counts[c]} for c in eligible],
        "city_results": city_results,
        "aggregate": {
            "next_poi": {
                "top1": mean_std(np_top1),
                "top5": mean_std(np_top5),
                "top10": mean_std(np_top10),
                "ndcg10": mean_std(np_ndcg10),
                "mrr": mean_std(np_mrr),
            },
            "user_identification": {
                "top1": mean_std(uid_top1),
            },
        },
    }

    print(json.dumps(out, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"saved {path}")


if __name__ == "__main__":
    main()