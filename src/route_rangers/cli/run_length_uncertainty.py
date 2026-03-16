#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from route_rangers.cli import run_length_sensitivity as ls
from route_rangers.eval.length_utils import (
    bin_name_for_length,
    expected_calibration_error,
    parse_bins,
)
from route_rangers.cli.run_benchmarks import forward_backbone, load_backbone


def parse_args():
    parser = argparse.ArgumentParser(description="Uncertainty analysis by length bucket.")
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
    parser.add_argument("--length_bins", type=str, default="")
    parser.add_argument("--ece_bins", type=int, default=10)
    parser.add_argument("--output", type=str, default="cache/length_uncertainty.json")
    return parser.parse_args()


def _new_bucket_store():
    return {
        "next_nll_sum": 0.0,
        "next_entropy_sum": 0.0,
        "next_tokens": 0.0,
        "dest_nll_sum": 0.0,
        "dest_entropy_sum": 0.0,
        "dest_conf": [],
        "dest_correct": [],
        "samples": 0,
    }


def main():
    args = parse_args()
    ls.set_seed(args.seed)
    if not Path(args.local_data).exists():
        raise FileNotFoundError(f"local_data not found: {args.local_data}")

    pack = load_backbone(args.checkpoint, device=args.device, override_max_len=args.max_len)
    records = ls.load_local_data(args.local_data)
    dataset = ls.FixedTrajectoryDataset(
        records, max_len=args.max_len, sample_limit=args.sample_limit
    )
    if len(dataset) < 10:
        raise RuntimeError(f"not enough samples: {len(dataset)}")

    lengths = np.asarray([int(s["raw_length"]) for s in dataset.samples], dtype=np.int64)
    bins, strategy = parse_bins(args.length_bins, lengths)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ls.collate_fixed,
    )

    store: Dict[str, dict] = {"short": _new_bucket_store(), "medium": _new_bucket_store(), "long": _new_bucket_store()}
    for batch in loader:
        outputs, _, t1, _, attention = forward_backbone(
            batch, pack, device=args.device, max_len=args.max_len, mask=None
        )
        t1 = t1.to(args.device)

        logits_step = outputs["step_logits"]["l1"]
        probs_step = torch.softmax(logits_step.float(), dim=-1)
        entropy_step = (-probs_step * torch.log(probs_step + 1e-9)).sum(dim=-1)

        logits_dest = outputs["dest_logits"]
        probs_dest = torch.softmax(logits_dest.float(), dim=-1)
        entropy_dest = (-probs_dest * torch.log(probs_dest + 1e-9)).sum(dim=-1)
        conf_dest = probs_dest.max(dim=-1).values

        last_idx = attention.sum(dim=1).long().clamp(min=1) - 1
        dest_targets = t1.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        dest_pred = logits_dest.argmax(dim=-1)
        dest_nll = F.cross_entropy(logits_dest.float(), dest_targets, reduction="none")

        raw_len = batch["raw_length"].numpy()
        for i in range(attention.shape[0]):
            bname = bin_name_for_length(int(raw_len[i]), bins)
            bucket = store[bname]

            vlen = int(attention[i].sum().item())
            if vlen > 1:
                step_logits_i = logits_step[i, : vlen - 1]
                step_targets_i = t1[i, 1:vlen]
                step_nll_i = F.cross_entropy(
                    step_logits_i.float(), step_targets_i, reduction="sum"
                ).item()
                step_entropy_i = entropy_step[i, : vlen - 1].sum().item()
                token_n = float(vlen - 1)
            else:
                step_nll_i = 0.0
                step_entropy_i = 0.0
                token_n = 0.0

            bucket["next_nll_sum"] += float(step_nll_i)
            bucket["next_entropy_sum"] += float(step_entropy_i)
            bucket["next_tokens"] += token_n
            bucket["dest_nll_sum"] += float(dest_nll[i].item())
            bucket["dest_entropy_sum"] += float(entropy_dest[i].item())
            bucket["dest_conf"].append(float(conf_dest[i].item()))
            bucket["dest_correct"].append(float((dest_pred[i] == dest_targets[i]).item()))
            bucket["samples"] += 1

    metrics = {}
    for bname, bucket in store.items():
        n = max(1, int(bucket["samples"]))
        next_tok = max(1.0, float(bucket["next_tokens"]))
        conf = np.asarray(bucket["dest_conf"], dtype=np.float64)
        corr = np.asarray(bucket["dest_correct"], dtype=np.float64)
        metrics[bname] = {
            "next_step_nll": float(bucket["next_nll_sum"] / next_tok),
            "next_step_entropy": float(bucket["next_entropy_sum"] / next_tok),
            "dest_nll": float(bucket["dest_nll_sum"] / n),
            "dest_entropy": float(bucket["dest_entropy_sum"] / n),
            "dest_ece": expected_calibration_error(conf, corr, n_bins=args.ece_bins),
            "samples": int(bucket["samples"]),
            "next_tokens": int(bucket["next_tokens"]),
        }

    out = {
        "checkpoint": args.checkpoint,
        "dataset": args.local_data,
        "samples": len(dataset),
        "bins": bins.tolist(),
        "bin_strategy": strategy,
        "ece_bins": args.ece_bins,
        "metrics": metrics,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved length uncertainty results: {out_path}")


if __name__ == "__main__":
    main()
