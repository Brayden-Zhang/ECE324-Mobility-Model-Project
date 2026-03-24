#!/usr/bin/env python3
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from route_rangers.cli import run_benchmarks as rb
from route_rangers.cli import run_length_sensitivity as ls
from route_rangers.eval.length_utils import bin_name_for_length, parse_bins


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run proposal baselines (mean-displacement + simple RNN) "
            "for length-sensitivity research."
        )
    )
    parser.add_argument("--local_data", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument(
        "--split_mode",
        type=str,
        default="both",
        choices=["both", "random", "temporal"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length_bins", type=str, default="")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64)
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--rnn_epochs", type=int, default=8)
    parser.add_argument("--rnn_lr", type=float, default=2e-3)
    parser.add_argument("--rnn_weight_decay", type=float, default=1e-4)
    parser.add_argument("--rnn_batch_size", type=int, default=512)
    parser.add_argument("--max_rnn_pairs", type=int, default=200000)
    parser.add_argument("--dest_prefix_ratio", type=float, default=0.5)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=str, default="cache/proposal_baselines.json")
    return parser.parse_args()


def set_seed(seed: int):
    rb.set_seed(seed)


def _coords_and_len(sample: dict) -> Tuple[np.ndarray, int]:
    coords = sample["coords"].numpy()
    attn = sample["attention_mask"].numpy()
    vlen = int(attn.sum())
    return coords, vlen


def _distance_m(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.shape[0] == 0:
        return 0.0
    pred_t = torch.from_numpy(pred.astype(np.float32))
    target_t = torch.from_numpy(target.astype(np.float32))
    d = rb.haversine_m(pred_t, target_t)
    return float(d.mean().item())


def _metric_from_pairs(pred: List[np.ndarray], true: List[np.ndarray]) -> Dict[str, float]:
    if not pred:
        return {"mae_m": 0.0, "rmse_m": 0.0, "n": 0}
    p = np.concatenate(pred, axis=0).astype(np.float32)
    t = np.concatenate(true, axis=0).astype(np.float32)
    d = rb.haversine_m(torch.from_numpy(p), torch.from_numpy(t)).numpy()
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d**2)))
    return {"mae_m": mae, "rmse_m": rmse, "n": int(d.shape[0])}


def evaluate_mean_displacement(
    dataset: ls.FixedTrajectoryDataset,
    indices: List[int],
    bins: np.ndarray,
    dest_prefix_ratio: float,
) -> Dict[str, dict]:
    next_pred: List[np.ndarray] = []
    next_true: List[np.ndarray] = []
    next_by_bucket: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {
        "short": ([], []),
        "medium": ([], []),
        "long": ([], []),
    }

    dest_pred: List[np.ndarray] = []
    dest_true: List[np.ndarray] = []
    dest_by_bucket: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {
        "short": ([], []),
        "medium": ([], []),
        "long": ([], []),
    }

    for idx in indices:
        sample = dataset[idx]
        coords, vlen = _coords_and_len(sample)
        if vlen <= 1:
            continue
        raw_len = int(sample["raw_length"])
        bucket = bin_name_for_length(raw_len, bins)

        # next-step baseline from running mean displacement
        # skip the first step where we have no motion history, so we only
        # evaluate on t>=2 where a mean displacement estimate is available.
        pred_steps = []
        true_steps = []
        for t in range(2, vlen):
            obs = coords[:t, :2]
            delta = obs[1:] - obs[:-1]
            mean_delta = delta.mean(axis=0)
            pred = obs[-1] + mean_delta
            pred_steps.append(pred[None, :])
            true_steps.append(coords[t : t + 1, :2])
        if pred_steps:
            p = np.concatenate(pred_steps, axis=0)
            y = np.concatenate(true_steps, axis=0)
            next_pred.append(p)
            next_true.append(y)
            next_by_bucket[bucket][0].append(p)
            next_by_bucket[bucket][1].append(y)

        # destination baseline from prefix mean displacement extrapolation
        prefix = max(2, int(math.ceil(vlen * float(dest_prefix_ratio))))
        prefix = min(prefix, vlen - 1)
        obs = coords[:prefix, :2]
        if obs.shape[0] >= 2:
            delta = obs[1:] - obs[:-1]
            mean_delta = delta.mean(axis=0)
        else:
            mean_delta = np.zeros((2,), dtype=np.float32)
        steps_left = max(0, vlen - prefix)
        pred_dest = obs[-1] + mean_delta * float(steps_left)
        true_dest = coords[vlen - 1, :2]
        dest_pred.append(pred_dest[None, :])
        dest_true.append(true_dest[None, :])
        dest_by_bucket[bucket][0].append(pred_dest[None, :])
        dest_by_bucket[bucket][1].append(true_dest[None, :])

    next_bucket_metrics = {
        b: _metric_from_pairs(v[0], v[1]) for b, v in next_by_bucket.items()
    }
    dest_bucket_metrics = {
        b: _metric_from_pairs(v[0], v[1]) for b, v in dest_by_bucket.items()
    }
    return {
        "next_location_regression_probe": {
            "test": _metric_from_pairs(next_pred, next_true),
            "test_by_length": next_bucket_metrics,
        },
        "destination_regression_probe": {
            "test": _metric_from_pairs(dest_pred, dest_true),
            "test_by_length": dest_bucket_metrics,
        },
    }


@dataclass
class SequenceItem:
    x: torch.Tensor  # [L, 2]
    y: torch.Tensor  # [2]
    bucket: str


class SequenceDataset(Dataset):
    def __init__(self, items: List[SequenceItem]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_sequence(batch: List[SequenceItem]):
    xs = [b.x for b in batch]
    ys = torch.stack([b.y for b in batch], dim=0)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    padded = pad_sequence(xs, batch_first=True)
    buckets = [b.bucket for b in batch]
    return padded, lengths, ys, buckets


class SimpleGRURegressor(torch.nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 64, layers: int = 1):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
        )
        self.head = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        h = h_n[-1]
        return self.head(h)


def build_rnn_items_next(
    dataset: ls.FixedTrajectoryDataset,
    indices: List[int],
    bins: np.ndarray,
    max_pairs: int,
) -> List[SequenceItem]:
    items: List[SequenceItem] = []
    for idx in indices:
        sample = dataset[idx]
        coords, vlen = _coords_and_len(sample)
        if vlen <= 1:
            continue
        bucket = bin_name_for_length(int(sample["raw_length"]), bins)
        for t in range(1, vlen):
            x = torch.from_numpy(coords[:t, :2].astype(np.float32))
            y = torch.from_numpy(coords[t, :2].astype(np.float32))
            items.append(SequenceItem(x=x, y=y, bucket=bucket))
            if max_pairs > 0 and len(items) >= max_pairs:
                return items
    return items


def build_rnn_items_destination(
    dataset: ls.FixedTrajectoryDataset,
    indices: List[int],
    bins: np.ndarray,
    prefix_ratio: float,
) -> List[SequenceItem]:
    items: List[SequenceItem] = []
    for idx in indices:
        sample = dataset[idx]
        coords, vlen = _coords_and_len(sample)
        if vlen <= 2:
            continue
        bucket = bin_name_for_length(int(sample["raw_length"]), bins)
        prefix = max(2, int(math.ceil(vlen * float(prefix_ratio))))
        prefix = min(prefix, vlen - 1)
        x = torch.from_numpy(coords[:prefix, :2].astype(np.float32))
        y = torch.from_numpy(coords[vlen - 1, :2].astype(np.float32))
        items.append(SequenceItem(x=x, y=y, bucket=bucket))
    return items


def train_rnn(
    train_items: List[SequenceItem],
    val_items: List[SequenceItem],
    args,
) -> SimpleGRURegressor:
    model = SimpleGRURegressor(
        hidden_dim=args.rnn_hidden_dim,
        layers=args.rnn_layers,
    ).to(args.device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.rnn_lr,
        weight_decay=args.rnn_weight_decay,
    )
    train_loader = DataLoader(
        SequenceDataset(train_items),
        batch_size=args.rnn_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_sequence,
    )
    val_loader = DataLoader(
        SequenceDataset(val_items),
        batch_size=args.rnn_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sequence,
    )

    best = None
    best_val = float("inf")
    for _ in range(max(1, args.rnn_epochs)):
        model.train()
        for x, lengths, y, _ in train_loader:
            x = x.to(args.device)
            lengths = lengths.to(args.device)
            y = y.to(args.device)
            pred = model(x, lengths)
            loss = torch.nn.functional.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, lengths, y, _ in val_loader:
                x = x.to(args.device)
                lengths = lengths.to(args.device)
                y = y.to(args.device)
                pred = model(x, lengths)
                val_losses.append(float(torch.nn.functional.mse_loss(pred, y).item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss < best_val:
            best_val = val_loss
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best is not None:
        model.load_state_dict(best)
    return model


def evaluate_rnn(
    model: SimpleGRURegressor,
    items: List[SequenceItem],
    args,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    loader = DataLoader(
        SequenceDataset(items),
        batch_size=args.rnn_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sequence,
    )
    pred_all: List[np.ndarray] = []
    true_all: List[np.ndarray] = []
    bucket_store: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {
        "short": ([], []),
        "medium": ([], []),
        "long": ([], []),
    }
    model.eval()
    with torch.no_grad():
        for x, lengths, y, buckets in loader:
            x = x.to(args.device)
            lengths = lengths.to(args.device)
            pred = model(x, lengths).cpu().numpy().astype(np.float32)
            true = y.numpy().astype(np.float32)
            pred_all.append(pred)
            true_all.append(true)
            for i, b in enumerate(buckets):
                if b not in bucket_store:
                    continue
                bucket_store[b][0].append(pred[i : i + 1])
                bucket_store[b][1].append(true[i : i + 1])
    overall = _metric_from_pairs(pred_all, true_all)
    by_length = {b: _metric_from_pairs(v[0], v[1]) for b, v in bucket_store.items()}
    return overall, by_length


def run_split(dataset, idx_triplet, bins, args) -> Dict[str, dict]:
    train_idx, val_idx, test_idx = idx_triplet

    # mean displacement baseline
    mean_metrics = evaluate_mean_displacement(
        dataset,
        indices=test_idx,
        bins=bins,
        dest_prefix_ratio=args.dest_prefix_ratio,
    )

    # RNN next-step
    next_train = build_rnn_items_next(dataset, train_idx, bins, args.max_rnn_pairs)
    next_val = build_rnn_items_next(
        dataset,
        val_idx,
        bins,
        max(1, args.max_rnn_pairs // 4) if args.max_rnn_pairs > 0 else 0,
    )
    next_test = build_rnn_items_next(
        dataset,
        test_idx,
        bins,
        max(1, args.max_rnn_pairs // 4) if args.max_rnn_pairs > 0 else 0,
    )
    next_model = train_rnn(next_train, next_val, args)
    next_train_m, _ = evaluate_rnn(next_model, next_train, args)
    next_val_m, _ = evaluate_rnn(next_model, next_val, args)
    next_test_m, next_test_by = evaluate_rnn(next_model, next_test, args)

    # RNN destination
    dest_train = build_rnn_items_destination(
        dataset, train_idx, bins, args.dest_prefix_ratio
    )
    dest_val = build_rnn_items_destination(dataset, val_idx, bins, args.dest_prefix_ratio)
    dest_test = build_rnn_items_destination(dataset, test_idx, bins, args.dest_prefix_ratio)
    dest_model = train_rnn(dest_train, dest_val, args)
    dest_train_m, _ = evaluate_rnn(dest_model, dest_train, args)
    dest_val_m, _ = evaluate_rnn(dest_model, dest_val, args)
    dest_test_m, dest_test_by = evaluate_rnn(dest_model, dest_test, args)

    return {
        "mean_displacement": mean_metrics,
        "simple_rnn": {
            "next_location_regression_probe": {
                "train": next_train_m,
                "val": next_val_m,
                "test": next_test_m,
                "test_by_length": next_test_by,
            },
            "destination_regression_probe": {
                "train": dest_train_m,
                "val": dest_val_m,
                "test": dest_test_m,
                "test_by_length": dest_test_by,
            },
        },
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    if not Path(args.local_data).exists():
        raise FileNotFoundError(f"local_data not found: {args.local_data}")

    records = rb.load_local_data(args.local_data)
    dataset = ls.FixedTrajectoryDataset(
        records, max_len=args.max_len, sample_limit=args.sample_limit
    )
    if len(dataset) < 10:
        raise RuntimeError(f"not enough samples: {len(dataset)}")

    raw_lengths = np.asarray([int(s["raw_length"]) for s in dataset.samples], dtype=np.int64)
    bins, strategy = parse_bins(args.length_bins, raw_lengths)

    split_modes = (
        ["random", "temporal"] if args.split_mode == "both" else [args.split_mode]
    )
    results = {
        "dataset": args.local_data,
        "samples": len(dataset),
        "bins": bins.tolist(),
        "bin_strategy": strategy,
        "settings": {
            "max_len": args.max_len,
            "rnn_hidden_dim": args.rnn_hidden_dim,
            "rnn_layers": args.rnn_layers,
            "rnn_epochs": args.rnn_epochs,
            "dest_prefix_ratio": args.dest_prefix_ratio,
        },
        "splits": {},
    }

    for mode in split_modes:
        split_idx = rb.split_indices(dataset, mode=mode, seed=args.seed)
        results["splits"][mode] = run_split(dataset, split_idx, bins, args)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved proposal baseline results: {out}")


if __name__ == "__main__":
    main()
