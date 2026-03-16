from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def build_zone_vocab(
    origin_zone_ids: Iterable[str],
    dest_zone_ids: Iterable[str],
    max_zones: int = 256,
) -> Dict[str, int]:
    counter = Counter()
    counter.update(origin_zone_ids)
    counter.update(dest_zone_ids)
    items = counter.most_common(max(1, int(max_zones)))
    return {zone: i for i, (zone, _) in enumerate(items)}


def compute_od_tensor(
    origin_idx: np.ndarray,
    dest_idx: np.ndarray,
    time_idx: np.ndarray,
    num_times: int,
    num_zones: int,
) -> np.ndarray:
    """
    Build OD count tensor with shape [T, O, D].
    Invalid indices (<0) are ignored.
    """
    od = np.zeros((num_times, num_zones, num_zones), dtype=np.float32)
    for o, d, t in zip(origin_idx, dest_idx, time_idx):
        if t < 0 or t >= num_times:
            continue
        if o < 0 or o >= num_zones:
            continue
        if d < 0 or d >= num_zones:
            continue
        od[int(t), int(o), int(d)] += 1.0
    return od


def metric_mae_rmse_mape(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-6,
) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    diff = pred - target
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    denom = np.maximum(np.abs(target), eps)
    mape = float(np.mean(np.abs(diff) / denom))
    return {"mae": mae, "rmse": rmse, "mape": mape}


def remap_to_vocab(values: Iterable[str], vocab: Dict[str, int]) -> np.ndarray:
    vals = list(values)
    out = np.full((len(vals),), -1, dtype=np.int64)
    for i, v in enumerate(vals):
        if v in vocab:
            out[i] = vocab[v]
    return out


def build_time_vocab(
    timestamps: np.ndarray,
    bin_seconds: int,
    max_bins: int = 0,
) -> Tuple[np.ndarray, Dict[int, int]]:
    ts = np.asarray(timestamps, dtype=np.float64)
    if ts.size == 0:
        return np.full((0,), -1, dtype=np.int64), {}
    bins = np.floor(ts / float(bin_seconds)).astype(np.int64)
    unique = sorted(set(int(x) for x in bins.tolist()))
    if max_bins > 0 and len(unique) > max_bins:
        unique = unique[:max_bins]
    vocab = {b: i for i, b in enumerate(unique)}
    idx = np.asarray([vocab.get(int(x), -1) for x in bins], dtype=np.int64)
    return idx, vocab
