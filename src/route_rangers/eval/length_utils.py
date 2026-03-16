from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import numpy as np


def parse_bins(length_bins: str, lengths: np.ndarray) -> Tuple[np.ndarray, str]:
    if length_bins:
        parts = [int(p.strip()) for p in length_bins.split(",") if p.strip()]
        parts = sorted(set(parts))
        if len(parts) >= 2:
            return np.asarray(parts[:2], dtype=np.int64), "fixed"
        if len(parts) == 1:
            return np.asarray([parts[0], parts[0]], dtype=np.int64), "fixed"
    if lengths.size == 0:
        return np.asarray([0, 0], dtype=np.int64), "quantile"
    q1, q2 = np.quantile(lengths, [0.33, 0.66])
    return np.asarray([int(q1), int(q2)], dtype=np.int64), "quantile"


def bin_name_for_length(length: int, bins: np.ndarray) -> str:
    if length <= int(bins[0]):
        return "short"
    if length <= int(bins[1]):
        return "medium"
    return "long"


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> float:
    if confidences.size == 0:
        return 0.0
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = float(confidences.shape[0])
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not np.any(mask):
            continue
        acc = float(correct[mask].mean())
        conf = float(confidences[mask].mean())
        weight = float(mask.sum()) / n
        ece += weight * abs(acc - conf)
    return float(ece)


def _ci_from_values(
    values: np.ndarray,
    ci_method: str = "seed",
    alpha: float = 0.05,
    bootstrap_iters: int = 2000,
    rng_seed: int = 0,
) -> Dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    if values.size == 1:
        return {
            "mean": mean,
            "std": std,
            "ci_low": mean,
            "ci_high": mean,
            "n": int(values.size),
        }
    if ci_method == "bootstrap":
        rng = np.random.default_rng(rng_seed)
        samples = np.empty((bootstrap_iters,), dtype=np.float64)
        for i in range(bootstrap_iters):
            idx = rng.integers(0, values.size, size=values.size)
            samples[i] = float(values[idx].mean())
        lo = float(np.quantile(samples, alpha / 2.0))
        hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    else:
        se = std / math.sqrt(float(values.size))
        z = 1.959963984540054  # ~N(0,1) 97.5%
        lo = mean - z * se
        hi = mean + z * se
    return {
        "mean": mean,
        "std": std,
        "ci_low": lo,
        "ci_high": hi,
        "n": int(values.size),
    }


def aggregate_length_metrics(
    seed_metrics: List[Dict[str, Dict[str, float]]],
    ci_method: str = "seed",
    alpha: float = 0.05,
    bootstrap_iters: int = 2000,
    rng_seed: int = 0,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Aggregate per-seed, per-bucket metrics into mean/std/CI.

    Expected input:
      seed_metrics[seed_idx][bucket_name][metric_name] = float
    """
    buckets = sorted({b for m in seed_metrics for b in m.keys()})
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for b in buckets:
        metric_names = sorted(
            {
                k
                for m in seed_metrics
                for k in m.get(b, {}).keys()
                if isinstance(m.get(b, {}).get(k), (int, float))
            }
        )
        out[b] = {}
        for metric in metric_names:
            vals = np.asarray(
                [float(m[b][metric]) for m in seed_metrics if b in m and metric in m[b]],
                dtype=np.float64,
            )
            out[b][metric] = _ci_from_values(
                vals,
                ci_method=ci_method,
                alpha=alpha,
                bootstrap_iters=bootstrap_iters,
                rng_seed=rng_seed,
            )
    return out


def gap_decision_from_seed_values(
    short_values: Iterable[float],
    long_values: Iterable[float],
    variability_k: float = 1.0,
    tolerance: float = 0.0,
) -> Dict[str, float | bool]:
    short = np.asarray(list(short_values), dtype=np.float64)
    long = np.asarray(list(long_values), dtype=np.float64)
    n = min(short.size, long.size)
    if n == 0:
        return {
            "gap_mean": 0.0,
            "gap_std": 0.0,
            "threshold": float(tolerance),
            "pass": True,
            "n": 0,
        }
    short = short[:n]
    long = long[:n]
    gaps = long - short
    gap_mean = float(gaps.mean())
    gap_std = float(gaps.std(ddof=1)) if n > 1 else 0.0
    pooled_std = 0.0
    if n > 1:
        pooled_std = float(np.sqrt((short.var(ddof=1) + long.var(ddof=1)) / 2.0))
    threshold = max(float(tolerance), float(variability_k) * pooled_std)
    is_pass = abs(gap_mean) <= threshold
    return {
        "gap_mean": gap_mean,
        "gap_std": gap_std,
        "threshold": threshold,
        "pass": bool(is_pass),
        "n": int(n),
    }
