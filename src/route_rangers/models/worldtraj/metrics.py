"""Trajectory-prediction evaluation metrics.

Standard metrics used to evaluate trajectory foundation models such as
UniTraj / WorldTraj:

* **ADE** (Average Displacement Error) — mean L2 distance across all
  agents and all predicted time-steps.
* **FDE** (Final Displacement Error) — mean L2 distance at the *last*
  predicted time-step only.
* **Miss Rate** — fraction of agents whose FDE exceeds a distance threshold
  (commonly 2 m).

All functions operate on NumPy arrays and are intentionally kept stateless
so they can be called freely inside loops or vectorised sweeps.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _l2_distances(
    predicted: NDArray[np.float64],
    ground_truth: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return per-agent, per-step L2 distances.

    Parameters
    ----------
    predicted:
        Shape ``(N, T, 2)``.
    ground_truth:
        Shape ``(N, T, 2)``.

    Returns
    -------
    NDArray[np.float64]
        Shape ``(N, T)`` — Euclidean distance at each step for each agent.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    if predicted.shape != ground_truth.shape:
        raise ValueError(
            f"predicted and ground_truth must have the same shape, "
            f"got {predicted.shape} vs {ground_truth.shape}"
        )
    diff = predicted - ground_truth          # (N, T, 2)
    return np.sqrt(np.sum(diff ** 2, axis=-1))  # (N, T)


def compute_ade(
    predicted: NDArray[np.float64],
    ground_truth: NDArray[np.float64],
) -> float:
    """Average Displacement Error (ADE) in metres.

    Parameters
    ----------
    predicted:
        Shape ``(N, T, 2)``.
    ground_truth:
        Shape ``(N, T, 2)``.

    Returns
    -------
    float
        Mean L2 displacement across all agents and all time-steps.
    """
    return float(np.mean(_l2_distances(predicted, ground_truth)))


def compute_fde(
    predicted: NDArray[np.float64],
    ground_truth: NDArray[np.float64],
) -> float:
    """Final Displacement Error (FDE) in metres.

    Parameters
    ----------
    predicted:
        Shape ``(N, T, 2)``.
    ground_truth:
        Shape ``(N, T, 2)``.

    Returns
    -------
    float
        Mean L2 displacement at the *last* time-step across all agents.
    """
    dists = _l2_distances(predicted, ground_truth)  # (N, T)
    return float(np.mean(dists[:, -1]))


def compute_miss_rate(
    predicted: NDArray[np.float64],
    ground_truth: NDArray[np.float64],
    threshold: float = 2.0,
) -> float:
    """Miss Rate (MR) — fraction of agents with FDE > threshold.

    Parameters
    ----------
    predicted:
        Shape ``(N, T, 2)``.
    ground_truth:
        Shape ``(N, T, 2)``.
    threshold:
        Distance threshold in metres (default 2.0 m, as used in Argoverse).

    Returns
    -------
    float
        Fraction of agents (in [0, 1]) whose final-step error exceeds the
        threshold.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    dists = _l2_distances(predicted, ground_truth)  # (N, T)
    final_dists = dists[:, -1]                      # (N,)
    return float(np.mean(final_dists > threshold))
