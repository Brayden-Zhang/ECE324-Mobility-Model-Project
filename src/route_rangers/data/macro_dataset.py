"""Macro-distribution dataset helpers for stage-2 training."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class MacroDistributionDataset(Dataset):
    """Dataset wrapper around the checked-in macro movement distribution npz."""

    def __init__(self, path: str, normalize: bool = True) -> None:
        with np.load(path, allow_pickle=True) as payload:
            dist = payload["dist"].astype(np.float32)
            region_idx = payload["region_idx"].astype(np.int64)
            time_idx = payload["time_idx"].astype(np.int64)

        if normalize:
            denom = np.clip(dist.sum(axis=-1, keepdims=True), a_min=1e-8, a_max=None)
            dist = dist / denom

        self.dist = torch.from_numpy(dist)
        self.region_idx = torch.from_numpy(region_idx)
        self.time_idx = torch.from_numpy(time_idx.astype(np.float32))
        self.region_vocab = int(region_idx.max()) + 1 if region_idx.size else 0
        self.dist_dim = int(dist.shape[-1]) if dist.ndim == 2 else 0

    def __len__(self) -> int:
        return int(self.dist.shape[0])

    def __getitem__(self, index: int) -> dict:
        return {
            "region_idx": self.region_idx[index],
            "time_ts": self.time_idx[index],
            "dist": self.dist[index],
        }
