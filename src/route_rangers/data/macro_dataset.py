from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class MacroMeta:
    regions: np.ndarray
    time_keys: np.ndarray
    categories: np.ndarray


def _month_to_ts(month_key: str) -> float:
    # month_key: YYYY-MM
    try:
        dt = datetime.strptime(month_key, "%Y-%m").replace(tzinfo=timezone.utc)
    except Exception:
        dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return float(dt.timestamp())


class MacroDistributionDataset(Dataset):
    def __init__(self, npz_path: str, normalize: bool = True):
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"macro npz not found: {npz_path}")
        data = np.load(path, allow_pickle=True)
        dist = data["dist"].astype(np.float32)
        region_idx = data["region_idx"].astype(np.int64)
        time_idx = data["time_idx"].astype(np.int64)
        self.meta = MacroMeta(
            regions=data["regions"],
            time_keys=data["time_keys"],
            categories=data["categories"],
        )

        if normalize:
            dist = np.clip(dist, 0.0, None)
            row_sum = dist.sum(axis=1, keepdims=True)
            dist = np.divide(dist, row_sum, out=np.zeros_like(dist), where=row_sum > 0)

        time_ts_by_idx = np.array(
            [_month_to_ts(str(k)) for k in self.meta.time_keys], dtype=np.float32
        )
        time_ts = time_ts_by_idx[time_idx]

        self.dist = dist
        self.region_idx = region_idx
        self.time_idx = time_idx
        self.time_ts = time_ts

    def __len__(self) -> int:
        return self.dist.shape[0]

    @property
    def region_vocab(self) -> int:
        return int(len(self.meta.regions))

    @property
    def dist_dim(self) -> int:
        return int(self.dist.shape[1])

    def __getitem__(self, idx: int) -> dict:
        return {
            "region_idx": torch.tensor(self.region_idx[idx], dtype=torch.long),
            "time_idx": torch.tensor(self.time_idx[idx], dtype=torch.long),
            "time_ts": torch.tensor(self.time_ts[idx], dtype=torch.float32),
            "dist": torch.tensor(self.dist[idx], dtype=torch.float32),
        }
