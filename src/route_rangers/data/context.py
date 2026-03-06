import json
import os
from typing import Optional

import numpy as np
import torch

try:
    import h3
except Exception:  # optional
    h3 = None


class OSMContextIndex:
    def __init__(self, path: Optional[str], feature_dim: int):
        self.path = path
        self.feature_dim = feature_dim
        self.index = None
        if path and os.path.exists(path):
            with open(path, "r") as f:
                raw = json.load(f)
            # json keys are strings; values are lists
            self.index = {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}

    def lookup(self, lat: np.ndarray, lon: np.ndarray, res: int) -> np.ndarray:
        if self.index is None:
            return np.zeros((lat.shape[0], self.feature_dim), dtype=np.float32)
        if h3 is None:
            raise RuntimeError("h3 is required to lookup OSM context by cell")
        feats = np.zeros((lat.shape[0], self.feature_dim), dtype=np.float32)
        for i in range(lat.shape[0]):
            cell = (
                h3.latlng_to_cell(lat[i], lon[i], res)
                if hasattr(h3, "latlng_to_cell")
                else h3.geo_to_h3(lat[i], lon[i], res)
            )
            key = str(cell)
            if key in self.index:
                feats[i] = self.index[key]
        return feats


def context_tensor_from_index(
    index: OSMContextIndex, coords: torch.Tensor, res: int
) -> torch.Tensor:
    if index is None:
        return None
    lat = coords[..., 0].detach().cpu().numpy()
    lon = coords[..., 1].detach().cpu().numpy()
    batch = []
    for b in range(lat.shape[0]):
        feats = index.lookup(lat[b], lon[b], res)
        batch.append(feats)
    stacked = np.stack(batch, axis=0)
    return torch.from_numpy(stacked).to(coords.device)
