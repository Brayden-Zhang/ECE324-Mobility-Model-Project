"""OSM-context compatibility helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class OSMContextIndex:
    """Minimal container for optional OSM context metadata."""

    def __init__(self, path: str | Path, dim: int) -> None:
        self.path = Path(path)
        self.dim = int(dim)
        self.payload: dict[str, Any] = {}
        if self.path.exists():
            try:
                self.payload = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.payload = {}


def context_tensor_from_index(
    index: OSMContextIndex,
    coords: torch.Tensor,
    res1: int,
) -> torch.Tensor:
    del res1
    return torch.zeros(
        (coords.shape[0], coords.shape[1], index.dim),
        device=coords.device,
        dtype=coords.dtype,
    )
