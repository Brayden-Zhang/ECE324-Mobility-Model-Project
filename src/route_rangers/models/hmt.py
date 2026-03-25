import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

try:
    import h3
except Exception:  # optional dependency
    h3 = None


@dataclass
class HMTConfig:
    mode: str = "h3"
    res0: int = 9
    res1: int = 7
    res2: int = 5
    vocab_l0: int = 16384
    vocab_l1: int = 4096
    vocab_l2: int = 1024
    hash_tokens: bool = True
    h3_vocab: str = ""


def _h3_cell(lat: float, lon: float, res: int):
    if h3 is None:
        raise RuntimeError("h3 is not installed; install h3 to use H3 tokenization")
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lon, res)
    return h3.geo_to_h3(lat, lon, res)


def _h3_to_int(cell) -> int:
    if isinstance(cell, int):
        return cell
    # Use a deterministic hash for string-like H3 cells to avoid low-bit collisions.
    if isinstance(cell, str):
        try:
            return int(cell, 16)
        except Exception:
            import hashlib

            h = hashlib.blake2b(cell.encode("utf-8"), digest_size=8).digest()
            return int.from_bytes(h, "little", signed=False)
    try:
        return int(cell)
    except Exception:
        import hashlib

        h = hashlib.blake2b(str(cell).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "little", signed=False)


class H3Tokenizer:
    def __init__(
        self,
        res0: int,
        res1: int,
        res2: int,
        vocab_sizes,
        hash_tokens: bool = True,
        h3_vocab: str = "",
    ):
        self.resolutions = (res0, res1, res2)
        self.vocab_sizes = vocab_sizes
        self.hash_tokens = hash_tokens
        self.h3_vocab = h3_vocab
        self.h3_to_id = None
        self.id_to_h3 = None
        if h3_vocab:
            import json
            from pathlib import Path

            vocab_path = Path(h3_vocab)
            with open(vocab_path, "r") as f:
                payload = json.load(f)
            cells_l0 = payload.get("cells_l0", [])
            cells_l1 = payload.get("cells_l1", [])
            cells_l2 = payload.get("cells_l2", [])
            self.id_to_h3 = (cells_l0, cells_l1, cells_l2)
            self.h3_to_id = (
                {str(cell): i for i, cell in enumerate(cells_l0)},
                {str(cell): i for i, cell in enumerate(cells_l1)},
                {str(cell): i for i, cell in enumerate(cells_l2)},
            )
            if (
                len(cells_l0) > vocab_sizes[0]
                or len(cells_l1) > vocab_sizes[1]
                or len(cells_l2) > vocab_sizes[2]
            ):
                raise ValueError(
                    "h3_vocab size exceeds configured vocab sizes; "
                    "rebuild vocab or increase vocab_l*"
                )

    def tokenize(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # coords: [L, 2] lat, lon
        l0 = np.zeros((coords.shape[0],), dtype=np.int64)
        l1 = np.zeros_like(l0)
        l2 = np.zeros_like(l0)
        for i, (lat, lon) in enumerate(coords):
            c0 = _h3_cell(lat, lon, self.resolutions[0])
            c1 = _h3_cell(lat, lon, self.resolutions[1])
            c2 = _h3_cell(lat, lon, self.resolutions[2])
            if self.h3_to_id is not None:
                # Map unknown cells to the last valid id to avoid out-of-range targets.
                l0[i] = self.h3_to_id[0].get(str(c0), max(0, self.vocab_sizes[0] - 1))
                l1[i] = self.h3_to_id[1].get(str(c1), max(0, self.vocab_sizes[1] - 1))
                l2[i] = self.h3_to_id[2].get(str(c2), max(0, self.vocab_sizes[2] - 1))
            elif self.hash_tokens:
                l0[i] = _h3_to_int(c0) % self.vocab_sizes[0]
                l1[i] = _h3_to_int(c1) % self.vocab_sizes[1]
                l2[i] = _h3_to_int(c2) % self.vocab_sizes[2]
            else:
                l0[i] = _h3_to_int(c0)
                l1[i] = _h3_to_int(c1)
                l2[i] = _h3_to_int(c2)
        return l0, l1, l2


class TimeFeatures(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        # cyclical time + elapsed position + irregular sampling statistics
        self.proj = nn.Linear(11, embed_dim)

    def forward(
        self, timestamps: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # timestamps: [B, L] in seconds
        ts = timestamps.float()
        if attention_mask is None:
            mask = torch.ones_like(ts)
        else:
            mask = attention_mask.float()

        day = 24 * 60 * 60
        week = 7 * day
        t_day = (ts % day) / day
        t_week = (ts % week) / week

        dt = torch.zeros_like(ts)
        dt[:, 1:] = (ts[:, 1:] - ts[:, :-1]).clamp(min=0.0)
        dt = dt * mask
        valid_dt = (dt > 0).float()
        mean_dt = (
            dt.sum(dim=1, keepdim=True)
            / valid_dt.sum(dim=1, keepdim=True).clamp(min=1.0)
        ).clamp(min=1e-3)
        dt_norm = dt / mean_dt
        log_dt = torch.log1p(dt)
        inv_dt = 1.0 / (1.0 + dt_norm)
        dt_jump = (dt_norm > 3.0).float()

        first_ts = ts[:, :1]
        last_ts = torch.where(mask > 0, ts, first_ts).max(dim=1, keepdim=True).values
        elapsed = (ts - first_ts) / (last_ts - first_ts + 1e-6)

        feats = torch.stack(
            [
                torch.sin(2 * math.pi * t_day),
                torch.cos(2 * math.pi * t_day),
                torch.sin(2 * math.pi * t_week),
                torch.cos(2 * math.pi * t_week),
                t_day,
                t_week,
                elapsed,
                log_dt,
                dt_norm,
                inv_dt,
                dt_jump,
            ],
            dim=-1,
        )
        feats = feats * mask.unsqueeze(-1)
        return self.proj(feats)


class HMTTokenizer(nn.Module):
    def __init__(self, config: HMTConfig, feature_dim: int, embed_dim: int):
        super().__init__()
        self.config = config
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.mode = "h3"
        self.h3_tokenizer = H3Tokenizer(
            config.res0,
            config.res1,
            config.res2,
            (config.vocab_l0, config.vocab_l1, config.vocab_l2),
            config.hash_tokens,
            config.h3_vocab,
        )

    def forward(
        self,
        coords: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # coords: [B, L, 2] lat, lon
        l0_list, l1_list, l2_list = [], [], []
        coords_np = coords.detach().cpu().numpy()
        for b in range(coords_np.shape[0]):
            l0, l1, l2 = self.h3_tokenizer.tokenize(coords_np[b])
            l0_list.append(l0)
            l1_list.append(l1)
            l2_list.append(l2)
        tokens_l0 = torch.from_numpy(np.stack(l0_list, axis=0)).to(coords.device)
        tokens_l1 = torch.from_numpy(np.stack(l1_list, axis=0)).to(coords.device)
        tokens_l2 = torch.from_numpy(np.stack(l2_list, axis=0)).to(coords.device)
        return (
            tokens_l0,
            tokens_l1,
            tokens_l2,
            torch.tensor(0.0, device=coords.device),
        )


