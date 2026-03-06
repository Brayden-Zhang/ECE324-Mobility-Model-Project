import math
from typing import Optional
import contextlib

import torch
from torch import nn


class SpaceTimeEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_freqs: int = 6, include_raw: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_freqs = num_freqs
        self.include_raw = include_raw
        in_dim = 0
        if include_raw:
            in_dim += 3
        in_dim += 3 * 2 * num_freqs
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        coords: torch.Tensor,
        timestamps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # coords: [B, L, 2] lat, lon (degrees), timestamps: [B, L] seconds
        # Run in float32 to avoid NaNs under AMP for large timestamps.
        if torch.is_autocast_enabled():
            autocast_ctx = torch.autocast(device_type=coords.device.type, enabled=False)
        else:
            autocast_ctx = contextlib.nullcontext()
        with autocast_ctx:
            coords_f = coords.float()
            ts_f = timestamps.float()
            lat = coords_f[..., 0:1]
            lon = coords_f[..., 1:2]

        # Normalize time to [0,1] within each sequence using attention_mask.
        if attention_mask is None:
            mask = torch.ones_like(ts_f)
        else:
            mask = attention_mask.float()
        first_ts = ts_f[:, :1]
        last_ts = torch.where(mask > 0, ts_f, first_ts).max(dim=1, keepdim=True).values
        t_norm = (ts_f - first_ts) / (last_ts - first_ts + 1e-6)
        t_norm = t_norm.unsqueeze(-1)

        # Convert lat/lon to radians to stabilize Fourier features.
        lat_rad = lat * (math.pi / 180.0)
        lon_rad = lon * (math.pi / 180.0)

        base = torch.cat([lat_rad, lon_rad, t_norm], dim=-1)
        feats = []
        if self.include_raw:
            feats.append(base)

        freq_bands = 2.0 ** torch.arange(
            self.num_freqs, device=coords.device, dtype=coords_f.dtype
        )
        for freq in freq_bands:
            scaled = base * freq
            feats.append(torch.sin(scaled))
            feats.append(torch.cos(scaled))

        out = torch.cat(feats, dim=-1)
        return self.proj(out).to(coords.dtype)
