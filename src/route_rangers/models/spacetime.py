"""Simple spatiotemporal encoding used by the compatibility HMT model."""

from __future__ import annotations

import torch
from torch import nn


class SpaceTimeEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_freqs: int = 6):
        super().__init__()
        self.num_freqs = num_freqs
        self.proj = nn.Linear(2 + 2 * num_freqs, embed_dim)

    def forward(
        self,
        coords: torch.Tensor,
        timestamps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ts = timestamps.float().unsqueeze(-1)
        pieces = [coords.float()]
        for freq_idx in range(self.num_freqs):
            freq = 2.0 ** freq_idx
            pieces.append(torch.sin(ts / freq))
            pieces.append(torch.cos(ts / freq))
        feats = torch.cat(pieces, dim=-1)
        if attention_mask is not None:
            feats = feats * attention_mask.unsqueeze(-1).float()
        return self.proj(feats)
