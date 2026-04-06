"""Lightweight flow-matching helpers used by the reproducibility snapshot."""

from __future__ import annotations

import torch
from torch import nn


class FlowMatchingHead(nn.Module):
    """Small compatibility head for meso-flow supervision."""

    def __init__(self, embed_dim: int, output_dim: int = 2):
        super().__init__()
        self.proj = nn.Linear(embed_dim, output_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


def sample_rectified_flow_targets(
    coords: torch.Tensor,
    noise_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return noisy coordinates and the residual target used by simple flow losses."""

    noise = torch.randn_like(coords) * noise_scale
    noisy = coords + noise
    target = coords - noisy
    return noisy, target


def flow_matching_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean-squared error with optional boolean masking."""

    diff = (pred - target) ** 2
    if mask is not None:
        diff = diff * mask.unsqueeze(-1).float()
        denom = mask.sum().clamp(min=1).float()
        return diff.sum() / denom
    return diff.mean()
