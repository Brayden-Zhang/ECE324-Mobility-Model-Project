from typing import Tuple

import torch
from torch import nn


class RectifiedFlowHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int = 2, mlp_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + out_dim + 1, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, out_dim),
        )

    def forward(
        self, hidden: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        # hidden: [B, L, D], x_t: [B, L, 2], t: [B, L, 1] or [B, 1, 1]
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        if t.shape[1] == 1:
            t = t.expand(-1, hidden.shape[1], -1)
        inp = torch.cat([hidden, x_t, t], dim=-1)
        return self.net(inp)


class FlowMatchingHead(RectifiedFlowHead):
    pass


def sample_rectified_flow_targets(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # x: [B, L, 2]
    b, seq_len, _ = x.shape
    t = torch.rand((b, 1, 1), device=x.device)
    x0 = torch.randn_like(x)
    x_t = t * x + (1 - t) * x0
    target_v = x - x0
    return x_t, target_v, t


def flow_matching_loss(
    pred_v: torch.Tensor, target_v: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    # mask: [B, L]
    mse = (pred_v - target_v).pow(2).sum(dim=-1)
    masked = torch.where(mask.bool(), mse, torch.zeros_like(mse))
    return masked.sum() / (mask.sum() + 1e-6)
