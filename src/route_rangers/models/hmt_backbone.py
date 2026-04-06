"""Compact backbone blocks used by the reproducibility snapshot."""

from __future__ import annotations

import torch
from torch import nn


class MaskedTransformer(nn.Module):
    """Small batch-first Transformer encoder.

    The compatibility implementation accepts an attention-mask argument to match
    the original API, but only applies standard self-attention without the full
    custom sparse masking stack used in the research code.
    """

    def __init__(
        self,
        embedding_dim: int,
        depth: int,
        num_heads: int,
        head_dim: int,
        feedforward_dim: int,
        dropout: float = 0.0,
        max_seq_len: int = 512,
    ):
        super().__init__()
        del head_dim, max_seq_len
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(
        self,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attn_mask
        return self.encoder(tokens)


class GraphTrajectoryEncoder(nn.Module):
    """Compatibility graph encoder that reuses the transformer backbone."""

    def __init__(
        self,
        embedding_dim: int,
        depth: int,
        num_heads: int,
        head_dim: int,
        feedforward_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = MaskedTransformer(
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            head_dim=head_dim,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        graph_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del graph_mask
        return self.encoder(tokens)
