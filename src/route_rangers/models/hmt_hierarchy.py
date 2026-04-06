"""Hierarchy helpers for the lightweight HMT model wrapper."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class HMTEncoder(nn.Module):
    """Embed hierarchical tokens plus optional auxiliary features."""

    def __init__(
        self,
        vocab_l0: int,
        vocab_l1: int,
        vocab_l2: int,
        embed_dim: int,
        context_dim: int = 0,
        trip_feat_dim: int = 0,
    ):
        super().__init__()
        self.embed_l0 = nn.Embedding(vocab_l0, embed_dim)
        self.embed_l1 = nn.Embedding(vocab_l1, embed_dim)
        self.embed_l2 = nn.Embedding(vocab_l2, embed_dim)
        self.time_proj = nn.Identity()
        self.context_proj = (
            nn.Linear(context_dim, embed_dim) if context_dim > 0 else None
        )
        self.trip_proj = (
            nn.Linear(trip_feat_dim, embed_dim) if trip_feat_dim > 0 else None
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tokens_l0: torch.Tensor,
        tokens_l1: torch.Tensor,
        tokens_l2: torch.Tensor,
        time_embed: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        trip_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = (
            self.embed_l0(tokens_l0)
            + self.embed_l1(tokens_l1)
            + self.embed_l2(tokens_l2)
            + self.time_proj(time_embed)
        )
        if self.context_proj is not None and context is not None:
            hidden = hidden + self.context_proj(context)
        if self.trip_proj is not None and trip_features is not None:
            hidden = hidden + self.trip_proj(trip_features)
        return self.norm(hidden)


class RegionTokenBuilder(nn.Module):
    """Aggregate step embeddings into per-region tokens."""

    def __init__(self, embedding: nn.Embedding, pad_id: int):
        super().__init__()
        del embedding
        self.pad_id = pad_id

    def forward(
        self,
        region_ids: torch.Tensor,
        step_embed: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del mask_ratio
        batch_size, seq_len, embed_dim = step_embed.shape
        device = step_embed.device

        mappings: list[dict[int, int]] = []
        max_regions = 0

        valid_steps = attention_mask.bool()
        for batch_idx in range(batch_size):
            mapping: dict[int, int] = {}
            ordered_ids: list[int] = []
            for step_idx in range(seq_len):
                if not valid_steps[batch_idx, step_idx]:
                    continue
                token_id = int(region_ids[batch_idx, step_idx].item())
                if token_id not in mapping:
                    mapping[token_id] = len(ordered_ids)
                    ordered_ids.append(token_id)
            mappings.append(mapping)
            max_regions = max(max_regions, len(ordered_ids))

        if max_regions == 0:
            empty_tokens = step_embed.new_zeros((batch_size, 0, embed_dim))
            empty_ids = region_ids.new_full((batch_size, 0), self.pad_id)
            empty_mask = attention_mask.new_zeros((batch_size, 0), dtype=torch.bool)
            step_to_region = region_ids.new_full((batch_size, seq_len), -1)
            empty_mlm = attention_mask.new_zeros((batch_size, 0), dtype=torch.bool)
            return empty_tokens, empty_ids, empty_mask, step_to_region, empty_mlm

        tokens = step_embed.new_zeros((batch_size, max_regions, embed_dim))
        ids_out = region_ids.new_full((batch_size, max_regions), self.pad_id)
        mask_out = torch.zeros(
            (batch_size, max_regions),
            device=device,
            dtype=torch.bool,
        )
        mlm_mask = torch.zeros(
            (batch_size, max_regions),
            device=device,
            dtype=torch.bool,
        )
        step_to_region = region_ids.new_full((batch_size, seq_len), -1)

        for batch_idx in range(batch_size):
            mapping = mappings[batch_idx]
            counts = torch.zeros(
                (len(mapping),),
                device=device,
                dtype=step_embed.dtype,
            )
            for token_id, region_idx in mapping.items():
                ids_out[batch_idx, region_idx] = token_id
                mask_out[batch_idx, region_idx] = True
            for step_idx in range(seq_len):
                if not valid_steps[batch_idx, step_idx]:
                    continue
                region_idx = mapping[int(region_ids[batch_idx, step_idx].item())]
                step_to_region[batch_idx, step_idx] = region_idx
                tokens[batch_idx, region_idx] += step_embed[batch_idx, step_idx]
                counts[region_idx] += 1.0
            if counts.numel() > 0:
                tokens[batch_idx, : counts.shape[0]] = (
                    tokens[batch_idx, : counts.shape[0]]
                    / counts.clamp(min=1.0).unsqueeze(-1)
                )

        return tokens, ids_out, mask_out, step_to_region, mlm_mask
