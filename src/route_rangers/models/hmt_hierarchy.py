from typing import Optional, Tuple

import torch
from torch import nn


class HMTEncoder(nn.Module):
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
        self.vocab_l0 = vocab_l0
        self.vocab_l1 = vocab_l1
        self.vocab_l2 = vocab_l2
        self.embed_l0 = nn.Embedding(vocab_l0 + 1, embed_dim)
        self.embed_l1 = nn.Embedding(vocab_l1 + 1, embed_dim)
        self.embed_l2 = nn.Embedding(vocab_l2 + 1, embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.context_mlp = None
        if context_dim > 0:
            self.context_mlp = nn.Sequential(
                nn.Linear(context_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
        self.trip_mlp = None
        if trip_feat_dim > 0:
            self.trip_mlp = nn.Sequential(
                nn.Linear(trip_feat_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )

    def forward(
        self,
        tokens_l0: torch.Tensor,
        tokens_l1: torch.Tensor,
        tokens_l2: torch.Tensor,
        time_embed: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        trip_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = (
            self.embed_l0(tokens_l0)
            + self.embed_l1(tokens_l1)
            + self.embed_l2(tokens_l2)
        )
        h = h + self.time_mlp(time_embed)
        if context is not None and self.context_mlp is not None:
            h = h + self.context_mlp(context)
        if trip_features is not None and self.trip_mlp is not None:
            h = h + self.trip_mlp(trip_features)
        return h


class RegionTokenBuilder(nn.Module):
    def __init__(self, embed_table: nn.Embedding, pad_id: int):
        super().__init__()
        self.embed_table = embed_table
        self.pad_id = pad_id

    def forward(
        self,
        tokens_level: torch.Tensor,
        step_embed: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_ratio: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = tokens_level.device
        bsz, seq_len = tokens_level.shape
        embed_dim = step_embed.shape[-1]

        region_ids_list = []
        region_steps_list = []
        step_to_region_list = []
        max_regions = 0

        for b in range(bsz):
            valid_len = int(attention_mask[b].sum().item())
            token_seq = tokens_level[b, :valid_len].tolist()
            region_map = {}
            region_ids = []
            region_steps = []
            step_to_region = torch.full((seq_len,), -1, device=device, dtype=torch.long)
            for t, tok in enumerate(token_seq):
                idx = region_map.get(tok)
                if idx is None:
                    idx = len(region_ids)
                    region_map[tok] = idx
                    region_ids.append(tok)
                    region_steps.append([])
                region_steps[idx].append(t)
                step_to_region[t] = idx
            region_ids_list.append(region_ids)
            region_steps_list.append(region_steps)
            step_to_region_list.append(step_to_region)
            max_regions = max(max_regions, len(region_ids))

        region_ids = torch.full(
            (bsz, max_regions), self.pad_id, device=device, dtype=torch.long
        )
        region_mask = torch.zeros((bsz, max_regions), device=device, dtype=torch.bool)
        region_agg = torch.zeros(
            (bsz, max_regions, embed_dim), device=step_embed.device
        )

        for b in range(bsz):
            ids = region_ids_list[b]
            if not ids:
                continue
            count = len(ids)
            region_ids[b, :count] = torch.tensor(ids, device=device, dtype=torch.long)
            region_mask[b, :count] = True
            step_emb = step_embed[b]
            agg = torch.zeros((count, embed_dim), device=step_embed.device)
            for idx, steps in enumerate(region_steps_list[b]):
                if not steps:
                    continue
                step_idx = torch.tensor(
                    steps, device=step_embed.device, dtype=torch.long
                )
                agg[idx] = step_emb.index_select(0, step_idx).mean(dim=0)
            region_agg[b, :count] = agg

        mlm_mask = torch.zeros((bsz, max_regions), device=device, dtype=torch.bool)
        if mask_ratio > 0 and max_regions > 0:
            rand = torch.rand((bsz, max_regions), device=device)
            mlm_mask = (rand < mask_ratio) & region_mask

        masked_ids = region_ids.clone()
        masked_ids[mlm_mask] = self.pad_id
        region_tokens = self.embed_table(masked_ids) + region_agg

        step_to_region = torch.stack(step_to_region_list, dim=0)
        return region_tokens, region_ids, region_mask, step_to_region, mlm_mask
