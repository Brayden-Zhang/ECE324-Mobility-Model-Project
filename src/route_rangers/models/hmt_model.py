import math
from typing import Optional, Dict

import torch
from torch import nn

from route_rangers.models.flow import FlowMatchingHead
from route_rangers.models.hmt_backbone import GraphTrajectoryEncoder, MaskedTransformer
from route_rangers.models.hmt_hierarchy import HMTEncoder, RegionTokenBuilder
from route_rangers.models.spacetime import SpaceTimeEncoder


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    mask = mask.unsqueeze(-1).float()
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class TrajectoryFMHMT(nn.Module):
    def __init__(
        self,
        vocab_l0: int,
        vocab_l1: int,
        vocab_l2: int,
        embed_dim: int = 256,
        depth: int = 8,
        heads: int = 8,
        dropout: float = 0.0,
        context_dim: int = 0,
        trip_feat_dim: int = 0,
        max_seq_len: int = 512,
        use_graph: bool = False,
        graph_layers: int = 2,
        graph_knn: int = 8,
        graph_temporal_window: int = 2,
        graph_same_region: bool = True,
        step_attention_window: int = 0,
        use_spacetime: bool = False,
        spacetime_freqs: int = 6,
        macro_region_vocab: int = 0,
        macro_dist_dim: int = 0,
    ):
        super().__init__()
        self.vocab_l0 = vocab_l0
        self.vocab_l1 = vocab_l1
        self.vocab_l2 = vocab_l2
        self.embed_dim = embed_dim
        self.use_graph = use_graph
        self.graph_knn = graph_knn
        self.graph_temporal_window = graph_temporal_window
        self.graph_same_region = graph_same_region
        self.step_attention_window = step_attention_window
        self.use_spacetime = use_spacetime
        self.macro_region_vocab = macro_region_vocab
        self.macro_dist_dim = macro_dist_dim

        self.encoder = HMTEncoder(
            vocab_l0=vocab_l0,
            vocab_l1=vocab_l1,
            vocab_l2=vocab_l2,
            embed_dim=embed_dim,
            context_dim=context_dim,
            trip_feat_dim=trip_feat_dim,
        )
        self.region_mid_builder = RegionTokenBuilder(self.encoder.embed_l1, pad_id=vocab_l1)
        self.region_coarse_builder = RegionTokenBuilder(self.encoder.embed_l2, pad_id=vocab_l2)

        self.transformer = MaskedTransformer(
            embed_dim,
            depth=depth,
            num_heads=heads,
            head_dim=embed_dim // heads,
            feedforward_dim=embed_dim * 4,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.graph_encoder = None
        if use_graph and graph_layers > 0:
            self.graph_encoder = GraphTrajectoryEncoder(
                embedding_dim=embed_dim,
                depth=graph_layers,
                num_heads=heads,
                head_dim=embed_dim // heads,
                feedforward_dim=embed_dim * 4,
                dropout=dropout,
            )
        self.norm = nn.LayerNorm(embed_dim)
        self.spacetime_encoder = None
        if self.use_spacetime:
            self.spacetime_encoder = SpaceTimeEncoder(embed_dim, num_freqs=spacetime_freqs)

        self.macro_region_embed = None
        self.macro_head = None
        if self.macro_region_vocab > 0 and self.macro_dist_dim > 0:
            self.macro_region_embed = nn.Embedding(self.macro_region_vocab, embed_dim)
            self.macro_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, self.macro_dist_dim),
            )

        self.head_l0 = nn.Linear(embed_dim, vocab_l0)
        self.head_l1 = nn.Linear(embed_dim, vocab_l1)
        self.head_l2 = nn.Linear(embed_dim, vocab_l2)
        self.region_head_l1 = nn.Linear(embed_dim, vocab_l1)
        self.region_head_l2 = nn.Linear(embed_dim, vocab_l2)

        self.dest_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, vocab_l1),
        )
        self.dest_uncertainty_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        self.flow_head = FlowMatchingHead(embed_dim)
        self.flow_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def build_step_graph(
        self,
        coords: Optional[torch.Tensor],
        step_mask: torch.Tensor,
        tokens_l1: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len = step_mask.shape
        graph = torch.zeros((bsz, seq_len, seq_len), device=step_mask.device, dtype=torch.bool)
        valid = step_mask.bool()

        for b in range(bsz):
            valid_idx = valid[b].nonzero(as_tuple=False).squeeze(-1)
            vlen = int(valid_idx.shape[0])
            if vlen == 0:
                continue

            graph[b, valid_idx, valid_idx] = True

            # Temporal locality edges around each step.
            if self.graph_temporal_window > 0:
                for t in range(vlen):
                    i = valid_idx[t].item()
                    left = max(0, t - self.graph_temporal_window)
                    right = min(vlen, t + self.graph_temporal_window + 1)
                    nbr = valid_idx[left:right]
                    graph[b, i, nbr] = True
                    graph[b, nbr, i] = True

            # Region ties encourage short paths in the same mid-level area.
            if self.graph_same_region and tokens_l1 is not None:
                tok = tokens_l1[b, valid_idx]
                same = tok.unsqueeze(1) == tok.unsqueeze(0)
                ii = valid_idx.unsqueeze(1).expand(vlen, vlen)
                jj = valid_idx.unsqueeze(0).expand(vlen, vlen)
                graph[b, ii, jj] = graph[b, ii, jj] | same

            # Spatial links via kNN on lat/lon.
            if coords is not None and self.graph_knn > 0 and vlen > 1:
                pts = coords[b, valid_idx]
                dist = torch.cdist(pts, pts)
                k = min(self.graph_knn + 1, vlen)
                nn_idx = torch.topk(dist, k=k, dim=-1, largest=False).indices
                for t in range(vlen):
                    i = valid_idx[t].item()
                    nbr = valid_idx[nn_idx[t]]
                    graph[b, i, nbr] = True
                    graph[b, nbr, i] = True
        return graph

    def build_attention_mask(
        self,
        step_mask: torch.Tensor,
        step_to_mid: torch.Tensor,
        mid_mask: torch.Tensor,
        step_to_coarse: torch.Tensor,
        coarse_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len = step_mask.shape
        mid_count = mid_mask.shape[1]
        coarse_count = coarse_mask.shape[1]
        total = seq_len + mid_count + coarse_count
        device = step_mask.device

        attn_mask = torch.zeros((bsz, total, total), device=device, dtype=torch.bool)
        for b in range(bsz):
            valid_steps = step_mask[b].bool()
            if valid_steps.any():
                step_idx = valid_steps.nonzero(as_tuple=False).squeeze(-1)
                if self.step_attention_window > 0:
                    step_idx_list = step_idx.tolist()
                    span = int(self.step_attention_window)
                    for pos, step_pos in enumerate(step_idx_list):
                        left = max(0, pos - span)
                        right = min(len(step_idx_list), pos + span + 1)
                        nbr = step_idx[left:right]
                        attn_mask[b, step_pos, nbr] = True
                else:
                    attn_mask[b][step_idx[:, None], step_idx[None, :]] = True
                for t in step_idx.tolist():
                    mid_idx = step_to_mid[b, t].item()
                    if mid_idx >= 0 and mid_idx < mid_count and mid_mask[b, mid_idx]:
                        mid_pos = seq_len + mid_idx
                        attn_mask[b, t, mid_pos] = True
                        attn_mask[b, mid_pos, t] = True
                    coarse_idx = step_to_coarse[b, t].item()
                    if coarse_idx >= 0 and coarse_idx < coarse_count and coarse_mask[b, coarse_idx]:
                        coarse_pos = seq_len + mid_count + coarse_idx
                        attn_mask[b, t, coarse_pos] = True
                        attn_mask[b, coarse_pos, t] = True

            for r in range(mid_count):
                if mid_mask[b, r]:
                    pos = seq_len + r
                    attn_mask[b, pos, pos] = True
            for r in range(coarse_count):
                if coarse_mask[b, r]:
                    pos = seq_len + mid_count + r
                    attn_mask[b, pos, pos] = True

            for pos in range(total):
                attn_mask[b, pos, pos] = True
        return attn_mask

    def meso_flow_logits(self, mid_hidden: torch.Tensor, mid_mask: torch.Tensor) -> torch.Tensor:
        if mid_hidden.numel() == 0:
            return mid_hidden.new_zeros((mid_hidden.shape[0], 0, 0))
        proj = self.flow_proj(mid_hidden)
        logits = torch.matmul(proj, proj.transpose(1, 2)) / math.sqrt(proj.shape[-1])
        if mid_mask is not None and mid_mask.numel() > 0:
            pair_mask = mid_mask.unsqueeze(1) & mid_mask.unsqueeze(2)
            logits = logits.masked_fill(~pair_mask, 0.0)
        return logits

    def macro_logits(self, region_idx: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        if self.macro_head is None or self.macro_region_embed is None:
            raise RuntimeError("macro_head is not configured")
        region_emb = self.macro_region_embed(region_idx)
        return self.macro_head(region_emb + time_embed)

    def forward(
        self,
        tokens_l0: torch.Tensor,
        tokens_l1: torch.Tensor,
        tokens_l2: torch.Tensor,
        time_embed: torch.Tensor,
        attention_mask: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        trip_features: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        region_mask_ratio: float = 0.0,
        region_source_l1: Optional[torch.Tensor] = None,
        region_source_l2: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        step_embed = self.encoder(
            tokens_l0,
            tokens_l1,
            tokens_l2,
            time_embed,
            context,
            trip_features=trip_features,
        )
        if self.spacetime_encoder is not None and coords is not None and timestamps is not None:
            step_embed = step_embed + self.spacetime_encoder(coords, timestamps, attention_mask)
        if self.graph_encoder is not None:
            graph_mask = self.build_step_graph(coords, attention_mask, tokens_l1)
            step_embed = self.graph_encoder(step_embed, graph_mask=graph_mask)

        if region_source_l1 is None:
            region_source_l1 = tokens_l1
        if region_source_l2 is None:
            region_source_l2 = tokens_l2

        mid_tokens, mid_ids, mid_mask, step_to_mid, mid_mlm_mask = self.region_mid_builder(
            region_source_l1,
            step_embed,
            attention_mask,
            mask_ratio=region_mask_ratio,
        )
        coarse_tokens, coarse_ids, coarse_mask, step_to_coarse, coarse_mlm_mask = self.region_coarse_builder(
            region_source_l2,
            step_embed,
            attention_mask,
            mask_ratio=region_mask_ratio,
        )

        full_tokens = torch.cat([step_embed, mid_tokens, coarse_tokens], dim=1)
        attn_mask = self.build_attention_mask(attention_mask, step_to_mid, mid_mask, step_to_coarse, coarse_mask)
        hidden = self.transformer(full_tokens, attn_mask=attn_mask)
        hidden = self.norm(hidden)

        seq_len = tokens_l0.shape[1]
        mid_count = mid_tokens.shape[1]
        step_hidden = hidden[:, :seq_len]
        mid_hidden = hidden[:, seq_len : seq_len + mid_count]
        coarse_hidden = hidden[:, seq_len + mid_count :]

        step_logits = {
            "l0": self.head_l0(step_hidden),
            "l1": self.head_l1(step_hidden),
            "l2": self.head_l2(step_hidden),
        }
        region_logits = {
            "l1": self.region_head_l1(mid_hidden),
            "l2": self.region_head_l2(coarse_hidden),
        }

        pooled_step = masked_mean(step_hidden, attention_mask)
        pooled_mid = masked_mean(mid_hidden, mid_mask) if mid_count > 0 else torch.zeros_like(pooled_step)
        dest_context = torch.cat([pooled_step, pooled_mid], dim=-1)
        dest_logits = self.dest_head(dest_context)
        dest_log_var = self.dest_uncertainty_head(dest_context).squeeze(-1).clamp(min=-6.0, max=6.0)

        return {
            "step_hidden": step_hidden,
            "mid_hidden": mid_hidden,
            "coarse_hidden": coarse_hidden,
            "step_logits": step_logits,
            "region_logits": region_logits,
            "dest_logits": dest_logits,
            "dest_log_var": dest_log_var,
            "mid_ids": mid_ids,
            "coarse_ids": coarse_ids,
            "mid_mask": mid_mask,
            "coarse_mask": coarse_mask,
            "mid_mlm_mask": mid_mlm_mask,
            "coarse_mlm_mask": coarse_mlm_mask,
            "step_to_mid": step_to_mid,
        }
