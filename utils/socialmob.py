"""
SocialMob: Multi-Agent Interaction Pre-training for Mobility Foundation Models.

Implements:
  1. RelationalAttention — attention with relative distance/velocity bias
  2. SocialConflictHead — predicts path intersection / avoidance between agents
  3. HierarchicalGroupTokenizer — coarse group-level tokenization for platoons/crowds
  4. SocialMobModel — full multi-agent MFM built on top of TrajectoryFM-HMT
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from utils.hmt_backbone import FeedForward, RotaryEmbedding
from utils.hmt_hierarchy import HMTEncoder, RegionTokenBuilder
from utils.hmt_backbone import MaskedTransformer
from utils.flow import FlowMatchingHead


# ---------------------------------------------------------------------------
# 1. Relational Attention: encodes pairwise distance + velocity differences
# ---------------------------------------------------------------------------

class RelationalBiasEncoder(nn.Module):
    """Encodes relative distance, velocity, and heading between all agent-step pairs
    into a scalar attention bias per head."""

    def __init__(self, num_heads: int, hidden_dim: int = 64):
        super().__init__()
        # 5 relational features: dx, dy, dist, dv_x, dv_y
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(
        self,
        coords: torch.Tensor,       # [B, N, 2]  (lat, lon)
        velocities: torch.Tensor,    # [B, N, 2]  (v_lat, v_lon)
    ) -> torch.Tensor:
        """Returns relational bias [B, num_heads, N, N]."""
        # Force float32 — coordinate deltas are tiny and lose precision in fp16
        orig_dtype = coords.dtype
        coords = coords.float()
        velocities = velocities.float()
        # pairwise differences
        dx = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B, N, N, 2]
        dist = dx.norm(dim=-1, keepdim=True)             # [B, N, N, 1]
        dv = velocities.unsqueeze(2) - velocities.unsqueeze(1)  # [B, N, N, 2]
        feats = torch.cat([dx, dist, dv], dim=-1)        # [B, N, N, 5]
        bias = self.net(feats.float())                    # [B, N, N, num_heads]
        return bias.permute(0, 3, 1, 2).to(orig_dtype)   # [B, H, N, N]


class RelationalAttention(nn.Module):
    """Multi-head attention with additive relational bias from pairwise spatial features."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        head_dim: int = 32,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        inner_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(embed_dim)
        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, embed_dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(head_dim, max_seq_len=max_seq_len)
        self.rel_bias = RelationalBiasEncoder(num_heads)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        velocities: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.norm(x)
        qkv = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = [t.view(b, n, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]

        # RoPE
        sin, cos = self.rotary(n)
        q1, q2 = q[..., :self.head_dim // 2], q[..., self.head_dim // 2:]
        k1, k2 = k[..., :self.head_dim // 2], k[..., self.head_dim // 2:]
        q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)

        # Attention scores + relational bias (compute in float32 for stability)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scale
        rel = self.rel_bias(coords, velocities)  # [B, H, N, N]
        scores = scores + rel.float()

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask > 0
            scores = scores.masked_fill(~attn_mask.unsqueeze(1), float("-inf"))

        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        probs = probs.nan_to_num(0.0)  # padding rows: softmax([-inf,...]) → NaN → 0
        probs = self.dropout(probs)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return self.to_out(out)


class RelationalTransformer(nn.Module):
    """Transformer stack with relational attention layers."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        head_dim: int,
        feedforward_dim: int,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                RelationalAttention(embed_dim, num_heads, head_dim, dropout, max_seq_len),
                FeedForward(embed_dim, feedforward_dim, dropout),
            ]))

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        velocities: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x, coords, velocities, attn_mask) + x
            x = ff(x) + x
        return x


# ---------------------------------------------------------------------------
# 2. Social Conflict Detection Head
# ---------------------------------------------------------------------------

class SocialConflictHead(nn.Module):
    """Predicts whether two agents' paths will intersect or require avoidance.

    Given embeddings for agent i and agent j, predicts a conflict score.
    Labels: 0 = no conflict, 1 = near-miss / intersection.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        agent_i: torch.Tensor,   # [B, P, D] pool of P agent-pair embeddings
        agent_j: torch.Tensor,   # [B, P, D]
    ) -> torch.Tensor:
        diff = agent_i - agent_j
        combined = torch.cat([agent_i, agent_j, diff], dim=-1)
        return self.net(combined)  # [B, P, 2]


# ---------------------------------------------------------------------------
# 3. Hierarchical Group Tokenizer
# ---------------------------------------------------------------------------

class GroupTokenBuilder(nn.Module):
    """Aggregates individual agent embeddings into group (platoon/crowd) tokens
    at a coarser scale using spatial proximity clustering."""

    def __init__(self, embed_dim: int, proximity_threshold: float = 0.001):
        super().__init__()
        self.embed_dim = embed_dim
        self.proximity_threshold = proximity_threshold
        self.group_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        agent_embeds: torch.Tensor,   # [B, A, D]  A = num agents
        agent_coords: torch.Tensor,   # [B, A, 2]  last known position per agent
        agent_mask: torch.Tensor,     # [B, A]     valid agents
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (group_tokens [B, G, D], group_mask [B, G], agent_to_group [B, A])."""
        bsz, num_agents, _ = agent_embeds.shape
        device = agent_embeds.device
        max_groups = 0
        group_assign_list = []
        group_embeds_list = []
        group_masks_list = []

        for b in range(bsz):
            valid = agent_mask[b].bool()
            valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
            n_valid = valid_idx.shape[0]
            if n_valid == 0:
                group_assign_list.append(torch.full((num_agents,), -1, device=device))
                group_embeds_list.append([])
                group_masks_list.append(0)
                continue

            # Simple agglomerative grouping by distance threshold
            coords_v = agent_coords[b, valid_idx]  # [n, 2]
            dist = torch.cdist(coords_v, coords_v)
            assigned = torch.full((n_valid,), -1, dtype=torch.long, device=device)
            gid = 0
            for i in range(n_valid):
                if assigned[i] >= 0:
                    continue
                assigned[i] = gid
                near = (dist[i] < self.proximity_threshold) & (assigned < 0)
                assigned[near] = gid
                gid += 1

            # Build group embeddings
            n_groups = gid
            g_embeds = torch.zeros(n_groups, self.embed_dim, device=device)
            for g in range(n_groups):
                members = valid_idx[(assigned == g).nonzero(as_tuple=False).squeeze(-1)]
                g_embeds[g] = agent_embeds[b, members].mean(dim=0)

            agent_assign = torch.full((num_agents,), -1, device=device, dtype=torch.long)
            agent_assign[valid_idx] = assigned
            group_assign_list.append(agent_assign)
            group_embeds_list.append(g_embeds)
            group_masks_list.append(n_groups)
            max_groups = max(max_groups, n_groups)

        if max_groups == 0:
            max_groups = 1
        group_tokens = torch.zeros(bsz, max_groups, self.embed_dim, device=device)
        group_mask = torch.zeros(bsz, max_groups, device=device, dtype=torch.bool)
        agent_to_group = torch.stack(group_assign_list, dim=0)

        for b in range(bsz):
            ng = group_masks_list[b]
            if ng > 0:
                group_tokens[b, :ng] = group_embeds_list[b]
                group_mask[b, :ng] = True

        group_tokens = self.group_proj(group_tokens)
        return group_tokens, group_mask, agent_to_group


# ---------------------------------------------------------------------------
# 4. Multi-Agent Dataset Utilities
# ---------------------------------------------------------------------------

def compute_velocities(coords: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Compute velocity vectors from coordinate deltas.
    coords: [B, L, 2], dt: [B, L] (seconds between consecutive steps).
    Returns: [B, L, 2] velocity in (lat/s, lon/s)."""
    v = torch.zeros_like(coords)
    delta = coords[:, 1:] - coords[:, :-1]
    safe_dt = dt[:, 1:].clamp(min=1e-3).unsqueeze(-1)
    v[:, 1:] = delta / safe_dt
    return v


def detect_conflicts_ground_truth(
    coords_a: torch.Tensor,  # [B, L, 2]
    coords_b: torch.Tensor,  # [B, L, 2]
    threshold: float = 0.0005,
) -> torch.Tensor:
    """Generate binary conflict labels: 1 if min pairwise distance < threshold."""
    # Compute pairwise distance between each time-step pair
    diff = coords_a.unsqueeze(2) - coords_b.unsqueeze(1)  # [B, L, L, 2]
    dist = diff.norm(dim=-1)  # [B, L, L]
    min_dist = dist.min(dim=-1).values.min(dim=-1).values  # [B]
    return (min_dist < threshold).long()


# ---------------------------------------------------------------------------
# 5. SocialMob Model
# ---------------------------------------------------------------------------

class SocialMobModel(nn.Module):
    """Multi-agent MFM that processes multiple agent trajectories jointly.

    Each agent is first encoded with a per-agent HMT encoder, then all agents'
    step-level embeddings are concatenated and processed by a RelationalTransformer
    that uses pairwise spatial/velocity biases.

    Additional heads:
    - Social conflict detection (pairwise)
    - Group-level token prediction
    - Standard masked token + flow matching (inherited from TrajectoryFM-HMT)
    """

    def __init__(
        self,
        vocab_l0: int = 16384,
        vocab_l1: int = 4096,
        vocab_l2: int = 1024,
        embed_dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        max_agents: int = 16,
        max_steps_per_agent: int = 64,
        dropout: float = 0.1,
        group_proximity: float = 0.001,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_agents = max_agents
        self.max_steps = max_steps_per_agent
        head_dim = embed_dim // heads

        # Per-agent encoder (shared weights across agents)
        self.agent_encoder = HMTEncoder(
            vocab_l0=vocab_l0,
            vocab_l1=vocab_l1,
            vocab_l2=vocab_l2,
            embed_dim=embed_dim,
        )

        # Agent ID embedding
        self.agent_id_embed = nn.Embedding(max_agents, embed_dim)

        # Relational transformer (processes all agents jointly)
        self.relational_transformer = RelationalTransformer(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=heads,
            head_dim=head_dim,
            feedforward_dim=embed_dim * 4,
            dropout=dropout,
            max_seq_len=max_agents * max_steps_per_agent,
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Group tokenizer
        self.group_builder = GroupTokenBuilder(embed_dim, proximity_threshold=group_proximity)

        # Prediction heads
        self.token_head = nn.Linear(embed_dim, vocab_l0)
        self.conflict_head = SocialConflictHead(embed_dim)
        self.flow_head = FlowMatchingHead(embed_dim)

        # Agent-level pooling → per-agent representation
        self.agent_pool_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

    def encode_agents(
        self,
        tokens_l0: torch.Tensor,   # [B, A, L]
        tokens_l1: torch.Tensor,   # [B, A, L]
        tokens_l2: torch.Tensor,   # [B, A, L]
        time_embed: torch.Tensor,  # [B, A, L, D]
        agent_mask: torch.Tensor,  # [B, A]
        step_mask: torch.Tensor,   # [B, A, L]
    ) -> torch.Tensor:
        """Encode each agent independently, return [B, A*L, D]."""
        bsz, num_agents, seq_len = tokens_l0.shape
        device = tokens_l0.device

        # Flatten agent dimension for shared encoder
        flat_l0 = tokens_l0.reshape(bsz * num_agents, seq_len)
        flat_l1 = tokens_l1.reshape(bsz * num_agents, seq_len)
        flat_l2 = tokens_l2.reshape(bsz * num_agents, seq_len)
        flat_time = time_embed.reshape(bsz * num_agents, seq_len, -1)

        # Per-agent encoding
        flat_embed = self.agent_encoder(flat_l0, flat_l1, flat_l2, flat_time)
        agent_embed = flat_embed.reshape(bsz, num_agents, seq_len, -1)

        # Add agent ID embedding
        agent_ids = torch.arange(num_agents, device=device).unsqueeze(0).expand(bsz, -1)
        aid_embed = self.agent_id_embed(agent_ids)  # [B, A, D]
        agent_embed = agent_embed + aid_embed.unsqueeze(2)

        # Flatten to [B, A*L, D]
        return agent_embed.reshape(bsz, num_agents * seq_len, -1)

    def forward(
        self,
        tokens_l0: torch.Tensor,   # [B, A, L]
        tokens_l1: torch.Tensor,   # [B, A, L]
        tokens_l2: torch.Tensor,   # [B, A, L]
        time_embed: torch.Tensor,  # [B, A, L, D]
        coords: torch.Tensor,      # [B, A, L, 2]
        dt: torch.Tensor,          # [B, A, L]
        agent_mask: torch.Tensor,  # [B, A]
        step_mask: torch.Tensor,   # [B, A, L]
        mask_indices: Optional[torch.Tensor] = None,  # [B, A, L] bool
    ) -> Dict[str, torch.Tensor]:
        bsz, num_agents, seq_len = tokens_l0.shape
        device = tokens_l0.device

        # 1) Per-agent encoding
        joint_embed = self.encode_agents(
            tokens_l0, tokens_l1, tokens_l2, time_embed, agent_mask, step_mask,
        )  # [B, A*L, D]

        # 2) Flatten coords + velocities for relational attention
        flat_coords = coords.reshape(bsz, num_agents * seq_len, 2)
        flat_dt = dt.reshape(bsz, num_agents * seq_len)
        flat_vel = compute_velocities(
            coords.reshape(bsz * num_agents, seq_len, 2),
            dt.reshape(bsz * num_agents, seq_len),
        ).reshape(bsz, num_agents * seq_len, 2)

        # Build joint attention mask
        flat_step_mask = step_mask.reshape(bsz, num_agents * seq_len).bool()
        joint_attn = flat_step_mask.unsqueeze(1) & flat_step_mask.unsqueeze(2)

        # 3) Relational Transformer
        hidden = self.relational_transformer(
            joint_embed, flat_coords, flat_vel, attn_mask=joint_attn,
        )
        hidden = self.norm(hidden)

        # 4) Masked token prediction
        token_logits = self.token_head(hidden)  # [B, A*L, V]

        # 5) Agent-level pooling for conflict detection
        agent_hidden = hidden.reshape(bsz, num_agents, seq_len, -1)
        # Mean-pool each agent's steps
        agent_step_mask = step_mask.unsqueeze(-1).float()
        agent_pool = (agent_hidden * agent_step_mask).sum(dim=2) / agent_step_mask.sum(dim=2).clamp(min=1)
        agent_pool = self.agent_pool_proj(agent_pool)  # [B, A, D]

        # 6) Build group tokens
        last_coords = coords[:, :, -1, :]  # [B, A, 2] last position per agent
        group_tokens, group_mask, agent_to_group = self.group_builder(
            agent_pool, last_coords, agent_mask,
        )

        # 7) Pairwise conflict logits (all pairs among valid agents)
        # Extract upper-triangle pairs
        pair_i_list, pair_j_list = [], []
        for a in range(num_agents):
            for b_idx in range(a + 1, num_agents):
                pair_i_list.append(a)
                pair_j_list.append(b_idx)
        if len(pair_i_list) > 0:
            pair_i = torch.tensor(pair_i_list, device=device)
            pair_j = torch.tensor(pair_j_list, device=device)
            emb_i = agent_pool[:, pair_i]  # [B, P, D]
            emb_j = agent_pool[:, pair_j]  # [B, P, D]
            conflict_logits = self.conflict_head(emb_i, emb_j)  # [B, P, 2]
        else:
            conflict_logits = torch.zeros(bsz, 0, 2, device=device)

        return {
            "token_logits": token_logits,
            "hidden": hidden,
            "agent_pool": agent_pool,
            "conflict_logits": conflict_logits,
            "group_tokens": group_tokens,
            "group_mask": group_mask,
            "agent_to_group": agent_to_group,
        }


# ---------------------------------------------------------------------------
# 6. Loss Functions
# ---------------------------------------------------------------------------

def socialmob_token_loss(
    logits: torch.Tensor,       # [B, A*L, V]
    targets: torch.Tensor,      # [B, A, L]
    mask_indices: torch.Tensor, # [B, A, L] bool — which positions are masked
    step_mask: torch.Tensor,    # [B, A, L]
) -> torch.Tensor:
    bsz, num_agents, seq_len = targets.shape
    flat_targets = targets.reshape(bsz, num_agents * seq_len)
    flat_mask = (mask_indices & step_mask.bool()).reshape(bsz, num_agents * seq_len)
    loss = F.cross_entropy(
        logits.permute(0, 2, 1),
        flat_targets,
        reduction="none",
    )
    masked_loss = torch.where(flat_mask, loss, torch.zeros_like(loss))
    return masked_loss.sum() / flat_mask.float().sum().clamp(min=1)


def socialmob_conflict_loss(
    conflict_logits: torch.Tensor,  # [B, P, 2]
    conflict_labels: torch.Tensor,  # [B, P]
    pair_mask: torch.Tensor,        # [B, P] valid pairs
) -> torch.Tensor:
    if conflict_logits.shape[1] == 0:
        return conflict_logits.new_tensor(0.0)
    loss = F.cross_entropy(
        conflict_logits.reshape(-1, 2),
        conflict_labels.reshape(-1),
        reduction="none",
    )
    loss = loss.reshape_as(conflict_labels)
    return (loss * pair_mask.float()).sum() / pair_mask.float().sum().clamp(min=1)
