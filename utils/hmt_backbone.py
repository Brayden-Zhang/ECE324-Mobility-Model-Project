import math
from typing import Optional

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, max_seq_len: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim))
        positions = torch.arange(max_seq_len).float()
        sinusoid_input = torch.einsum("i, j -> i j", positions, inv_freq)
        self.register_buffer("sin", sinusoid_input.sin(), persistent=False)
        self.register_buffer("cos", sinusoid_input.cos(), persistent=False)

    def forward(self, seq_len: int):
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
        return sin, cos


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskedAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        max_seq_len: int = 512,
    ):
        super().__init__()
        inner_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.norm = nn.LayerNorm(embedding_dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(embedding_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, embedding_dim), nn.Dropout(dropout))

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, _ = x.shape
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, n, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]

        sin, cos = self.rotary_emb(n)

        q1, q2 = q[..., : self.head_dim // 2], q[..., self.head_dim // 2 :]
        k1, k2 = k[..., : self.head_dim // 2], k[..., self.head_dim // 2 :]
        q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)

        attn_scores = torch.matmul(q_rot, k_rot.transpose(-1, -2)) * self.scale
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask > 0
            attn_scores = attn_scores.masked_fill(~attn_mask.unsqueeze(1), float("-inf"))

        attn_probs = self.attend(attn_scores)
        attn_probs = attn_probs.nan_to_num(0.0)  # padding rows: softmax([-inf,...]) → NaN → 0
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(b, n, self.num_heads * self.head_dim)
        return self.to_out(out)


class MaskedTransformer(nn.Module):
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
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        MaskedAttention(
                            embedding_dim,
                            num_heads=num_heads,
                            head_dim=head_dim,
                            dropout=dropout,
                            max_seq_len=max_seq_len,
                        ),
                        FeedForward(embedding_dim, feedforward_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for attn_layer, ff_layer in self.layers:
            x = attn_layer(x, attn_mask=attn_mask) + x
            x = ff_layer(x) + x
        return x


class GraphAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(embedding_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, embedding_dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, graph_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        h = self.norm(x)
        qkv = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = [t.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if graph_mask is not None:
            mask = graph_mask
            if mask.dtype != torch.bool:
                mask = mask > 0
            eye = torch.eye(seq_len, device=x.device, dtype=torch.bool).unsqueeze(0)
            mask = mask | eye
            min_score = -65504.0 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(~mask.unsqueeze(1), min_score)

        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        return self.to_out(out)


class GraphTrajectoryEncoder(nn.Module):
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
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        GraphAttention(
                            embedding_dim=embedding_dim,
                            num_heads=num_heads,
                            head_dim=head_dim,
                            dropout=dropout,
                        ),
                        FeedForward(embedding_dim, feedforward_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, graph_mask: torch.Tensor) -> torch.Tensor:
        for attn_layer, ff_layer in self.layers:
            x = attn_layer(x, graph_mask=graph_mask) + x
            x = ff_layer(x) + x
        return x
