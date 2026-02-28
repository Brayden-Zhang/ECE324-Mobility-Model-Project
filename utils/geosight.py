"""
GeoSight: Multi-modal Urban Perception for Physically-Grounded MFMs.

Implements:
  1. VisionSpatialEncoder — extracts semantic features from street-view / satellite imagery
  2. CrossModalAlignment — projects visual features into trajectory embedding space
  3. VisualSpatialMatchingHead — predicts visual features from movement patterns (and vice versa)
  4. GeoSightModel — full multi-modal MFM built on TrajectoryFM-HMT backbone
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from utils.hmt_backbone import FeedForward, MaskedTransformer
from utils.hmt_hierarchy import HMTEncoder, RegionTokenBuilder
from utils.flow import FlowMatchingHead


# ---------------------------------------------------------------------------
# 1. Vision Spatial Encoder
# ---------------------------------------------------------------------------

class FrozenVisionEncoder(nn.Module):
    """Wrapper around a frozen vision model (CLIP ViT, DINOv2, etc.).

    For prototyping, we simulate this with a learned CNN encoder on
    small image patches. In production, replace with:
        from transformers import CLIPModel
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    """

    def __init__(self, vision_dim: int = 512, pretrained: bool = False):
        super().__init__()
        self.vision_dim = vision_dim
        self.pretrained = pretrained

        if pretrained:
            # Placeholder for real CLIP loading
            # In practice: self.clip = CLIPModel.from_pretrained(...)
            self.encoder = nn.Identity()
            self.proj = nn.Linear(768, vision_dim)  # CLIP ViT-B outputs 768
        else:
            # Lightweight CNN for prototyping without heavy vision models
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.proj = nn.Linear(128, vision_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 3, H, W] → [B, vision_dim]"""
        if self.pretrained:
            with torch.no_grad():
                feats = self.encoder(images)
            return self.proj(feats)
        else:
            feats = self.encoder(images)
            return self.proj(feats)


class VisionSpatialEncoder(nn.Module):
    """Encodes per-location visual features from imagery.

    Given a set of location images (street-view, satellite), produces
    a visual feature vector per spatial token that can be aligned with
    trajectory embeddings.
    """

    def __init__(self, vision_dim: int = 512, embed_dim: int = 256, use_pretrained: bool = False):
        super().__init__()
        self.vision_encoder = FrozenVisionEncoder(vision_dim, pretrained=use_pretrained)
        self.vision_dim = vision_dim
        self.embed_dim = embed_dim

        # Semantic feature decomposition: extract interpretable urban features
        self.semantic_heads = nn.ModuleDict({
            "walkability": nn.Linear(vision_dim, 1),
            "commercial_density": nn.Linear(vision_dim, 1),
            "greenery": nn.Linear(vision_dim, 1),
            "road_width": nn.Linear(vision_dim, 1),
            "building_height": nn.Linear(vision_dim, 1),
            "pedestrian_density": nn.Linear(vision_dim, 1),
        })

        # Project concatenated semantic scores + raw vision to embed_dim
        semantic_dim = len(self.semantic_heads)
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + semantic_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        images: [B, 3, H, W] or [B, N, 3, H, W] (N locations per batch)
        Returns: (visual_embed [B, (N,) D], semantic_scores dict)
        """
        if images.dim() == 5:
            B, N, C, H, W = images.shape
            images_flat = images.reshape(B * N, C, H, W)
            vision_feats = self.vision_encoder(images_flat)
            vision_feats = vision_feats.reshape(B, N, -1)
        else:
            vision_feats = self.vision_encoder(images)
            if vision_feats.dim() == 2:
                vision_feats = vision_feats.unsqueeze(1)  # [B, 1, D]

        # Semantic decomposition
        semantics = {}
        scores_list = []
        for name, head in self.semantic_heads.items():
            score = head(vision_feats)  # [B, N, 1]
            semantics[name] = score.squeeze(-1)
            scores_list.append(score)
        semantic_concat = torch.cat(scores_list, dim=-1)  # [B, N, S]

        # Fuse vision + semantics
        fused = self.fusion(torch.cat([vision_feats, semantic_concat], dim=-1))
        return fused, semantics


# ---------------------------------------------------------------------------
# 2. Cross-Modal Alignment Module
# ---------------------------------------------------------------------------

class CrossModalAligner(nn.Module):
    """Projects visual features into the trajectory embedding space using
    contrastive learning (InfoNCE) and optional MSE alignment."""

    def __init__(self, embed_dim: int, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

        # Projection heads for contrastive alignment
        self.traj_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        traj_embed: torch.Tensor,    # [B, D]
        vision_embed: torch.Tensor,  # [B, D]
    ) -> Dict[str, torch.Tensor]:
        """Compute cross-modal alignment loss and similarity."""
        z_t = F.normalize(self.traj_proj(traj_embed), dim=-1)
        z_v = F.normalize(self.vision_proj(vision_embed), dim=-1)

        # InfoNCE
        logits = z_t @ z_v.T / self.temperature  # [B, B]
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_t2v = F.cross_entropy(logits, labels)
        loss_v2t = F.cross_entropy(logits.T, labels)
        contrastive_loss = (loss_t2v + loss_v2t) / 2

        # MSE alignment (soft target)
        mse_loss = F.mse_loss(z_t, z_v)

        return {
            "contrastive_loss": contrastive_loss,
            "mse_loss": mse_loss,
            "similarity": logits,
        }


# ---------------------------------------------------------------------------
# 3. Visual-Spatial Matching Heads
# ---------------------------------------------------------------------------

class VisualSpatialMatchingHead(nn.Module):
    """Bidirectional prediction:
    - traj_to_vision: predict visual semantics from trajectory patterns
    - vision_to_traj: predict movement statistics from visual context
    """

    def __init__(self, embed_dim: int, num_semantic_features: int = 6, num_movement_stats: int = 8):
        super().__init__()
        self.num_semantic = num_semantic_features
        self.num_movement = num_movement_stats

        # Trajectory → Visual semantics
        self.traj_to_visual = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_semantic_features),
        )

        # Visual → Movement statistics
        self.visual_to_movement = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_movement_stats),
        )

    def forward(
        self,
        traj_embed: torch.Tensor,     # [B, D]
        vision_embed: torch.Tensor,   # [B, D]
        semantic_targets: Optional[torch.Tensor] = None,  # [B, S]
        movement_targets: Optional[torch.Tensor] = None,  # [B, M]
    ) -> Dict[str, torch.Tensor]:
        pred_visual = self.traj_to_visual(traj_embed)
        pred_movement = self.visual_to_movement(vision_embed)

        result = {
            "pred_visual": pred_visual,
            "pred_movement": pred_movement,
        }

        if semantic_targets is not None:
            result["visual_loss"] = F.mse_loss(pred_visual, semantic_targets)
        if movement_targets is not None:
            result["movement_loss"] = F.mse_loss(pred_movement, movement_targets)

        return result


# ---------------------------------------------------------------------------
# 4. GeoSight Model
# ---------------------------------------------------------------------------

class GeoSightModel(nn.Module):
    """Multi-modal MFM combining trajectory and visual urban perception.

    Architecture:
    1. HMT encoder for trajectory tokens (reused from TrajectoryFM-HMT)
    2. Vision encoder for location-specific imagery
    3. Cross-modal fusion transformer
    4. Multiple prediction heads (token MLM, visual-spatial matching, flow)
    """

    def __init__(
        self,
        vocab_l0: int = 16384,
        vocab_l1: int = 4096,
        vocab_l2: int = 1024,
        embed_dim: int = 256,
        depth: int = 8,
        heads: int = 8,
        vision_dim: int = 512,
        max_seq_len: int = 200,
        dropout: float = 0.1,
        use_pretrained_vision: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        head_dim = embed_dim // heads

        # Trajectory encoder
        self.traj_encoder = HMTEncoder(
            vocab_l0=vocab_l0,
            vocab_l1=vocab_l1,
            vocab_l2=vocab_l2,
            embed_dim=embed_dim,
        )

        # Vision encoder
        self.vision_encoder = VisionSpatialEncoder(
            vision_dim=vision_dim,
            embed_dim=embed_dim,
            use_pretrained=use_pretrained_vision,
        )

        # Cross-modal fusion: concatenate trajectory tokens + visual tokens
        # then process with shared transformer
        self.fusion_proj = nn.Linear(embed_dim, embed_dim)
        self.type_embed = nn.Embedding(2, embed_dim)  # 0=traj, 1=visual

        self.transformer = MaskedTransformer(
            embedding_dim=embed_dim,
            depth=depth,
            num_heads=heads,
            head_dim=head_dim,
            feedforward_dim=embed_dim * 4,
            dropout=dropout,
            max_seq_len=max_seq_len + 64,  # extra for visual tokens
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Prediction heads
        self.token_head = nn.Linear(embed_dim, vocab_l0)
        self.cross_modal_aligner = CrossModalAligner(embed_dim)
        self.matching_head = VisualSpatialMatchingHead(embed_dim)
        self.flow_head = FlowMatchingHead(embed_dim)

    def forward(
        self,
        tokens_l0: torch.Tensor,        # [B, L]
        tokens_l1: torch.Tensor,        # [B, L]
        tokens_l2: torch.Tensor,        # [B, L]
        time_embed: torch.Tensor,       # [B, L, D]
        attention_mask: torch.Tensor,   # [B, L]
        location_images: Optional[torch.Tensor] = None,  # [B, N_img, 3, H, W]
        image_mask: Optional[torch.Tensor] = None,       # [B, N_img]
    ) -> Dict[str, torch.Tensor]:
        bsz, seq_len = tokens_l0.shape
        device = tokens_l0.device

        # 1) Trajectory encoding
        traj_embed = self.traj_encoder(tokens_l0, tokens_l1, tokens_l2, time_embed)
        traj_embed = traj_embed + self.type_embed(torch.zeros(bsz, seq_len, device=device, dtype=torch.long))

        # 2) Vision encoding (if images provided)
        if location_images is not None:
            vis_embed, semantics = self.vision_encoder(location_images)
            num_vis = vis_embed.shape[1]
            vis_embed = self.fusion_proj(vis_embed)
            vis_embed = vis_embed + self.type_embed(torch.ones(bsz, num_vis, device=device, dtype=torch.long))

            # Build joint attention mask
            if image_mask is None:
                image_mask = torch.ones(bsz, num_vis, device=device, dtype=torch.bool)
            joint_mask_size = seq_len + num_vis
            joint_attn = torch.zeros(bsz, joint_mask_size, joint_mask_size, device=device, dtype=torch.bool)
            # Traj tokens attend to each other
            for b in range(bsz):
                valid = attention_mask[b].bool()
                step_idx = valid.nonzero(as_tuple=False).squeeze(-1)
                if step_idx.numel() > 0:
                    joint_attn[b][step_idx[:, None], step_idx[None, :]] = True
                # Visual tokens attend to each other
                vis_valid = image_mask[b].bool()
                vis_idx = vis_valid.nonzero(as_tuple=False).squeeze(-1) + seq_len
                if vis_idx.numel() > 0:
                    joint_attn[b][vis_idx[:, None], vis_idx[None, :]] = True
                # Cross-attention: traj ↔ visual
                if step_idx.numel() > 0 and vis_idx.numel() > 0:
                    joint_attn[b][step_idx[:, None], vis_idx[None, :]] = True
                    joint_attn[b][vis_idx[:, None], step_idx[None, :]] = True
            # Self-attention
            for i in range(joint_mask_size):
                joint_attn[:, i, i] = True

            # Concatenate and process
            joint_tokens = torch.cat([traj_embed, vis_embed], dim=1)
            hidden = self.transformer(joint_tokens, attn_mask=joint_attn)
            hidden = self.norm(hidden)

            traj_hidden = hidden[:, :seq_len]
            vis_hidden = hidden[:, seq_len:]
        else:
            # Traj-only mode
            traj_attn = attention_mask.bool().unsqueeze(1) & attention_mask.bool().unsqueeze(2)
            hidden = self.transformer(traj_embed, attn_mask=traj_attn)
            hidden = self.norm(hidden)
            traj_hidden = hidden
            vis_hidden = None
            semantics = None

        # 3) Prediction heads
        token_logits = self.token_head(traj_hidden)

        # Trajectory-level pooling
        mask_f = attention_mask.unsqueeze(-1).float()
        traj_pool = (traj_hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        result = {
            "token_logits": token_logits,
            "traj_hidden": traj_hidden,
            "traj_pool": traj_pool,
        }

        if vis_hidden is not None:
            vis_mask_f = image_mask.unsqueeze(-1).float()
            vis_pool = (vis_hidden * vis_mask_f).sum(dim=1) / vis_mask_f.sum(dim=1).clamp(min=1)
            result["vis_hidden"] = vis_hidden
            result["vis_pool"] = vis_pool
            result["semantics"] = semantics

            # Cross-modal alignment
            align = self.cross_modal_aligner(traj_pool, vis_pool)
            result["contrastive_loss"] = align["contrastive_loss"]
            result["mse_align_loss"] = align["mse_loss"]

        return result


# ---------------------------------------------------------------------------
# 5. Loss Functions
# ---------------------------------------------------------------------------

def geosight_token_loss(
    logits: torch.Tensor,   # [B, L, V]
    targets: torch.Tensor,  # [B, L]
    mask_indices: torch.Tensor,  # [B, L] bool
    step_mask: torch.Tensor,     # [B, L]
) -> torch.Tensor:
    valid = mask_indices & step_mask.bool()
    loss = F.cross_entropy(logits.permute(0, 2, 1), targets, reduction="none")
    return torch.where(valid, loss, torch.zeros_like(loss)).sum() / valid.float().sum().clamp(min=1)


def geosight_visual_matching_loss(
    matching_head: VisualSpatialMatchingHead,
    traj_pool: torch.Tensor,
    vis_pool: torch.Tensor,
    semantic_targets: torch.Tensor,
    movement_targets: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    return matching_head(traj_pool, vis_pool, semantic_targets, movement_targets)


def compute_movement_statistics(
    coords: torch.Tensor,      # [B, L, 2]
    timestamps: torch.Tensor,  # [B, L]
    mask: torch.Tensor,        # [B, L]
) -> torch.Tensor:
    """Extract 8-dim movement statistics per trajectory for visual→movement prediction."""
    dt = torch.zeros_like(timestamps)
    dt[:, 1:] = (timestamps[:, 1:] - timestamps[:, :-1]).clamp(min=1e-3)
    delta = torch.zeros_like(coords)
    delta[:, 1:] = coords[:, 1:] - coords[:, :-1]
    speed = delta.norm(dim=-1) / dt.clamp(min=1e-3)
    speed = speed * mask

    valid_count = mask.sum(dim=1).clamp(min=1)
    mean_speed = (speed * mask).sum(dim=1) / valid_count
    std_speed = torch.sqrt(((speed - mean_speed.unsqueeze(1)).pow(2) * mask).sum(dim=1) / valid_count)

    # Heading changes
    heading = torch.atan2(delta[..., 0], delta[..., 1] + 1e-8)
    d_heading = torch.zeros_like(heading)
    d_heading[:, 1:] = heading[:, 1:] - heading[:, :-1]
    mean_turn = (d_heading.abs() * mask).sum(dim=1) / valid_count

    # Displacement
    start = coords[:, 0]
    end_idx = (mask.cumsum(dim=1) * mask).argmax(dim=1)
    end = coords[torch.arange(coords.shape[0]), end_idx]
    displacement = (end - start).norm(dim=-1)

    # Path length
    path_len = (delta.norm(dim=-1) * mask).sum(dim=1)
    straightness = displacement / (path_len + 1e-6)

    # Duration
    duration = (timestamps * mask).max(dim=1).values - timestamps[:, 0]

    stats = torch.stack([
        mean_speed, std_speed, mean_turn, displacement,
        path_len, straightness, duration, valid_count,
    ], dim=-1)
    # Normalize to prevent loss explosion when used as prediction targets
    stats = (stats - stats.mean(dim=0, keepdim=True)) / (stats.std(dim=0, keepdim=True) + 1e-6)
    return stats
