"""
Metabolic-Mobility: Multi-Objective Foundation Models for Urban Sustainability.

Implements:
  1. MetabolicTokenizer — extends trajectory tokens with energy/emission features
  2. UrbanMetabolismEncoder — fuses mobility with environmental sensor streams
  3. ScenarioDiscoveryHead — "What-if" analysis for urban planning
  4. MetabolicMobilityModel — full sustainability-aware MFM
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from utils.hmt_backbone import FeedForward, MaskedTransformer
from utils.hmt_hierarchy import HMTEncoder
from utils.flow import FlowMatchingHead


# ---------------------------------------------------------------------------
# 1. Metabolic Tokenizer — enriched (x, y, t) with energy + emissions
# ---------------------------------------------------------------------------

class TransportModeEncoder(nn.Module):
    """Encodes transport mode with associated emission/energy profiles.

    Modes: 0=walk, 1=bike, 2=bus, 3=car, 4=subway, 5=unknown
    """
    NUM_MODES = 6

    # Approximate emission factors (gCO2/km) and energy cost (kJ/km)
    EMISSION_FACTORS = [0.0, 0.0, 89.0, 192.0, 41.0, 100.0]
    ENERGY_FACTORS = [250.0, 100.0, 800.0, 2500.0, 400.0, 1000.0]

    def __init__(self, embed_dim: int):
        super().__init__()
        self.mode_embed = nn.Embedding(self.NUM_MODES, embed_dim)

        # Learnable emission/energy factor adjustments
        self.emission_base = nn.Parameter(
            torch.tensor(self.EMISSION_FACTORS, dtype=torch.float32)
        )
        self.energy_base = nn.Parameter(
            torch.tensor(self.ENERGY_FACTORS, dtype=torch.float32)
        )
        # Contextual adjustment (by speed, traffic conditions)
        self.adjustment = nn.Sequential(
            nn.Linear(3, 32),  # speed, hour_of_day, congestion
            nn.GELU(),
            nn.Linear(32, 2),  # emission_adj, energy_adj
        )

    def forward(
        self,
        mode: torch.Tensor,       # [B, L] mode indices
        speed: torch.Tensor,      # [B, L] m/s
        hour: torch.Tensor,       # [B, L] hour [0-23]
        congestion: torch.Tensor,  # [B, L] congestion index [0-1]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (mode_embedding, emission_rate, energy_rate)."""
        emb = self.mode_embed(mode)

        base_emission = self.emission_base[mode]  # [B, L]
        base_energy = self.energy_base[mode]      # [B, L]

        context = torch.stack([speed, hour / 24.0, congestion], dim=-1)
        adj = self.adjustment(context)
        adj_emission = adj[..., 0]  # [B, L]
        adj_energy = adj[..., 1]    # [B, L]

        emission_rate = (base_emission + adj_emission).clamp(min=0)
        energy_rate = (base_energy + adj_energy).clamp(min=0)

        return emb, emission_rate, energy_rate


class MetabolicTokenizer(nn.Module):
    """Extended tokenizer that represents (x, y, t, mode, emission, energy).

    Builds on the H3 spatial tokenizer and adds metabolic dimensions.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.mode_encoder = TransportModeEncoder(embed_dim)

        # Energy/emission to embedding projection
        self.metabolic_proj = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )

    def forward(
        self,
        mode: torch.Tensor,
        speed: torch.Tensor,
        hour: torch.Tensor,
        congestion: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns metabolic embedding and per-step metrics."""
        mode_emb, emissions, energy = self.mode_encoder(mode, speed, hour, congestion)
        metabolic_feats = torch.stack([emissions, energy], dim=-1)
        metabolic_emb = self.metabolic_proj(metabolic_feats)

        return mode_emb + metabolic_emb, {
            "emissions": emissions,
            "energy": energy,
        }


# ---------------------------------------------------------------------------
# 2. Urban Metabolism Encoder — fuses environmental sensor streams
# ---------------------------------------------------------------------------

class EnvironmentalSensorEncoder(nn.Module):
    """Encodes environmental sensor readings (air quality, noise, energy).

    Input features per spatial region per timestep:
    - PM2.5 (μg/m³)
    - NO₂ (ppb)
    - O₃ (ppb)
    - Noise level (dB)
    - Temperature (°C)
    - Power grid load (MW)
    - Solar radiation (W/m²)
    - Wind speed (m/s)
    """
    NUM_SENSORS = 8

    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(self.NUM_SENSORS, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Temporal aggregation (e.g., 24h window into single embedding)
        self.temporal_agg = nn.GRU(embed_dim, embed_dim, batch_first=True)

    def forward(
        self,
        sensor_data: torch.Tensor,  # [B, T, S] or [B, R, T, S]
    ) -> torch.Tensor:
        """Returns environmental embedding."""
        if sensor_data.dim() == 4:
            B, R, T, S = sensor_data.shape
            flat = sensor_data.reshape(B * R, T, S)
            encoded = self.encoder(flat)  # [B*R, T, D]
            _, h = self.temporal_agg(encoded)
            return h.squeeze(0).reshape(B, R, -1)
        else:
            encoded = self.encoder(sensor_data)  # [B, T, D]
            _, h = self.temporal_agg(encoded)
            return h.squeeze(0)  # [B, D]


class UrbanMetabolismEncoder(nn.Module):
    """Fuses trajectory embeddings with environmental context.

    Uses cross-attention: trajectory queries attend to environmental keys.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.env_encoder = EnvironmentalSensorEncoder(embed_dim)

        # Cross-attention: traj attends to environment
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.cross_ff = FeedForward(embed_dim, embed_dim * 4, dropout)

    def forward(
        self,
        traj_embed: torch.Tensor,        # [B, L, D]
        env_data: Optional[torch.Tensor],  # [B, R, T, S] or [B, T, S]
        traj_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if env_data is None:
            return traj_embed

        env_embed = self.env_encoder(env_data)
        if env_embed.dim() == 2:
            env_embed = env_embed.unsqueeze(1)  # [B, 1, D]

        # Cross-attention
        residual = traj_embed
        h = self.cross_norm(traj_embed)
        h, _ = self.cross_attn(h, env_embed, env_embed)
        h = h + residual
        h = self.cross_ff(h) + h

        return h


# ---------------------------------------------------------------------------
# 3. Scenario Discovery Head — "What-if" Analysis
# ---------------------------------------------------------------------------

class ScenarioDiscoveryHead(nn.Module):
    """Predicts city-wide impact of urban interventions.

    Given:
    - Current mobility pattern embedding
    - Intervention description embedding
    Predicts:
    - Change in CO₂ emissions (%)
    - Change in energy consumption (%)
    - Change in noise levels (%)
    - Change in average travel time (%)
    - Change in modal split distribution
    - Overall sustainability score
    """

    NUM_INTERVENTIONS = 8  # pedestrianize, add_bike_lane, congestion_charge, etc.
    NUM_IMPACT_DIMS = 6

    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.intervention_embed = nn.Embedding(self.NUM_INTERVENTIONS, embed_dim)

        # Region selector: which region(s) are affected
        self.region_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Impact predictor
        self.impact_net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.NUM_IMPACT_DIMS),
        )

        # Uncertainty estimator
        self.uncertainty_net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.NUM_IMPACT_DIMS),
        )

        # Modal split predictor (6 modes)
        self.modal_split_net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, TransportModeEncoder.NUM_MODES),
        )

    def forward(
        self,
        mobility_embed: torch.Tensor,    # [B, D] pooled mobility pattern
        env_embed: torch.Tensor,          # [B, D] environmental context
        intervention_id: torch.Tensor,    # [B] which intervention
        region_embeds: Optional[torch.Tensor] = None,  # [B, R, D] per-region embeddings
    ) -> Dict[str, torch.Tensor]:
        interv_emb = self.intervention_embed(intervention_id)  # [B, D]
        context = torch.cat([mobility_embed, env_embed, interv_emb], dim=-1)

        # Predict impact
        impact = self.impact_net(context)  # [B, 6]
        uncertainty = self.uncertainty_net(context).exp().clamp(min=1e-4, max=10.0)

        # Modal split after intervention
        modal_logits = self.modal_split_net(context)
        modal_split = torch.softmax(modal_logits, dim=-1)

        result = {
            "impact": impact,
            "impact_names": [
                "co2_change_%", "energy_change_%", "noise_change_%",
                "travel_time_change_%", "congestion_change_%", "sustainability_score",
            ],
            "uncertainty": uncertainty,
            "modal_split": modal_split,
        }

        # Region-level impact gating
        if region_embeds is not None:
            B, R, D = region_embeds.shape
            interv_expand = interv_emb.unsqueeze(1).expand(B, R, D)
            gate_input = torch.cat([region_embeds, interv_expand], dim=-1)
            region_weights = self.region_gate(gate_input).squeeze(-1)  # [B, R]
            result["region_weights"] = region_weights

        return result


# ---------------------------------------------------------------------------
# 4. Metabolic-Mobility Model
# ---------------------------------------------------------------------------

class MetabolicMobilityModel(nn.Module):
    """Sustainability-aware Mobility Foundation Model.

    Combines:
    1. HMT trajectory encoder with metabolic token enrichment
    2. Urban metabolism encoder (environmental sensor fusion)
    3. Standard transformer backbone
    4. Multi-objective prediction heads (token, flow, emissions, energy, scenario)
    """

    def __init__(
        self,
        vocab_l0: int = 16384,
        vocab_l1: int = 4096,
        vocab_l2: int = 1024,
        embed_dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        max_seq_len: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        head_dim = embed_dim // heads

        # Core trajectory encoder
        self.traj_encoder = HMTEncoder(
            vocab_l0=vocab_l0,
            vocab_l1=vocab_l1,
            vocab_l2=vocab_l2,
            embed_dim=embed_dim,
        )

        # Metabolic enrichment
        self.metabolic_tokenizer = MetabolicTokenizer(embed_dim)

        # Environmental fusion
        self.metabolism_encoder = UrbanMetabolismEncoder(
            embed_dim, num_heads=heads // 2, dropout=dropout,
        )

        # Transformer backbone
        self.transformer = MaskedTransformer(
            embedding_dim=embed_dim,
            depth=depth,
            num_heads=heads,
            head_dim=head_dim,
            feedforward_dim=embed_dim * 4,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Prediction heads
        self.token_head = nn.Linear(embed_dim, vocab_l0)
        self.flow_head = FlowMatchingHead(embed_dim)

        # Emission prediction head (per-step)
        self.emission_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

        # Energy prediction head (per-step)
        self.energy_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

        # Scenario discovery
        self.scenario_head = ScenarioDiscoveryHead(embed_dim)

    def forward(
        self,
        tokens_l0: torch.Tensor,         # [B, L]
        tokens_l1: torch.Tensor,         # [B, L]
        tokens_l2: torch.Tensor,         # [B, L]
        time_embed: torch.Tensor,        # [B, L, D]
        attention_mask: torch.Tensor,    # [B, L]
        transport_mode: Optional[torch.Tensor] = None,   # [B, L]
        speed: Optional[torch.Tensor] = None,            # [B, L]
        hour: Optional[torch.Tensor] = None,             # [B, L]
        congestion: Optional[torch.Tensor] = None,       # [B, L]
        env_data: Optional[torch.Tensor] = None,         # [B, R, T, S] or [B, T, S]
    ) -> Dict[str, torch.Tensor]:
        bsz, seq_len = tokens_l0.shape
        device = tokens_l0.device

        # 1) Base trajectory encoding
        step_embed = self.traj_encoder(tokens_l0, tokens_l1, tokens_l2, time_embed)

        # 2) Add metabolic enrichment
        metabolic_metrics = {}
        if transport_mode is not None:
            if speed is None:
                speed = torch.zeros(bsz, seq_len, device=device)
            if hour is None:
                hour = torch.zeros(bsz, seq_len, device=device)
            if congestion is None:
                congestion = torch.zeros(bsz, seq_len, device=device)

            metabolic_emb, metabolic_metrics = self.metabolic_tokenizer(
                transport_mode, speed, hour, congestion,
            )
            step_embed = step_embed + metabolic_emb

        # 3) Fuse with environmental data
        step_embed = self.metabolism_encoder(step_embed, env_data, attention_mask)

        # 4) Transformer
        attn_mask = attention_mask.bool().unsqueeze(1) & attention_mask.bool().unsqueeze(2)
        hidden = self.transformer(step_embed, attn_mask=attn_mask)
        hidden = self.norm(hidden)

        # 5) Predictions
        token_logits = self.token_head(hidden)
        pred_emissions = self.emission_head(hidden).squeeze(-1)
        pred_energy = self.energy_head(hidden).squeeze(-1)

        # Pooled representation
        mask_f = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        result = {
            "token_logits": token_logits,
            "hidden": hidden,
            "pooled": pooled,
            "pred_emissions": pred_emissions,
            "pred_energy": pred_energy,
            "metabolic_metrics": metabolic_metrics,
        }

        return result

    def scenario_analysis(
        self,
        pooled_embed: torch.Tensor,
        env_embed: torch.Tensor,
        intervention_id: torch.Tensor,
        region_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run what-if scenario analysis."""
        return self.scenario_head(pooled_embed, env_embed, intervention_id, region_embeds)


# ---------------------------------------------------------------------------
# 5. Loss Functions
# ---------------------------------------------------------------------------

def metabolic_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask_indices: torch.Tensor,
    step_mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask_indices & step_mask.bool()
    loss = F.cross_entropy(logits.permute(0, 2, 1), targets, reduction="none")
    return torch.where(valid, loss, torch.zeros_like(loss)).sum() / valid.float().sum().clamp(min=1)


def emission_prediction_loss(
    pred: torch.Tensor,      # [B, L]
    target: torch.Tensor,    # [B, L]
    mask: torch.Tensor,      # [B, L]
) -> torch.Tensor:
    loss = F.huber_loss(pred, target, reduction="none", delta=1.0)
    return torch.where(mask.bool(), loss, torch.zeros_like(loss)).sum() / mask.sum().clamp(min=1)


def energy_prediction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    loss = F.huber_loss(pred, target, reduction="none", delta=1.0)
    return torch.where(mask.bool(), loss, torch.zeros_like(loss)).sum() / mask.sum().clamp(min=1)


def scenario_impact_loss(
    pred_impact: torch.Tensor,
    true_impact: torch.Tensor,
    uncertainty: torch.Tensor,
) -> torch.Tensor:
    """Heteroscedastic loss for scenario prediction with uncertainty."""
    # Negative log-likelihood of Gaussian
    log_var = uncertainty.log()
    nll = 0.5 * ((pred_impact - true_impact).pow(2) / uncertainty + log_var)
    return nll.mean()


def multi_objective_loss(
    losses: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> torch.Tensor:
    """Weighted sum of multiple objectives."""
    total = torch.tensor(0.0, device=next(iter(losses.values())).device)
    for name, loss in losses.items():
        w = weights.get(name, 1.0)
        total = total + w * loss
    return total
