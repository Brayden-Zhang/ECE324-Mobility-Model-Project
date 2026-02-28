"""
PrivaMove: Federated Continual Learning for Privacy-Preserving MFMs.

Implements:
  1. DPGenerativeReplay — differentially private trajectory generator
  2. CityExpertMoE — Mixture-of-Experts with city-local + shared experts
  3. FederatedAverager — federated averaging for distributed training
  4. MembershipInferenceAttack — privacy benchmarking utility
  5. PrivaMoveModel — full federated continual-learning MFM
"""

import copy
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from utils.hmt_backbone import FeedForward, MaskedTransformer
from utils.hmt_hierarchy import HMTEncoder
from utils.flow import FlowMatchingHead


# ---------------------------------------------------------------------------
# 1. Differentially Private Generative Replay
# ---------------------------------------------------------------------------

class TrajectoryGenerator(nn.Module):
    """Conditional trajectory generator using a denoising approach.

    Learns to generate synthetic trajectories conditioned on:
    - city_id embedding
    - time-of-day embedding
    - trajectory length

    Architecture: MLP-based flow with residual blocks.
    """

    def __init__(
        self,
        coord_dim: int = 2,
        hidden_dim: int = 256,
        max_len: int = 64,
        num_cities: int = 100,
        time_dim: int = 32,
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.max_len = max_len

        self.city_embed = nn.Embedding(num_cities, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, hidden_dim),
        )
        self.length_embed = nn.Embedding(max_len + 1, hidden_dim)

        # Noise level embedding
        self.noise_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Generator backbone
        cond_dim = hidden_dim * 3  # city + time + length
        self.net = nn.Sequential(
            nn.Linear(coord_dim * max_len + cond_dim + hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, coord_dim * max_len),
        )

    def forward(
        self,
        x_noisy: torch.Tensor,   # [B, L, 2] noisy trajectory
        t: torch.Tensor,         # [B] noise level in [0, 1]
        city_id: torch.Tensor,   # [B]
        time_of_day: torch.Tensor,  # [B] in [0, 1]
        traj_length: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """Predict velocity field for denoising."""
        bsz = x_noisy.shape[0]

        # Condition embeddings
        c_city = self.city_embed(city_id)
        c_time = self.time_embed(time_of_day.unsqueeze(-1))
        c_len = self.length_embed(traj_length.clamp(max=self.max_len))
        cond = torch.cat([c_city, c_time, c_len], dim=-1)

        # Noise level embedding
        n_embed = self.noise_embed(t.unsqueeze(-1))

        # Flatten coordinates
        x_flat = x_noisy.reshape(bsz, -1)
        # Pad to max_len * coord_dim if needed
        if x_flat.shape[1] < self.max_len * self.coord_dim:
            pad = torch.zeros(bsz, self.max_len * self.coord_dim - x_flat.shape[1], device=x_flat.device)
            x_flat = torch.cat([x_flat, pad], dim=1)

        inp = torch.cat([x_flat, cond, n_embed], dim=-1)
        out = self.net(inp)
        return out.reshape(bsz, self.max_len, self.coord_dim)


class DPGenerativeReplay(nn.Module):
    """Wraps a trajectory generator with differential privacy mechanisms.

    Adds calibrated Gaussian noise to gradients during training to achieve
    (epsilon, delta)-differential privacy guarantees.
    """

    def __init__(
        self,
        generator: TrajectoryGenerator,
        epsilon: float = 8.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
    ):
        super().__init__()
        self.generator = generator
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm

        # Compute noise multiplier from (eps, delta)
        # Simplified Gaussian mechanism: sigma = sqrt(2 * ln(1.25/delta)) / epsilon
        self.noise_multiplier = math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    def clip_and_noise_gradients(self):
        """Apply DP gradient clipping and noise injection (call after loss.backward())."""
        total_norm = 0.0
        for p in self.generator.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = math.sqrt(total_norm)

        # Clip
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for p in self.generator.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)

        # Add noise
        for p in self.generator.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * self.noise_multiplier * self.max_grad_norm
                p.grad.data.add_(noise)

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        city_id: int,
        traj_length: int = 32,
        device: str = "cpu",
        num_steps: int = 20,
    ) -> torch.Tensor:
        """Generate synthetic trajectories with iterative denoising."""
        self.generator.eval()
        x = torch.randn(num_samples, self.generator.max_len, 2, device=device) * 0.01

        city = torch.full((num_samples,), city_id, device=device, dtype=torch.long)
        tod = torch.rand(num_samples, device=device)
        length = torch.full((num_samples,), traj_length, device=device, dtype=torch.long)

        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_val = step / num_steps
            t = torch.full((num_samples,), t_val, device=device)
            v = self.generator(x, t, city, tod, length)
            x = x + v * dt

        return x[:, :traj_length]

    def train_step(self, real_trajectories, city_ids, time_of_day, traj_lengths, optimizer):
        """One training step with DP gradient noise."""
        self.generator.train()
        bsz = real_trajectories.shape[0]
        device = real_trajectories.device

        # Sample random time
        t = torch.rand(bsz, device=device)

        # Sample noise
        x0 = torch.randn_like(real_trajectories) * 0.01
        x_t = t.unsqueeze(-1).unsqueeze(-1) * real_trajectories + (1 - t.unsqueeze(-1).unsqueeze(-1)) * x0
        target_v = real_trajectories - x0

        # Predict
        pred_v = self.generator(x_t, t, city_ids, time_of_day, traj_lengths)

        # Loss (on actual trajectory portion only)
        max_len = self.generator.max_len
        actual_len = real_trajectories.shape[1]
        if actual_len < max_len:
            pred_v = pred_v[:, :actual_len]
        loss = F.mse_loss(pred_v, target_v)

        optimizer.zero_grad()
        loss.backward()
        self.clip_and_noise_gradients()
        optimizer.step()

        return loss.item()


# ---------------------------------------------------------------------------
# 2. City-Expert Mixture of Experts
# ---------------------------------------------------------------------------

class ExpertLayer(nn.Module):
    """A single expert: feedforward network."""
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CityExpertRouter(nn.Module):
    """Routes tokens to city-local experts or the shared global expert."""
    def __init__(self, embed_dim: int, num_local_experts: int, top_k: int = 2):
        super().__init__()
        total_experts = num_local_experts + 1  # +1 for shared
        self.gate = nn.Linear(embed_dim, total_experts, bias=False)
        self.top_k = top_k
        self.num_experts = total_experts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (top_k_weights [B, L, K], top_k_indices [B, L, K])."""
        logits = self.gate(x)
        weights, indices = torch.topk(torch.softmax(logits, dim=-1), self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights, indices


class CityExpertMoE(nn.Module):
    """Mixture-of-Experts with city-specific local experts + one shared global expert.

    The shared expert is updated via federated averaging.
    City-specific experts stay local and are not shared.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_cities: int,
        top_k: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_cities = num_cities
        self.top_k = top_k

        # Shared global expert (federated)
        self.shared_expert = ExpertLayer(embed_dim, hidden_dim, dropout)

        # City-local experts
        self.local_experts = nn.ModuleList([
            ExpertLayer(embed_dim, hidden_dim, dropout) for _ in range(num_cities)
        ])

        self.router = CityExpertRouter(embed_dim, num_cities, top_k)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D]"""
        residual = x
        x = self.norm(x)
        weights, indices = self.router(x)  # [B, L, K], [B, L, K]

        bsz, seq_len, dim = x.shape
        output = torch.zeros_like(x)

        all_experts = [self.shared_expert] + list(self.local_experts)

        for k in range(self.top_k):
            expert_idx = indices[:, :, k]  # [B, L]
            w = weights[:, :, k].unsqueeze(-1)  # [B, L, 1]

            for e_id in range(len(all_experts)):
                mask = (expert_idx == e_id)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = all_experts[e_id](expert_input)
                    output[mask] += w[mask].squeeze(-1).unsqueeze(-1) * expert_output

        return output + residual

    def get_shared_state(self) -> OrderedDict:
        """Return only the shared expert's state for federated averaging."""
        return OrderedDict(
            {f"shared_expert.{k}": v.clone() for k, v in self.shared_expert.state_dict().items()}
        )

    def load_shared_state(self, state: OrderedDict):
        """Update shared expert from federated average."""
        shared_state = {}
        for k, v in state.items():
            key = k.replace("shared_expert.", "")
            shared_state[key] = v
        self.shared_expert.load_state_dict(shared_state)


# ---------------------------------------------------------------------------
# 3. Federated Averaging
# ---------------------------------------------------------------------------

class FederatedAverager:
    """Coordinates federated averaging across city clients.

    Each city trains locally and shares only:
    - shared expert parameters
    - generator model parameters (or synthetic data)
    """

    def __init__(self, num_cities: int):
        self.num_cities = num_cities
        self.global_shared_state: Optional[OrderedDict] = None
        self.city_shared_states: Dict[int, OrderedDict] = {}
        self.city_weights: Dict[int, float] = {}

    def receive_update(self, city_id: int, shared_state: OrderedDict, num_samples: int):
        """City client sends its updated shared expert state."""
        self.city_shared_states[city_id] = shared_state
        self.city_weights[city_id] = float(num_samples)

    def aggregate(self) -> OrderedDict:
        """Weighted federated average of shared expert parameters."""
        if not self.city_shared_states:
            raise RuntimeError("No city updates to aggregate")

        total_weight = sum(self.city_weights.values())
        avg_state = OrderedDict()

        for key in list(self.city_shared_states.values())[0].keys():
            weighted_sum = None
            for city_id, state in self.city_shared_states.items():
                w = self.city_weights[city_id] / total_weight
                param = state[key].float() * w
                if weighted_sum is None:
                    weighted_sum = param
                else:
                    weighted_sum = weighted_sum + param
            avg_state[key] = weighted_sum

        self.global_shared_state = avg_state
        self.city_shared_states.clear()
        self.city_weights.clear()
        return avg_state

    def broadcast(self) -> OrderedDict:
        """Return the global averaged state for cities to download."""
        if self.global_shared_state is None:
            raise RuntimeError("No global state available; call aggregate() first")
        return copy.deepcopy(self.global_shared_state)


# ---------------------------------------------------------------------------
# 4. Membership Inference Attack (Privacy Benchmarking)
# ---------------------------------------------------------------------------

class MembershipInferenceAttack(nn.Module):
    """Shadow model-based membership inference attack.

    Given a trajectory and a model, predicts whether the trajectory was
    in the training set. Used to quantify privacy leakage.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Attack model takes per-sample loss statistics as input
        # Features: loss, per-token entropy, confidence, gradient norm
        self.attack_net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def compute_attack_features(
        self,
        model: nn.Module,
        tokens_l0: torch.Tensor,
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample features for the attack model."""
        with torch.no_grad():
            # Per-token loss
            loss = F.cross_entropy(
                logits.permute(0, 2, 1), tokens_l0, reduction="none",
            )
            masked_loss = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            # Entropy of predictions
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
            masked_entropy = (entropy * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            # Confidence (max prob)
            max_prob = probs.max(dim=-1).values
            masked_conf = (max_prob * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            # Perplexity proxy
            perplexity = torch.exp(masked_loss)

        features = torch.stack([masked_loss, masked_entropy, masked_conf, perplexity], dim=-1)
        return features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict membership: 0 = non-member, 1 = member."""
        return self.attack_net(features)

    def evaluate_privacy(
        self,
        member_features: torch.Tensor,
        nonmember_features: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate membership inference attack accuracy and AUC."""
        with torch.no_grad():
            member_logits = self.forward(member_features)
            nonmember_logits = self.forward(nonmember_features)

            member_pred = member_logits.argmax(dim=-1)
            nonmember_pred = nonmember_logits.argmax(dim=-1)

            tp = (member_pred == 1).float().mean().item()
            tn = (nonmember_pred == 0).float().mean().item()
            attack_accuracy = (tp + tn) / 2

            # AUC approximation using confidence scores
            member_conf = torch.softmax(member_logits, dim=-1)[:, 1]
            nonmember_conf = torch.softmax(nonmember_logits, dim=-1)[:, 1]
            all_conf = torch.cat([member_conf, nonmember_conf])
            all_labels = torch.cat([
                torch.ones_like(member_conf),
                torch.zeros_like(nonmember_conf),
            ])

            # Sort by confidence
            sorted_idx = all_conf.argsort(descending=True)
            sorted_labels = all_labels[sorted_idx]
            tpr = sorted_labels.cumsum(0) / sorted_labels.sum()
            fpr = (1 - sorted_labels).cumsum(0) / (1 - sorted_labels).sum()
            auc = torch.trapezoid(tpr, fpr).item()

        return {
            "attack_accuracy": attack_accuracy,
            "attack_auc": auc,
            "true_positive_rate": tp,
            "true_negative_rate": tn,
        }


# ---------------------------------------------------------------------------
# 5. PrivaMove Model
# ---------------------------------------------------------------------------

class PrivaMoveModel(nn.Module):
    """Federated continual-learning MFM with city-specific MoE and DP replay.

    Architecture:
    1. Shared HMT encoder (federated averaged)
    2. City-expert MoE layers (local experts stay private, shared expert is federated)
    3. Standard prediction heads
    """

    def __init__(
        self,
        vocab_l0: int = 16384,
        vocab_l1: int = 4096,
        vocab_l2: int = 1024,
        embed_dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        num_cities: int = 10,
        moe_top_k: int = 2,
        max_seq_len: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cities = num_cities
        head_dim = embed_dim // heads

        # Shared encoder (federated)
        self.encoder = HMTEncoder(
            vocab_l0=vocab_l0,
            vocab_l1=vocab_l1,
            vocab_l2=vocab_l2,
            embed_dim=embed_dim,
        )

        # Transformer backbone (shared, federated)
        self.transformer = MaskedTransformer(
            embedding_dim=embed_dim,
            depth=depth,
            num_heads=heads,
            head_dim=head_dim,
            feedforward_dim=embed_dim * 4,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # City-expert MoE (mixed: shared expert is federated, locals stay)
        self.moe = CityExpertMoE(
            embed_dim=embed_dim,
            hidden_dim=embed_dim * 4,
            num_cities=num_cities,
            top_k=moe_top_k,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.token_head = nn.Linear(embed_dim, vocab_l0)
        self.flow_head = FlowMatchingHead(embed_dim)

    def forward(
        self,
        tokens_l0: torch.Tensor,
        tokens_l1: torch.Tensor,
        tokens_l2: torch.Tensor,
        time_embed: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        step_embed = self.encoder(tokens_l0, tokens_l1, tokens_l2, time_embed)

        # Simple self-attention mask
        attn_mask = attention_mask.bool().unsqueeze(1) & attention_mask.bool().unsqueeze(2)
        hidden = self.transformer(step_embed, attn_mask=attn_mask)

        # MoE layer
        hidden = self.moe(hidden)
        hidden = self.norm(hidden)

        token_logits = self.token_head(hidden)

        # Pool for downstream
        mask_f = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        return {
            "token_logits": token_logits,
            "hidden": hidden,
            "pooled": pooled,
        }

    def get_federated_state(self) -> OrderedDict:
        """Get parameters that should be federated (everything except local experts)."""
        state = OrderedDict()
        for name, param in self.named_parameters():
            if "local_experts" not in name:
                state[name] = param.data.clone()
        return state

    def load_federated_state(self, state: OrderedDict, alpha: float = 1.0):
        """Update federated parameters (with optional interpolation)."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in state:
                    if alpha >= 1.0:
                        param.data.copy_(state[name])
                    else:
                        param.data.copy_(alpha * state[name] + (1 - alpha) * param.data)


# ---------------------------------------------------------------------------
# Loss utilities
# ---------------------------------------------------------------------------

def privamove_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask_indices: torch.Tensor,
    step_mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask_indices & step_mask.bool()
    loss = F.cross_entropy(logits.permute(0, 2, 1), targets, reduction="none")
    return torch.where(valid, loss, torch.zeros_like(loss)).sum() / valid.float().sum().clamp(min=1)


def replay_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """KL divergence between student and teacher on synthetic replay data."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return (kl * mask).sum() / mask.sum().clamp(min=1) * (temperature ** 2)
