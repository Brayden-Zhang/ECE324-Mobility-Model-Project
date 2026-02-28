"""
SocialMob Training Script — Multi-Agent Interaction Pre-training.

Usage:
  python train_socialmob.py --max_steps 5000 --batch_size 16 --max_agents 8

Supports:
  - Synthetic multi-agent data generation for quick prototyping
  - Real multi-agent datasets (provide --data_dir with per-scene parquets)
  - Joint training with masked token prediction + social conflict detection
  - Wandb / console logging
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils.hmt import HMTConfig, HMTTokenizer, TimeFeatures
from utils.socialmob import (
    SocialMobModel,
    socialmob_token_loss,
    socialmob_conflict_loss,
    compute_velocities,
    detect_conflicts_ground_truth,
)
from utils.flow import sample_rectified_flow_targets, flow_matching_loss


# ---------------------------------------------------------------------------
# Synthetic Multi-Agent Dataset
# ---------------------------------------------------------------------------

class SyntheticMultiAgentDataset(Dataset):
    """Generates random multi-agent scenes for prototyping.

    Each scene has A agents, each with L GPS steps. Some agents walk in
    converging paths to create "conflict" ground truth.
    """

    def __init__(
        self,
        num_scenes: int = 10000,
        max_agents: int = 8,
        max_steps: int = 32,
        conflict_prob: float = 0.3,
        seed: int = 42,
    ):
        super().__init__()
        self.num_scenes = num_scenes
        self.max_agents = max_agents
        self.max_steps = max_steps
        self.conflict_prob = conflict_prob
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, idx):
        A = self.rng.randint(2, self.max_agents + 1)
        L = self.max_steps

        # Generate random starting positions
        starts = self.rng.uniform(-0.01, 0.01, (A, 2)).astype(np.float32)
        headings = self.rng.uniform(0, 2 * np.pi, (A,)).astype(np.float32)
        speeds = self.rng.uniform(0.00001, 0.0001, (A,)).astype(np.float32)

        coords = np.zeros((A, L, 2), dtype=np.float32)
        for a in range(A):
            for t in range(L):
                if t == 0:
                    coords[a, t] = starts[a]
                else:
                    noise = self.rng.normal(0, speeds[a] * 0.1, 2)
                    coords[a, t, 0] = coords[a, t - 1, 0] + speeds[a] * np.cos(headings[a]) + noise[0]
                    coords[a, t, 1] = coords[a, t - 1, 1] + speeds[a] * np.sin(headings[a]) + noise[1]

        # Inject conflicts (converging paths)
        for a in range(1, A):
            if self.rng.random() < self.conflict_prob:
                target_a = self.rng.randint(0, a)
                meet_t = self.rng.randint(L // 2, L)
                target_pos = coords[target_a, min(meet_t, L - 1)]
                start_pos = coords[a, 0]
                for t in range(meet_t):
                    frac = t / max(meet_t, 1)
                    coords[a, t] = start_pos * (1 - frac) + target_pos * frac
                    coords[a, t] += self.rng.normal(0, 0.00005, 2)

        # Timestamps: uniform 1-second intervals
        timestamps = np.arange(L, dtype=np.float32).reshape(1, L).repeat(A, axis=0)
        timestamps += self.rng.uniform(0, 100, (A, 1)).astype(np.float32)

        # Pad to max_agents
        padded_coords = np.zeros((self.max_agents, L, 2), dtype=np.float32)
        padded_ts = np.zeros((self.max_agents, L), dtype=np.float32)
        agent_mask = np.zeros(self.max_agents, dtype=np.float32)
        step_mask = np.zeros((self.max_agents, L), dtype=np.float32)

        padded_coords[:A] = coords
        padded_ts[:A] = timestamps
        agent_mask[:A] = 1.0
        step_mask[:A] = 1.0

        return {
            "coords": torch.from_numpy(padded_coords),       # [A, L, 2]
            "timestamps": torch.from_numpy(padded_ts),        # [A, L]
            "agent_mask": torch.from_numpy(agent_mask),       # [A]
            "step_mask": torch.from_numpy(step_mask),         # [A, L]
            "num_agents": A,
        }


def collate_multi_agent(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = torch.tensor(vals)
    return out


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train SocialMob multi-agent MFM")
    p.add_argument("--data_dir", type=str, default="", help="path to real multi-agent data")
    p.add_argument("--synthetic_scenes", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_agents", type=int, default=8)
    p.add_argument("--max_agent_steps", type=int, default=32)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--vocab_l0", type=int, default=16384)
    p.add_argument("--vocab_l1", type=int, default=4096)
    p.add_argument("--vocab_l2", type=int, default=1024)
    p.add_argument("--mask_ratio", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--conflict_weight", type=float, default=1.0)
    p.add_argument("--token_weight", type=float, default=1.0)
    p.add_argument("--flow_weight", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/socialmob")
    p.add_argument("--amp", action="store_true", default=False)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr(step, warmup, total, min_ratio=0.1):
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))


def tokenize_coords(coords, tokenizer_h3, device):
    """Tokenize multi-agent coords [B, A, L, 2] → token tensors [B, A, L]."""
    bsz, A, L, _ = coords.shape
    coords_np = coords.cpu().numpy()
    all_l0 = np.zeros((bsz, A, L), dtype=np.int64)
    all_l1 = np.zeros((bsz, A, L), dtype=np.int64)
    all_l2 = np.zeros((bsz, A, L), dtype=np.int64)
    for b in range(bsz):
        for a in range(A):
            l0, l1, l2 = tokenizer_h3.tokenize(coords_np[b, a])
            all_l0[b, a] = l0
            all_l1[b, a] = l1
            all_l2[b, a] = l2
    return (
        torch.from_numpy(all_l0).to(device),
        torch.from_numpy(all_l1).to(device),
        torch.from_numpy(all_l2).to(device),
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device

    print("=" * 60)
    print("SocialMob: Multi-Agent Interaction Pre-training")
    print("=" * 60)

    # Build dataset
    dataset = SyntheticMultiAgentDataset(
        num_scenes=args.synthetic_scenes,
        max_agents=args.max_agents,
        max_steps=args.max_agent_steps,
        seed=args.seed,
    )
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_multi_agent, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multi_agent, num_workers=0)

    # Build tokenizer (H3)
    from utils.hmt import H3Tokenizer
    h3_tok = H3Tokenizer(
        res0=9, res1=7, res2=5,
        vocab_sizes=(args.vocab_l0, args.vocab_l1, args.vocab_l2),
        hash_tokens=True,
    )

    # Build time encoder
    time_encoder = TimeFeatures(args.embed_dim).to(device)

    # Build model
    model = SocialMobModel(
        vocab_l0=args.vocab_l0,
        vocab_l1=args.vocab_l1,
        vocab_l2=args.vocab_l2,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        max_agents=args.max_agents,
        max_steps_per_agent=args.max_agent_steps,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(time_encoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(enabled=args.amp and device == "cuda")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Training loop
    step = 0
    model.train()
    time_encoder.train()

    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break

            coords = batch["coords"].to(device)       # [B, A, L, 2]
            timestamps = batch["timestamps"].to(device) # [B, A, L]
            agent_mask = batch["agent_mask"].to(device) # [B, A]
            step_mask = batch["step_mask"].to(device)   # [B, A, L]

            bsz, A, L, _ = coords.shape

            # Tokenize
            tokens_l0, tokens_l1, tokens_l2 = tokenize_coords(coords, h3_tok, device)

            # Time embeddings
            flat_ts = timestamps.reshape(bsz * A, L)
            flat_step_mask = step_mask.reshape(bsz * A, L)
            flat_time = time_encoder(flat_ts, flat_step_mask)
            time_embed = flat_time.reshape(bsz, A, L, -1)

            # Generate masks for MLM
            mask_indices = torch.rand(bsz, A, L, device=device) < args.mask_ratio
            mask_indices = mask_indices & step_mask.bool()

            # Compute dt
            dt = torch.zeros_like(timestamps)
            dt[:, :, 1:] = (timestamps[:, :, 1:] - timestamps[:, :, :-1]).clamp(min=0)

            # Forward
            lr_scale = cosine_lr(step, args.warmup_steps, args.max_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * lr_scale

            with torch.amp.autocast(device_type=device, enabled=args.amp and device == "cuda"):
                out = model(
                    tokens_l0, tokens_l1, tokens_l2,
                    time_embed, coords, dt,
                    agent_mask, step_mask, mask_indices,
                )

                # Token loss
                loss_token = socialmob_token_loss(
                    out["token_logits"], tokens_l0, mask_indices, step_mask,
                )

                # Conflict loss — generate ground truth pairs
                num_pairs = out["conflict_logits"].shape[1]
                if num_pairs > 0:
                    pair_idx = []
                    for a in range(A):
                        for b_idx in range(a + 1, A):
                            pair_idx.append((a, b_idx))
                    conflict_labels = []
                    pair_valid = []
                    for a_i, a_j in pair_idx[:num_pairs]:
                        labels = detect_conflicts_ground_truth(
                            coords[:, a_i], coords[:, a_j],
                        )  # [B]
                        valid = agent_mask[:, a_i] * agent_mask[:, a_j]
                        conflict_labels.append(labels)
                        pair_valid.append(valid)
                    conflict_labels = torch.stack(conflict_labels, dim=1)  # [B, P]
                    pair_valid = torch.stack(pair_valid, dim=1)            # [B, P]
                    loss_conflict = socialmob_conflict_loss(
                        out["conflict_logits"], conflict_labels, pair_valid,
                    )
                else:
                    loss_conflict = torch.tensor(0.0, device=device)

                # Flow matching loss on hidden states
                flat_coords_2d = coords.reshape(bsz, A * L, 2)
                flat_step_mask_2d = step_mask.reshape(bsz, A * L)
                x_t, target_v, t_flow = sample_rectified_flow_targets(flat_coords_2d)
                pred_v = model.flow_head(out["hidden"], x_t, t_flow)
                loss_flow = flow_matching_loss(pred_v, target_v, flat_step_mask_2d)

                total_loss = (
                    args.token_weight * loss_token
                    + args.conflict_weight * loss_conflict
                    + args.flow_weight * loss_flow
                )

            # NaN guard
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"  [Step {step+1}] NaN/Inf loss detected — skipping batch")
                optimizer.zero_grad()
                step += 1
                continue

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            step += 1

            if step % 50 == 0:
                print(
                    f"[Step {step}/{args.max_steps}] "
                    f"loss={total_loss.item():.4f} "
                    f"token={loss_token.item():.4f} "
                    f"conflict={loss_conflict.item():.4f} "
                    f"flow={loss_flow.item():.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            if step % args.save_interval == 0:
                ckpt_path = os.path.join(args.ckpt_dir, f"socialmob_step{step}.pt")
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "time_encoder": time_encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

            if step % args.eval_interval == 0:
                model.eval()
                time_encoder.eval()
                val_losses = []
                with torch.no_grad():
                    for i, vbatch in enumerate(val_loader):
                        if i >= 10:
                            break
                        vc = vbatch["coords"].to(device)
                        vt = vbatch["timestamps"].to(device)
                        va = vbatch["agent_mask"].to(device)
                        vs = vbatch["step_mask"].to(device)
                        vB, vA, vL, _ = vc.shape
                        vl0, vl1, vl2 = tokenize_coords(vc, h3_tok, device)
                        vft = time_encoder(vt.reshape(vB * vA, vL), vs.reshape(vB * vA, vL))
                        vte = vft.reshape(vB, vA, vL, -1)
                        vm = torch.rand(vB, vA, vL, device=device) < args.mask_ratio
                        vm = vm & vs.bool()
                        vdt = torch.zeros_like(vt)
                        vdt[:, :, 1:] = (vt[:, :, 1:] - vt[:, :, :-1]).clamp(min=0)
                        vout = model(vl0, vl1, vl2, vte, vc, vdt, va, vs, vm)
                        vloss = socialmob_token_loss(vout["token_logits"], vl0, vm, vs)
                        val_losses.append(vloss.item())
                avg_val = np.mean(val_losses) if val_losses else 0.0
                print(f"  [Eval] val_token_loss={avg_val:.4f}")
                model.train()
                time_encoder.train()

    # Final save
    final_path = os.path.join(args.ckpt_dir, "socialmob_final.pt")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "time_encoder": time_encoder.state_dict(),
        "args": vars(args),
    }, final_path)
    print(f"Training complete. Final model: {final_path}")


if __name__ == "__main__":
    main()
