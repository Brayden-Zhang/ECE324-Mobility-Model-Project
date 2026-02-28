"""
GeoSight Training Script — Multi-modal Urban Perception Pre-training.

Usage:
  python train_geosight.py --max_steps 5000 --batch_size 32

Supports:
  - Synthetic trajectory + imagery data for prototyping
  - Real paired (trajectory, street-view image) datasets
  - Joint training: masked token prediction + cross-modal alignment + visual-spatial matching
"""

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils.hmt import HMTConfig, H3Tokenizer, TimeFeatures
from utils.geosight import (
    GeoSightModel,
    geosight_token_loss,
    compute_movement_statistics,
)
from utils.flow import sample_rectified_flow_targets, flow_matching_loss


# ---------------------------------------------------------------------------
# Synthetic Vision-Trajectory Dataset
# ---------------------------------------------------------------------------

class SyntheticGeoSightDataset(Dataset):
    """Generates paired (trajectory, image) data for prototyping.

    Trajectories are random walks; images are synthetic 32x32 RGB patches
    whose color channels correlate with movement statistics (e.g., faster
    trajectories → brighter images) to create a learnable cross-modal signal.
    """

    def __init__(
        self,
        num_samples: int = 20000,
        max_len: int = 64,
        num_images_per_sample: int = 4,
        img_size: int = 32,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.max_len = max_len
        self.num_images = num_images_per_sample
        self.img_size = img_size
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.RandomState(self.rng.randint(0, 2**31) + idx)
        L = rng.randint(self.max_len // 2, self.max_len + 1)

        # Random walk trajectory
        speed = rng.uniform(0.00001, 0.0005)
        heading = rng.uniform(0, 2 * np.pi)
        coords = np.zeros((self.max_len, 2), dtype=np.float32)
        for t in range(L):
            if t == 0:
                coords[t] = rng.uniform(-0.01, 0.01, 2)
            else:
                heading += rng.normal(0, 0.3)
                coords[t, 0] = coords[t - 1, 0] + speed * np.cos(heading)
                coords[t, 1] = coords[t - 1, 1] + speed * np.sin(heading)

        timestamps = np.zeros(self.max_len, dtype=np.float32)
        timestamps[:L] = np.cumsum(rng.uniform(1, 60, L))

        mask = np.zeros(self.max_len, dtype=np.float32)
        mask[:L] = 1.0

        # Generate synthetic images correlated with trajectory stats
        mean_speed_val = speed * 1000  # scale for visualization
        turn_rate = 0.3  # from heading noise
        images = np.zeros((self.num_images, 3, self.img_size, self.img_size), dtype=np.float32)
        for i in range(self.num_images):
            # R channel: speed, G channel: turn rate, B channel: random texture
            images[i, 0] = np.clip(mean_speed_val + rng.normal(0, 0.1, (self.img_size, self.img_size)), 0, 1)
            images[i, 1] = np.clip(turn_rate + rng.normal(0, 0.1, (self.img_size, self.img_size)), 0, 1)
            images[i, 2] = rng.uniform(0, 1, (self.img_size, self.img_size))

        # Semantic targets (6-dim: walkability, commercial, greenery, road_width, building_height, pedestrian)
        semantic_targets = np.array([
            0.5 + 0.3 * mean_speed_val,  # walkability correlates with speed
            rng.uniform(0, 1),
            rng.uniform(0, 1),
            0.3 + 0.5 * mean_speed_val,  # road width correlates with speed
            rng.uniform(0.1, 0.9),
            rng.uniform(0.1, 0.9),
        ], dtype=np.float32)

        image_mask = np.ones(self.num_images, dtype=np.float32)

        return {
            "coords": torch.from_numpy(coords),
            "timestamps": torch.from_numpy(timestamps),
            "attention_mask": torch.from_numpy(mask),
            "images": torch.from_numpy(images),
            "image_mask": torch.from_numpy(image_mask),
            "semantic_targets": torch.from_numpy(semantic_targets),
        }


def collate_geosight(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch])
    return out


# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train GeoSight multi-modal MFM")
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--num_images", type=int, default=4)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--vision_dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--vocab_l0", type=int, default=16384)
    p.add_argument("--vocab_l1", type=int, default=4096)
    p.add_argument("--vocab_l2", type=int, default=1024)
    p.add_argument("--mask_ratio", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--token_weight", type=float, default=1.0)
    p.add_argument("--contrastive_weight", type=float, default=1.0)
    p.add_argument("--matching_weight", type=float, default=0.5)
    p.add_argument("--flow_weight", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/geosight")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--use_pretrained_vision", action="store_true", default=False)
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


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device

    print("=" * 60)
    print("GeoSight: Multi-modal Urban Perception Pre-training")
    print("=" * 60)

    # Dataset
    dataset = SyntheticGeoSightDataset(
        num_samples=args.num_samples,
        max_len=args.max_len,
        num_images_per_sample=args.num_images,
        seed=args.seed,
    )
    n_val = max(1, len(dataset) // 10)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_geosight, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_geosight, num_workers=0)

    # Tokenizer
    h3_tok = H3Tokenizer(
        res0=9, res1=7, res2=5,
        vocab_sizes=(args.vocab_l0, args.vocab_l1, args.vocab_l2),
        hash_tokens=True,
    )

    # Time encoder
    time_encoder = TimeFeatures(args.embed_dim).to(device)

    # Model
    model = GeoSightModel(
        vocab_l0=args.vocab_l0,
        vocab_l1=args.vocab_l1,
        vocab_l2=args.vocab_l2,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        vision_dim=args.vision_dim,
        max_seq_len=args.max_len,
        dropout=args.dropout,
        use_pretrained_vision=args.use_pretrained_vision,
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

    step = 0
    model.train()
    time_encoder.train()

    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break

            coords = batch["coords"].to(device)              # [B, L, 2]
            timestamps = batch["timestamps"].to(device)       # [B, L]
            mask = batch["attention_mask"].to(device)          # [B, L]
            images = batch["images"].to(device)                # [B, N, 3, H, W]
            image_mask = batch["image_mask"].to(device)        # [B, N]
            semantic_targets = batch["semantic_targets"].to(device)  # [B, 6]

            bsz, L, _ = coords.shape

            # Tokenize
            coords_np = coords.cpu().numpy()
            l0_all, l1_all, l2_all = [], [], []
            for b in range(bsz):
                l0, l1, l2 = h3_tok.tokenize(coords_np[b])
                l0_all.append(l0)
                l1_all.append(l1)
                l2_all.append(l2)
            tokens_l0 = torch.from_numpy(np.stack(l0_all)).to(device)
            tokens_l1 = torch.from_numpy(np.stack(l1_all)).to(device)
            tokens_l2 = torch.from_numpy(np.stack(l2_all)).to(device)

            # Time embedding
            time_embed = time_encoder(timestamps, mask)

            # MLM mask
            mask_indices = (torch.rand(bsz, L, device=device) < args.mask_ratio) & mask.bool()

            # LR schedule
            lr_scale = cosine_lr(step, args.warmup_steps, args.max_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * lr_scale

            with torch.amp.autocast(device_type=device, enabled=args.amp and device == "cuda"):
                out = model(
                    tokens_l0, tokens_l1, tokens_l2,
                    time_embed, mask,
                    location_images=images,
                    image_mask=image_mask.bool(),
                )

                # Token prediction loss
                loss_token = geosight_token_loss(
                    out["token_logits"], tokens_l0, mask_indices, mask,
                )

                # Cross-modal contrastive loss
                loss_contrastive = out.get("contrastive_loss", torch.tensor(0.0, device=device))

                # Visual-spatial matching
                loss_matching = torch.tensor(0.0, device=device)
                if "vis_pool" in out:
                    movement_stats = compute_movement_statistics(coords, timestamps, mask)
                    # Normalize semantic targets to zero-mean unit-var for stable MSE
                    sem_norm = (semantic_targets - semantic_targets.mean(dim=0, keepdim=True)) / (semantic_targets.std(dim=0, keepdim=True) + 1e-6)
                    match_out = model.matching_head(
                        out["traj_pool"], out["vis_pool"],
                        sem_norm, movement_stats,
                    )
                    loss_matching = match_out.get("visual_loss", torch.tensor(0.0, device=device)) + \
                                    match_out.get("movement_loss", torch.tensor(0.0, device=device))

                # Flow matching loss
                x_t, target_v, t_flow = sample_rectified_flow_targets(coords)
                pred_v = model.flow_head(out["traj_hidden"], x_t, t_flow)
                loss_flow = flow_matching_loss(pred_v, target_v, mask)

                total_loss = (
                    args.token_weight * loss_token
                    + args.contrastive_weight * loss_contrastive
                    + args.matching_weight * loss_matching
                    + args.flow_weight * loss_flow
                )

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
                    f"contrastive={loss_contrastive.item():.4f} "
                    f"matching={loss_matching.item():.4f} "
                    f"flow={loss_flow.item():.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            if step % args.save_interval == 0:
                ckpt = os.path.join(args.ckpt_dir, f"geosight_step{step}.pt")
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "time_encoder": time_encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, ckpt)
                print(f"  Saved: {ckpt}")

            if step % args.eval_interval == 0:
                model.eval()
                time_encoder.eval()
                val_losses = []
                with torch.no_grad():
                    for i, vb in enumerate(val_loader):
                        if i >= 10:
                            break
                        vc = vb["coords"].to(device)
                        vts = vb["timestamps"].to(device)
                        vm = vb["attention_mask"].to(device)
                        vi = vb["images"].to(device)
                        vim = vb["image_mask"].to(device).bool()
                        vB, vL = vc.shape[:2]
                        vc_np = vc.cpu().numpy()
                        vl0, vl1, vl2 = [], [], []
                        for b in range(vB):
                            a, bb, c = h3_tok.tokenize(vc_np[b])
                            vl0.append(a); vl1.append(bb); vl2.append(c)
                        vl0 = torch.from_numpy(np.stack(vl0)).to(device)
                        vl1 = torch.from_numpy(np.stack(vl1)).to(device)
                        vl2 = torch.from_numpy(np.stack(vl2)).to(device)
                        vte = time_encoder(vts, vm)
                        vmask = (torch.rand(vB, vL, device=device) < args.mask_ratio) & vm.bool()
                        vout = model(vl0, vl1, vl2, vte, vm, vi, vim)
                        vloss = geosight_token_loss(vout["token_logits"], vl0, vmask, vm)
                        val_losses.append(vloss.item())
                print(f"  [Eval] val_token_loss={np.mean(val_losses):.4f}")
                model.train()
                time_encoder.train()

    final_path = os.path.join(args.ckpt_dir, "geosight_final.pt")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "time_encoder": time_encoder.state_dict(),
        "args": vars(args),
    }, final_path)
    print(f"Training complete. Final model: {final_path}")


if __name__ == "__main__":
    main()
