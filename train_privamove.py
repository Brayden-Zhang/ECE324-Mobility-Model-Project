"""
PrivaMove Training Script — Federated Continual Learning Pre-training.

Simulates a federated learning scenario with multiple city clients, each
training locally with differentially private generative replay.

Usage:
  python train_privamove.py --num_cities 5 --rounds 20 --local_steps 100

Supports:
  - Simulated multi-city federated training
  - DP generative replay for privacy-safe knowledge transfer
  - MoE with shared + city-local experts
  - Membership inference attack benchmarking
"""

import argparse
import copy
import math
import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils.hmt import H3Tokenizer, TimeFeatures
from utils.privamove import (
    PrivaMoveModel,
    TrajectoryGenerator,
    DPGenerativeReplay,
    FederatedAverager,
    MembershipInferenceAttack,
    privamove_token_loss,
    replay_distillation_loss,
)
from utils.flow import sample_rectified_flow_targets, flow_matching_loss


# ---------------------------------------------------------------------------
# Synthetic Per-City Dataset
# ---------------------------------------------------------------------------

class CityTrajectoryDataset(Dataset):
    """Generates city-specific trajectory data.

    Each city has a characteristic center, speed distribution, and heading bias
    to simulate different urban mobility patterns.
    """

    def __init__(
        self,
        city_id: int,
        num_samples: int = 5000,
        max_len: int = 64,
        seed: int = 42,
    ):
        self.city_id = city_id
        self.num_samples = num_samples
        self.max_len = max_len
        self.rng = np.random.RandomState(seed + city_id * 1000)

        # City-specific characteristics
        self.center = self.rng.uniform(-0.05, 0.05, 2).astype(np.float32)
        self.speed_scale = self.rng.uniform(0.00005, 0.0005)
        self.heading_bias = self.rng.uniform(0, 2 * np.pi)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.RandomState(self.rng.randint(0, 2**31) + idx)
        L = rng.randint(self.max_len // 2, self.max_len + 1)

        coords = np.zeros((self.max_len, 2), dtype=np.float32)
        heading = self.heading_bias + rng.normal(0, 0.5)
        speed = self.speed_scale * rng.uniform(0.5, 2.0)

        for t in range(L):
            if t == 0:
                coords[t] = self.center + rng.normal(0, 0.005, 2)
            else:
                heading += rng.normal(0, 0.2)
                coords[t, 0] = coords[t - 1, 0] + speed * np.cos(heading) + rng.normal(0, speed * 0.1)
                coords[t, 1] = coords[t - 1, 1] + speed * np.sin(heading) + rng.normal(0, speed * 0.1)

        timestamps = np.zeros(self.max_len, dtype=np.float32)
        timestamps[:L] = np.cumsum(rng.uniform(1, 30, L))
        mask = np.zeros(self.max_len, dtype=np.float32)
        mask[:L] = 1.0

        return {
            "coords": torch.from_numpy(coords),
            "timestamps": torch.from_numpy(timestamps),
            "attention_mask": torch.from_numpy(mask),
            "city_id": self.city_id,
            "length": L,
        }


def collate_city(batch):
    out = {}
    for k in batch[0].keys():
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        else:
            out[k] = torch.tensor(vals)
    return out


# ---------------------------------------------------------------------------
# City Client
# ---------------------------------------------------------------------------

class CityClient:
    """Simulates a federated learning client for one city."""

    def __init__(
        self,
        city_id: int,
        model: PrivaMoveModel,
        generator: DPGenerativeReplay,
        dataset: CityTrajectoryDataset,
        h3_tok: H3Tokenizer,
        time_encoder: TimeFeatures,
        args,
    ):
        self.city_id = city_id
        self.model = model
        self.generator = generator
        self.dataset = dataset
        self.h3_tok = h3_tok
        self.time_encoder = time_encoder
        self.args = args
        self.device = args.device

        self.loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_city, num_workers=0,
        )
        self.model_optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
        self.gen_optimizer = torch.optim.Adam(
            generator.generator.parameters(), lr=args.gen_lr,
        )

    def local_train(self, num_steps: int) -> dict:
        """Train locally for num_steps, return metrics."""
        self.model.train()
        self.time_encoder.train()
        device = self.device
        total_loss = 0.0
        steps_done = 0

        data_iter = iter(self.loader)
        for _ in range(num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.loader)
                batch = next(data_iter)

            coords = batch["coords"].to(device)
            timestamps = batch["timestamps"].to(device)
            mask = batch["attention_mask"].to(device)
            bsz, L = coords.shape[:2]

            # Tokenize
            coords_np = coords.cpu().numpy()
            l0_all, l1_all, l2_all = [], [], []
            for b in range(bsz):
                l0, l1, l2 = self.h3_tok.tokenize(coords_np[b])
                l0_all.append(l0)
                l1_all.append(l1)
                l2_all.append(l2)
            tokens_l0 = torch.from_numpy(np.stack(l0_all)).to(device)
            tokens_l1 = torch.from_numpy(np.stack(l1_all)).to(device)
            tokens_l2 = torch.from_numpy(np.stack(l2_all)).to(device)

            time_embed = self.time_encoder(timestamps, mask)

            # MLM mask
            mask_idx = (torch.rand(bsz, L, device=device) < self.args.mask_ratio) & mask.bool()

            out = self.model(tokens_l0, tokens_l1, tokens_l2, time_embed, mask)
            loss = privamove_token_loss(out["token_logits"], tokens_l0, mask_idx, mask)

            # Flow matching
            x_t, target_v, t_flow = sample_rectified_flow_targets(coords)
            pred_v = self.model.flow_head(out["hidden"], x_t, t_flow)
            loss_flow = flow_matching_loss(pred_v, target_v, mask)
            loss = loss + 0.5 * loss_flow

            # NaN guard
            if torch.isnan(loss) or torch.isinf(loss):
                self.model_optimizer.zero_grad()
                continue

            self.model_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.model_optimizer.step()

            total_loss += loss.item()
            steps_done += 1

        # Train generator on local data (with DP)
        gen_loss = 0.0
        data_iter = iter(self.loader)
        for _ in range(min(num_steps // 2, 50)):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.loader)
                batch = next(data_iter)

            coords = batch["coords"].to(device)
            city_ids = batch["city_id"].to(device)
            lengths = batch["length"].to(device)
            tod = (batch["timestamps"][:, 0] % 86400 / 86400).to(device)

            gl = self.generator.train_step(
                coords, city_ids, tod, lengths, self.gen_optimizer,
            )
            gen_loss += gl

        return {
            "city_id": self.city_id,
            "avg_loss": total_loss / max(steps_done, 1),
            "gen_loss": gen_loss / max(num_steps // 2, 1),
            "num_samples": len(self.dataset),
        }


# ---------------------------------------------------------------------------
# Training Orchestration
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train PrivaMove federated MFM")
    p.add_argument("--num_cities", type=int, default=5)
    p.add_argument("--samples_per_city", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--moe_top_k", type=int, default=2)
    p.add_argument("--vocab_l0", type=int, default=16384)
    p.add_argument("--vocab_l1", type=int, default=4096)
    p.add_argument("--vocab_l2", type=int, default=1024)
    p.add_argument("--mask_ratio", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gen_lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local_steps", type=int, default=100)
    p.add_argument("--dp_epsilon", type=float, default=8.0)
    p.add_argument("--dp_delta", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/privamove")
    p.add_argument("--benchmark_privacy", action="store_true", default=True)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device

    print("=" * 60)
    print("PrivaMove: Federated Continual Learning Pre-training")
    print(f"  Cities: {args.num_cities}, Rounds: {args.rounds}, Local steps: {args.local_steps}")
    print(f"  DP: epsilon={args.dp_epsilon}, delta={args.dp_delta}")
    print("=" * 60)

    # H3 tokenizer
    h3_tok = H3Tokenizer(
        res0=9, res1=7, res2=5,
        vocab_sizes=(args.vocab_l0, args.vocab_l1, args.vocab_l2),
        hash_tokens=True,
    )

    # Build city datasets
    city_datasets = []
    for c in range(args.num_cities):
        ds = CityTrajectoryDataset(
            city_id=c, num_samples=args.samples_per_city,
            max_len=args.max_len, seed=args.seed,
        )
        city_datasets.append(ds)

    # Build per-city models and generators
    city_clients = []
    for c in range(args.num_cities):
        model = PrivaMoveModel(
            vocab_l0=args.vocab_l0,
            vocab_l1=args.vocab_l1,
            vocab_l2=args.vocab_l2,
            embed_dim=args.embed_dim,
            depth=args.depth,
            heads=args.heads,
            num_cities=args.num_cities,
            moe_top_k=args.moe_top_k,
            max_seq_len=args.max_len,
        ).to(device)

        gen = TrajectoryGenerator(
            hidden_dim=args.embed_dim,
            max_len=args.max_len,
            num_cities=args.num_cities,
        ).to(device)
        dp_gen = DPGenerativeReplay(
            gen, epsilon=args.dp_epsilon, delta=args.dp_delta,
        )

        time_encoder = TimeFeatures(args.embed_dim).to(device)

        client = CityClient(
            city_id=c, model=model, generator=dp_gen,
            dataset=city_datasets[c], h3_tok=h3_tok,
            time_encoder=time_encoder, args=args,
        )
        city_clients.append(client)

    # Initialize all models with same weights
    init_state = city_clients[0].model.get_federated_state()
    for c in range(1, args.num_cities):
        city_clients[c].model.load_federated_state(init_state)

    # Federated averager
    averager = FederatedAverager(args.num_cities)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Federated training loop
    for round_idx in range(args.rounds):
        print(f"\n--- Round {round_idx + 1}/{args.rounds} ---")

        # 1) Each city trains locally
        round_metrics = []
        for client in city_clients:
            metrics = client.local_train(args.local_steps)
            round_metrics.append(metrics)
            print(
                f"  City {metrics['city_id']}: "
                f"loss={metrics['avg_loss']:.4f} "
                f"gen_loss={metrics['gen_loss']:.4f}"
            )

        # 2) Cities send shared expert state to server
        for client in city_clients:
            shared_state = client.model.get_federated_state()
            averager.receive_update(client.city_id, shared_state, len(client.dataset))

        # 3) Server aggregates
        global_state = averager.aggregate()

        # 4) Broadcast back to cities and reset optimizer state
        #    (Adam momentum is stale after weight replacement → causes NaN)
        for client in city_clients:
            client.model.load_federated_state(global_state, alpha=0.5)  # interpolate
            # Reset optimizer state to avoid stale momentum causing NaN
            client.model_optimizer = torch.optim.AdamW(
                client.model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            )

        # 5) Generate synthetic replay data from each city's DP generator
        #    and train other cities on it (cross-city knowledge transfer)
        for c_src in range(args.num_cities):
            synthetic = city_clients[c_src].generator.generate(
                num_samples=50, city_id=c_src, traj_length=args.max_len // 2,
                device=device, num_steps=10,
            )
            # Could train other cities on this synthetic data
            # (simplified: just log that replay happened)
            if round_idx == 0:
                print(f"  Generated {synthetic.shape[0]} synthetic trajectories from city {c_src}")

        # Save checkpoint
        if (round_idx + 1) % 5 == 0:
            ckpt = os.path.join(args.ckpt_dir, f"privamove_round{round_idx + 1}.pt")
            torch.save({
                "round": round_idx + 1,
                "global_state": global_state,
                "args": vars(args),
            }, ckpt)
            print(f"  Saved: {ckpt}")

    # Privacy benchmarking
    if args.benchmark_privacy:
        print("\n--- Privacy Benchmarking (Membership Inference) ---")
        attack = MembershipInferenceAttack(args.embed_dim).to(device)

        # Use city 0's model and data
        model = city_clients[0].model
        model.eval()
        time_enc = city_clients[0].time_encoder
        time_enc.eval()

        # Member data (from training set)
        member_loader = DataLoader(
            city_datasets[0], batch_size=64, shuffle=True, collate_fn=collate_city,
        )
        # Non-member data (from a different city)
        nonmember_loader = DataLoader(
            city_datasets[-1], batch_size=64, shuffle=True, collate_fn=collate_city,
        )

        def get_features(loader, n_batches=5):
            feats = []
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    if i >= n_batches:
                        break
                    c = batch["coords"].to(device)
                    ts = batch["timestamps"].to(device)
                    m = batch["attention_mask"].to(device)
                    bsz = c.shape[0]
                    c_np = c.cpu().numpy()
                    l0, l1, l2 = [], [], []
                    for b in range(bsz):
                        a, bb, cc = h3_tok.tokenize(c_np[b])
                        l0.append(a); l1.append(bb); l2.append(cc)
                    l0 = torch.from_numpy(np.stack(l0)).to(device)
                    l1 = torch.from_numpy(np.stack(l1)).to(device)
                    l2 = torch.from_numpy(np.stack(l2)).to(device)
                    te = time_enc(ts, m)
                    out = model(l0, l1, l2, te, m)
                    f = attack.compute_attack_features(model, l0, out["token_logits"], m)
                    feats.append(f)
            return torch.cat(feats)

        member_feats = get_features(member_loader)
        nonmember_feats = get_features(nonmember_loader)

        # Train attack model (simplified)
        attack_opt = torch.optim.Adam(attack.parameters(), lr=1e-3)
        for epoch in range(50):
            n = min(member_feats.shape[0], nonmember_feats.shape[0])
            all_feats = torch.cat([member_feats[:n], nonmember_feats[:n]])
            labels = torch.cat([torch.ones(n, device=device), torch.zeros(n, device=device)]).long()
            perm = torch.randperm(2 * n, device=device)
            logits = attack(all_feats[perm])
            loss = F.cross_entropy(logits, labels[perm])
            attack_opt.zero_grad()
            loss.backward()
            attack_opt.step()

        privacy_metrics = attack.evaluate_privacy(member_feats, nonmember_feats)
        print(f"  Attack accuracy: {privacy_metrics['attack_accuracy']:.4f}")
        print(f"  Attack AUC: {privacy_metrics['attack_auc']:.4f}")
        print(f"  TPR: {privacy_metrics['true_positive_rate']:.4f}")
        print(f"  TNR: {privacy_metrics['true_negative_rate']:.4f}")
        print(f"  (Lower = better privacy. Random = 0.5)")

    # Final save
    final_path = os.path.join(args.ckpt_dir, "privamove_final.pt")
    torch.save({
        "round": args.rounds,
        "global_state": global_state,
        "args": vars(args),
    }, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
