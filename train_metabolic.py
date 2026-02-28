"""
Metabolic-Mobility Training Script — Urban Sustainability Pre-training.

Usage:
  python train_metabolic.py --max_steps 5000 --batch_size 32

Pre-trains a sustainability-aware MFM with:
  - Masked token prediction (spatial)
  - Emission/energy prediction
  - Flow matching
  - Scenario discovery (what-if analysis)
"""

import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils.hmt import H3Tokenizer, TimeFeatures
from utils.metabolic import (
    MetabolicMobilityModel,
    TransportModeEncoder,
    metabolic_token_loss,
    emission_prediction_loss,
    energy_prediction_loss,
    scenario_impact_loss,
    multi_objective_loss,
)
from utils.flow import sample_rectified_flow_targets, flow_matching_loss


# ---------------------------------------------------------------------------
# Synthetic Sustainability Dataset
# ---------------------------------------------------------------------------

class SyntheticMetabolicDataset(Dataset):
    """Generates trajectory data enriched with transport mode, emissions, energy,
    and environmental sensor readings.

    Each sample simulates a commuter trip with realistic emission profiles
    based on transport mode and traffic conditions.
    """

    def __init__(
        self,
        num_samples: int = 20000,
        max_len: int = 64,
        num_env_timesteps: int = 24,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.max_len = max_len
        self.num_env_t = num_env_timesteps
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.RandomState(self.rng.randint(0, 2**31) + idx)
        L = rng.randint(self.max_len // 2, self.max_len + 1)

        # Transport mode (may switch mid-trip)
        primary_mode = rng.randint(0, TransportModeEncoder.NUM_MODES)
        modes = np.full(self.max_len, primary_mode, dtype=np.int64)
        if rng.random() < 0.3:  # 30% chance of mode switch
            switch_point = rng.randint(L // 3, 2 * L // 3)
            secondary_mode = rng.randint(0, TransportModeEncoder.NUM_MODES)
            modes[switch_point:L] = secondary_mode

        # Speed profile based on mode
        mode_speeds = {0: 1.5, 1: 4.0, 2: 8.0, 3: 12.0, 4: 15.0, 5: 5.0}  # m/s
        base_speed = mode_speeds.get(primary_mode, 5.0)
        speeds = np.zeros(self.max_len, dtype=np.float32)
        for t in range(L):
            speeds[t] = max(0, base_speed * rng.uniform(0.5, 1.5) + rng.normal(0, 1))

        # Trajectory coordinates
        coords = np.zeros((self.max_len, 2), dtype=np.float32)
        heading = rng.uniform(0, 2 * np.pi)
        for t in range(L):
            if t == 0:
                coords[t] = rng.uniform(-0.01, 0.01, 2)
            else:
                heading += rng.normal(0, 0.1)
                step = speeds[t] * 1e-5  # scale to lat/lon
                coords[t, 0] = coords[t - 1, 0] + step * np.cos(heading)
                coords[t, 1] = coords[t - 1, 1] + step * np.sin(heading)

        # Time
        start_hour = rng.uniform(0, 24)
        timestamps = np.zeros(self.max_len, dtype=np.float32)
        timestamps[:L] = start_hour * 3600 + np.cumsum(rng.uniform(1, 30, L))
        hours = np.zeros(self.max_len, dtype=np.float32)
        hours[:L] = (timestamps[:L] / 3600) % 24

        # Congestion (peaks at 8am, 5pm)
        congestion = np.zeros(self.max_len, dtype=np.float32)
        for t in range(L):
            h = hours[t]
            morning_peak = np.exp(-((h - 8) ** 2) / 4)
            evening_peak = np.exp(-((h - 17) ** 2) / 4)
            congestion[t] = min(1.0, (morning_peak + evening_peak) * 0.8 + rng.uniform(0, 0.2))

        # Ground truth emissions and energy
        emission_factors = [0.0, 0.0, 89.0, 192.0, 41.0, 100.0]
        energy_factors = [250.0, 100.0, 800.0, 2500.0, 400.0, 1000.0]
        emissions = np.zeros(self.max_len, dtype=np.float32)
        energy = np.zeros(self.max_len, dtype=np.float32)
        for t in range(L):
            mode = modes[t]
            dist_km = speeds[t] * 0.001  # approximate
            cong_factor = 1.0 + congestion[t] * 0.5
            emissions[t] = emission_factors[mode] * dist_km * cong_factor
            energy[t] = energy_factors[mode] * dist_km * cong_factor

        # Environmental sensor data (8 features, 24 timestep window)
        env_data = np.zeros((self.num_env_t, 8), dtype=np.float32)
        for t in range(self.num_env_t):
            h = t
            env_data[t, 0] = max(0, 25 + 30 * np.exp(-((h - 8)**2)/4) + rng.normal(0, 5))  # PM2.5
            env_data[t, 1] = max(0, 15 + 20 * np.exp(-((h - 17)**2)/4) + rng.normal(0, 3))  # NO2
            env_data[t, 2] = max(0, 30 + 20 * np.sin(np.pi * h / 12) + rng.normal(0, 5))   # O3
            env_data[t, 3] = max(0, 50 + 15 * (1 + congestion[min(t, L - 1)]) + rng.normal(0, 3))  # Noise
            env_data[t, 4] = 15 + 10 * np.sin(np.pi * h / 12) + rng.normal(0, 2)  # Temp
            env_data[t, 5] = max(0, 100 + 50 * (congestion[min(t, L - 1)]) + rng.normal(0, 10))  # Power
            env_data[t, 6] = max(0, 500 * np.sin(np.pi * max(h - 6, 0) / 12) + rng.normal(0, 30))  # Solar
            env_data[t, 7] = max(0, 3 + rng.normal(0, 1.5))  # Wind

        mask = np.zeros(self.max_len, dtype=np.float32)
        mask[:L] = 1.0

        return {
            "coords": torch.from_numpy(coords),
            "timestamps": torch.from_numpy(timestamps),
            "attention_mask": torch.from_numpy(mask),
            "transport_mode": torch.from_numpy(modes),
            "speed": torch.from_numpy(speeds),
            "hour": torch.from_numpy(hours),
            "congestion": torch.from_numpy(congestion),
            "emissions": torch.from_numpy(emissions),
            "energy": torch.from_numpy(energy),
            "env_data": torch.from_numpy(env_data),
        }


def collate_metabolic(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch])
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Metabolic-Mobility sustainability MFM")
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=64)
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
    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--token_weight", type=float, default=1.0)
    p.add_argument("--emission_weight", type=float, default=1.0)
    p.add_argument("--energy_weight", type=float, default=1.0)
    p.add_argument("--flow_weight", type=float, default=0.5)
    p.add_argument("--scenario_weight", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/metabolic")
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


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device

    print("=" * 60)
    print("Metabolic-Mobility: Urban Sustainability Pre-training")
    print("=" * 60)

    # Dataset
    dataset = SyntheticMetabolicDataset(
        num_samples=args.num_samples,
        max_len=args.max_len,
        seed=args.seed,
    )
    n_val = max(1, len(dataset) // 10)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate_metabolic, num_workers=2)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=collate_metabolic, num_workers=0)

    # Tokenizer
    h3_tok = H3Tokenizer(
        res0=9, res1=7, res2=5,
        vocab_sizes=(args.vocab_l0, args.vocab_l1, args.vocab_l2),
        hash_tokens=True,
    )

    # Time encoder
    time_encoder = TimeFeatures(args.embed_dim).to(device)

    # Model
    model = MetabolicMobilityModel(
        vocab_l0=args.vocab_l0,
        vocab_l1=args.vocab_l1,
        vocab_l2=args.vocab_l2,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        max_seq_len=args.max_len,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(time_encoder.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
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

            coords = batch["coords"].to(device)
            timestamps = batch["timestamps"].to(device)
            mask = batch["attention_mask"].to(device)
            t_mode = batch["transport_mode"].to(device)
            speed = batch["speed"].to(device)
            hour = batch["hour"].to(device)
            congestion = batch["congestion"].to(device)
            gt_emissions = batch["emissions"].to(device)
            gt_energy = batch["energy"].to(device)
            env_data = batch["env_data"].to(device)

            # Normalize emission/energy targets to prevent large loss values
            em_mean, em_std = gt_emissions[mask.bool()].mean(), gt_emissions[mask.bool()].std().clamp(min=1e-6)
            en_mean, en_std = gt_energy[mask.bool()].mean(), gt_energy[mask.bool()].std().clamp(min=1e-6)
            gt_emissions_norm = (gt_emissions - em_mean) / em_std
            gt_energy_norm = (gt_energy - en_mean) / en_std

            bsz, L = coords.shape[:2]

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

            time_embed = time_encoder(timestamps, mask)

            mask_indices = (torch.rand(bsz, L, device=device) < args.mask_ratio) & mask.bool()

            lr_scale = cosine_lr(step, args.warmup_steps, args.max_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * lr_scale

            with torch.amp.autocast(device_type=device, enabled=args.amp and device == "cuda"):
                out = model(
                    tokens_l0, tokens_l1, tokens_l2,
                    time_embed, mask,
                    transport_mode=t_mode,
                    speed=speed,
                    hour=hour,
                    congestion=congestion,
                    env_data=env_data,
                )

                # Token loss
                loss_token = metabolic_token_loss(
                    out["token_logits"], tokens_l0, mask_indices, mask,
                )

                # Emission prediction loss (normalized)
                loss_emission = emission_prediction_loss(
                    out["pred_emissions"], gt_emissions_norm, mask,
                )

                # Energy prediction loss (normalized)
                loss_energy = energy_prediction_loss(
                    out["pred_energy"], gt_energy_norm, mask,
                )

                # Flow matching
                x_t, target_v, t_flow = sample_rectified_flow_targets(coords)
                pred_v = model.flow_head(out["hidden"], x_t, t_flow)
                loss_flow = flow_matching_loss(pred_v, target_v, mask)

                # Scenario discovery loss (using random interventions)
                intervention_ids = torch.randint(
                    0, model.scenario_head.NUM_INTERVENTIONS, (bsz,), device=device,
                )
                # Use environmental encoder pooled output as env_embed
                env_embed_for_scenario = model.metabolism_encoder.env_encoder(env_data)
                if env_embed_for_scenario.dim() == 3:
                    env_embed_for_scenario = env_embed_for_scenario.mean(dim=1)

                scenario_out = model.scenario_analysis(
                    out["pooled"], env_embed_for_scenario, intervention_ids,
                )
                # Synthetic scenario targets (in practice: from simulation models)
                fake_impact = torch.randn_like(scenario_out["impact"]) * 0.1
                loss_scenario = scenario_impact_loss(
                    scenario_out["impact"], fake_impact, scenario_out["uncertainty"],
                )

                # Multi-objective combination
                total_loss = multi_objective_loss(
                    {
                        "token": loss_token,
                        "emission": loss_emission,
                        "energy": loss_energy,
                        "flow": loss_flow,
                        "scenario": loss_scenario,
                    },
                    {
                        "token": args.token_weight,
                        "emission": args.emission_weight,
                        "energy": args.energy_weight,
                        "flow": args.flow_weight,
                        "scenario": args.scenario_weight,
                    },
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
                    f"emission={loss_emission.item():.4f} "
                    f"energy={loss_energy.item():.4f} "
                    f"flow={loss_flow.item():.4f} "
                    f"scenario={loss_scenario.item():.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            if step % args.save_interval == 0:
                ckpt = os.path.join(args.ckpt_dir, f"metabolic_step{step}.pt")
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
                val_metrics = {"token": [], "emission": [], "energy": []}
                with torch.no_grad():
                    for i, vb in enumerate(val_loader):
                        if i >= 10:
                            break
                        vc = vb["coords"].to(device)
                        vts = vb["timestamps"].to(device)
                        vm = vb["attention_mask"].to(device)
                        vtm = vb["transport_mode"].to(device)
                        vsp = vb["speed"].to(device)
                        vhr = vb["hour"].to(device)
                        vcg = vb["congestion"].to(device)
                        vge = vb["emissions"].to(device)
                        vgn = vb["energy"].to(device)
                        venv = vb["env_data"].to(device)
                        vB, vL = vc.shape[:2]
                        vc_np = vc.cpu().numpy()
                        vl0, vl1, vl2 = [], [], []
                        for b in range(vB):
                            a, bb, cc = h3_tok.tokenize(vc_np[b])
                            vl0.append(a); vl1.append(bb); vl2.append(cc)
                        vl0 = torch.from_numpy(np.stack(vl0)).to(device)
                        vl1 = torch.from_numpy(np.stack(vl1)).to(device)
                        vl2 = torch.from_numpy(np.stack(vl2)).to(device)
                        vte = time_encoder(vts, vm)
                        vmask = (torch.rand(vB, vL, device=device) < args.mask_ratio) & vm.bool()
                        vout = model(vl0, vl1, vl2, vte, vm, vtm, vsp, vhr, vcg, venv)
                        val_metrics["token"].append(
                            metabolic_token_loss(vout["token_logits"], vl0, vmask, vm).item()
                        )
                        val_metrics["emission"].append(
                            emission_prediction_loss(vout["pred_emissions"], vge, vm).item()
                        )
                        val_metrics["energy"].append(
                            energy_prediction_loss(vout["pred_energy"], vgn, vm).item()
                        )
                print(
                    f"  [Eval] token={np.mean(val_metrics['token']):.4f} "
                    f"emission={np.mean(val_metrics['emission']):.4f} "
                    f"energy={np.mean(val_metrics['energy']):.4f}"
                )
                model.train()
                time_encoder.train()

    # Final save
    final_path = os.path.join(args.ckpt_dir, "metabolic_final.pt")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "time_encoder": time_encoder.state_dict(),
        "args": vars(args),
    }, final_path)
    print(f"Training complete. Final model: {final_path}")

    # Demo: scenario analysis
    print("\n--- Scenario Discovery Demo ---")
    model.eval()
    demo_batch = next(iter(val_loader))
    with torch.no_grad():
        dc = demo_batch["coords"][:4].to(device)
        dts = demo_batch["timestamps"][:4].to(device)
        dm = demo_batch["attention_mask"][:4].to(device)
        dtm = demo_batch["transport_mode"][:4].to(device)
        dsp = demo_batch["speed"][:4].to(device)
        dhr = demo_batch["hour"][:4].to(device)
        dcg = demo_batch["congestion"][:4].to(device)
        denv = demo_batch["env_data"][:4].to(device)

        dc_np = dc.cpu().numpy()
        dl0, dl1, dl2 = [], [], []
        for b in range(4):
            a, bb, cc = h3_tok.tokenize(dc_np[b])
            dl0.append(a); dl1.append(bb); dl2.append(cc)
        dl0 = torch.from_numpy(np.stack(dl0)).to(device)
        dl1 = torch.from_numpy(np.stack(dl1)).to(device)
        dl2 = torch.from_numpy(np.stack(dl2)).to(device)
        dte = time_encoder(dts, dm)

        dout = model(dl0, dl1, dl2, dte, dm, dtm, dsp, dhr, dcg, denv)

        interventions = ["pedestrianize", "add_bike_lane", "congestion_charge",
                        "add_transit", "reduce_speed_limit", "green_corridor",
                        "ev_mandate", "remote_work_incentive"]

        env_emb = model.metabolism_encoder.env_encoder(denv)
        if env_emb.dim() == 3:
            env_emb = env_emb.mean(dim=1)

        for interv_id in range(min(4, len(interventions))):
            iid = torch.full((4,), interv_id, device=device, dtype=torch.long)
            sout = model.scenario_analysis(dout["pooled"], env_emb, iid)
            impact = sout["impact"].mean(dim=0)
            print(f"  Intervention: {interventions[interv_id]}")
            for j, name in enumerate(sout["impact_names"]):
                print(f"    {name}: {impact[j].item():+.2f}")
            print(f"    Modal split: {sout['modal_split'].mean(dim=0).cpu().numpy().round(3)}")
            print()


if __name__ == "__main__":
    main()
