import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from route_rangers.baselines import Normalize, TrajectoryDataset, UniTraj


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a UniTraj-compatible baseline on local WorldTrace-format data"
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--task", type=str, default="both", choices=["both", "recovery", "prediction"]
    )
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--pred_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def _haversine_m(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() == 0:
        return pred.new_zeros((0,))
    deg2rad = math.pi / 180.0
    lat1 = pred[:, 0] * deg2rad
    lon1 = pred[:, 1] * deg2rad
    lat2 = target[:, 0] * deg2rad
    lon2 = target[:, 1] * deg2rad
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    )
    c = 2 * torch.asin(torch.clamp(a.sqrt(), max=1.0))
    return 6371000.0 * c


def _build_mask_indices(
    attention: torch.Tensor,
    max_len: int,
    task: str,
    mask_ratio: float,
    pred_steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    bsz = attention.shape[0]
    if task == "prediction":
        m = pred_steps
    else:
        m = max(1, int(max_len * mask_ratio))
    indices = torch.zeros((bsz, m), dtype=torch.long, device=attention.device)
    for b in range(bsz):
        vlen = int(attention[b].sum().item())
        if task == "prediction":
            if vlen >= pred_steps:
                idx = list(range(vlen - pred_steps, vlen))
            else:
                idx = list(range(max_len - pred_steps, max_len))
        else:
            num = max(1, int(vlen * mask_ratio)) if vlen > 0 else 1
            num = min(num, m)
            if vlen > 0:
                perm = torch.randperm(
                    vlen, generator=generator, device=attention.device
                ).tolist()
                idx = perm[:num]
            else:
                idx = []
            if num < m:
                remaining = [i for i in range(max_len) if i not in idx]
                extra = np.random.choice(
                    remaining, size=m - num, replace=False
                ).tolist()
                idx.extend(extra)
        indices[b] = torch.tensor(idx, dtype=torch.long, device=attention.device)
    return indices


def _unnormalize(
    traj: torch.Tensor, original: torch.Tensor, transform: Normalize
) -> torch.Tensor:
    # traj: [B, 2, L]
    out = traj.clone()
    if transform is not None:
        std = transform.std.to(out.device).view(1, 2, 1)
        mean = transform.mean.to(out.device).view(1, 2, 1)
        out = out * std + mean
    out = out.transpose(1, 2)
    out = out + original[:, None, :]
    return out


def evaluate(
    loader: DataLoader, model: UniTraj, transform: Normalize, args, task: str
) -> Tuple[float, float, int]:
    device = args.device
    total_abs = 0.0
    total_sq = 0.0
    total_n = 0
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 11)

    for batch in loader:
        traj = batch["trajectory"].to(device)  # [B, 2, L]
        intervals = batch["intervals"].to(device)
        attention = batch["attention_mask"].to(device)
        original = batch["original"].to(device)  # [B, 2]

        indices = _build_mask_indices(
            attention,
            args.max_len,
            task=task,
            mask_ratio=args.mask_ratio,
            pred_steps=args.pred_steps,
            generator=generator,
        )

        with torch.no_grad():
            pred_traj, _ = model(traj, intervals, indices)

        pred_coords = _unnormalize(pred_traj, original, transform)
        true_coords = _unnormalize(traj, original, transform)

        # gather masked points
        mask = torch.zeros(
            (traj.shape[0], args.max_len), dtype=torch.bool, device=device
        )
        for b in range(mask.shape[0]):
            mask[b, indices[b]] = True
        mask = mask & attention.bool()

        pred_sel = pred_coords[mask]
        true_sel = true_coords[mask]
        d = _haversine_m(pred_sel, true_sel)
        if d.numel() == 0:
            continue
        total_abs += float(d.sum().item())
        total_sq += float((d**2).sum().item())
        total_n += int(d.numel())

    if total_n == 0:
        return 0.0, 0.0, 0
    mae = total_abs / total_n
    rmse = math.sqrt(total_sq / total_n)
    return mae, rmse, total_n


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    model = UniTraj(
        trajectory_length=args.max_len,
        patch_size=1,
        embedding_dim=128,
        encoder_layers=8,
        encoder_heads=4,
        decoder_layers=4,
        decoder_heads=4,
        mask_ratio=args.mask_ratio,
    ).to(device)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=False)

    model.eval()
    transform = Normalize()
    dataset = TrajectoryDataset(
        data_path=args.data_path, max_len=args.max_len, transform=transform
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    tasks = ["recovery", "prediction"] if args.task == "both" else [args.task]
    results = {
        "checkpoint": args.checkpoint,
        "dataset": args.data_path,
        "coord_order": {
            "trajectory_tensor_order": "lon_lat",
            "distance_metric_assumed_order": "lat_lon",
            "normalized_output_order": "unitraj_compatible_baseline",
        },
        "settings": {
            "max_len": args.max_len,
            "mask_ratio": args.mask_ratio,
            "pred_steps": args.pred_steps,
        },
        "metrics": {},
    }
    for task in tasks:
        mae, rmse, n = evaluate(loader, model, transform, args, task)
        print(f"{task}: mae_m={mae:.2f} rmse_m={rmse:.2f} n={n}")
        results["metrics"][task] = {"mae_m": mae, "rmse_m": rmse, "n": n}

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            import json

            json.dump(results, f, indent=2)
        print(f"saved UniTraj external eval results: {out}")


if __name__ == "__main__":
    main()
