import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from utils.dataset import Normalize, TrajectoryDataset
from utils.unitraj import UniTraj


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train UniTraj on a local WorldTrace-format dataset"
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="external/unitraj/outputs")
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(n: int, val_ratio: float, seed: int):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    if val_ratio <= 0:
        return idx, []
    split = max(1, int(n * (1 - val_ratio)))
    return idx[:split], idx[split:]


def loss_fn(pred, traj, mask, attention):
    return torch.mean((pred - traj) ** 2 * mask * attention) / 0.5


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device

    transform = Normalize()
    dataset = TrajectoryDataset(
        data_path=args.data_path,
        max_len=args.max_len,
        transform=transform,
        mask_ratio=args.mask_ratio,
    )
    train_idx, val_idx = split_indices(len(dataset), args.val_ratio, args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = None
    if val_idx:
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

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

    optim = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pt"
    last_path = out_dir / "last_model.pt"
    best_val = math.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            traj = batch["trajectory"].to(device)
            attention = batch["attention_mask"].to(device)
            intervals = batch["intervals"].to(device)
            indices = batch["indices"]
            attention = attention.unsqueeze(1).expand_as(traj)
            pred_traj, mask = model(traj, intervals, indices)
            loss = loss_fn(pred_traj, traj, mask, attention)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            train_losses.append(loss.item())

        avg_train = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    traj = batch["trajectory"].to(device)
                    attention = batch["attention_mask"].to(device)
                    intervals = batch["intervals"].to(device)
                    indices = batch["indices"]
                    attention = attention.unsqueeze(1).expand_as(traj)
                    pred_traj, mask = model(traj, intervals, indices)
                    loss = loss_fn(pred_traj, traj, mask, attention)
                    val_losses.append(loss.item())
            val_loss = float(np.mean(val_losses)) if val_losses else None

        print(
            f"epoch={epoch} train_loss={avg_train:.6f} "
            f"val_loss={val_loss if val_loss is not None else 'n/a'}"
        )

        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
        torch.save(model.state_dict(), last_path)

    print(f"saved best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
