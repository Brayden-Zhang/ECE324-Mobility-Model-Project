import argparse
import time
from pathlib import Path

import torch

from route_rangers.cli import run_benchmarks as rb


def parse_args():
    parser = argparse.ArgumentParser(description="Model size and throughput stats")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--disable_graph", action="store_true")
    return parser.parse_args()


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def main():
    args = parse_args()
    pack = rb.load_backbone(
        args.checkpoint,
        device=args.device,
        override_max_len=args.max_len,
        disable_graph=args.disable_graph,
    )

    model_params = count_params(pack.model)
    tokenizer_params = count_params(pack.tokenizer)
    time_params = count_params(pack.time_encoder)

    coords = torch.zeros((args.batch_size, args.max_len, 2), device=args.device)
    coords[..., 0] = 40.0
    coords[..., 1] = -73.0
    timestamps = (
        torch.arange(args.max_len, device=args.device)
        .float()
        .unsqueeze(0)
        .repeat(args.batch_size, 1)
    )
    attention = torch.ones((args.batch_size, args.max_len), device=args.device)
    batch = {
        "coords": coords,
        "timestamps": timestamps,
        "attention_mask": attention,
        "start_ts": torch.zeros((args.batch_size,), device=args.device),
    }

    for _ in range(args.warmup):
        _ = rb.forward_backbone(
            batch, pack, device=args.device, max_len=args.max_len, mask=None
        )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(args.iters):
        _ = rb.forward_backbone(
            batch, pack, device=args.device, max_len=args.max_len, mask=None
        )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.time() - start
    per_iter = elapsed / max(1, args.iters)

    print(f"checkpoint: {args.checkpoint}")
    print(f"model_params: {model_params}")
    print(f"tokenizer_params: {tokenizer_params}")
    print(f"time_encoder_params: {time_params}")
    print(f"batch_size: {args.batch_size}")
    print(f"max_len: {args.max_len}")
    print(f"device: {args.device}")
    print(f"forward_seconds_per_iter: {per_iter:.4f}")


if __name__ == "__main__":
    main()
