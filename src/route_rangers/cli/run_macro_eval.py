import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from route_rangers.cli import run_benchmarks as rb
from route_rangers.data.macro_dataset import MacroDistributionDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate macro distribution head")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--macro_data", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    pack = rb.load_backbone(args.checkpoint, device=args.device, override_max_len=200, disable_graph=False)
    if pack.model.macro_head is None:
        raise RuntimeError("macro_head is not configured in checkpoint")

    dataset = MacroDistributionDataset(args.macro_data, normalize=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    total_kl = 0.0
    total_js = 0.0
    total_l1 = 0.0
    total_acc = 0.0
    total_n = 0

    pack.model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.max_batches > 0 and i >= args.max_batches:
                break
            region_idx = batch["region_idx"].to(args.device)
            time_ts = batch["time_ts"].to(args.device)
            dist = batch["dist"].to(args.device)

            attn = torch.ones((region_idx.shape[0], 1), device=args.device)
            time_embed = pack.time_encoder(time_ts.unsqueeze(1), attn)[:, 0]
            logits = pack.model.macro_logits(region_idx, time_embed)
            logp = F.log_softmax(logits.float(), dim=-1)
            pred = logp.exp()

            kl = F.kl_div(logp, dist, reduction="batchmean")
            m = 0.5 * (pred + dist)
            js = 0.5 * (
                F.kl_div(torch.log(pred + 1e-8), m, reduction="batchmean")
                + F.kl_div(torch.log(dist + 1e-8), m, reduction="batchmean")
            )
            l1 = torch.abs(pred - dist).mean()

            top1 = (pred.argmax(dim=-1) == dist.argmax(dim=-1)).float().mean()

            bs = dist.shape[0]
            total_kl += float(kl.item()) * bs
            total_js += float(js.item()) * bs
            total_l1 += float(l1.item()) * bs
            total_acc += float(top1.item()) * bs
            total_n += int(bs)

    metrics = {
        "macro_kl": total_kl / max(1, total_n),
        "macro_js": total_js / max(1, total_n),
        "macro_l1": total_l1 / max(1, total_n),
        "macro_top1": total_acc / max(1, total_n),
        "n": total_n,
    }

    print(metrics)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                {
                    "checkpoint": args.checkpoint,
                    "macro_data": args.macro_data,
                    "metrics": metrics,
                },
                f,
                indent=2,
            )
        print(f"saved {out}")


if __name__ == "__main__":
    main()
