import argparse
import json
from copy import deepcopy
from pathlib import Path

from route_rangers.cli import run_unitraj_eval as ue


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run UniTraj-style eval across multiple datasets"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", required=True, help="Local data paths")
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cuda" if ue.torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument(
        "--split_mode", type=str, default="both", choices=["both", "random", "temporal"]
    )
    parser.add_argument(
        "--task", type=str, default="both", choices=["both", "recovery", "prediction"]
    )
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--pred_steps", type=int, default=5)
    parser.add_argument(
        "--centroid_level", type=str, default="l0", choices=["l0", "l1", "l2"]
    )
    parser.add_argument("--centroid_samples", type=int, default=0)
    parser.add_argument("--centroid_fraction", type=float, default=0.0)
    parser.add_argument("--coord_noise_std_m", type=float, default=0.0)
    parser.add_argument("--input_drop_ratio", type=float, default=0.0)
    parser.add_argument("--disable_graph", action="store_true")
    parser.add_argument(
        "--output", type=str, default="cache/unitraj_transfer_suite.json"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = {
        "checkpoint": args.checkpoint,
        "datasets": {},
    }
    for dataset in args.datasets:
        run_args = deepcopy(args)
        run_args.local_data = dataset
        results["datasets"][dataset] = ue.run_eval(run_args)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved transfer suite results: {out}")


if __name__ == "__main__":
    main()
