#!/usr/bin/env python3
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a reproducible local foundation-model evaluation suite (HMT + UniTraj-comparable metrics)."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="HMT checkpoint path")
    parser.add_argument("--local_data", type=str, required=True, help="WorldTrace-format local eval data (.pkl/.parquet)")
    parser.add_argument("--output_dir", type=str, default="cache/foundation_suite", help="Output directory for JSON results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--split_mode", type=str, default="both", choices=["both", "random", "temporal", "all"])
    parser.add_argument("--run_data_efficiency", action="store_true", help="Also run centroid-data-efficiency sweep")
    parser.add_argument("--run_invariance", action="store_true", help="Also run invariance suite")
    parser.add_argument(
        "--run_external_unitraj",
        action="store_true",
        help="Also run external UniTraj baseline eval (requires external/unitraj checkout/checkpoint).",
    )
    parser.add_argument("--unitraj_data_path", type=str, default="", help="Data path for external UniTraj eval")
    parser.add_argument("--unitraj_checkpoint", type=str, default="", help="Checkpoint for external UniTraj eval")
    parser.add_argument("--name", type=str, default="latest", help="Run tag used in output filenames")
    return parser.parse_args()


def run_cmd(cmd: List[str]):
    print(">>>", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def _cmd(pyfile: str, params: Dict[str, object]) -> List[str]:
    cmd = [sys.executable, str(ROOT / pyfile)]
    for k, v in params.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
            continue
        if isinstance(v, (list, tuple)):
            cmd.append(f"--{k}")
            cmd.extend([str(x) for x in v])
            continue
        cmd.extend([f"--{k}", str(v)])
    return cmd


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = {
        "checkpoint": args.checkpoint,
        "local_data": args.local_data,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
    }

    outputs = {
        "benchmarks": str(output_dir / f"benchmark_{args.name}.json"),
        "unitraj_eval": str(output_dir / f"unitraj_eval_{args.name}.json"),
        "unitraj_eval_robust": str(output_dir / f"unitraj_eval_robust_{args.name}.json"),
        "length_sensitivity": str(output_dir / f"length_sensitivity_{args.name}.json"),
    }

    commands: List[List[str]] = []
    commands.append(
        _cmd(
            "scripts/run_benchmarks.py",
            {
                **base,
                "split_mode": "both" if args.split_mode == "all" else args.split_mode,
                "output": outputs["benchmarks"],
            },
        )
    )
    commands.append(
        _cmd(
            "scripts/run_unitraj_eval.py",
            {
                **base,
                "split_mode": args.split_mode,
                "task": "both",
                "exclude_unknown": True,
                "output": outputs["unitraj_eval"],
            },
        )
    )
    commands.append(
        _cmd(
            "scripts/run_unitraj_eval.py",
            {
                **base,
                "split_mode": args.split_mode,
                "task": "both",
                "exclude_unknown": True,
                "coord_noise_std_m": 30,
                "input_drop_ratio": 0.2,
                "output": outputs["unitraj_eval_robust"],
            },
        )
    )
    commands.append(
        _cmd(
            "scripts/run_length_sensitivity.py",
            {
                **base,
                "mask_ratio": 0.3,
                "dest_mask_last_k": 1,
                "output": outputs["length_sensitivity"],
            },
        )
    )

    if args.run_data_efficiency:
        outputs["data_efficiency"] = str(output_dir / f"unitraj_data_efficiency_{args.name}.json")
        commands.append(
            _cmd(
                "scripts/run_data_efficiency.py",
                {
                    **base,
                    "fractions": [0.05, 0.1, 0.2, 0.5, 1.0],
                    "split_mode": "both" if args.split_mode == "all" else args.split_mode,
                    "task": "both",
                    "output": outputs["data_efficiency"],
                },
            )
        )

    if args.run_invariance:
        outputs["invariance"] = str(output_dir / f"invariance_{args.name}.json")
        commands.append(
            _cmd(
                "scripts/run_invariance_suite.py",
                {
                    **base,
                    "output": outputs["invariance"],
                },
            )
        )

    if args.run_external_unitraj:
        if not args.unitraj_data_path:
            raise ValueError("--unitraj_data_path is required with --run_external_unitraj")
        outputs["unitraj_external"] = str(output_dir / f"unitraj_external_eval_{args.name}.json")
        commands.append(
            _cmd(
                "scripts/run_unitraj_external_eval.py",
                {
                    "data_path": args.unitraj_data_path,
                    "checkpoint": args.unitraj_checkpoint,
                    "max_len": args.max_len,
                    "batch_size": args.batch_size,
                    "device": args.device,
                    "task": "both",
                    "output": outputs["unitraj_external"],
                },
            )
        )

    for cmd in commands:
        run_cmd(cmd)

    manifest_path = output_dir / f"suite_manifest_{args.name}.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "local_data": args.local_data,
                "output_dir": str(output_dir),
                "outputs": outputs,
                "commands": commands,
            },
            f,
            indent=2,
        )
    print(f"saved suite manifest: {manifest_path}")


if __name__ == "__main__":
    main()
