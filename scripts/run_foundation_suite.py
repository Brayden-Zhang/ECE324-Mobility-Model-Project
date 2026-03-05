#!/usr/bin/env python3
import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
ROBUST_COORD_NOISE_STD_M = 30
ROBUST_INPUT_DROP_RATIO = 0.2
LENGTH_SENSITIVITY_MASK_RATIO = 0.3
LENGTH_SENSITIVITY_DEST_MASK_LAST_K = 1
DATA_EFFICIENCY_FRACTIONS = [0.05, 0.1, 0.2, 0.5, 1.0]


@dataclass
class SuiteDefaults:
    batch_size: int = 32
    max_len: int = 200
    sample_limit: int = 0
    probe_epochs: int = 6
    probe_batch_size: int = 2048
    quick_cpu_batch_size: int = 8
    quick_cpu_max_len: int = 64
    quick_cpu_sample_limit: int = 128
    quick_cpu_probe_epochs: int = 2
    quick_cpu_probe_batch_size: int = 256


def detect_default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a reproducible local foundation-model evaluation suite (HMT + UniTraj-comparable metrics)."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="HMT checkpoint path")
    parser.add_argument("--local_data", type=str, required=True, help="WorldTrace-format local eval data (.pkl/.parquet)")
    parser.add_argument("--output_dir", type=str, default="cache/foundation_suite", help="Output directory for JSON results")
    parser.add_argument("--device", type=str, default=detect_default_device())
    parser.add_argument("--batch_size", type=int, default=SuiteDefaults.batch_size)
    parser.add_argument("--max_len", type=int, default=SuiteDefaults.max_len)
    parser.add_argument("--sample_limit", type=int, default=SuiteDefaults.sample_limit)
    parser.add_argument("--split_mode", type=str, default="both", choices=["both", "random", "temporal"])
    parser.add_argument("--run_data_efficiency", action="store_true", help="Also run centroid-data-efficiency sweep")
    parser.add_argument("--run_invariance", action="store_true", help="Also run invariance suite")
    parser.add_argument(
        "--run_external_unitraj",
        action="store_true",
        help="Also run external UniTraj baseline eval (requires external/unitraj checkout/checkpoint).",
    )
    parser.add_argument("--unitraj_data_path", type=str, default="", help="Data path for external UniTraj eval")
    parser.add_argument("--unitraj_checkpoint", type=str, default="", help="Checkpoint for external UniTraj eval")
    parser.add_argument(
        "--quick_cpu_smoke",
        action="store_true",
        help="Use low-cost defaults for quick CPU verification on a small sample.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--name", type=str, default="latest", help="Run tag used in output filenames")
    return parser.parse_args()


def run_cmd(cmd: List[str], dry_run: bool = False):
    print(">>>", " ".join(shlex.quote(c) for c in cmd))
    if not dry_run:
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


def _apply_quick_cpu_mode(args) -> None:
    if not args.quick_cpu_smoke:
        return
    args.device = "cpu"
    if args.batch_size == SuiteDefaults.batch_size:
        args.batch_size = SuiteDefaults.quick_cpu_batch_size
    if args.max_len == SuiteDefaults.max_len:
        args.max_len = SuiteDefaults.quick_cpu_max_len
    if args.sample_limit == SuiteDefaults.sample_limit:
        args.sample_limit = SuiteDefaults.quick_cpu_sample_limit


def _validate_args(args) -> None:
    if args.sample_limit < 0:
        raise ValueError(f"--sample_limit must be non-negative (>= 0), got {args.sample_limit}")
    if args.run_external_unitraj:
        if not args.unitraj_data_path:
            raise ValueError("--unitraj_data_path is required with --run_external_unitraj")
        if not args.unitraj_checkpoint:
            raise ValueError("--unitraj_checkpoint is required with --run_external_unitraj")
        unitraj_ckpt = Path(args.unitraj_checkpoint)
        if not unitraj_ckpt.exists():
            raise FileNotFoundError(f"unitraj checkpoint not found: {unitraj_ckpt}")


def _base_params(args) -> Dict[str, object]:
    return {
        "checkpoint": args.checkpoint,
        "local_data": args.local_data,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "sample_limit": args.sample_limit,
    }


def _build_outputs(output_dir: Path, run_name: str) -> Dict[str, str]:
    return {
        "benchmarks": str(output_dir / f"benchmark_{run_name}.json"),
        "unitraj_eval": str(output_dir / f"unitraj_eval_{run_name}.json"),
        "unitraj_eval_robust": str(output_dir / f"unitraj_eval_robust_{run_name}.json"),
        "length_sensitivity": str(output_dir / f"length_sensitivity_{run_name}.json"),
    }


def _build_commands(args, base: Dict[str, object], outputs: Dict[str, str]) -> List[List[str]]:
    commands: List[List[str]] = []
    probe_epochs = SuiteDefaults.quick_cpu_probe_epochs if args.quick_cpu_smoke else SuiteDefaults.probe_epochs
    probe_batch_size = (
        SuiteDefaults.quick_cpu_probe_batch_size if args.quick_cpu_smoke else SuiteDefaults.probe_batch_size
    )

    commands.append(
        _cmd(
            "scripts/run_benchmarks.py",
            {
                **base,
                "split_mode": args.split_mode,
                "probe_epochs": probe_epochs,
                "probe_batch_size": probe_batch_size,
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
                # Robust setting mirrors scripts/slurm_eval_full.sh and keeps
                # HMT-vs-HMT comparisons on the same perturbation protocol.
                "coord_noise_std_m": ROBUST_COORD_NOISE_STD_M,
                "input_drop_ratio": ROBUST_INPUT_DROP_RATIO,
                "output": outputs["unitraj_eval_robust"],
            },
        )
    )
    commands.append(
        _cmd(
            "scripts/run_length_sensitivity.py",
            {
                **base,
                # Use script defaults explicitly so the suite is reproducible.
                "mask_ratio": LENGTH_SENSITIVITY_MASK_RATIO,
                # Masking the last destination token avoids trivial copying.
                "dest_mask_last_k": LENGTH_SENSITIVITY_DEST_MASK_LAST_K,
                "output": outputs["length_sensitivity"],
            },
        )
    )

    if args.run_data_efficiency:
        outputs["data_efficiency"] = str(Path(args.output_dir) / f"unitraj_data_efficiency_{args.name}.json")
        commands.append(
            _cmd(
                "scripts/run_data_efficiency.py",
                {
                    **base,
                    # Standard coarse-to-full fractions used by slurm_eval_full.sh.
                    "fractions": DATA_EFFICIENCY_FRACTIONS,
                    "split_mode": args.split_mode,
                    "task": "both",
                    "output": outputs["data_efficiency"],
                },
            )
        )

    if args.run_invariance:
        outputs["invariance"] = str(Path(args.output_dir) / f"invariance_{args.name}.json")
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
        outputs["unitraj_external"] = str(Path(args.output_dir) / f"unitraj_external_eval_{args.name}.json")
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
    return commands


def main():
    args = parse_args()
    _apply_quick_cpu_mode(args)
    _validate_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = _base_params(args)
    outputs = _build_outputs(output_dir, args.name)
    commands = _build_commands(args, base, outputs)

    for cmd in commands:
        run_cmd(cmd, dry_run=args.dry_run)

    manifest_path = output_dir / f"suite_manifest_{args.name}.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "local_data": args.local_data,
                "output_dir": str(output_dir),
                "outputs": outputs,
                "commands": commands,
                "quick_cpu_smoke": bool(args.quick_cpu_smoke),
                "dry_run": bool(args.dry_run),
            },
            f,
            indent=2,
        )
    print(f"saved suite manifest: {manifest_path}")


if __name__ == "__main__":
    main()
