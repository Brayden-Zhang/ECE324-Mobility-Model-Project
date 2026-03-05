#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def parse_args():
    parser = argparse.ArgumentParser(description="Run a unified downstream benchmark suite for trajectory foundation checkpoints.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--local_data", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="cache/foundation_suite")
    parser.add_argument("--output", type=str, default="cache/foundation_suite/summary.json")
    return parser.parse_args()


def run_task(script_name: str, args: argparse.Namespace, output_path: Path) -> Dict:
    cmd = [
        sys.executable,
        str(SCRIPTS / script_name),
        "--checkpoint",
        args.checkpoint,
        "--local_data",
        args.local_data,
        "--max_len",
        str(args.max_len),
        "--batch_size",
        str(args.batch_size),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--sample_limit",
        str(args.sample_limit),
        "--output",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    with open(output_path, "r") as f:
        return json.load(f)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = {
        "benchmarks": "run_benchmarks.py",
        "length_sensitivity": "run_length_sensitivity.py",
        "travel_time": "run_travel_time_estimation.py",
        "trip_classification": "run_trip_classification.py",
        "anomaly_detection": "run_anomaly_detection.py",
        "similarity_retrieval": "run_similarity_retrieval.py",
    }

    summary = {
        "checkpoint": args.checkpoint,
        "dataset": args.local_data,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "tasks": {},
    }

    for name, script in tasks.items():
        task_output = output_dir / f"{name}.json"
        summary["tasks"][name] = run_task(script, args, task_output)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved foundation suite summary: {out_path}")


if __name__ == "__main__":
    main()
