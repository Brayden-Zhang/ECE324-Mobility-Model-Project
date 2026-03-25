"""Generate paper-facing tables and metric snapshots from checked-in reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from route_rangers.reporting import write_paper_artifacts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables and JSON metrics for the paper."
    )
    parser.add_argument("--output_dir", type=str, default="reports")
    parser.add_argument(
        "--report_json", type=str, default="reports/foundation_downstream_report.json"
    )
    parser.add_argument("--foundation_doc", type=str, default="docs/foundation_evals.md")
    parser.add_argument(
        "--unitraj_csv",
        type=str,
        default="reports/unitraj_hmt_external_comparison_20260324.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = write_paper_artifacts(
        Path(args.output_dir),
        report_json=Path(args.report_json),
        foundation_doc=Path(args.foundation_doc),
        unitraj_csv=Path(args.unitraj_csv),
    )
    print(f"wrote paper artifacts to {Path(args.output_dir).resolve()}")
    print(f"length_gap_dest_top1={metrics['length'].get('gap_dest_top1')}")


if __name__ == "__main__":
    main()
