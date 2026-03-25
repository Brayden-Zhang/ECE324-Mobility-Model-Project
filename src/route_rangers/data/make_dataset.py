"""Utilities for staging source data into the repository data layout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from route_rangers.config import RAW_DATA_DIR, ensure_project_directories


def stage_raw_file(source: Path) -> Path:
    """Copy a source file into the raw data directory."""

    ensure_project_directories()
    destination = RAW_DATA_DIR / source.name
    shutil.copy2(source, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage a raw dataset into data/raw.")
    parser.add_argument(
        "--source",
        type=Path,
        help="Path to a local file that should be copied into data/raw.",
    )
    args = parser.parse_args()

    ensure_project_directories()
    if args.source is None:
        print(f"Raw data directory is ready at {RAW_DATA_DIR}")
        return

    staged_path = stage_raw_file(args.source)
    print(f"Staged raw dataset at {staged_path}")


if __name__ == "__main__":
    main()
