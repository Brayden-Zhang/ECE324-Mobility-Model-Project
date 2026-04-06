"""Compatibility entrypoint for training the bundled UniTraj baseline."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    src_dir = Path(__file__).resolve().parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from route_rangers.cli.unitraj.train_unitraj import main as train_main

    train_main()


if __name__ == "__main__":
    main()
