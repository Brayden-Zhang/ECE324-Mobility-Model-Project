"""Visualization entry point for exported figures."""

from route_rangers.config import FIGURES_DIR, ensure_project_directories


def main() -> None:
    ensure_project_directories()
    print(f"Implement report plots here and export figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
