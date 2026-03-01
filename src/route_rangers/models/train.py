"""Training entry point for the mobility model."""

from route_rangers.config import MODELS_DIR, ensure_project_directories


def main() -> None:
    ensure_project_directories()
    print(f"Implement model training here and save artifacts to {MODELS_DIR}")


if __name__ == "__main__":
    main()
