"""Entry points for feature generation."""

from route_rangers.config import PROCESSED_DATA_DIR, ensure_project_directories


def main() -> None:
    ensure_project_directories()
    print(
        "Implement feature engineering here and write model-ready outputs to "
        f"{PROCESSED_DATA_DIR}"
    )


if __name__ == "__main__":
    main()
