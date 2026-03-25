"""Shared paths for paper artifacts, reports, data, and checkpoints."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"
PAPER_FIGURES_DIR = DOCS_DIR / "figures"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def ensure_project_directories() -> list[Path]:
    """Create the main project directories if they do not already exist."""

    directories = [
        RAW_DATA_DIR,
        EXTERNAL_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        CHECKPOINTS_DIR,
        REPORTS_DIR,
        PAPER_FIGURES_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return directories
