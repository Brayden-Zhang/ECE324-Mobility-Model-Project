# Data

Use the subdirectories in this folder consistently:

- `raw/`: immutable source datasets captured as received.
- `external/`: third-party files that are not generated in this repository.
- `interim/`: temporary outputs from cleaning and joining steps.
- `processed/`: final model-ready datasets.

Large datasets should stay out of Git unless they are intentionally small and versioned.
