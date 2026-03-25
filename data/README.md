# Data

Use the subdirectories in this folder consistently:

- `raw/`: immutable source datasets captured as received.
- `external/`: third-party files that are not generated in this repository.
- `interim/`: temporary outputs from cleaning and joining steps.
- `processed/`: final model-ready datasets.

Large datasets should stay out of Git unless they are intentionally small and versioned.

Tracked sample assets in this repo include `worldtrace_sample.pkl`,
`worldtrace_sample_nyc.pkl`, `poi_mobility_sample.pkl`, and `osm_context.json`.

## Repository data catalog

For an always-human-readable map of all currently tracked data assets and how each
asset is used in evaluation and figure generation, see:

- `data/DATA_CATALOG.md`

This catalog is the source of truth for:

- which files are raw vs processed,
- which assets are lightweight and intentionally versioned,
- which large/raw files are local-only and should remain outside normal commits.
