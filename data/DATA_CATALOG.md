# Data Catalog

This file organizes the raw and processed data assets used by this repository.
Keep it updated whenever datasets are added, removed, or renamed.

## Directory conventions

- `data/raw/`: immutable source drops and one-time imports.
- `data/external/`: third-party side assets not generated here.
- `data/interim/`: temporary or partially processed joins.
- `data/processed/`: finalized model-ready assets.

## Versioned lightweight assets (tracked in git)

- `data/worldtrace_sample.pkl`: compact WorldTrace-derived sample used by local evals.
- `data/worldtrace_sample_nyc.pkl`: NYC-focused WorldTrace sample for transfer checks.
- `data/poi_mobility_sample.pkl`: sample POI trajectory dataset for next-POI probes.
- `data/osm_context.json`: compact OSM-derived context used by context-aware modeling.
- `data/h3_vocab_worldtrace_full.json`: H3 token vocabulary snapshot for stable token ids.
- `data/hdx/movement-distribution/processed/movement_distribution_12m_monthly.npz`: processed macro movement tensor.
- `data/hdx/movement-distribution/processed/movement_distribution_12m_monthly.meta.json`: metadata for the processed macro tensor.
- `data/hdx/commuting-zones/data-for-good-at-meta-commuting-zones-march-2023.csv`: commuting-zone labels used by zone probes.

## Local large/raw assets (normally not tracked)

- `data/worldtrace_full/Trajectory.zip`: full WorldTrace archive for large-scale runs.
- `data/hdx/movement-distribution/*.zip`: raw monthly movement-distribution archives.
- `data/raw/`: any new raw imports should be staged here first and documented before processing.

## Data-to-evaluation mapping

- UniTraj-style evaluation: `data/worldtrace_sample.pkl`
- Transfer suite: `data/worldtrace_sample.pkl`, `data/worldtrace_sample_nyc.pkl`
- Next POI evaluation: `data/poi_mobility_sample.pkl`
- Macro distribution eval: `data/hdx/movement-distribution/processed/movement_distribution_12m_monthly.npz`
- Commuting-zone probe: `data/hdx/commuting-zones/data-for-good-at-meta-commuting-zones-march-2023.csv`

## Reproducibility note

For paper-facing output provenance (which report files generate which tables/figures),
see `reports/EVAL_CATALOG.md`.
