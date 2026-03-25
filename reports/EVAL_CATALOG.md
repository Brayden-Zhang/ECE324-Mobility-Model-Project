# Evaluation Catalog

This file organizes completed evaluations and paper-facing outputs in `reports/`.
It is designed to make figure/table provenance obvious for new contributors.

## Core evaluation artifacts

- `foundation/foundation_downstream_report.json`: machine-readable aggregate of downstream tasks.
- `foundation/foundation_downstream_report.csv`: flat table export for quick filtering/sheets use.
- `foundation/foundation_downstream_report.md`: markdown summary used for narrative inspection.
- `baselines/unitraj_hmt_external_comparison_20260324.csv`: HMT vs external UniTraj meter-scale comparison.

## Paper artifact snapshots

- `paper/paper_metrics.json`: consolidated metrics consumed by figure/table generators.
- `paper/paper_length_table.tex`: length sensitivity table included by `docs/paper.tex`.
- `paper/paper_robustness_table.tex`: robustness diagnostics table for the main results.
- `paper/paper_unitraj_main.tex`: primary HMT vs UniTraj table.
- `paper/paper_unitraj_clean.tex`: appendix clean-metric UniTraj-style comparison table.
- `paper/paper_macro_table.tex`: macro mobility metrics table.
- `paper/paper_compute_table.tex`: scheduler resource budget table.

## Figure/table provenance map

- `docs/figures/results_length_lsg.pdf` <- `reports/paper/paper_metrics.json` (`length`)
- `docs/figures/results_robustness_suite.pdf` <- `reports/paper/paper_metrics.json` (`invariance`, `reverse_order`, `change_detection`)
- `docs/figures/results_meter_scale.pdf` <- `reports/paper/paper_metrics.json` (`unitraj_clean`)
- `docs/paper.tex` table includes:
  - `../reports/paper/paper_length_table.tex`
  - `../reports/paper/paper_robustness_table.tex`
  - `../reports/paper/paper_unitraj_main.tex`
  - `../reports/paper/paper_unitraj_clean.tex`
  - `../reports/paper/paper_macro_table.tex`
  - `../reports/paper/paper_compute_table.tex`

## Regeneration commands

```bash
PYTHONPATH=src python -m route_rangers.cli.generate_paper_artifacts
PYTHONPATH=src python -m route_rangers.visualization.plot_results
cd docs && latexmk -pdf -interaction=nonstopmode paper.tex
```

Or run the combined target:

```bash
make paper
```

## Update policy

When adding a new evaluation:

1. Add output artifact(s) under `reports/` with stable names.
2. Add an entry here describing metric scope and intended use.
3. If used by the paper, add/refresh provenance links in this catalog.
