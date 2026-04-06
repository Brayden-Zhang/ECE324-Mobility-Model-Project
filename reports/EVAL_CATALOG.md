# Evaluation Catalog

This file maps the checked-in evaluation artifacts in `reports/` to the summary
tables and figures that can be regenerated from this repository snapshot.

## Core evaluation artifacts

- `foundation/foundation_downstream_report.json`: machine-readable aggregate of
  downstream tasks.
- `foundation/foundation_downstream_report.csv`: flat export for spreadsheet use.
- `foundation/foundation_downstream_report.md`: compact aggregate report.
- `foundation/foundation_eval_summary.md`: maintained sectioned summary used by
  the paper-artifact generator.
- `baselines/unitraj_hmt_external_comparison_20260324.csv`: HMT vs external
  UniTraj meter-scale comparison.

## Regeneration commands

```bash
PYTHONPATH=src python -m route_rangers.cli.generate_paper_artifacts
PYTHONPATH=src python -m route_rangers.visualization.plot_results
```

Or use:

```bash
make verify-paper
```

## Update policy

When adding a new checked-in evaluation:

1. Write stable outputs under `reports/`.
2. Add the artifact here with a short description.
3. Update `reports/foundation/foundation_eval_summary.md` if the result feeds the
   paper-style artifact pipeline.
