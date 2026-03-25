# Reports

This directory stores lightweight checked-in result summaries that support the paper:

- `foundation_downstream_report.*`: canonical downstream summary exports.
- `foundation_downstream_report_full_table.tex`: LaTeX-ready table export.
- `foundation_downstream_report_task_summary.tex`: compact task summary table.
- `unitraj_hmt_external_comparison_20260324.csv`: meter-scale comparison snapshot.

Heavy intermediate caches belong in `cache/` and are intentionally not tracked.

## Evaluation organization

Use `reports/EVAL_CATALOG.md` as the canonical index for:

- all current evaluation artifacts,
- which metrics each artifact stores,
- where each paper table/figure draws its values from.
