# Baselines Beyond UniTraj (Mobility/Trajectory Foundation Models)

This matrix captures the comparison landscape around the paper without requiring
vendored third-party repositories in this repo.

Legend:
- `ready-now`: can run immediately with current repo + data/compute assumptions.
- `ready-with-adapter`: runnable after data conversion or lightweight glue code.
- `blocked-external`: cannot run yet because code/weights are not practically available.

## Runnable-Today Comparison Matrix

| baseline | implementation path | ckpt availability | mapping to our evals | runnable status today | concrete blocker / note |
|---|---|---|---|---|---|
| UniTraj-compatible baseline | `src/route_rangers/baselines` | user-supplied or locally trained | UniTraj-style recovery/prediction (clean + robust) | ready-with-checkpoint | train via `python -m route_rangers.cli.unitraj.train_unitraj` or evaluate a provided checkpoint |
| TrajGPT | github.com/ktxlh/TrajGPT | no public pretrained listed | next-location, infilling (after token/data alignment) | ready-with-adapter | needs conversion from WorldTrace/H3 format to TrajGPT input schema |
| MobilityGPT | github.com/ammarhydr/MobilityGPT | no public pretrained listed | next-location/generation probes | ready-with-adapter | expects map-matched road-network-style inputs |
| TrajFM | anonymous.4open.science/r/TrajFM-30E4 | unknown | infilling, next-location, transfer | blocked-external | weights/checkpoint availability unclear |
| GPS-MTM | repo link not confirmed | unknown | infilling, next-stop, anomaly | blocked-external | reproducible code entrypoint not pinned yet |
| MoveGPT | paper reference only | unknown | paper-specific multi-task suite | blocked-external | no stable public implementation discovered |
| MoveFM-R | anonymous.4open.science/r/MoveFM-R-CDE7 | unknown | semantic/zero-shot transfer claims | blocked-external | needs repo audit + runnable training/eval scripts |

## Priority Expansion Order

1. Promote `ready-with-adapter` baselines first after the bundled UniTraj-compatible baseline.
2. Keep `blocked-external` rows in matrix as "not runnable today" until code + weights are verified.
3. For each promoted baseline, map into existing report tasks before adding new metrics.

## Alignment Notes

- Common metrics: accuracy@k for next-location, MAE/RMSE (or ADE/FDE when applicable), plus KL/JS/L1 for macro distributions.
- Transfer: region/city holdout transfer and task transfer (infilling vs next-location).
- Fairness rule: whenever possible, run baseline-native evaluation and also run on our local WorldTrace-formatted slice.
