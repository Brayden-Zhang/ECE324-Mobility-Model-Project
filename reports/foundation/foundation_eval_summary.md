# Foundation Evaluation Summary

This file is the maintained, human-readable summary that backs the paper-style
artifact generation workflow in `route_rangers.reporting.paper_artifacts`.

It intentionally lives under `reports/` because the current reproducibility
snapshot tracks evaluated outputs and report-ready summaries, not the older
`docs/` paper source tree.

<!-- RESULTS:BEGIN -->
### Benchmarks (random split, test)
| run | recon@l1 | next@top1 | dest@top1 |
|---|---|---|---|
| 2517827 | 0.828 | 0.767 | 0.317 |

### Length sensitivity
latest: `length_sensitivity_2517829.json`
bins: [160, 601] (quantile)

| run | recon@l1 short | recon@l1 long | gap | dest@top1 short | dest@top1 long | gap | n |
|---|---|---|---|---|---|---|---|
| 2517829 | 0.965 | 0.828 | -0.137 | 0.776 | 0.640 | -0.137 | 488100 |

### Invariance suite
latest: `invariance_suite_2517830.json`
- prefix dest_top1: 0.25:0.597, 0.50:0.633, 0.75:0.669, 1.00:0.718
- time-shift dest_top1: 0:0.718, 43200:0.716, 86400:0.718
- downsample dest_top1: 0.50:0.663, 0.25:0.625

### Embedding retrieval
latest: `embedding_retrieval_2517831.json`
- top1: 0.5120000243186951
- top5: 0.6259999871253967

### Reverse-order stress
latest: `reverse_order_2517833.json`
- original: {'top1': 0.718, 'top5': 0.9, 'nll': 1.0126158671379089}
- reversed: {'top1': 0.575, 'top5': 0.6025, 'nll': 6.752220626831055}
- delta: {'top1': -0.14300000000000002, 'top5': -0.2975, 'nll': 5.739604759693146}

### Change detection
latest: `change_detection_2517834.json`
- pos_mean_dist: 0.03529779613018036
- neg_mean_dist: 0.48614659905433655
- auc: 0.957029

### Macro distribution head
latest: `macro_eval_2300197.json`
- macro_kl: 0.4930346369709155
- macro_js: 0.3149266023822457
- macro_l1: 0.21622090613321798
- macro_top1: 0.4099692685822438
- n: 488100.0

### UniTraj-compatible baseline
latest: `unitraj_hmt_external_comparison_20260324.csv`
- recovery: mae_m=17.16, rmse_m=47.15, n=1200
- prediction: mae_m=77.92, rmse_m=159.92, n=1200
<!-- RESULTS:END -->
