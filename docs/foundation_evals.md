# Foundation-Model Evaluation Suite

This document collects the evaluation scripts that align with UniTraj-style metrics and deeper foundation-model evidence.

Evaluation artifact index:

- Data asset organization: `data/DATA_CATALOG.md`
- Report/output organization: `reports/EVAL_CATALOG.md`

## Update Results Blocks

After running any eval that writes JSON to `cache/`, refresh the summary tables:

```bash
PYTHONPATH=src python -m route_rangers.cli.collect_results
```

## Unified downstream report package

Generate a single publishable downstream report table (Markdown/CSV/JSON) with
baseline deltas from all available cache artifacts:

```bash
PYTHONPATH=src python -m route_rangers.cli.generate_foundation_report \
  --cache_dir cache \
  --output_dir reports
```

Outputs:
- `reports/foundation/foundation_downstream_report.md`
- `reports/foundation/foundation_downstream_report.csv`
- `reports/foundation/foundation_downstream_report.json`

## UniTraj-style recovery and prediction

Trajectory recovery masks random points; trajectory prediction masks the last `K` points. Metrics are MAE/RMSE in meters.

```bash
PYTHONPATH=src python -m route_rangers.cli.run_unitraj_eval \
  --checkpoint checkpoints/hmt_step_15000.pt \
  --local_data data/samples/worldtrace_sample.pkl \
  --split_mode both \
  --task both \
  --output cache/unitraj_eval_step15000.json
```

Summarize:
```bash
PYTHONPATH=src python -m route_rangers.cli.summarize_unitraj_eval cache/unitraj_eval_step15000.json
```

## Transfer across datasets (zero-shot)

Run the same evaluation on multiple datasets without fine-tuning.

```bash
PYTHONPATH=src python -m route_rangers.cli.run_transfer_suite \
  --checkpoint checkpoints/hmt_step_15000.pt \
  --datasets data/samples/worldtrace_sample.pkl data/other_dataset.pkl \
  --output cache/unitraj_transfer_suite.json
```

## Data-efficiency sweep

Build centroid mappings from a fraction of the training split to simulate data efficiency.

```bash
PYTHONPATH=src python -m route_rangers.cli.run_data_efficiency \
  --checkpoint checkpoints/hmt_step_15000.pt \
  --local_data data/samples/worldtrace_sample.pkl \
  --fractions 0.05 0.1 0.2 0.5 1.0 \
  --output cache/unitraj_data_efficiency.json
```

## Robustness stress tests

Use input noise and additional dropouts to evaluate robustness.

```bash
PYTHONPATH=src python -m route_rangers.cli.run_unitraj_eval \
  --checkpoint checkpoints/hmt_step_15000.pt \
  --local_data data/samples/worldtrace_sample.pkl \
  --coord_noise_std_m 30 \
  --input_drop_ratio 0.2 \
  --output cache/unitraj_eval_robust.json
```

## Length Sensitivity (short/medium/long)

Evaluate how metrics differ across short/medium/long trajectory-length buckets. By default, destination metrics mask the final point to avoid trivial copying (`--dest_mask_last_k 1`).

```bash
PYTHONPATH=src python -m route_rangers.cli.run_length_sensitivity \
  --checkpoint checkpoints/hmt_ablate_lenweight_step_15000.pt \
  --local_data data/samples/worldtrace_sample.pkl \
  --max_len 200 \
  --sample_limit 2000 \
  --dest_mask_last_k 1 \
  --output cache/length_sensitivity_lenweight_step15000.json
```

CPU smoke run (faster, lower-fidelity):
```bash
PYTHONPATH=src python -m route_rangers.cli.run_length_sensitivity \
  --checkpoint checkpoints/hmt_ablate_lenweight_step_15000.pt \
  --local_data data/samples/worldtrace_sample.pkl \
  --max_len 64 \
  --sample_limit 100 \
  --dest_mask_last_k 1 \
  --output cache/length_sensitivity_lenweight_step15000_smoke.json
```

## Macro distribution head (Movement Distribution)

Evaluate the macro head on monthly distance-category distributions.

```bash
PYTHONPATH=src python -m route_rangers.cli.run_macro_eval \
  --checkpoint checkpoints/hmt_stage2_final_step_100000.pt \
  --macro_data data/processed/macro/movement_distribution_12m_monthly.npz \
  --output cache/macro_eval.json
```

## Commuting zone destination probe

Predict the destination commuting zone from pooled step embeddings.

```bash
PYTHONPATH=src python -m route_rangers.cli.run_commuting_zone_probe \
  --checkpoint checkpoints/hmt_stage2_final_step_100000.pt \
  --local_data data/samples/worldtrace_sample.pkl \
  --cz_csv data/raw/hdx/commuting-zones/data-for-good-at-meta-commuting-zones-march-2023.csv \
  --output cache/commuting_zone_probe.json
```

## Next POI prediction (MoveGPT-style)

Evaluate next-POI ranking metrics (Acc@1/5/10, Recall@K, NDCG@10, MRR), plus optional user identification.

```bash
PYTHONPATH=src python -m route_rangers.cli.run_next_poi_eval \
  --checkpoint checkpoints/hmt_step_15000.pt \
  --local_data data/samples/poi_mobility_sample.pkl \
  --max_len 200 \
  --split_mode temporal \
  --output cache/next_poi_eval.json
```

## Cross-city transfer (leave-one-city-out)

Run zero-shot transfer across cities by holding out one city at a time.

```bash
PYTHONPATH=src python -m route_rangers.cli.run_cross_city_transfer \
  --checkpoint checkpoints/hmt_step_15000.pt \
  --local_data data/samples/poi_mobility_sample.pkl \
  --min_city_records 100 \
  --output cache/cross_city_transfer.json
```

## Compute efficiency

Quick parameter count and throughput stats:

```bash
PYTHONPATH=src python -m route_rangers.cli.model_stats \
  --checkpoint checkpoints/hmt_step_15000.pt \
  --batch_size 16 --max_len 200
```

## UniTraj-compatible baseline

The repository bundles a UniTraj-compatible baseline implementation under
`route_rangers.baselines`. Evaluate any compatible checkpoint on the same tasks:

```bash
PYTHONPATH=src python -m route_rangers.cli.run_unitraj_external_eval \
  --data_path data/samples/worldtrace_sample.pkl \
  --checkpoint checkpoints/unitraj.pt \
  --task both
```

If you need a simpler interchange format, export WorldTrace into a CSV and adapt as needed.

```bash
PYTHONPATH=src python -m route_rangers.cli.unitraj.export_worldtrace_csv \
  --input data/samples/worldtrace_sample.pkl \
  --output data/worldtrace_sample.csv
```


## Latest Results
Evaluation dataset note (latest UniTraj-style runs): `data/samples/worldtrace_sample.pkl` (2,000 trajectories, `split_mode=all`, `max_len=200`, `mask_ratio=0.5`, `pred_steps=5`).
<!-- RESULTS:BEGIN -->
### Benchmarks (random split, test)

| run | recon@l1 | next@top1 | dest@top1 |
|---|---|---|---|
| 2108484 | 0.000 | 1.000 | 0.000 |
| step10000_newmetrics_32 | 1.000 | 0.000 | 0.000 |
| step10000_full_20260207_114420 | 1.000 | 1.000 | 1.000 |
| ablate_nograph_2239975 | 0.985 | 0.797 | 0.453 |
| ablate_notrip_2239977 | 0.925 | 0.793 | 0.213 |
| ablate_noflow_2239978 | 0.896 | 0.746 | 0.667 |
| local_context_2241225 | 0.000 | 1.000 | 1.000 |
| local_nocontext_2241275 | 0.000 | 1.000 | 1.000 |
| 2240183 | 0.863 | 0.799 | 0.193 |
| baseline_2240191 | 0.863 | 0.799 | 0.193 |
| baseline_2239996 | 0.863 | 0.799 | 0.193 |
| ablate_lenweight_2239980 | 0.854 | 0.798 | 0.220 |
| ablate_hash_2239979 | 1.000 | 1.000 | 1.000 |
| 2260434 | 0.000 | 0.800 | 0.580 |
| 2300197 | 0.863 | 0.797 | 0.487 |
| 2300206 | 0.863 | 0.797 | 0.487 |
| 2300207 | 0.863 | 0.797 | 0.487 |
| 2300208 | 0.863 | 0.797 | 0.487 |
| 2300205 | 0.863 | 0.797 | 0.487 |
| 2517827 | 0.828 | 0.767 | 0.317 |

### Length sensitivity

latest: `length_sensitivity_lenweight_step15000_maxlen64_n100_destmask1.json`
bins: [160, 601] (quantile)

| run | recon@l1 short | recon@l1 long | gap | dest@top1 short | dest@top1 long | gap | n |
|---|---|---|---|---|---|---|---|
| lenweight_step15000_maxlen64_n100_destmask1 | 0.868 | 0.752 | -0.117 | 0.697 | 0.375 | -0.322 | 100 |
| full_step15000_maxlen64_n100_destmask1 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 100 |
| nograph_step15000_maxlen64_n100_destmask1 | 0.982 | 0.919 | -0.063 | 0.727 | 0.375 | -0.352 | 100 |

### Invariance suite

latest: `invariance_suite_2517830.json`
- prefix dest_top1: 0.25:0.597, 0.50:0.633, 0.75:0.669, 1.00:0.718
- time-shift dest_top1: 0:0.718, 43200:0.716, 86400:0.718
- downsample dest_top1: 0.50:0.663, 0.25:0.625

### Embedding retrieval

latest: `embedding_retrieval_2517831.json`
- top1: 0.5120000243186951
- top5: 0.6259999871253967
- samples: 1000

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

### UniTraj-style eval (centroid)

latest: `unitraj_eval_regression_best_20260324_170752.json`
- split: all
- recovery: mae_m=5.4, rmse_m=10.5, n=166673
- prediction: mae_m=15.8, rmse_m=35.5, n=10000

### UniTraj-style eval (centroid, robust)

latest: `unitraj_eval_robust_2517827.json`
- split: random
- recovery: mae_m=33.4, rmse_m=47.3, n=24725
- prediction: mae_m=180.0, rmse_m=222.8, n=1500

### UniTraj-style eval (regression)

**clean** (`unitraj_eval_regression_best_20260324_170752.json`)
- recovery: mae_m=5.4, rmse_m=10.5, n=166673
- prediction: mae_m=15.8, rmse_m=35.5, n=10000
**robust** (`unitraj_eval_regression_robust_best_20260324_170752.json`)
- recovery: mae_m=32.9, rmse_m=44.7, n=166673
- prediction: mae_m=185.3, rmse_m=230.5, n=10000

### Data efficiency

latest: `unitraj_data_efficiency_2300197.json`
- recovery mae_m: 0.05:9050630.8, 0.1:8777055.3, 0.2:8588700.9, 0.5:6253402.1, 1.0:4951738.4
- prediction mae_m: 0.05:8988683.7, 0.1:8830886.3, 0.2:8763472.4, 0.5:6180781.2, 1.0:5435990.0

### Transfer suite

latest: `unitraj_transfer_suite_2261966.json`
- worldtrace_sample.pkl, split=random, recovery_mae_m=8740435.3, pred_mae_m=9169197.7
- worldtrace_sample_nyc.pkl, split=random, recovery_mae_m=6876262.9, pred_mae_m=5289742.2

### Macro distribution head

latest: `macro_eval_2300197.json`
- macro_kl: 0.4930346369709155
- macro_js: 0.3149266023822457
- macro_l1: 0.21622090613321798
- macro_top1: 0.4099692685822438
- n: 488100

### Commuting zone probe

No commuting-zone results yet.

### External UniTraj baseline

latest: `unitraj_external_eval_best_20260324_170752.json`
- recovery: mae_m=17.2, rmse_m=47.2, n=132536
- prediction: mae_m=77.9, rmse_m=159.9, n=10000

### Travel Time Estimation

latest: `travel_time_2517828.json`
| prefix_ratio | MAE (s) | RMSE (s) | MAPE | R² |
|---|---|---|---|---|
| 0.25 | 173.4 | 211.3 | 0.661 | 0.000 |
| 0.5 | 173.6 | 205.0 | 0.666 | 0.000 |
| 0.75 | 176.3 | 208.0 | 0.677 | 0.000 |
| 1.0 | 177.1 | 206.2 | 0.657 | 0.000 |

### Anomaly Detection

latest: `anomaly_detection_2517828.json`
| anomaly_type | AUROC | PR-AUC |
|---|---|---|
| noise | 0.441 | 0.457 |
| reverse | 0.882 | 0.881 |
| swap | 0.458 | 0.460 |
| detour | 0.481 | 0.485 |
| combined | 0.579 | 0.610 |

### Trip Classification

latest: `trip_classification_2517828.json`
| task | top1 | top5 | n_classes |
|---|---|---|---|
| speed_mode | 0.417 | 1.000 | 4 |
| duration_bucket | 0.710 | 1.000 | 4 |
| distance_bucket | 0.587 | 1.000 | 4 |

### Similarity Retrieval

latest: `similarity_retrieval_2517828.json`
- **geographic_consistency**: origin_mean_dist_km=1377.800, dest_mean_dist_km=1377.729, spatial_mean_dist_km=1377.436
- **self_retrieval**: self_retrieval_top1=0.795, self_retrieval_top10=0.878

### Next POI Prediction

No next-POI results yet.

### Cross-City Transfer

No cross-city transfer results yet.
<!-- RESULTS:END -->
