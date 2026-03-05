# Foundation Evaluation Plan (HMT vs UniTraj-style Baselines)

This document is the canonical experiment map for TrajectoryFM-HMT. It is organized to make runs reproducible, comparable to UniTraj-style evaluations, and explicit about long-range/length-sensitive behavior.

## 1) Datasets and Splits

### Primary pretraining/eval data
- **WorldTrace** (default): loaded via HuggingFace (`--data_mode hf_zip`) or local pickle/parquet (`--local_data`).
- Expected trajectory format in local files: list/array of points with time fields (`trajectory|traj|points`, `time|times|timestamp`).

### Additional transfer/robustness data (optional)
- Transfer suite uses multiple local datasets via `scripts/run_transfer_suite.py --datasets ...`.
- Macro-flow and commuting-zone probes rely on optional HDX-derived files:
  - `data/hdx/movement-distribution/processed/movement_distribution_12m_monthly.npz`
  - `data/hdx/commuting-zones/...csv`

### Split policy
- Training supports deterministic local `train/val/test` splits.
- Evaluation scripts support **random** and **temporal** split modes (`--split_mode both` or `random|temporal`) for in-distribution and temporal-OOD views.

## 2) Core Evaluation Bundle (recommended minimum)

Run all core evaluations locally with one command:

```bash
python scripts/run_foundation_suite.py \
  --checkpoint checkpoints/hmt_step_15000.pt \
  --local_data data/worldtrace_sample.pkl \
  --split_mode both \
  --name hmt_main
```

This runs and stores:
- `run_benchmarks.py` (reconstruction + next/destination probes)
- `run_unitraj_eval.py` (UniTraj-style recovery/prediction MAE/RMSE)
- robust UniTraj-style eval (`coord_noise_std_m=30`, `input_drop_ratio=0.2`)
- `run_length_sensitivity.py` (short/medium/long bins + long-short gap)

Outputs are written under `cache/foundation_suite/` with a JSON manifest.

## 3) UniTraj-Comparable Protocol

### HMT evaluated with UniTraj-style metrics
```bash
python scripts/run_unitraj_eval.py \
  --checkpoint checkpoints/hmt_step_15000.pt \
  --local_data data/worldtrace_sample.pkl \
  --split_mode both \
  --task both \
  --exclude_unknown \
  --output cache/unitraj_eval_hmt.json
```

### External UniTraj baseline on same metric family
```bash
python scripts/run_unitraj_external_eval.py \
  --data_path data/worldtrace_sample.pkl \
  --checkpoint external/unitraj/UniTraj/models/best_model.pt \
  --task both \
  --output cache/unitraj_external_eval.json
```

### Side-by-side comparison CSV
```bash
python scripts/compare_unitraj_results.py \
  --hmt cache/unitraj_eval_hmt.json \
  --unitraj cache/unitraj_external_eval.json \
  --mode random --split test
```

## 4) Ablations (clear matrix)

Use `scripts/slurm_train_hmt_ablate.sh` with these toggles:

| Ablation | Key env toggle(s) | Hypothesis |
|---|---|---|
| No graph | `USE_GRAPH=0` | Degrades long-range spatial relation handling |
| No trip features | `NO_TRIP_FEATURES=1` | Hurts context-aware destination prediction |
| Space-time encoder on/off | `SPACE_TIME_ENCODER=1/0` | Tests explicit periodic encoding value |
| Length-weighted loss | `LENGTH_WEIGHTED_LOSS=1` | Should improve long-trajectory quality |
| Flow objective weight | `FLOW_WEIGHT=<value>` | Controls generative trajectory fidelity |
| Region hierarchy weights | `REGION_WEIGHT`, `CONSISTENCY_WEIGHT` | Tests hierarchical consistency benefit |

For each ablation checkpoint, run:
1. `run_benchmarks.py`
2. `run_unitraj_eval.py`
3. `run_length_sensitivity.py`

Then compare long-short gaps (`length_sensitivity_gap`) and MAE/RMSE deltas.

## 5) Long-Range / Length-Sensitive Evidence

Primary evidence script:

```bash
python scripts/run_length_sensitivity.py \
  --checkpoint checkpoints/hmt_step_15000.pt \
  --local_data data/worldtrace_sample.pkl \
  --dest_mask_last_k 1 \
  --output cache/length_sensitivity_hmt.json
```

Key reported values:
- `metrics.short|medium|long.recon_acc_l1`
- `metrics.short|medium|long.dest_top1`
- `length_sensitivity_gap.recon_acc_l1`
- `length_sensitivity_gap.dest_top1`

Claim support should rely on:
- Positive long-short destination/reconstruction gaps vs baselines
- Smaller degradation under temporal split and robust-noise/drop settings
- Better recovery/prediction MAE/RMSE at larger `max_len`

## 6) Full SLURM pipeline (paper-scale)

For cluster runs, use:
```bash
bash scripts/slurm_neurips_master.sh
```

This orchestrates training, suite evaluation, downstream tasks, ablations, and UniTraj comparisons.

## Latest Results
<!-- RESULTS:BEGIN -->
(pending)
<!-- RESULTS:END -->
