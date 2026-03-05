
## 🚀 TrajectoryFM-HMT (Hierarchical Mobility Tokens + Flow Matching)

This repo  includes an experimental TrajectoryFM implementation that learns hierarchical mobility tokens (HMT) and trains a flow-matching decoder for probabilistic trajectory generation. TrajectoryFM-HMT is a separate hierarchical foundation model


### Key additions
- **HMT tokenizer** (`utils/hmt.py`): hierarchical discrete tokens via H3 indexing or learned VQ codebooks.
- **Hierarchical encoder + region tokens** (`utils/hmt_hierarchy.py`): explicit mid/coarse region tokens with aggregation from step-level features.
- **Hierarchical backbone** (`utils/hmt_backbone.py`): masked transformer with region-aware attention (not UniTraj).
- **Trajectory graph encoder** (`utils/hmt_backbone.py` + `utils/hmt_model.py`): sparse graph attention over temporal, spatial kNN, and same-region edges.
- **Flow-matching head** (`utils/flow.py`): rectified-flow velocity field conditioned on HMT step tokens.
- **HF dataset loader** (`utils/hmt_dataset.py`): streaming WorldTrace support.
- **Training entrypoint** (`train_hmt.py`): multi-objective training (step + region MLM, flow matching, destination classification, and micro↔meso flow consistency).

### Quick start (streaming from HuggingFace)
```bash
python train_hmt.py --data_mode hf_zip --worldtrace_file Trajectory.zip --take 10000 --batch_size 64 --max_len 200 --tokenizer h3 --hash_tokens
```

### Mini smoke-train on SLURM (recommended first run)
Runs a small end-to-end job (train + optional benchmarks) to validate the pipeline before launching a long foundation run:
```bash
sbatch scripts/slurm_train_hmt_mini.sh
```
Hierarchy knobs you can tune: `--region_mask_ratio`, `--region_weight`, `--consistency_weight`.
Graph knobs you can tune: `--use_graph`, `--graph_layers`, `--graph_knn`, `--graph_temporal_window`.
Trip-context features (enabled by default): `--use_trip_features` / `--no_trip_features`.
Robustness knobs you can tune:
- `--mask_ratio_min`, `--mask_curriculum_steps` (masking curriculum)
- `--span_mask_prob`, `--span_lambda` (span masking for contiguous trajectory dropout)
- `--region_mask_ratio_min`, `--region_mask_curriculum_steps` (region-level curriculum masking)
- `--length_adaptive_masking`, `--length_mask_alpha` (length-adaptive token masking for short/long trajectory balance)
- `--use_length_adapter` (length-conditioned hidden gating before token heads)
- `--coord_noise_std` (GPS jitter augmentation)
- `--step_attention_window` (local sparse step-step attention, set `>0` to reduce quadratic attention cost)
If your environment requires system Arrow (e.g., Compute Canada), load it before running:
```bash
module load arrow/21.0.0
```
To avoid filling home-cache on large downloads, set:
```bash
export HF_HOME=$PWD/.hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
```

### Full WorldTrace training (recommended)
`--take` now defaults to full split when omitted (or set to `0`).
```bash
python train_hmt.py \
  --data_mode hf_zip \
  --worldtrace_file Trajectory.zip \
  --shuffle_buffer 1000 \
  --batch_size 64 \
  --max_len 200 \
  --max_steps 200000 \
  --eval_interval 2000 \
  --save_interval 5000 \
  --lr 2e-4 \
  --weight_decay 0.05 \
  --warmup_steps 5000 \
  --accum_steps 2 \
  --tokenizer h3 \
  --hash_tokens \
  --use_graph \
  --graph_layers 2 \
  --graph_knn 8 \
  --graph_temporal_window 2
```

### Optional: download a local parquet for offline training
```bash
python scripts/download_worldtrace.py --split train --output data/worldtrace --max_samples 500000
python train_hmt.py --local_data data/worldtrace/train.parquet --batch_size 64
```

### Reproducible local experiments (train/val/test + saved JSON metrics)
`train_hmt.py` now supports deterministic local splits (`--val_ratio`, `--test_ratio`), richer eval metrics (`token_acc_l0/l1/l2`, `dest_top1/top5`), and result export via `--results_path`.

Baseline (VQ tokenizer, no graph):
```bash
python train_hmt.py \
  --data_mode local \
  --local_data data/worldtrace_sample.pkl \
  --tokenizer vq \
  --vocab_l0 8192 --vocab_l1 2048 --vocab_l2 512 \
  --batch_size 32 --max_len 64 --max_steps 80 \
  --embed_dim 128 --depth 3 --heads 4 \
  --no_amp --cpu_threads 8 \
  --results_path cache/results_vq_base80.json
```

Graph ablation:
```bash
python train_hmt.py \
  --data_mode local \
  --local_data data/worldtrace_sample.pkl \
  --tokenizer vq \
  --vocab_l0 8192 --vocab_l1 2048 --vocab_l2 512 \
  --batch_size 24 --max_len 64 --max_steps 60 \
  --embed_dim 128 --depth 3 --heads 4 \
  --use_graph --graph_layers 1 --graph_knn 6 --graph_temporal_window 2 \
  --no_amp --cpu_threads 8 \
  --results_path cache/results_vq_graph.json
```

### Benchmark suite (next-location, destination, reconstruction, temporal OOD)
Run locally from a checkpoint:
```bash
python scripts/run_benchmarks.py \
  --checkpoint checkpoints/hmt_step_5000.pt \
  --local_data data/worldtrace_sample.pkl \
  --split_mode both \
  --output cache/benchmark_results.json
```
If you need a faster CPU run, disable graph edges during benchmarking:
```bash
python scripts/run_benchmarks.py \
  --checkpoint checkpoints/hmt_step_5000.pt \
  --local_data data/worldtrace_sample.pkl \
  --disable_graph \
  --output cache/benchmark_results_fast.json
```

### One-command downstream benchmark suite
Run downstream probes plus core benchmark and length-sensitivity in one command:
```bash
python scripts/run_foundation_eval_suite.py \
  --checkpoint checkpoints/hmt_step_5000.pt \
  --local_data data/worldtrace_sample.pkl \
  --output cache/foundation_suite/summary.json
```

### Optional: build OSM context features
```bash
python scripts/build_osm_context.py --north 40.9 --south 40.4 --east -73.7 --west -74.3 --res 7 --output data/osm_context.json
python train_hmt.py --osm_context data/osm_context.json --osm_context_dim 16
```
For SLURM, `scripts/slurm_train_hmt.sh` can auto-build this file if you export:
```bash
export BUILD_OSM_CONTEXT=1
export OSM_NORTH=40.9 OSM_SOUTH=40.4 OSM_EAST=-73.7 OSM_WEST=-74.3
sbatch scripts/slurm_train_hmt.sh
```
Run benchmarks on SLURM:
```bash
export CKPT_PATH=checkpoints/hmt_step_5000.pt
export LOCAL_DATA=data/worldtrace_sample.pkl
sbatch scripts/slurm_benchmark_hmt.sh
```

### UniTraj-style evaluation (recovery/prediction MAE/RMSE)
See `docs/foundation_evals.md` for the full suite. Minimal run:
```bash
python scripts/run_unitraj_eval.py \
  --checkpoint checkpoints/hmt_step_15000.pt \
  --local_data data/worldtrace_sample.pkl \
  --split_mode both \
  --task both \
  --output cache/unitraj_eval_step15000.json
```
