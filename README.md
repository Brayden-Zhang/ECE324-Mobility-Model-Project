# TrajectoryFM-HMT



## Repository layout

- `src/route_rangers/`: canonical Python package.
- `train_hmt.py`: main HMT training entrypoint.
- `scripts/`: SLURM launchers and lightweight orchestration wrappers.
- `docs/paper.tex`: paper source.
- `docs/paper.pdf`: compiled paper snapshot.
- `docs/figures/`: figures used by the paper.
- `reports/`: checked-in summary tables and result exports referenced by the paper workflow.
- `data/`: small sample assets and dataset helpers.
- `checkpoints/`: local checkpoint drop zone. We keep documentation here, not model weights.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

Run tests:

```bash
make test
```

Run the full paper verification pipeline:

```bash
make verify-paper
```

Optional on cluster environments that require Arrow:

```bash
module load arrow/21.0.0
```

## Reproducing the paper

Generate the paper figures:

```bash
make figures
```

Build the PDF from `docs/paper.tex`:

```bash
make paper
```

`make verify-paper` runs unit tests and then rebuilds the paper artifacts, figures, and PDF in one pass.

The paper-facing figure generator is implemented in `src/route_rangers/visualization/plot_results.py`, with `scripts/generate_all_plots.py` kept as a thin convenience wrapper.

## Data

Small reproducible assets kept in the repo:

- `data/worldtrace_sample.pkl`
- `data/worldtrace_sample_nyc.pkl`
- `data/poi_mobility_sample.pkl`
- `data/osm_context.json`

Optional large local assets are intentionally not tracked, for example:

- `data/worldtrace_full/Trajectory.zip`
- HDX movement-distribution files under `data/hdx/`
- local checkpoints under `checkpoints/`

Download helper:

```bash
PYTHONPATH=src python -m route_rangers.cli.download_worldtrace \
  --split train \
  --output data/worldtrace \
  --max_samples 500000
```

## Training

Quick local run:

```bash
python train_hmt.py \
  --data_mode local \
  --local_data data/worldtrace_sample.pkl \
  --max_len 200 \
  --batch_size 32 \
  --max_steps 1000 \
  --tokenizer h3 \
  --no_hash_tokens \
  --results_path cache/train_local_results.json
```

Full local run from a WorldTrace zip:

```bash
python train_hmt.py \
  --data_mode hf_zip \
  --worldtrace_local_path data/worldtrace_full/Trajectory.zip \
  --worldtrace_file Trajectory.zip \
  --batch_size 64 \
  --max_len 200 \
  --max_steps 200000 \
  --eval_interval 2000 \
  --save_interval 5000 \
  --tokenizer h3 \
  --no_hash_tokens
```

## Evaluation

Core examples:

```bash
PYTHONPATH=src python -m route_rangers.cli.run_benchmarks \
  --checkpoint checkpoints/<your_hmt_checkpoint>.pt \
  --local_data data/worldtrace_sample.pkl \
  --split_mode both \
  --output cache/benchmark_results.json
```

```bash
PYTHONPATH=src python -m route_rangers.cli.run_length_sensitivity \
  --checkpoint checkpoints/<your_hmt_checkpoint>.pt \
  --local_data data/worldtrace_sample.pkl \
  --output cache/length_sensitivity_latest.json
```

```bash
PYTHONPATH=src python -m route_rangers.cli.run_unitraj_eval \
  --checkpoint checkpoints/<your_hmt_checkpoint>.pt \
  --local_data data/worldtrace_sample.pkl \
  --task both \
  --split_mode all \
  --output cache/unitraj_eval_regression.json
```

For the full experiment surface, see `docs/foundation_evals.md` and `docs/neurips_experiments.md`.

## Baseline compatibility

The UniTraj-compatible baseline used for comparison lives inside the package under `route_rangers.baselines`. The repo no longer vendors third-party baseline repositories or checkpoints; provide the checkpoint path you want to evaluate explicitly or place it under `checkpoints/`.

## Notes

- Import project code from `route_rangers.*`.
- Run packaged tools as `python -m route_rangers.cli.<tool>` with `PYTHONPATH=src` unless the package is installed in editable mode.
- `cache/`, local checkpoints, SLURM logs, and LaTeX build artifacts are intentionally ignored.

## Release assets for checkpoints

Large model checkpoints are intentionally not committed to git history and are not required to build or test this repository.

When publishing a checkpoint for reproducibility:

1. Create a GitHub Release tag (for example `v1.0-checkpoint`).
2. Upload the checkpoint file as a release asset (for example `hmt_worldtrace_stage1_step_25000.pt`).
3. Reference the resulting release asset URL in PR notes or in this README for traceability.

This avoids Git LFS quota bottlenecks while keeping code history clean and mergeable.

Current checkpoint release:

- Release page: `https://github.com/Brayden-Zhang/ECE324-Mobility-Model-Project/releases/tag/v1.0-checkpoint`
- Direct checkpoint asset: `https://github.com/Brayden-Zhang/ECE324-Mobility-Model-Project/releases/download/v1.0-checkpoint/hmt_worldtrace_stage1_step_25000.pt`
