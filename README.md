# Route Rangers: TrajectoryFM-HMT

Implementation of the TrajectoryFM-HMT pipeline for trajectory representation learning, length-sensitivity analysis, and UniTraj-style evaluation.

## What is canonical now

The repository was reorganized so the real Python package lives in `src/route_rangers`.

- Model code: `src/route_rangers/models/`
- Data pipeline: `src/route_rangers/data/`
- CLI/evaluation scripts: `src/route_rangers/cli/`

All maintained Python entrypoints now live under `src/route_rangers`; shell launchers call these modules directly.

## Repository layout

- `train_hmt.py`: main HMT training entrypoint
- `src/route_rangers/models/`: HMT tokenizer/backbone/model/flow modules
- `src/route_rangers/data/`: dataset loaders, preprocessing, macro dataset, OSM context
- `src/route_rangers/cli/`: benchmark and evaluation entrypoints
- `scripts/*.sh`: SLURM launch scripts
- `docs/`: paper and experiment notes
- `cache/`: evaluation outputs
- `checkpoints/`: model checkpoints

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

Optional (cluster environments that need Arrow module):

```bash
module load arrow/21.0.0
```

## Data

Expected local files used in this repo:

- `data/worldtrace_sample.pkl` (main small benchmark set)
- `data/worldtrace_sample_nyc.pkl` (transfer/secondary)
- `data/worldtrace_full/Trajectory.zip` (large local zip, optional)

Download helper (optional):

```bash
PYTHONPATH=src python -m route_rangers.cli.download_worldtrace --split train --output data/worldtrace --max_samples 500000
```

## Training

### Quick local run (sample data)

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

### Full run from local WorldTrace zip

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

### Core benchmark suite

```bash
PYTHONPATH=src python -m route_rangers.cli.run_benchmarks \
  --checkpoint checkpoints/hmt_step_15000.pt \
  --local_data data/worldtrace_sample.pkl \
  --split_mode both \
  --output cache/benchmark_results.json
```

### Length sensitivity

```bash
PYTHONPATH=src python -m route_rangers.cli.run_length_sensitivity \
  --checkpoint checkpoints/hmt_step_15000.pt \
  --local_data data/worldtrace_sample.pkl \
  --output cache/length_sensitivity_latest.json
```

### Invariance / robustness suite

```bash
PYTHONPATH=src python -m route_rangers.cli.run_invariance_suite \
  --checkpoint checkpoints/hmt_step_15000.pt \
  --local_data data/worldtrace_sample.pkl \
  --output cache/invariance_suite.json
```

### UniTraj-style regression evaluation

```bash
PYTHONPATH=src python -m route_rangers.cli.run_unitraj_eval \
  --checkpoint checkpoints/hmt_step_15000.pt \
  --local_data data/worldtrace_sample.pkl \
  --task both \
  --split_mode all \
  --output cache/unitraj_eval_regression.json
```

For a broader set of evaluation commands, see `docs/foundation_evals.md` and `docs/neurips_experiments.md`.

## SLURM launchers

Common launch scripts:

- `scripts/slurm_train_hmt.sh`
- `scripts/slurm_train_hmt_nohash_full.sh`
- `scripts/slurm_benchmark_latest.sh`
- `scripts/slurm_length_sensitivity.sh`
- `scripts/slurm_unitraj_eval_regression.sh`

## Notes

- Import project modules from `route_rangers.*`.
- Run CLI tools via `python -m route_rangers.cli.<tool>` with `PYTHONPATH=src` unless installed in editable mode.
