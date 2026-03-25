#!/usr/bin/env bash
# Train UniTraj on WorldTrace sample with configurable hyperparams.
# Outputs checkpoints to OUTPUT_DIR (best_model.pt, last_model.pt).

#SBATCH --job-name=unitraj-train
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-unitraj-train-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FALLBACK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${SLURM_SUBMIT_DIR:-${FALLBACK_ROOT}}"

if [[ ! -f "${ROOT_DIR}/src/route_rangers/cli/unitraj/train_unitraj.py" ]]; then
  ROOT_DIR="${FALLBACK_ROOT}"
fi
cd "${ROOT_DIR}"

module load arrow/21.0.0

if [[ -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"
else
  echo "ERROR: .venv not found at ${ROOT_DIR}/.venv" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/worldtrace_sample.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/checkpoints/unitraj}"

PYTHONPATH=src python -m route_rangers.cli.unitraj.train_unitraj \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_len "${MAX_LEN:-200}" \
  --mask_ratio "${MASK_RATIO:-0.5}" \
  --batch_size "${BATCH_SIZE:-256}" \
  --epochs "${EPOCHS:-50}" \
  --lr "${LR:-1e-3}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --val_ratio "${VAL_RATIO:-0.1}"
