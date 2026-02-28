#!/usr/bin/env bash
# Evaluate a UniTraj checkpoint on WorldTrace sample with UniTraj-style metrics.

#SBATCH --job-name=unitraj-eval
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-unitraj-eval-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FALLBACK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${SLURM_SUBMIT_DIR:-${FALLBACK_ROOT}}"

if [[ ! -f "${ROOT_DIR}/scripts/run_unitraj_external_eval.py" ]]; then
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
CKPT_PATH="${CKPT_PATH:-${ROOT_DIR}/external/unitraj/outputs/best_model.pt}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/cache/unitraj_external_eval_${SLURM_JOB_ID}.json}"

python scripts/run_unitraj_external_eval.py \
  --data_path "${DATA_PATH}" \
  --checkpoint "${CKPT_PATH}" \
  --max_len "${MAX_LEN:-200}" \
  --mask_ratio "${MASK_RATIO:-0.5}" \
  --pred_steps "${PRED_STEPS:-5}" \
  --task "${TASK:-both}" \
  --output "${OUTPUT}"
