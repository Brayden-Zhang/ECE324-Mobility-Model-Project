#!/usr/bin/env bash
# Evaluate the latest checkpoint with UniTraj-style metrics.

#SBATCH --job-name=hmt-unitraj-latest
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-hmt-unitraj-latest-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FALLBACK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${SLURM_SUBMIT_DIR:-${FALLBACK_ROOT}}"

if [[ ! -f "${ROOT_DIR}/src/route_rangers/cli/run_unitraj_eval.py" ]]; then
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

LOCAL_DATA="${LOCAL_DATA:-${ROOT_DIR}/data/worldtrace_sample.pkl}"
SPLIT_MODE="${SPLIT_MODE:-all}"
TASK="${TASK:-both}"
CKPT_GLOB="${CKPT_GLOB:-${ROOT_DIR}/checkpoints/hmt_*_step_*.pt ${ROOT_DIR}/checkpoints/hmt_*_final_step_*.pt}"

CKPT_PATH="$(ls -1t ${CKPT_GLOB} 2>/dev/null | head -n1 || true)"
if [[ -z "${CKPT_PATH}" ]]; then
  echo "ERROR: no checkpoints found for glob: ${CKPT_GLOB}" >&2
  exit 1
fi

name="$(basename "${CKPT_PATH}" .pt)"
out="${ROOT_DIR}/cache/unitraj_eval_${name}_${SLURM_JOB_ID}.json"

PYTHONPATH=src python -m route_rangers.cli.run_unitraj_eval \
  --checkpoint "${CKPT_PATH}" \
  --local_data "${LOCAL_DATA}" \
  --split_mode "${SPLIT_MODE}" \
  --task "${TASK}" \
  --exclude_unknown \
  --mask_ratio "${MASK_RATIO:-0.5}" \
  --pred_steps "${PRED_STEPS:-5}" \
  --output "${out}"
