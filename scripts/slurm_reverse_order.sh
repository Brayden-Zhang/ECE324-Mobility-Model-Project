#!/usr/bin/env bash
# Reverse-order stress test.

#SBATCH --job-name=trajfm-reverse
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-reverse-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FALLBACK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${SLURM_SUBMIT_DIR:-${FALLBACK_ROOT}}"

if [[ ! -f "${ROOT_DIR}/src/route_rangers/cli/run_reverse_order.py" ]]; then
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

CKPT_GLOB="${CKPT_GLOB:-${ROOT_DIR}/checkpoints/hmt_*_final_step_*.pt ${ROOT_DIR}/checkpoints/hmt_*_step_*.pt}"
CKPT_PATH="${CKPT_PATH:-$(ls -1t ${CKPT_GLOB} 2>/dev/null | head -n1 || true)}"
if [[ -z "${CKPT_PATH}" ]]; then
  echo "ERROR: no checkpoint found." >&2
  exit 1
fi

LOCAL_DATA="${LOCAL_DATA:-${ROOT_DIR}/data/worldtrace_sample.pkl}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/cache/reverse_order_${SLURM_JOB_ID}.json}"
OUTPUT="${OUTPUT//%j/${SLURM_JOB_ID}}"

PYTHONPATH=src python -m route_rangers.cli.run_reverse_order \
  --checkpoint "${CKPT_PATH}" \
  --local_data "${LOCAL_DATA}" \
  --output "${OUTPUT}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --max_len "${MAX_LEN:-200}"
