#!/usr/bin/env bash
# Evaluate TrajectoryFM-HMT checkpoints with UniTraj-style metrics (MAE/RMSE).

#SBATCH --job-name=hmt-unitraj
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-hmt-unitraj-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FALLBACK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${SLURM_SUBMIT_DIR:-${FALLBACK_ROOT}}"

if [[ ! -f "${ROOT_DIR}/scripts/run_unitraj_eval.py" ]]; then
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

if [[ -n "${CKPT_LIST:-}" ]]; then
  IFS=',' read -r -a CKPTS <<< "${CKPT_LIST}"
else
  CKPTS=()
  for p in "${ROOT_DIR}"/checkpoints/hmt_step_5000.pt "${ROOT_DIR}"/checkpoints/hmt_step_10000.pt "${ROOT_DIR}"/checkpoints/hmt_step_15000.pt; do
    [[ -f "${p}" ]] && CKPTS+=("${p}")
  done
fi

if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "ERROR: no checkpoints found. Set CKPT_LIST or add checkpoints." >&2
  exit 1
fi

for ckpt in "${CKPTS[@]}"; do
  name="$(basename "${ckpt}" .pt)"
  out="${ROOT_DIR}/cache/unitraj_eval_${name}_${SLURM_JOB_ID}.json"
  python scripts/run_unitraj_eval.py \
    --checkpoint "${ckpt}" \
    --local_data "${LOCAL_DATA}" \
    --split_mode "${SPLIT_MODE}" \
    --task "${TASK}" \
    --exclude_unknown \
    --mask_ratio "${MASK_RATIO:-0.5}" \
    --pred_steps "${PRED_STEPS:-5}" \
    --output "${out}"
done
