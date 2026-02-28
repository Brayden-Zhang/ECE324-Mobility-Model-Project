#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"
else
  echo "ERROR: .venv not found at ${ROOT_DIR}/.venv" >&2
  exit 1
fi

CKPT="$(ls -1t checkpoints/hmt_stage2_big_*final_step_*.pt 2>/dev/null | head -n1 || true)"
if [[ -z "${CKPT}" ]]; then
  echo "ERROR: no stage2 checkpoint found under checkpoints/hmt_stage2_big_*final_step_*.pt" >&2
  exit 1
fi

python scripts/run_macro_eval.py \
  --checkpoint "${CKPT}" \
  --macro_data "data/hdx/movement-distribution/processed/movement_distribution_12m_monthly.npz" \
  --batch_size 512 \
  --output "cache/macro_eval_${SLURM_JOB_ID}.json"
