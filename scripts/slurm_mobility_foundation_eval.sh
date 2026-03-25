#!/usr/bin/env bash
# MoveGPT-style mobility evaluation: next-POI + cross-city transfer.

#SBATCH --job-name=mobility-foundation-eval
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-mobility-foundation-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FALLBACK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${SLURM_SUBMIT_DIR:-${FALLBACK_ROOT}}"
cd "${ROOT_DIR}"

module load arrow/21.0.0
source "${ROOT_DIR}/.venv/bin/activate"
export PYTHONUNBUFFERED=1

CKPT_PATH="${CKPT_PATH:-$(ls -1t "${ROOT_DIR}"/checkpoints/hmt_nohash_full_step_*.pt 2>/dev/null | head -n1 || true)}"
if [[ -z "${CKPT_PATH}" ]]; then
    echo "ERROR: no checkpoint found." >&2
    exit 1
fi

POI_DATA="${POI_DATA:-${ROOT_DIR}/data/samples/poi_mobility_sample.pkl}"
JOB_ID="${SLURM_JOB_ID:-local}"

echo "=== Mobility Foundation Eval ==="
echo "Checkpoint: ${CKPT_PATH}"
echo "POI Data: ${POI_DATA}"
echo "Job ID: ${JOB_ID}"

echo ">>> Next-POI eval"
PYTHONPATH=src python -m route_rangers.cli.run_next_poi_eval \
  --checkpoint "${CKPT_PATH}" \
  --local_data "${POI_DATA}" \
  --split_mode "${SPLIT_MODE:-temporal}" \
  --max_len "${MAX_LEN:-200}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --probe_epochs "${PROBE_EPOCHS:-8}" \
  --probe_lr "${PROBE_LR:-2e-3}" \
  --output "${ROOT_DIR}/cache/next_poi_eval_${JOB_ID}.json"

echo ">>> Cross-city transfer"
PYTHONPATH=src python -m route_rangers.cli.run_cross_city_transfer \
  --checkpoint "${CKPT_PATH}" \
  --local_data "${POI_DATA}" \
  --max_len "${MAX_LEN:-200}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --probe_epochs "${PROBE_EPOCHS:-8}" \
  --probe_lr "${PROBE_LR:-2e-3}" \
  --min_city_records "${MIN_CITY_RECORDS:-100}" \
  --output "${ROOT_DIR}/cache/cross_city_transfer_${JOB_ID}.json"

echo ">>> Refresh report blocks"
PYTHONPATH=src python -m route_rangers.cli.collect_results
PYTHONPATH=src python -m route_rangers.cli.generate_foundation_report --cache_dir cache --output_dir reports

echo "=== Mobility foundation evaluation complete ==="
