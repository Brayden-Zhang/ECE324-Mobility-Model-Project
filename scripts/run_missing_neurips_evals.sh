#!/usr/bin/env bash
# Launch the two currently-missing NeurIPS mobility-foundation experiments:
# - next_poi_eval
# - cross_city_transfer
#
# Usage:
#   bash scripts/run_missing_neurips_evals.sh /abs/or/relative/path/to/poi_mobility_sample.pkl
#
# Optional env vars:
#   CKPT_PATH=/path/to/checkpoint.pt
#   SPLIT_MODE=temporal|random
#   MAX_LEN=200
#   BATCH_SIZE=32
#   PROBE_EPOCHS=8
#   PROBE_LR=2e-3
#   MIN_CITY_RECORDS=100

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

POI_DATA_INPUT="${1:-${POI_DATA:-}}"
if [[ -z "${POI_DATA_INPUT}" ]]; then
  echo "ERROR: missing POI dataset path argument." >&2
  echo "Usage: bash scripts/run_missing_neurips_evals.sh /path/to/poi_mobility_sample.pkl" >&2
  exit 1
fi

if [[ ! -f "${POI_DATA_INPUT}" ]]; then
  echo "ERROR: POI dataset not found at ${POI_DATA_INPUT}" >&2
  exit 1
fi

CKPT_PATH="${CKPT_PATH:-$(ls -1t "${ROOT_DIR}"/checkpoints/hmt_nohash_full_step_*.pt 2>/dev/null | head -n1 || true)}"
if [[ -z "${CKPT_PATH}" || ! -f "${CKPT_PATH}" ]]; then
  echo "ERROR: no checkpoint found. Set CKPT_PATH explicitly." >&2
  exit 1
fi

EXPORT_VARS="ALL,CKPT_PATH=${CKPT_PATH},POI_DATA=${POI_DATA_INPUT},SPLIT_MODE=${SPLIT_MODE:-temporal},MAX_LEN=${MAX_LEN:-200},BATCH_SIZE=${BATCH_SIZE:-32},PROBE_EPOCHS=${PROBE_EPOCHS:-8},PROBE_LR=${PROBE_LR:-2e-3},MIN_CITY_RECORDS=${MIN_CITY_RECORDS:-100}"

echo "Submitting mobility foundation eval job..."
echo "Checkpoint: ${CKPT_PATH}"
echo "POI data: ${POI_DATA_INPUT}"

OUT=$(sbatch --export="${EXPORT_VARS}" scripts/slurm_mobility_foundation_eval.sh)
echo "${OUT}"

echo "After completion, refresh consolidated reports with:"
echo "  export PYTHONPATH=src"
echo "  python -m route_rangers.cli.collect_results"
echo "  python -m route_rangers.cli.generate_foundation_report --cache_dir cache --output_dir reports"
