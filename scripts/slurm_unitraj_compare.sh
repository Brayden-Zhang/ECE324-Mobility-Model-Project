#!/usr/bin/env bash
# Compare latest TrajectoryFM UniTraj-style eval against the UniTraj-compatible baseline.

#SBATCH --job-name=unitraj-compare
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-unitraj-compare-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FALLBACK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${SLURM_SUBMIT_DIR:-${FALLBACK_ROOT}}"
[[ -f "${ROOT_DIR}/src/route_rangers/cli/compare_unitraj_results.py" ]] || ROOT_DIR="${FALLBACK_ROOT}"
cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"
fi

JOB_ID="${SLURM_JOB_ID:-local}"
MODE="${MODE:-all}"
SPLIT="${SPLIT:-test}"
HMT_JSON="${HMT_JSON:-$(ls -1t "${ROOT_DIR}"/cache/unitraj_eval_regression_[0-9]*.json 2>/dev/null | head -n1 || true)}"
UNITRAJ_JSON="${UNITRAJ_JSON:-$(ls -1t "${ROOT_DIR}"/cache/unitraj_external_eval_*.json 2>/dev/null | head -n1 || true)}"
OUT_CSV="${OUT_CSV:-${ROOT_DIR}/cache/unitraj_compare_${JOB_ID}.csv}"

if [[ -z "${HMT_JSON}" || ! -f "${HMT_JSON}" ]]; then
  echo "ERROR: HMT eval JSON not found. Set HMT_JSON=/path/to/unitraj_eval_*.json" >&2
  exit 1
fi
if [[ -z "${UNITRAJ_JSON}" || ! -f "${UNITRAJ_JSON}" ]]; then
  echo "ERROR: UniTraj external eval JSON not found. Set UNITRAJ_JSON=/path/to/unitraj_external_eval_*.json" >&2
  exit 1
fi

echo "=== UniTraj Comparison ==="
echo "HMT JSON: ${HMT_JSON}"
echo "UniTraj JSON: ${UNITRAJ_JSON}"
echo "Mode: ${MODE}, Split: ${SPLIT}"
echo "Output: ${OUT_CSV}"

PYTHONPATH=src python -m route_rangers.cli.compare_unitraj_results \
  --hmt "${HMT_JSON}" \
  --unitraj "${UNITRAJ_JSON}" \
  --mode "${MODE}" \
  --split "${SPLIT}" \
  > "${OUT_CSV}"

echo "=== Done ==="
