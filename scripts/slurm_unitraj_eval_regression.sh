#!/usr/bin/env bash
# UniTraj eval with regression-based coordinate prediction.
# This produces fair MAE/RMSE comparisons with UniTraj by using a learned
# regression head instead of discrete token centroids.

#SBATCH --job-name=unitraj-reg
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-unitraj-reg-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FALLBACK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${SLURM_SUBMIT_DIR:-${FALLBACK_ROOT}}"
[[ -f "${ROOT_DIR}/scripts/run_unitraj_eval.py" ]] || ROOT_DIR="${FALLBACK_ROOT}"
cd "${ROOT_DIR}"

module load arrow/21.0.0
source "${ROOT_DIR}/.venv/bin/activate"
export PYTHONUNBUFFERED=1

CKPT_PATH="${CKPT_PATH:-$(ls -1t "${ROOT_DIR}"/checkpoints/hmt_nohash_full_step_*.pt 2>/dev/null | head -n1 || true)}"
if [[ -z "${CKPT_PATH}" ]]; then
    echo "ERROR: no checkpoint found." >&2
    exit 1
fi

LOCAL_DATA="${LOCAL_DATA:-${ROOT_DIR}/data/worldtrace_sample.pkl}"
JOB_ID="${SLURM_JOB_ID:-local}"
SPLIT_MODE="${SPLIT_MODE:-all}"
TASK="${TASK:-both}"

echo "=== UniTraj Regression Eval ==="
echo "Checkpoint: ${CKPT_PATH}"
echo "Data: ${LOCAL_DATA}"
echo "Split mode: ${SPLIT_MODE}"
echo "Task: ${TASK}"

# Regression-based eval (primary - fair comparison with UniTraj)
python scripts/run_unitraj_eval.py \
  --checkpoint "${CKPT_PATH}" \
  --local_data "${LOCAL_DATA}" \
  --split_mode "${SPLIT_MODE}" \
  --task "${TASK}" \
  --use_regression \
  --regression_epochs 10 \
  --regression_lr 2e-3 \
  --output "${ROOT_DIR}/cache/unitraj_eval_regression_${JOB_ID}.json"

# Also run with exclude_unknown and H3 centroids for comparison
python scripts/run_unitraj_eval.py \
  --checkpoint "${CKPT_PATH}" \
  --local_data "${LOCAL_DATA}" \
  --split_mode "${SPLIT_MODE}" \
  --task "${TASK}" \
  --no-use_regression \
  --exclude_unknown \
  --centroid_level l2 \
  --output "${ROOT_DIR}/cache/unitraj_eval_l2_excl_${JOB_ID}.json"

# Robust eval with regression
python scripts/run_unitraj_eval.py \
  --checkpoint "${CKPT_PATH}" \
  --local_data "${LOCAL_DATA}" \
  --split_mode "${SPLIT_MODE}" \
  --task "${TASK}" \
  --use_regression \
  --coord_noise_std_m 30 \
  --input_drop_ratio 0.2 \
  --output "${ROOT_DIR}/cache/unitraj_eval_regression_robust_${JOB_ID}.json"

echo "=== Done ==="
