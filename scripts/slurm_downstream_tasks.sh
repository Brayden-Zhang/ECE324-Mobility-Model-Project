#!/usr/bin/env bash
# Run all downstream task evaluations.
# Tasks: travel-time estimation, anomaly detection, trip classification, similarity retrieval.

#SBATCH --job-name=downstream-tasks
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-downstream-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FALLBACK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${SLURM_SUBMIT_DIR:-${FALLBACK_ROOT}}"
[[ -f "${ROOT_DIR}/src/route_rangers/cli/run_travel_time_estimation.py" ]] || ROOT_DIR="${FALLBACK_ROOT}"
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

echo "=== Downstream Task Suite ==="
echo "Checkpoint: ${CKPT_PATH}"
echo "Data: ${LOCAL_DATA}"
echo "Job ID: ${JOB_ID}"
echo ""

# ---- Travel Time Estimation ----
echo ">>> Travel Time Estimation"
PYTHONPATH=src python -m route_rangers.cli.run_travel_time_estimation \
  --checkpoint "${CKPT_PATH}" \
  --local_data "${LOCAL_DATA}" \
  --prefix_ratios 0.25 0.50 0.75 1.00 \
  --probe_epochs "${PROBE_EPOCHS:-10}" \
  --probe_lr "${PROBE_LR:-2e-3}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --max_len "${MAX_LEN:-200}" \
  --output "${ROOT_DIR}/cache/travel_time_${JOB_ID}.json"
echo ""

# ---- Anomaly Detection ----
echo ">>> Anomaly Detection"
PYTHONPATH=src python -m route_rangers.cli.run_anomaly_detection \
  --checkpoint "${CKPT_PATH}" \
  --local_data "${LOCAL_DATA}" \
  --anomaly_ratio 0.1 \
  --batch_size "${BATCH_SIZE:-32}" \
  --max_len "${MAX_LEN:-200}" \
  --output "${ROOT_DIR}/cache/anomaly_detection_${JOB_ID}.json"
echo ""

# ---- Trip Classification ----
echo ">>> Trip Classification"
PYTHONPATH=src python -m route_rangers.cli.run_trip_classification \
  --checkpoint "${CKPT_PATH}" \
  --local_data "${LOCAL_DATA}" \
  --probe_epochs "${PROBE_EPOCHS:-10}" \
  --probe_lr "${PROBE_LR:-1e-3}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --max_len "${MAX_LEN:-200}" \
  --output "${ROOT_DIR}/cache/trip_classification_${JOB_ID}.json"
echo ""

# ---- Similarity Retrieval ----
echo ">>> Similarity Retrieval"
PYTHONPATH=src python -m route_rangers.cli.run_similarity_retrieval \
  --checkpoint "${CKPT_PATH}" \
  --local_data "${LOCAL_DATA}" \
  --top_k 10 \
  --batch_size "${BATCH_SIZE:-32}" \
  --max_len "${MAX_LEN:-200}" \
  --output "${ROOT_DIR}/cache/similarity_retrieval_${JOB_ID}.json"
echo ""

echo "=== All downstream tasks complete ==="
