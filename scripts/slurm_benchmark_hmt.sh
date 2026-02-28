#!/usr/bin/env bash
#SBATCH --job-name=trajfm-bench
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-bench-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FALLBACK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${SLURM_SUBMIT_DIR:-${FALLBACK_ROOT}}"

if [[ ! -f "${ROOT_DIR}/scripts/run_benchmarks.py" ]]; then
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

export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${ROOT_DIR}/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE}}"
export PYTHONUNBUFFERED=1

CKPT_PATH="${CKPT_PATH:-$(ls -1t "${ROOT_DIR}"/checkpoints/hmt_final_step_*.pt "${ROOT_DIR}"/checkpoints/hmt_step_*.pt 2>/dev/null | head -n1 || true)}"
if [[ -z "${CKPT_PATH}" ]]; then
  echo "ERROR: no checkpoint found; set CKPT_PATH explicitly." >&2
  exit 1
fi

LOCAL_DATA="${LOCAL_DATA:-${ROOT_DIR}/data/worldtrace_sample.pkl}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/cache/benchmark_${SLURM_JOB_ID}.json}"
DISABLE_GRAPH_ARGS=()
if [[ "${DISABLE_GRAPH_BENCHMARK:-0}" == "1" ]]; then
  DISABLE_GRAPH_ARGS=(--disable_graph)
fi

python scripts/run_benchmarks.py \
  --checkpoint "${CKPT_PATH}" \
  --local_data "${LOCAL_DATA}" \
  --output "${OUTPUT}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --max_len "${MAX_LEN:-200}" \
  --probe_epochs "${PROBE_EPOCHS:-6}" \
  --probe_lr "${PROBE_LR:-2e-3}" \
  --probe_weight_decay "${PROBE_WEIGHT_DECAY:-1e-4}" \
  --max_probe_points "${MAX_PROBE_POINTS:-200000}" \
  --split_mode "${SPLIT_MODE:-both}" \
  "${DISABLE_GRAPH_ARGS[@]}"
