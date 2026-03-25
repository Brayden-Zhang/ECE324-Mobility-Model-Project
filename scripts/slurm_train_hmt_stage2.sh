#!/usr/bin/env bash
# Stage-2 training with macro distribution multitask.

#SBATCH --job-name=trajfm-hmt-stage2
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-hmt-stage2-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FALLBACK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${SLURM_SUBMIT_DIR:-${FALLBACK_ROOT}}"

if [[ ! -f "${ROOT_DIR}/train_hmt.py" ]]; then
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
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${ROOT_DIR}/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE}}"

DEFAULT_WORLDTRACE_LOCAL_PATH=""
if [[ -f "${ROOT_DIR}/data/worldtrace_full/Trajectory.zip" ]]; then
  DEFAULT_WORLDTRACE_LOCAL_PATH="${ROOT_DIR}/data/worldtrace_full/Trajectory.zip"
fi
WORLDTRACE_LOCAL_PATH="${WORLDTRACE_LOCAL_PATH:-${DEFAULT_WORLDTRACE_LOCAL_PATH}}"
HF_NAME="${HF_NAME:-OpenTrace/WorldTrace}"
WORLDTRACE_FILE="${WORLDTRACE_FILE:-Trajectory.zip}"
WORLDTRACE_ARGS=(--hf_name "${HF_NAME}" --worldtrace_file "${WORLDTRACE_FILE}")
if [[ -n "${WORLDTRACE_LOCAL_PATH}" ]]; then
  WORLDTRACE_ARGS+=(--worldtrace_local_path "${WORLDTRACE_LOCAL_PATH}")
fi

RESUME="${RESUME:-}"
if [[ -z "${RESUME}" ]]; then
  RESUME="$(ls -1t "${ROOT_DIR}"/checkpoints/hmt_*_final_step_*.pt "${ROOT_DIR}"/checkpoints/hmt_*_step_*.pt 2>/dev/null | head -n1 || true)"
fi
if [[ -z "${RESUME}" ]]; then
  echo "ERROR: no checkpoint found for stage2; set RESUME explicitly." >&2
  exit 1
fi

RESUME_OPT_ARGS=()
if [[ "${RESUME_OPTIMIZER:-0}" == "1" ]]; then
  RESUME_OPT_ARGS=(--resume_optimizer)
fi

MACRO_DATA="${MACRO_DATA:-${ROOT_DIR}/data/hdx/movement-distribution/processed/movement_distribution_12m_monthly.npz}"
if [[ ! -f "${MACRO_DATA}" ]]; then
  echo "ERROR: macro_data not found: ${MACRO_DATA}" >&2
  exit 1
fi

SPACE_TIME_ARGS=()
if [[ "${SPACE_TIME_ENCODER:-0}" == "1" ]]; then
  SPACE_TIME_ARGS=(--space_time_encoder --space_time_freqs "${SPACE_TIME_FREQS:-6}")
fi

JOB_ID="${SLURM_JOB_ID:-local}"
RESULTS_PATH="${RESULTS_PATH:-${ROOT_DIR}/cache/train_${CKPT_PREFIX:-hmt_stage2}_${JOB_ID}.json}"

python train_hmt.py \
  --data_mode hf_zip \
  "${WORLDTRACE_ARGS[@]}" \
  --shuffle_buffer "${SHUFFLE_BUFFER:-1000}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --max_len "${MAX_LEN:-200}" \
  --max_steps "${MAX_STEPS:-100000}" \
  --eval_interval "${EVAL_INTERVAL:-2000}" \
  --save_interval "${SAVE_INTERVAL:-5000}" \
  --embed_dim "${EMBED_DIM:-256}" \
  --depth "${DEPTH:-8}" \
  --heads "${HEADS:-8}" \
  --lr "${LR:-1e-4}" \
  --weight_decay "${WEIGHT_DECAY:-0.05}" \
  --warmup_steps "${WARMUP_STEPS:-2000}" \
  --accum_steps "${ACCUM_STEPS:-2}" \
  --use_graph \
  --graph_layers "${GRAPH_LAYERS:-2}" \
  --graph_knn "${GRAPH_KNN:-8}" \
  --graph_temporal_window "${GRAPH_TEMPORAL_WINDOW:-2}" \
  --tokenizer h3 \
  --no_hash_tokens \
  --h3_vocab "${H3_VOCAB:-${ROOT_DIR}/data/h3_vocab_worldtrace_full.json}" \
  --ckpt_prefix "${CKPT_PREFIX:-hmt_stage2}" \
  --resume "${RESUME}" \
  "${RESUME_OPT_ARGS[@]}" \
  --macro_data "${MACRO_DATA}" \
  --macro_batch_size "${MACRO_BATCH_SIZE:-256}" \
  --macro_mix_prob "${MACRO_MIX_PROB:-0.5}" \
  --macro_weight "${MACRO_WEIGHT:-1.0}" \
  --results_path "${RESULTS_PATH}" \
  "${SPACE_TIME_ARGS[@]}"
