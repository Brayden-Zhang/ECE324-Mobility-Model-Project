#!/usr/bin/env bash
# Train HMT on full WorldTrace (hf_zip) with non-hashed H3 tokens and a fixed H3 vocab.

#SBATCH --job-name=hmt-nohash-full
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-hmt-nohash-full-%j.out

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

H3_VOCAB="${H3_VOCAB:-${ROOT_DIR}/data/h3_vocab_worldtrace_full.json}"

VOCAB_L0="${VOCAB_L0:-16384}"
VOCAB_L1="${VOCAB_L1:-4096}"
VOCAB_L2="${VOCAB_L2:-1024}"

MAX_CELLS_L0="${MAX_CELLS_L0:-${VOCAB_L0}}"
MAX_CELLS_L1="${MAX_CELLS_L1:-${VOCAB_L1}}"
MAX_CELLS_L2="${MAX_CELLS_L2:-${VOCAB_L2}}"

EMBED_DIM="${EMBED_DIM:-256}"
DEPTH="${DEPTH:-8}"
HEADS="${HEADS:-8}"
DROPOUT="${DROPOUT:-0.0}"
STEP_ATTENTION_WINDOW="${STEP_ATTENTION_WINDOW:-0}"

# Optional stronger graph preset for better long-range structure modeling.
# Enable with: STRONG_GRAPH_PRESET=1 sbatch scripts/slurm_train_hmt_nohash_full.sh
if [[ "${STRONG_GRAPH_PRESET:-0}" == "1" ]]; then
  GRAPH_LAYERS="${GRAPH_LAYERS:-4}"
  GRAPH_KNN="${GRAPH_KNN:-16}"
  GRAPH_TEMPORAL_WINDOW="${GRAPH_TEMPORAL_WINDOW:-4}"
  STEP_ATTENTION_WINDOW="${STEP_ATTENTION_WINDOW:-64}"
  SPACE_TIME_ENCODER="${SPACE_TIME_ENCODER:-1}"
  SPACE_TIME_FREQS="${SPACE_TIME_FREQS:-8}"
fi

SPACE_TIME_ARGS=()
if [[ "${SPACE_TIME_ENCODER:-0}" == "1" ]]; then
  SPACE_TIME_ARGS=(--space_time_encoder --space_time_freqs "${SPACE_TIME_FREQS:-6}")
fi

RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
  RESUME_ARGS=(--resume "${RESUME}")
  if [[ "${RESUME_OPTIMIZER:-0}" == "1" ]]; then
    RESUME_ARGS+=(--resume_optimizer)
  fi
fi

JOB_ID="${SLURM_JOB_ID:-local}"
RESULTS_PATH="${RESULTS_PATH:-${ROOT_DIR}/cache/train_${CKPT_PREFIX:-hmt_nohash_full}_${JOB_ID}.json}"

PYTHONPATH=src python -m route_rangers.cli.build_h3_vocab \
  --data_mode hf_zip \
  --hf_name "${HF_NAME:-OpenTrace/WorldTrace}" \
  --worldtrace_file "${WORLDTRACE_FILE:-Trajectory.zip}" \
  --worldtrace_local_path "${WORLDTRACE_LOCAL_PATH}" \
  --res0 "${RES0:-9}" \
  --res1 "${RES1:-7}" \
  --res2 "${RES2:-5}" \
  --max_cells_l0 "${MAX_CELLS_L0}" \
  --max_cells_l1 "${MAX_CELLS_L1}" \
  --max_cells_l2 "${MAX_CELLS_L2}" \
  --output "${H3_VOCAB}"

python train_hmt.py \
  --data_mode hf_zip \
  --hf_name "${HF_NAME:-OpenTrace/WorldTrace}" \
  --worldtrace_file "${WORLDTRACE_FILE:-Trajectory.zip}" \
  --worldtrace_local_path "${WORLDTRACE_LOCAL_PATH}" \
  --take "${TAKE:-0}" \
  --shuffle_buffer "${SHUFFLE_BUFFER:-1000}" \
  --tokenizer h3 \
  --no_hash_tokens \
  --h3_vocab "${H3_VOCAB}" \
  --vocab_l0 "${VOCAB_L0}" \
  --vocab_l1 "${VOCAB_L1}" \
  --vocab_l2 "${VOCAB_L2}" \
  --ckpt_prefix "${CKPT_PREFIX:-hmt_nohash_full}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --max_len "${MAX_LEN:-200}" \
  --max_steps "${MAX_STEPS:-200000}" \
  --eval_interval "${EVAL_INTERVAL:-2000}" \
  --save_interval "${SAVE_INTERVAL:-5000}" \
  --embed_dim "${EMBED_DIM}" \
  --depth "${DEPTH}" \
  --heads "${HEADS}" \
  --dropout "${DROPOUT}" \
  --step_attention_window "${STEP_ATTENTION_WINDOW}" \
  --lr "${LR:-2e-4}" \
  --weight_decay "${WEIGHT_DECAY:-0.05}" \
  --warmup_steps "${WARMUP_STEPS:-5000}" \
  --accum_steps "${ACCUM_STEPS:-2}" \
  --use_graph \
  --graph_layers "${GRAPH_LAYERS:-2}" \
  --graph_knn "${GRAPH_KNN:-8}" \
  --graph_temporal_window "${GRAPH_TEMPORAL_WINDOW:-2}" \
  --results_path "${RESULTS_PATH}" \
  "${SPACE_TIME_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
