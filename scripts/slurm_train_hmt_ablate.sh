#!/usr/bin/env bash
# Generic ablation training script for HMT on full WorldTrace (hf_zip) with fixed H3 vocab.

#SBATCH --job-name=hmt-ablate
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-hmt-ablate-%j.out

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

H3_VOCAB="${H3_VOCAB:-${ROOT_DIR}/data/h3_vocab_worldtrace_full.json}"
VOCAB_L0="${VOCAB_L0:-16384}"
VOCAB_L1="${VOCAB_L1:-4096}"
VOCAB_L2="${VOCAB_L2:-1024}"

MAX_CELLS_L0="${MAX_CELLS_L0:-${VOCAB_L0}}"
MAX_CELLS_L1="${MAX_CELLS_L1:-${VOCAB_L1}}"
MAX_CELLS_L2="${MAX_CELLS_L2:-${VOCAB_L2}}"

TOKEN_ARGS=(--tokenizer h3)
if [[ "${HASH_TOKENS:-0}" == "1" ]]; then
  TOKEN_ARGS+=(--hash_tokens)
else
  if [[ ! -f "${H3_VOCAB}" ]]; then
    PYTHONPATH=src python -m route_rangers.cli.build_h3_vocab \
      --data_mode hf_zip \
      --hf_name "${HF_NAME:-OpenTrace/WorldTrace}" \
      --worldtrace_file "${WORLDTRACE_FILE:-Trajectory.zip}" \
      --worldtrace_local_path "${WORLDTRACE_LOCAL_PATH:-}" \
      --res0 "${RES0:-9}" \
      --res1 "${RES1:-7}" \
      --res2 "${RES2:-5}" \
      --max_cells_l0 "${MAX_CELLS_L0}" \
      --max_cells_l1 "${MAX_CELLS_L1}" \
      --max_cells_l2 "${MAX_CELLS_L2}" \
      --output "${H3_VOCAB}"
  fi
  TOKEN_ARGS+=(--no_hash_tokens --h3_vocab "${H3_VOCAB}")
fi

GRAPH_ARGS=()
if [[ "${USE_GRAPH:-1}" == "1" ]]; then
  GRAPH_ARGS=(--use_graph --graph_layers "${GRAPH_LAYERS:-2}" --graph_knn "${GRAPH_KNN:-8}" --graph_temporal_window "${GRAPH_TEMPORAL_WINDOW:-2}")
fi

SPACE_TIME_ARGS=()
if [[ "${SPACE_TIME_ENCODER:-0}" == "1" ]]; then
  SPACE_TIME_ARGS=(--space_time_encoder --space_time_freqs "${SPACE_TIME_FREQS:-6}")
fi

TRIP_ARGS=()
if [[ "${NO_TRIP_FEATURES:-0}" == "1" ]]; then
  TRIP_ARGS=(--no_trip_features)
fi

LOSS_ARGS=()
if [[ "${LENGTH_WEIGHTED_LOSS:-0}" == "1" ]]; then
  LOSS_ARGS=(--length_weighted_loss)
fi

python train_hmt.py \
  --data_mode hf_zip \
  --hf_name "${HF_NAME:-OpenTrace/WorldTrace}" \
  --worldtrace_file "${WORLDTRACE_FILE:-Trajectory.zip}" \
  --worldtrace_local_path "${WORLDTRACE_LOCAL_PATH:-}" \
  --take "${TAKE:-0}" \
  --shuffle_buffer "${SHUFFLE_BUFFER:-1000}" \
  "${TOKEN_ARGS[@]}" \
  --vocab_l0 "${VOCAB_L0}" \
  --vocab_l1 "${VOCAB_L1}" \
  --vocab_l2 "${VOCAB_L2}" \
  --ckpt_prefix "${CKPT_PREFIX:-hmt_ablate}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --max_len "${MAX_LEN:-200}" \
  --max_steps "${MAX_STEPS:-60000}" \
  --eval_interval "${EVAL_INTERVAL:-2000}" \
  --save_interval "${SAVE_INTERVAL:-5000}" \
  --embed_dim "${EMBED_DIM:-256}" \
  --depth "${DEPTH:-8}" \
  --heads "${HEADS:-8}" \
  --dropout "${DROPOUT:-0.0}" \
  --step_attention_window "${STEP_ATTENTION_WINDOW:-0}" \
  --lr "${LR:-2e-4}" \
  --weight_decay "${WEIGHT_DECAY:-0.05}" \
  --warmup_steps "${WARMUP_STEPS:-5000}" \
  --accum_steps "${ACCUM_STEPS:-2}" \
  --flow_weight "${FLOW_WEIGHT:-1.0}" \
  --token_weight "${TOKEN_WEIGHT:-1.0}" \
  --dest_weight "${DEST_WEIGHT:-0.3}" \
  --region_weight "${REGION_WEIGHT:-0.5}" \
  --consistency_weight "${CONSISTENCY_WEIGHT:-0.2}" \
  "${GRAPH_ARGS[@]}" \
  "${SPACE_TIME_ARGS[@]}" \
  "${TRIP_ARGS[@]}" \
  "${LOSS_ARGS[@]}"
