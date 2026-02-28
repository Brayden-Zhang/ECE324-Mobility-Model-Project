#!/usr/bin/env bash
# Train HMT with non-hashed H3 tokens using a fixed H3 vocabulary.

#SBATCH --job-name=hmt-nohash
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-hmt-nohash-%j.out

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

LOCAL_DATA="${LOCAL_DATA:-${ROOT_DIR}/data/worldtrace_sample.pkl}"
VOCAB_DATA="${VOCAB_DATA:-${LOCAL_DATA}}"
H3_VOCAB="${H3_VOCAB:-${ROOT_DIR}/data/h3_vocab_worldtrace_sample.json}"
VOCAB_L0="${VOCAB_L0:-16384}"
VOCAB_L1="${VOCAB_L1:-4096}"
VOCAB_L2="${VOCAB_L2:-1024}"
EMBED_DIM="${EMBED_DIM:-256}"
DEPTH="${DEPTH:-8}"
HEADS="${HEADS:-8}"
DROPOUT="${DROPOUT:-0.0}"
STEP_ATTENTION_WINDOW="${STEP_ATTENTION_WINDOW:-0}"
SPACE_TIME_ARGS=()
if [[ "${SPACE_TIME_ENCODER:-0}" == "1" ]]; then
  SPACE_TIME_ARGS=(--space_time_encoder --space_time_freqs "${SPACE_TIME_FREQS:-6}")
fi

python scripts/build_h3_vocab.py \
  --local_data "${VOCAB_DATA}" \
  --res0 "${RES0:-9}" \
  --res1 "${RES1:-7}" \
  --res2 "${RES2:-5}" \
  --max_cells_l0 "${MAX_CELLS_L0:-${VOCAB_L0}}" \
  --max_cells_l1 "${MAX_CELLS_L1:-${VOCAB_L1}}" \
  --max_cells_l2 "${MAX_CELLS_L2:-${VOCAB_L2}}" \
  --output "${H3_VOCAB}"

python train_hmt.py \
  --data_mode local \
  --local_data "${LOCAL_DATA}" \
  --tokenizer h3 \
  --no_hash_tokens \
  --h3_vocab "${H3_VOCAB}" \
  --vocab_l0 "${VOCAB_L0}" \
  --vocab_l1 "${VOCAB_L1}" \
  --vocab_l2 "${VOCAB_L2}" \
  --ckpt_prefix "${CKPT_PREFIX:-hmt_nohash}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --max_len "${MAX_LEN:-200}" \
  --max_steps "${MAX_STEPS:-20000}" \
  --eval_interval "${EVAL_INTERVAL:-1000}" \
  --save_interval "${SAVE_INTERVAL:-5000}" \
  --embed_dim "${EMBED_DIM}" \
  --depth "${DEPTH}" \
  --heads "${HEADS}" \
  --dropout "${DROPOUT}" \
  --step_attention_window "${STEP_ATTENTION_WINDOW}" \
  --lr "${LR:-2e-4}" \
  --weight_decay "${WEIGHT_DECAY:-0.05}" \
  --warmup_steps "${WARMUP_STEPS:-500}" \
  --accum_steps "${ACCUM_STEPS:-1}" \
  --use_graph \
  --graph_layers "${GRAPH_LAYERS:-2}" \
  --graph_knn "${GRAPH_KNN:-8}" \
  --graph_temporal_window "${GRAPH_TEMPORAL_WINDOW:-2}" \
  "${SPACE_TIME_ARGS[@]}"
