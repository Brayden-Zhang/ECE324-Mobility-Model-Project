#!/usr/bin/env bash
# Train HMT on a local pickle/parquet dataset (for quick ablations).

#SBATCH --job-name=hmt-local
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-hmt-local-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FALLBACK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${SLURM_SUBMIT_DIR:-${FALLBACK_ROOT}}"

if [[ ! -f "${ROOT_DIR}/train_hmt.py" ]]; then
  ROOT_DIR="${FALLBACK_ROOT}"
fi
cd "${ROOT_DIR}"

module load arrow/21.0.0 || true

if [[ -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"
else
  echo "ERROR: .venv not found at ${ROOT_DIR}/.venv" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

LOCAL_DATA="${LOCAL_DATA:-${ROOT_DIR}/data/worldtrace_sample_nyc.pkl}"
if [[ ! -f "${LOCAL_DATA}" ]]; then
  echo "ERROR: local_data not found: ${LOCAL_DATA}" >&2
  exit 1
fi

OSM_ARGS=()
if [[ -n "${OSM_CONTEXT_PATH:-}" && -f "${OSM_CONTEXT_PATH}" ]]; then
  OSM_ARGS=(--osm_context "${OSM_CONTEXT_PATH}" --osm_context_dim "${OSM_CONTEXT_DIM:-16}")
fi

python train_hmt.py \
  --data_mode local \
  --local_data "${LOCAL_DATA}" \
  --val_ratio "${VAL_RATIO:-0.2}" \
  --test_ratio "${TEST_RATIO:-0.2}" \
  --tokenizer h3 \
  --hash_tokens \
  --ckpt_prefix "${CKPT_PREFIX:-hmt_local}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --max_len "${MAX_LEN:-200}" \
  --max_steps "${MAX_STEPS:-2000}" \
  --eval_interval "${EVAL_INTERVAL:-200}" \
  --save_interval "${SAVE_INTERVAL:-500}" \
  --embed_dim "${EMBED_DIM:-256}" \
  --depth "${DEPTH:-6}" \
  --heads "${HEADS:-8}" \
  --dropout "${DROPOUT:-0.0}" \
  --lr "${LR:-2e-4}" \
  --weight_decay "${WEIGHT_DECAY:-0.05}" \
  --warmup_steps "${WARMUP_STEPS:-200}" \
  --accum_steps "${ACCUM_STEPS:-1}" \
  --use_graph \
  --graph_layers "${GRAPH_LAYERS:-2}" \
  --graph_knn "${GRAPH_KNN:-8}" \
  --graph_temporal_window "${GRAPH_TEMPORAL_WINDOW:-2}" \
  "${OSM_ARGS[@]}"
