#!/usr/bin/env bash
#SBATCH --job-name=trajfm-hmt
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

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
  # Use the repo-local venv even if sbatch is launched from elsewhere.
  source "${ROOT_DIR}/.venv/bin/activate"
else
  echo "ERROR: .venv not found at ${ROOT_DIR}/.venv" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${ROOT_DIR}/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE}}"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

HF_NAME="${HF_NAME:-OpenTrace/WorldTrace}"
WORLDTRACE_FILE="${WORLDTRACE_FILE:-Trajectory.zip}"
WORLDTRACE_LOCAL_PATH="${WORLDTRACE_LOCAL_PATH:-}"
TAKE="${TAKE:-0}"
WORLDTRACE_ARGS=(--hf_name "${HF_NAME}" --worldtrace_file "${WORLDTRACE_FILE}")
if [[ -n "${WORLDTRACE_LOCAL_PATH}" ]]; then
  WORLDTRACE_ARGS+=(--worldtrace_local_path "${WORLDTRACE_LOCAL_PATH}")
fi

OSM_CONTEXT_PATH="${OSM_CONTEXT_PATH:-${ROOT_DIR}/data/osm_context.json}"
OSM_ARGS=()
if [[ "${BUILD_OSM_CONTEXT:-0}" == "1" && ! -f "${OSM_CONTEXT_PATH}" ]]; then
  module load proj/9.4.1 || true
  PYTHONPATH=src python -m route_rangers.cli.build_osm_context \
    --north "${OSM_NORTH:-40.9}" \
    --south "${OSM_SOUTH:-40.4}" \
    --east "${OSM_EAST:--73.7}" \
    --west "${OSM_WEST:--74.3}" \
    --res "${OSM_RES:-7}" \
    --output "${OSM_CONTEXT_PATH}"
fi
if [[ -f "${OSM_CONTEXT_PATH}" ]]; then
  OSM_ARGS=(--osm_context "${OSM_CONTEXT_PATH}" --osm_context_dim "${OSM_CONTEXT_DIM:-16}")
fi

MAX_STEPS="${MAX_STEPS:-200000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LEN="${MAX_LEN:-200}"
RESULTS_PATH="${RESULTS_PATH:-${ROOT_DIR}/cache/slurm_train_metrics_${SLURM_JOB_ID}.json}"
CKPT_PREFIX="${CKPT_PREFIX:-hmt}"
SHUFFLE_BUFFER="${SHUFFLE_BUFFER:-1000}"

EMBED_DIM="${EMBED_DIM:-256}"
DEPTH="${DEPTH:-8}"
HEADS="${HEADS:-8}"
DROPOUT="${DROPOUT:-0.0}"
STEP_ATTENTION_WINDOW="${STEP_ATTENTION_WINDOW:-0}"
SPACE_TIME_ARGS=()
if [[ "${SPACE_TIME_ENCODER:-0}" == "1" ]]; then
  SPACE_TIME_ARGS=(--space_time_encoder --space_time_freqs "${SPACE_TIME_FREQS:-6}")
fi

python train_hmt.py \
  --data_mode hf_zip \
  "${WORLDTRACE_ARGS[@]}" \
  --take "${TAKE}" \
  --shuffle_buffer "${SHUFFLE_BUFFER}" \
  --batch_size "${BATCH_SIZE}" \
  --max_len "${MAX_LEN}" \
  --max_steps "${MAX_STEPS}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --save_interval "${SAVE_INTERVAL}" \
  --embed_dim "${EMBED_DIM}" \
  --depth "${DEPTH}" \
  --heads "${HEADS}" \
  --dropout "${DROPOUT}" \
  --step_attention_window "${STEP_ATTENTION_WINDOW}" \
  --weight_decay 0.05 \
  --warmup_steps 5000 \
  --accum_steps 2 \
  --tokenizer h3 \
  --hash_tokens \
  --use_graph \
  --graph_layers 2 \
  --graph_knn 8 \
  --graph_temporal_window 2 \
  --ckpt_prefix "${CKPT_PREFIX}" \
  --results_path "${RESULTS_PATH}" \
  "${OSM_ARGS[@]}" \
  "${SPACE_TIME_ARGS[@]}"

if [[ "${RUN_BENCHMARKS:-0}" == "1" ]]; then
  CKPT_PATH="${BENCH_CKPT_PATH:-$(ls -1t "${ROOT_DIR}"/checkpoints/hmt_final_step_*.pt "${ROOT_DIR}"/checkpoints/hmt_step_*.pt 2>/dev/null | head -n1 || true)}"
  if [[ -z "${CKPT_PATH}" ]]; then
    echo "No checkpoint found for benchmark run; skipping."
    exit 0
  fi
  BENCH_DATA="${BENCH_LOCAL_DATA:-${ROOT_DIR}/data/worldtrace_sample.pkl}"
  BENCH_OUT="${BENCH_OUTPUT:-${ROOT_DIR}/cache/benchmark_${SLURM_JOB_ID}.json}"
  BENCH_GRAPH_ARGS=()
  if [[ "${BENCH_DISABLE_GRAPH:-0}" == "1" ]]; then
    BENCH_GRAPH_ARGS=(--disable_graph)
  fi
  PYTHONPATH=src python -m route_rangers.cli.run_benchmarks \
    --checkpoint "${CKPT_PATH}" \
    --local_data "${BENCH_DATA}" \
    --output "${BENCH_OUT}" \
    --batch_size "${BENCH_BATCH_SIZE:-32}" \
    --max_len "${BENCH_MAX_LEN:-200}" \
    --probe_epochs "${BENCH_PROBE_EPOCHS:-6}" \
    --probe_lr "${BENCH_PROBE_LR:-2e-3}" \
    --probe_weight_decay "${BENCH_PROBE_WEIGHT_DECAY:-1e-4}" \
    --max_probe_points "${BENCH_MAX_PROBE_POINTS:-200000}" \
    --split_mode "${BENCH_SPLIT_MODE:-both}" \
    "${BENCH_GRAPH_ARGS[@]}"
fi
