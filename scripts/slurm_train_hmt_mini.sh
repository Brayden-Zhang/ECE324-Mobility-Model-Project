#!/usr/bin/env bash
# Mini end-to-end smoke training on WorldTrace (quick architecture check).
#
# Typical usage:
#   sbatch scripts/slurm_train_hmt_mini.sh
#
# Common overrides (env vars):
#   TAKE=20000 MAX_STEPS=300 BATCH_SIZE=64 MAX_LEN=128 EMBED_DIM=128 DEPTH=4 HEADS=4
#   WORLDTRACE_LOCAL_PATH=/path/to/Trajectory.zip
#   RUN_BENCHMARKS=0

#SBATCH --job-name=hmt-mini
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-hmt-mini-%j.out

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
export WANDB_MODE=disabled
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${ROOT_DIR}/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE}}"

JOB_ID="${SLURM_JOB_ID:-local}"

TAKE="${TAKE:-20000}"
MAX_STEPS="${MAX_STEPS:-300}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-150}"
SHUFFLE_BUFFER="${SHUFFLE_BUFFER:-500}"

BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LEN="${MAX_LEN:-128}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"

EMBED_DIM="${EMBED_DIM:-128}"
DEPTH="${DEPTH:-4}"
HEADS="${HEADS:-4}"
DROPOUT="${DROPOUT:-0.0}"

WORLDTRACE_FILE="${WORLDTRACE_FILE:-Trajectory.zip}"
WORLDTRACE_LOCAL_PATH="${WORLDTRACE_LOCAL_PATH:-}"
if [[ -z "${WORLDTRACE_LOCAL_PATH}" && -f "${ROOT_DIR}/data/worldtrace_full/Trajectory.zip" ]]; then
  WORLDTRACE_LOCAL_PATH="${ROOT_DIR}/data/worldtrace_full/Trajectory.zip"
fi

WORLDTRACE_ARGS=(--hf_name "${HF_NAME:-OpenTrace/WorldTrace}" --worldtrace_file "${WORLDTRACE_FILE}")
if [[ -n "${WORLDTRACE_LOCAL_PATH}" ]]; then
  WORLDTRACE_ARGS+=(--worldtrace_local_path "${WORLDTRACE_LOCAL_PATH}")
fi

TOKEN_ARGS=(--tokenizer h3)
H3_VOCAB_DEFAULT="${ROOT_DIR}/data/h3_vocab_worldtrace_full.json"
if [[ "${HASH_TOKENS:-0}" == "1" ]]; then
  TOKEN_ARGS+=(--hash_tokens)
else
  if [[ -f "${H3_VOCAB_DEFAULT}" ]]; then
    TOKEN_ARGS+=(--no_hash_tokens --h3_vocab "${H3_VOCAB:-${H3_VOCAB_DEFAULT}}")
  else
    TOKEN_ARGS+=(--hash_tokens)
  fi
fi

GRAPH_ARGS=()
if [[ "${USE_GRAPH:-1}" == "1" ]]; then
  GRAPH_ARGS=(
    --use_graph
    --graph_layers "${GRAPH_LAYERS:-1}"
    --graph_knn "${GRAPH_KNN:-6}"
    --graph_temporal_window "${GRAPH_TEMPORAL_WINDOW:-2}"
  )
fi

CKPT_PREFIX="${CKPT_PREFIX:-hmt_mini_${JOB_ID}}"
RESULTS_PATH="${RESULTS_PATH:-${ROOT_DIR}/cache/mini_train_metrics_${JOB_ID}.json}"

python train_hmt.py \
  --data_mode hf_zip \
  "${WORLDTRACE_ARGS[@]}" \
  --take "${TAKE}" \
  --shuffle_buffer "${SHUFFLE_BUFFER}" \
  "${TOKEN_ARGS[@]}" \
  --ckpt_prefix "${CKPT_PREFIX}" \
  --batch_size "${BATCH_SIZE}" \
  --max_len "${MAX_LEN}" \
  --max_steps "${MAX_STEPS}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --save_interval "${SAVE_INTERVAL}" \
  --embed_dim "${EMBED_DIM}" \
  --depth "${DEPTH}" \
  --heads "${HEADS}" \
  --dropout "${DROPOUT}" \
  --lr "${LR:-2e-4}" \
  --weight_decay "${WEIGHT_DECAY:-0.05}" \
  --warmup_steps "${WARMUP_STEPS:-100}" \
  --accum_steps "${ACCUM_STEPS}" \
  --max_eval_batches "${MAX_EVAL_BATCHES:-10}" \
  "${GRAPH_ARGS[@]}" \
  --results_path "${RESULTS_PATH}"

if [[ "${RUN_BENCHMARKS:-1}" == "1" ]]; then
  CKPT_PATH="${BENCH_CKPT_PATH:-$(ls -1t "${ROOT_DIR}"/checkpoints/"${CKPT_PREFIX}"_final_step_*.pt "${ROOT_DIR}"/checkpoints/"${CKPT_PREFIX}"_step_*.pt 2>/dev/null | head -n1 || true)}"
  if [[ -z "${CKPT_PATH}" ]]; then
    echo "No checkpoint found for benchmark run; skipping."
    exit 0
  fi
  BENCH_DATA="${BENCH_LOCAL_DATA:-${ROOT_DIR}/data/worldtrace_sample.pkl}"
  BENCH_OUT="${BENCH_OUTPUT:-${ROOT_DIR}/cache/mini_benchmark_${JOB_ID}.json}"
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
    --probe_epochs "${BENCH_PROBE_EPOCHS:-4}" \
    --probe_lr "${BENCH_PROBE_LR:-2e-3}" \
    --probe_weight_decay "${BENCH_PROBE_WEIGHT_DECAY:-1e-4}" \
    --max_probe_points "${BENCH_MAX_PROBE_POINTS:-100000}" \
    --split_mode "${BENCH_SPLIT_MODE:-both}" \
    "${BENCH_GRAPH_ARGS[@]}"
fi

