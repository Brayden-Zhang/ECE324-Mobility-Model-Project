#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONUNBUFFERED=1

default_checkpoint() {
  local ckpt
  ckpt="$(ls -1t "${ROOT_DIR}"/checkpoints/*.pt 2>/dev/null | head -n1 || true)"
  if [[ -n "${ckpt}" ]]; then
    echo "${ckpt}"
    return
  fi
  echo "${ROOT_DIR}/checkpoints/hmt_step_15000.pt"
}

default_local_data() {
  for p in \
    "${ROOT_DIR}/data/worldtrace_sample.pkl" \
    "${ROOT_DIR}/data/worldtrace_sample.parquet" \
    "${ROOT_DIR}/data/worldtrace_sample_nyc.pkl" \
    "${ROOT_DIR}/data/worldtrace/train.parquet"; do
    if [[ -f "${p}" ]]; then
      echo "${p}"
      return
    fi
  done
  echo "${ROOT_DIR}/data/worldtrace_sample.pkl"
}

CHECKPOINT="${CHECKPOINT:-$(default_checkpoint)}"
LOCAL_DATA="${LOCAL_DATA:-$(default_local_data)}"
MAX_LEN="${MAX_LEN:-200}"
SEEDS="${SEEDS:-42,43,44}"
JOB_TAG="${JOB_TAG:-latest}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
  echo "Hint: set CHECKPOINT=/abs/path/to/your_checkpoint.pt" >&2
  echo "Available checkpoints:" >&2
  ls -1 "${ROOT_DIR}"/checkpoints/*.pt 2>/dev/null || echo "  (none)" >&2
  exit 1
fi
if [[ ! -f "${LOCAL_DATA}" ]]; then
  echo "ERROR: local_data not found: ${LOCAL_DATA}" >&2
  echo "Hint: set LOCAL_DATA=/abs/path/to/worldtrace_sample.pkl" >&2
  echo "You can also download parquet with:" >&2
  echo "  PYTHONPATH=src python -m route_rangers.cli.download_worldtrace --output data/worldtrace --max_samples 200000" >&2
  exit 1
fi

mkdir -p cache

PYTHONPATH=src python -m route_rangers.cli.run_benchmarks \
  --checkpoint "${CHECKPOINT}" \
  --local_data "${LOCAL_DATA}" \
  --max_len "${MAX_LEN}" \
  --split_mode both \
  --output "cache/benchmark_${JOB_TAG}.json"

PYTHONPATH=src python -m route_rangers.cli.run_proposal_baselines \
  --local_data "${LOCAL_DATA}" \
  --max_len "${MAX_LEN}" \
  --split_mode both \
  --output "cache/proposal_baselines_${JOB_TAG}.json"

PYTHONPATH=src python -m route_rangers.cli.run_length_sensitivity \
  --checkpoint "${CHECKPOINT}" \
  --local_data "${LOCAL_DATA}" \
  --max_len "${MAX_LEN}" \
  --seeds "${SEEDS}" \
  --ci_method seed \
  --output "cache/length_sensitivity_${JOB_TAG}.json"

PYTHONPATH=src python -m route_rangers.cli.run_length_uncertainty \
  --checkpoint "${CHECKPOINT}" \
  --local_data "${LOCAL_DATA}" \
  --max_len "${MAX_LEN}" \
  --output "cache/length_uncertainty_${JOB_TAG}.json"

PYTHONPATH=src python -m route_rangers.cli.run_embedding_length_probe \
  --checkpoint "${CHECKPOINT}" \
  --local_data "${LOCAL_DATA}" \
  --max_len "${MAX_LEN}" \
  --split_mode both \
  --output "cache/embedding_length_probe_${JOB_TAG}.json"

PYTHONPATH=src python -m route_rangers.cli.run_od_matrix_eval \
  --checkpoint "${CHECKPOINT}" \
  --local_data "${LOCAL_DATA}" \
  --max_len "${MAX_LEN}" \
  --split_mode both \
  --output "cache/od_eval_${JOB_TAG}.json"

PYTHONPATH=src python -m route_rangers.cli.collect_results \
  --cache_dir cache \
  --foundation_doc docs/foundation_evals.md \
  --paper_doc docs/neurips_paper_draft.md

echo "paper suite complete"
