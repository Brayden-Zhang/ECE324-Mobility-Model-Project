#!/usr/bin/env bash
# Submit a staged "elite" WorldTrace pipeline:
# 1. Resume full WorldTrace pretraining from the strongest local no-hash checkpoint.
# 2. Run stage-2 macro/downstream multitask fine-tuning from that checkpoint.
# 3. Launch the evaluation/downstream suite and the UniTraj-compatible baseline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

DRY_RUN="${DRY_RUN:-0}"

submit_job() {
  local desc="$1"
  shift

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY RUN] ${desc}: sbatch $*" >&2
    echo "DRY_${RANDOM}"
    return 0
  fi

  local out
  out=$(sbatch "$@" 2>&1)
  local jid
  jid=$(awk '/Submitted batch job/ {print $4}' <<<"${out}" | tail -n1)
  if [[ -z "${jid}" ]]; then
    echo "ERROR: failed to submit ${desc}" >&2
    echo "${out}" >&2
    return 1
  fi
  echo "[SUBMITTED] ${desc}: job ${jid}" >&2
  echo "${jid}"
}

submit_afterok() {
  local desc="$1"
  local dep_jid="$2"
  shift 2

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY RUN] ${desc} (after ${dep_jid}): sbatch --dependency=afterok:${dep_jid} $*" >&2
    echo "DRY_${RANDOM}"
    return 0
  fi

  local out
  out=$(sbatch --dependency=afterok:"${dep_jid}" "$@" 2>&1)
  local jid
  jid=$(awk '/Submitted batch job/ {print $4}' <<<"${out}" | tail -n1)
  if [[ -z "${jid}" ]]; then
    echo "ERROR: failed to submit ${desc}" >&2
    echo "${out}" >&2
    return 1
  fi
  echo "[SUBMITTED] ${desc} (after ${dep_jid}): job ${jid}" >&2
  echo "${jid}"
}

build_export() {
  local value="ALL"
  local kv
  for kv in "$@"; do
    if [[ -n "${kv}" ]]; then
      value="${value},${kv}"
    fi
  done
  printf '%s' "${value}"
}

DEFAULT_WORLDTRACE_LOCAL_PATH=""
if [[ -f "${ROOT_DIR}/data/worldtrace_full/Trajectory.zip" ]]; then
  DEFAULT_WORLDTRACE_LOCAL_PATH="${ROOT_DIR}/data/worldtrace_full/Trajectory.zip"
fi

WORLDTRACE_LOCAL_PATH="${WORLDTRACE_LOCAL_PATH:-${DEFAULT_WORLDTRACE_LOCAL_PATH}}"
H3_VOCAB_DEFAULT="${ROOT_DIR}/data/h3_vocab_worldtrace_full.json"
MACRO_DATA="${MACRO_DATA:-${ROOT_DIR}/data/hdx/movement-distribution/processed/movement_distribution_12m_monthly.npz}"
LOCAL_DATA="${LOCAL_DATA:-${ROOT_DIR}/data/worldtrace_sample.pkl}"
POI_DATA="${POI_DATA:-${ROOT_DIR}/data/poi_mobility_sample.pkl}"
UNITRAJ_CKPT="${UNITRAJ_CKPT:-${ROOT_DIR}/checkpoints/unitraj.pt}"

BASE_CKPT="${BASE_CKPT:-$(ls -1t "${ROOT_DIR}"/checkpoints/hmt_nohash_full*_final_step_*.pt "${ROOT_DIR}"/checkpoints/hmt_nohash_full*_step_*.pt 2>/dev/null | head -n1 || true)}"
if [[ -z "${BASE_CKPT}" || ! -f "${BASE_CKPT}" ]]; then
  echo "ERROR: no base HMT full-data checkpoint found. Set BASE_CKPT explicitly." >&2
  exit 1
fi
if [[ -z "${WORLDTRACE_LOCAL_PATH}" || ! -f "${WORLDTRACE_LOCAL_PATH}" ]]; then
  echo "ERROR: WorldTrace local zip not found. Set WORLDTRACE_LOCAL_PATH explicitly." >&2
  exit 1
fi
if [[ ! -f "${H3_VOCAB_DEFAULT}" && -z "${H3_VOCAB:-}" ]]; then
  echo "ERROR: H3 vocab not found at ${H3_VOCAB_DEFAULT}" >&2
  exit 1
fi
if [[ ! -f "${MACRO_DATA}" ]]; then
  echo "ERROR: macro dataset not found at ${MACRO_DATA}" >&2
  exit 1
fi

eval "$(
  .venv/bin/python - "${BASE_CKPT}" "${H3_VOCAB_DEFAULT}" "${WORLDTRACE_LOCAL_PATH}" <<'PY'
import shlex
import sys
import torch

path, default_vocab, default_worldtrace = sys.argv[1:4]
ckpt = torch.load(path, map_location="cpu")
args = ckpt.get("args", {})

def quote(name, value):
    print(f"{name}={shlex.quote(str(value))}")

def as_flag(value):
    return "1" if bool(value) else "0"

quote("RESUME_STEP", ckpt.get("step", 0))
quote("RESUME_EMBED_DIM", args.get("embed_dim", 256))
quote("RESUME_DEPTH", args.get("depth", 8))
quote("RESUME_HEADS", args.get("heads", 8))
quote("RESUME_BATCH_SIZE", args.get("batch_size", 64))
quote("RESUME_ACCUM_STEPS", args.get("accum_steps", 2))
quote("RESUME_GRAPH_LAYERS", args.get("graph_layers", 2))
quote("RESUME_GRAPH_KNN", args.get("graph_knn", 8))
quote("RESUME_GRAPH_TEMPORAL_WINDOW", args.get("graph_temporal_window", 2))
quote("RESUME_STEP_ATTENTION_WINDOW", args.get("step_attention_window", 0))
quote("RESUME_SPACE_TIME_ENCODER", as_flag(args.get("space_time_encoder", False)))
quote("RESUME_SPACE_TIME_FREQS", args.get("space_time_freqs", 6))
quote("RESUME_H3_VOCAB", args.get("h3_vocab") or default_vocab)
quote("RESUME_WORLDTRACE_LOCAL_PATH", args.get("worldtrace_local_path") or default_worldtrace)
PY
)"

H3_VOCAB="${H3_VOCAB:-${RESUME_H3_VOCAB}}"
WORLDTRACE_LOCAL_PATH="${WORLDTRACE_LOCAL_PATH:-${RESUME_WORLDTRACE_LOCAL_PATH}}"

STAGE1_PREFIX="${STAGE1_PREFIX:-hmt_elite_worldtrace_stage1}"
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-60000}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-${RESUME_BATCH_SIZE}}"
STAGE1_ACCUM_STEPS="${STAGE1_ACCUM_STEPS:-${RESUME_ACCUM_STEPS}}"
STAGE1_EMBED_DIM="${STAGE1_EMBED_DIM:-${RESUME_EMBED_DIM}}"
STAGE1_DEPTH="${STAGE1_DEPTH:-${RESUME_DEPTH}}"
STAGE1_HEADS="${STAGE1_HEADS:-${RESUME_HEADS}}"
STAGE1_GRAPH_LAYERS="${STAGE1_GRAPH_LAYERS:-${RESUME_GRAPH_LAYERS}}"
STAGE1_GRAPH_KNN="${STAGE1_GRAPH_KNN:-${RESUME_GRAPH_KNN}}"
STAGE1_GRAPH_TEMPORAL_WINDOW="${STAGE1_GRAPH_TEMPORAL_WINDOW:-${RESUME_GRAPH_TEMPORAL_WINDOW}}"
STAGE1_STEP_ATTENTION_WINDOW="${STAGE1_STEP_ATTENTION_WINDOW:-${RESUME_STEP_ATTENTION_WINDOW}}"
STAGE1_SPACE_TIME_ENCODER="${STAGE1_SPACE_TIME_ENCODER:-${RESUME_SPACE_TIME_ENCODER}}"
STAGE1_SPACE_TIME_FREQS="${STAGE1_SPACE_TIME_FREQS:-${RESUME_SPACE_TIME_FREQS}}"
STAGE1_FINAL_CKPT="${ROOT_DIR}/checkpoints/${STAGE1_PREFIX}_final_step_${STAGE1_MAX_STEPS}.pt"

STAGE2_PREFIX="${STAGE2_PREFIX:-hmt_elite_worldtrace_stage2}"
STAGE2_EXTRA_STEPS="${STAGE2_EXTRA_STEPS:-15000}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-$((STAGE1_MAX_STEPS + STAGE2_EXTRA_STEPS))}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-${STAGE1_BATCH_SIZE}}"
STAGE2_ACCUM_STEPS="${STAGE2_ACCUM_STEPS:-${STAGE1_ACCUM_STEPS}}"
STAGE2_EMBED_DIM="${STAGE2_EMBED_DIM:-${STAGE1_EMBED_DIM}}"
STAGE2_DEPTH="${STAGE2_DEPTH:-${STAGE1_DEPTH}}"
STAGE2_HEADS="${STAGE2_HEADS:-${STAGE1_HEADS}}"
STAGE2_GRAPH_LAYERS="${STAGE2_GRAPH_LAYERS:-${STAGE1_GRAPH_LAYERS}}"
STAGE2_GRAPH_KNN="${STAGE2_GRAPH_KNN:-${STAGE1_GRAPH_KNN}}"
STAGE2_GRAPH_TEMPORAL_WINDOW="${STAGE2_GRAPH_TEMPORAL_WINDOW:-${STAGE1_GRAPH_TEMPORAL_WINDOW}}"
STAGE2_SPACE_TIME_ENCODER="${STAGE2_SPACE_TIME_ENCODER:-${STAGE1_SPACE_TIME_ENCODER}}"
STAGE2_SPACE_TIME_FREQS="${STAGE2_SPACE_TIME_FREQS:-${STAGE1_SPACE_TIME_FREQS}}"
STAGE2_FINAL_CKPT="${ROOT_DIR}/checkpoints/${STAGE2_PREFIX}_final_step_${STAGE2_MAX_STEPS}.pt"

echo "=========================================="
echo "Elite WorldTrace Pipeline"
echo "Root: ${ROOT_DIR}"
echo "DRY_RUN: ${DRY_RUN}"
echo "Base checkpoint: ${BASE_CKPT}"
echo "WorldTrace zip: ${WORLDTRACE_LOCAL_PATH}"
echo "Stage1 final: ${STAGE1_FINAL_CKPT}"
echo "Stage2 final: ${STAGE2_FINAL_CKPT}"
echo "=========================================="

stage1_export="$(build_export \
  "WORLDTRACE_LOCAL_PATH=${WORLDTRACE_LOCAL_PATH}" \
  "H3_VOCAB=${H3_VOCAB}" \
  "RESUME=${BASE_CKPT}" \
  "RESUME_OPTIMIZER=${RESUME_OPTIMIZER:-1}" \
  "CKPT_PREFIX=${STAGE1_PREFIX}" \
  "MAX_STEPS=${STAGE1_MAX_STEPS}" \
  "BATCH_SIZE=${STAGE1_BATCH_SIZE}" \
  "ACCUM_STEPS=${STAGE1_ACCUM_STEPS}" \
  "EMBED_DIM=${STAGE1_EMBED_DIM}" \
  "DEPTH=${STAGE1_DEPTH}" \
  "HEADS=${STAGE1_HEADS}" \
  "GRAPH_LAYERS=${STAGE1_GRAPH_LAYERS}" \
  "GRAPH_KNN=${STAGE1_GRAPH_KNN}" \
  "GRAPH_TEMPORAL_WINDOW=${STAGE1_GRAPH_TEMPORAL_WINDOW}" \
  "STEP_ATTENTION_WINDOW=${STAGE1_STEP_ATTENTION_WINDOW}" \
  "SPACE_TIME_ENCODER=${STAGE1_SPACE_TIME_ENCODER}" \
  "SPACE_TIME_FREQS=${STAGE1_SPACE_TIME_FREQS}" \
  "EVAL_INTERVAL=${STAGE1_EVAL_INTERVAL:-2000}" \
  "SAVE_INTERVAL=${STAGE1_SAVE_INTERVAL:-5000}" \
  "MAX_LEN=${STAGE1_MAX_LEN:-200}" \
)"
stage1_jid="$(
  submit_job \
    "Stage-1 full WorldTrace continuation" \
    --export="${stage1_export}" \
    scripts/slurm_train_hmt_nohash_full.sh
)"

stage2_export="$(build_export \
  "WORLDTRACE_LOCAL_PATH=${WORLDTRACE_LOCAL_PATH}" \
  "H3_VOCAB=${H3_VOCAB}" \
  "RESUME=${STAGE1_FINAL_CKPT}" \
  "RESUME_OPTIMIZER=${STAGE2_RESUME_OPTIMIZER:-0}" \
  "CKPT_PREFIX=${STAGE2_PREFIX}" \
  "MAX_STEPS=${STAGE2_MAX_STEPS}" \
  "BATCH_SIZE=${STAGE2_BATCH_SIZE}" \
  "ACCUM_STEPS=${STAGE2_ACCUM_STEPS}" \
  "EMBED_DIM=${STAGE2_EMBED_DIM}" \
  "DEPTH=${STAGE2_DEPTH}" \
  "HEADS=${STAGE2_HEADS}" \
  "GRAPH_LAYERS=${STAGE2_GRAPH_LAYERS}" \
  "GRAPH_KNN=${STAGE2_GRAPH_KNN}" \
  "GRAPH_TEMPORAL_WINDOW=${STAGE2_GRAPH_TEMPORAL_WINDOW}" \
  "SPACE_TIME_ENCODER=${STAGE2_SPACE_TIME_ENCODER}" \
  "SPACE_TIME_FREQS=${STAGE2_SPACE_TIME_FREQS}" \
  "MACRO_DATA=${MACRO_DATA}" \
  "MACRO_MIX_PROB=${MACRO_MIX_PROB:-0.5}" \
  "MACRO_WEIGHT=${MACRO_WEIGHT:-1.0}" \
  "MACRO_BATCH_SIZE=${MACRO_BATCH_SIZE:-256}" \
  "EVAL_INTERVAL=${STAGE2_EVAL_INTERVAL:-2000}" \
  "SAVE_INTERVAL=${STAGE2_SAVE_INTERVAL:-5000}" \
  "MAX_LEN=${STAGE2_MAX_LEN:-200}" \
)"
stage2_jid="$(
  submit_afterok \
    "Stage-2 macro/downstream fine-tune" \
    "${stage1_jid}" \
    --export="${stage2_export}" \
    scripts/slurm_train_hmt_stage2.sh
)"

eval_export="$(build_export \
  "CKPT_PATH=${STAGE2_FINAL_CKPT}" \
  "LOCAL_DATA=${LOCAL_DATA}" \
  "MAX_LEN=${EVAL_MAX_LEN:-200}" \
  "BATCH_SIZE=${EVAL_BATCH_SIZE:-32}" \
)"
eval_jid="$(
  submit_afterok \
    "Full evaluation suite" \
    "${stage2_jid}" \
    --export="${eval_export}" \
    scripts/slurm_eval_full.sh
)"

downstream_jid="$(
  submit_afterok \
    "Downstream task suite" \
    "${stage2_jid}" \
    --export="${eval_export}" \
    scripts/slurm_downstream_tasks.sh
)"

length_jid="$(
  submit_afterok \
    "Length sensitivity" \
    "${stage2_jid}" \
    --export="${eval_export}" \
    scripts/slurm_length_sensitivity.sh
)"

invariance_jid="$(
  submit_afterok \
    "Invariance suite" \
    "${stage2_jid}" \
    --export="${eval_export}" \
    scripts/slurm_invariance_suite.sh
)"

retrieval_jid="$(
  submit_afterok \
    "Embedding retrieval" \
    "${stage2_jid}" \
    --export="${eval_export}" \
    scripts/slurm_embedding_retrieval.sh
)"

mobility_export="$(build_export \
  "CKPT_PATH=${STAGE2_FINAL_CKPT}" \
  "POI_DATA=${POI_DATA}" \
  "MAX_LEN=${EVAL_MAX_LEN:-200}" \
  "BATCH_SIZE=${EVAL_BATCH_SIZE:-32}" \
)"
mobility_jid="$(
  submit_afterok \
    "Mobility foundation eval (next-POI + cross-city)" \
    "${stage2_jid}" \
    --export="${mobility_export}" \
    scripts/slurm_mobility_foundation_eval.sh
)"

reverse_jid="$(
  submit_afterok \
    "Reverse-order stress test" \
    "${stage2_jid}" \
    --export="${eval_export}" \
    scripts/slurm_reverse_order.sh
)"

change_jid="$(
  submit_afterok \
    "Change-detection suite" \
    "${stage2_jid}" \
    --export="${eval_export}" \
    scripts/slurm_change_detection.sh
)"

unitraj_reg_jid="$(
  submit_afterok \
    "UniTraj-style regression eval" \
    "${stage2_jid}" \
    --export="${eval_export}" \
    scripts/slurm_unitraj_eval_regression.sh
)"

unitraj_ext_export="$(build_export \
  "DATA_PATH=${LOCAL_DATA}" \
  "CKPT_PATH=${UNITRAJ_CKPT}" \
)"
unitraj_ext_jid="$(
  submit_job \
    "External UniTraj baseline eval" \
    --export="${unitraj_ext_export}" \
    scripts/slurm_unitraj_eval_external.sh
)"

collect_dep="${eval_jid}:${downstream_jid}:${length_jid}:${invariance_jid}:${retrieval_jid}:${mobility_jid}:${reverse_jid}:${change_jid}:${unitraj_reg_jid}:${unitraj_ext_jid}"
collect_wrap="cd ${ROOT_DIR} && module load arrow/21.0.0 && source .venv/bin/activate && PYTHONPATH=src python -m route_rangers.cli.collect_results"
collect_jid="$(
  submit_afterok \
    "Collect results" \
    "${collect_dep}" \
    --time="${COLLECT_TIME:-00:15:00}" \
    --cpus-per-task="${COLLECT_CPUS:-1}" \
    --mem="${COLLECT_MEM:-4G}" \
    --wrap="${collect_wrap}"
)"

echo ""
echo "Submitted jobs:"
echo "  stage1:   ${stage1_jid}"
echo "  stage2:   ${stage2_jid}"
echo "  eval:     ${eval_jid}"
echo "  downstream: ${downstream_jid}"
echo "  length:   ${length_jid}"
echo "  invariance: ${invariance_jid}"
echo "  retrieval: ${retrieval_jid}"
echo "  mobility:  ${mobility_jid}"
echo "  reverse:  ${reverse_jid}"
echo "  change:   ${change_jid}"
echo "  unitraj_reg: ${unitraj_reg_jid}"
echo "  unitraj_ext: ${unitraj_ext_jid}"
echo "  collect:  ${collect_jid}"
