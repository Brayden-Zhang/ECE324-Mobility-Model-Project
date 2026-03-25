#!/usr/bin/env bash
# Master NeurIPS experiment pipeline.
# Submits all training and evaluation jobs with proper dependencies.
#
# Usage:
#   bash scripts/slurm_neurips_master.sh          # submit everything
#   DRY_RUN=1 bash scripts/slurm_neurips_master.sh  # print commands without submitting
#
# This script orchestrates:
# 1. Resume stage-1 training (200K steps)
# 2. Run full eval suite on best checkpoint
# 3. Run all downstream tasks
# 4. Run ablation evals
# 5. UniTraj comparison with regression-based metrics
# 6. Collect and update results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

DRY_RUN="${DRY_RUN:-0}"

submit() {
    local desc="$1"
    shift
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY RUN] $desc: sbatch $*" >&2
        echo "DRY_$RANDOM"
    else
        local out
        out=$(sbatch "$@" 2>&1)
        local jid
        jid=$(echo "$out" | grep -oP '\d+' | tail -1)
        echo "[SUBMITTED] $desc: job $jid" >&2
        echo "$jid"
    fi
}

submit_dep() {
    local desc="$1"
    local dep_jid="$2"
    shift 2
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY RUN] $desc (after $dep_jid): sbatch --dependency=afterok:$dep_jid $*" >&2
        echo "DRY_$RANDOM"
    else
        local out
        out=$(sbatch --dependency=afterok:"${dep_jid}" "$@" 2>&1)
        local jid
        jid=$(echo "$out" | grep -oP '\d+' | tail -1)
        echo "[SUBMITTED] $desc (after $dep_jid): job $jid" >&2
        echo "$jid"
    fi
}

echo "=========================================="
echo "  NeurIPS Master Pipeline"
echo "  Root: ${ROOT_DIR}"
echo "  DRY_RUN: ${DRY_RUN}"
echo "=========================================="

# ── 1. Resume stage-1 training ──────────────────────────────────────────
echo ""
echo "── Stage 1: Resume pre-training ──"

LATEST_CKPT=$(ls -1t "${ROOT_DIR}"/checkpoints/hmt_nohash_full_step_*.pt 2>/dev/null | head -n1 || true)
if [[ -z "${LATEST_CKPT}" ]]; then
    echo "WARNING: no nohash_full checkpoint found; skipping resume"
    TRAIN_JID=""
else
    echo "Resuming from: ${LATEST_CKPT}"
    TRAIN_JID=$(submit "Stage-1 resume training" \
        scripts/slurm_train_hmt_nohash_full.sh \
        --export=ALL,RESUME="${LATEST_CKPT}",RESUME_OPTIMIZER=1,MAX_STEPS=200000,SAVE_INTERVAL=10000,EVAL_INTERVAL=5000)
fi

# ── 2. Stage-2 macro head training ──────────────────────────────────────
echo ""
echo "── Stage 2: Macro distribution head ──"

STAGE2_CKPT=$(ls -1t "${ROOT_DIR}"/checkpoints/hmt_stage2_*_step_*.pt 2>/dev/null | head -n1 || true)
if [[ -n "${TRAIN_JID}" ]] && [[ "${TRAIN_JID}" != DRY_* ]]; then
    STAGE2_JID=$(submit_dep "Stage-2 macro training" "${TRAIN_JID}" \
        scripts/slurm_train_hmt_stage2.sh \
        --export=ALL,MAX_STEPS=50000,SAVE_INTERVAL=10000,EMBED_DIM=256,MACRO_MIX_PROB=0.5)
else
    # Run immediately with existing checkpoint
    STAGE2_JID=$(submit "Stage-2 macro training (existing ckpt)" \
        scripts/slurm_train_hmt_stage2.sh \
        --export=ALL,MAX_STEPS=50000,SAVE_INTERVAL=10000,EMBED_DIM=256,MACRO_MIX_PROB=0.5)
fi

# ── 3. Full evaluation suite ────────────────────────────────────────────
echo ""
echo "── Full Evaluation Suite ──"

# Determine best checkpoint for eval
EVAL_CKPT="${LATEST_CKPT:-${ROOT_DIR}/checkpoints/hmt_nohash_full_step_25000.pt}"

# 3a. Core benchmarks (reconstruction, probes)
BENCH_JID=$(submit "Core benchmarks" \
    scripts/slurm_eval_full.sh \
    --export=ALL,CKPT_PATH="${EVAL_CKPT}",SPLIT_MODE=both,TRANSFER_DATASETS="${ROOT_DIR}/data/worldtrace_sample.pkl ${ROOT_DIR}/data/worldtrace_sample_nyc.pkl")

# 3b. UniTraj eval with regression (fair comparison)
UNITRAJ_REG_JID=$(submit "UniTraj eval (regression)" \
    scripts/slurm_unitraj_eval_regression.sh \
    --export=ALL,CKPT_PATH="${EVAL_CKPT}")

# 3c. Length sensitivity
LENGTH_JID=$(submit "Length sensitivity" \
    scripts/slurm_length_sensitivity.sh \
    --export=ALL,CKPT_PATH="${EVAL_CKPT}")

# 3d. Invariance suite
INVAR_JID=$(submit "Invariance suite" \
    scripts/slurm_invariance_suite.sh \
    --export=ALL,CKPT_PATH="${EVAL_CKPT}")

# 3e. Embedding retrieval
RETRIEV_JID=$(submit "Embedding retrieval" \
    scripts/slurm_embedding_retrieval.sh \
    --export=ALL,CKPT_PATH="${EVAL_CKPT}")

# 3f. Change detection
CHANGE_JID=$(submit "Change detection" \
    scripts/slurm_change_detection.sh \
    --export=ALL,CKPT_PATH="${EVAL_CKPT}")

# 3g. Reverse order stress
REVERSE_JID=$(submit "Reverse order stress" \
    scripts/slurm_reverse_order.sh \
    --export=ALL,CKPT_PATH="${EVAL_CKPT}")

# ── 4. Downstream tasks ─────────────────────────────────────────────────
echo ""
echo "── Downstream Tasks ──"

DOWNSTREAM_JID=$(submit "Downstream tasks" \
    scripts/slurm_downstream_tasks.sh \
    --export=ALL,CKPT_PATH="${EVAL_CKPT}")

# ── 5. Ablation evals ───────────────────────────────────────────────────
echo ""
echo "── Ablation Evaluations ──"

for ABLATE_CKPT in \
    "${ROOT_DIR}/checkpoints/hmt_ablate_nograph_step_15000.pt" \
    "${ROOT_DIR}/checkpoints/hmt_ablate_noflow_step_15000.pt" \
    "${ROOT_DIR}/checkpoints/hmt_ablate_notrip_step_15000.pt" \
    "${ROOT_DIR}/checkpoints/hmt_ablate_lenweight_step_15000.pt"; do
    if [[ -f "${ABLATE_CKPT}" ]]; then
        NAME=$(basename "${ABLATE_CKPT}" .pt | sed 's/hmt_//')
        submit "Ablation eval: ${NAME}" \
            scripts/slurm_benchmark_latest.sh \
            --export=ALL,CKPT_PATH="${ABLATE_CKPT}"
    fi
done

# ── 6. External UniTraj baseline ────────────────────────────────────────
echo ""
echo "── External UniTraj Baseline ──"

UNITRAJ_EXT_CKPT="${ROOT_DIR}/checkpoints/unitraj.pt"
if [[ -f "${UNITRAJ_EXT_CKPT}" ]]; then
    submit "External UniTraj eval" \
        scripts/slurm_unitraj_eval_external.sh \
        --export=ALL
else
    echo "WARNING: External UniTraj checkpoint not found at ${UNITRAJ_EXT_CKPT}"
fi

# ── 7. Collect results (runs after all evals) ───────────────────────────
echo ""
echo "── Post-processing ──"
echo "After all jobs complete, run:"
echo "  PYTHONPATH=src python -m route_rangers.cli.collect_results"
echo ""
echo "=========================================="
echo "  Pipeline submitted!"
echo "=========================================="
