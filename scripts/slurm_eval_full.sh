#!/usr/bin/env bash
# Run a full evaluation suite on a checkpoint.

#SBATCH --job-name=trajfm-eval-full
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-eval-full-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FALLBACK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${SLURM_SUBMIT_DIR:-${FALLBACK_ROOT}}"

if [[ ! -f "${ROOT_DIR}/scripts/run_benchmarks.py" ]]; then
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

CKPT_GLOB="${CKPT_GLOB:-${ROOT_DIR}/checkpoints/hmt_*_final_step_*.pt ${ROOT_DIR}/checkpoints/hmt_*_step_*.pt}"
CKPT_PATH="${CKPT_PATH:-$(ls -1t ${CKPT_GLOB} 2>/dev/null | head -n1 || true)}"
if [[ -z "${CKPT_PATH}" ]]; then
  echo "ERROR: no checkpoint found." >&2
  exit 1
fi

LOCAL_DATA="${LOCAL_DATA:-${ROOT_DIR}/data/worldtrace_sample.pkl}"
MACRO_DATA="${MACRO_DATA:-${ROOT_DIR}/data/hdx/movement-distribution/processed/movement_distribution_12m_monthly.npz}"
CZ_CSV="${CZ_CSV:-${ROOT_DIR}/data/hdx/commuting-zones/data-for-good-at-meta-commuting-zones-march-2023.csv}"
JOB_ID="${SLURM_JOB_ID}"

python scripts/run_benchmarks.py \
  --checkpoint "${CKPT_PATH}" \
  --local_data "${LOCAL_DATA}" \
  --output "${ROOT_DIR}/cache/benchmark_${JOB_ID}.json" \
  --batch_size "${BATCH_SIZE:-32}" \
  --max_len "${MAX_LEN:-200}" \
  --probe_epochs "${PROBE_EPOCHS:-6}" \
  --probe_lr "${PROBE_LR:-2e-3}" \
  --probe_weight_decay "${PROBE_WEIGHT_DECAY:-1e-4}" \
  --max_probe_points "${MAX_PROBE_POINTS:-200000}" \
  --split_mode "${SPLIT_MODE:-both}"

python scripts/run_unitraj_eval.py \
  --checkpoint "${CKPT_PATH}" \
  --local_data "${LOCAL_DATA}" \
  --split_mode "${SPLIT_MODE:-both}" \
  --task both \
  --exclude_unknown \
  --use_regression \
  --output "${ROOT_DIR}/cache/unitraj_eval_${JOB_ID}.json"

python scripts/run_unitraj_eval.py \
  --checkpoint "${CKPT_PATH}" \
  --local_data "${LOCAL_DATA}" \
  --split_mode "${SPLIT_MODE:-both}" \
  --task both \
  --exclude_unknown \
  --use_regression \
  --coord_noise_std_m 30 \
  --input_drop_ratio 0.2 \
  --output "${ROOT_DIR}/cache/unitraj_eval_robust_${JOB_ID}.json"

python scripts/run_data_efficiency.py \
  --checkpoint "${CKPT_PATH}" \
  --local_data "${LOCAL_DATA}" \
  --fractions 0.05 0.1 0.2 0.5 1.0 \
  --output "${ROOT_DIR}/cache/unitraj_data_efficiency_${JOB_ID}.json"

if [[ -f "${MACRO_DATA}" ]]; then
  python scripts/run_macro_eval.py \
    --checkpoint "${CKPT_PATH}" \
    --macro_data "${MACRO_DATA}" \
    --batch_size "${MACRO_BATCH_SIZE:-512}" \
    --output "${ROOT_DIR}/cache/macro_eval_${JOB_ID}.json"
fi

if [[ -f "${CZ_CSV}" ]]; then
  python scripts/run_commuting_zone_probe.py \
    --checkpoint "${CKPT_PATH}" \
    --local_data "${LOCAL_DATA}" \
    --cz_csv "${CZ_CSV}" \
    --output "${ROOT_DIR}/cache/commuting_zone_probe_${JOB_ID}.json"
fi

if [[ -n "${TRANSFER_DATASETS:-}" ]]; then
  python scripts/run_transfer_suite.py \
    --checkpoint "${CKPT_PATH}" \
    --datasets ${TRANSFER_DATASETS} \
    --output "${ROOT_DIR}/cache/unitraj_transfer_suite_${JOB_ID}.json"
fi

python scripts/model_stats.py \
  --checkpoint "${CKPT_PATH}" \
  --batch_size "${STAT_BATCH_SIZE:-16}" \
  --max_len "${MAX_LEN:-200}" \
  --device "cuda"
