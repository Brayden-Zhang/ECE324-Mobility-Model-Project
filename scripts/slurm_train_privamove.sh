#!/usr/bin/env bash
#SBATCH --job-name=privamove
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_l40s_b1,gpubase_l40s_b2,gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=slurm-privamove-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FALLBACK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${SLURM_SUBMIT_DIR:-${FALLBACK_ROOT}}"
[[ -f "${ROOT_DIR}/train_privamove.py" ]] || ROOT_DIR="${FALLBACK_ROOT}"
cd "${ROOT_DIR}"

source "${ROOT_DIR}/.venv/bin/activate"

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

python train_privamove.py \
  --num_cities "${NUM_CITIES:-5}" \
  --samples_per_city "${SAMPLES_PER_CITY:-5000}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --max_len "${MAX_LEN:-64}" \
  --embed_dim "${EMBED_DIM:-256}" \
  --depth "${DEPTH:-4}" \
  --heads "${HEADS:-8}" \
  --moe_top_k 2 \
  --mask_ratio 0.3 \
  --lr 1e-4 \
  --gen_lr 5e-4 \
  --weight_decay 0.05 \
  --grad_clip 1.0 \
  --rounds "${ROUNDS:-20}" \
  --local_steps "${LOCAL_STEPS:-100}" \
  --dp_epsilon "${DP_EPSILON:-8.0}" \
  --dp_delta 1e-5 \
  --seed 42 \
  --ckpt_dir "${ROOT_DIR}/checkpoints/privamove" \
  --benchmark_privacy
