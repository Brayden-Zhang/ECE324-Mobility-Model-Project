#!/usr/bin/env bash
#SBATCH --job-name=geosight
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_l40s_b1,gpubase_l40s_b2,gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=slurm-geosight-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FALLBACK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${SLURM_SUBMIT_DIR:-${FALLBACK_ROOT}}"
[[ -f "${ROOT_DIR}/train_geosight.py" ]] || ROOT_DIR="${FALLBACK_ROOT}"
cd "${ROOT_DIR}"

source "${ROOT_DIR}/.venv/bin/activate"

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

python train_geosight.py \
  --num_samples "${NUM_SAMPLES:-50000}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --max_len "${MAX_LEN:-64}" \
  --num_images "${NUM_IMAGES:-4}" \
  --embed_dim "${EMBED_DIM:-256}" \
  --vision_dim "${VISION_DIM:-512}" \
  --depth "${DEPTH:-6}" \
  --heads "${HEADS:-8}" \
  --dropout 0.1 \
  --mask_ratio 0.3 \
  --lr 1e-4 \
  --weight_decay 0.05 \
  --warmup_steps 300 \
  --max_steps "${MAX_STEPS:-5000}" \
  --eval_interval 200 \
  --save_interval 1000 \
  --grad_clip 1.0 \
  --token_weight 1.0 \
  --contrastive_weight 1.0 \
  --matching_weight 0.5 \
  --flow_weight 0.5 \
  --seed 42 \
  --ckpt_dir "${ROOT_DIR}/checkpoints/geosight" \
  --amp
