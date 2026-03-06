#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FALLBACK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${SLURM_SUBMIT_DIR:-${FALLBACK_ROOT}}"
if [[ ! -f "${ROOT_DIR}/train_hmt.py" ]]; then
  ROOT_DIR="${FALLBACK_ROOT}"
fi
cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"
else
  echo "ERROR: .venv not found at ${ROOT_DIR}/.venv" >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set." >&2
  exit 1
fi

python - <<'PY'
from huggingface_hub import snapshot_download
import os

repo_id = "OpenTrace/WorldTrace"
local_dir = "data/worldtrace_full"
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=["*Trajectory.zip"],
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    token=os.environ.get("HF_TOKEN"),
)
print(f"Downloaded into {local_dir}")
PY

WT_ZIP="data/worldtrace_full/Trajectory.zip"
if [[ ! -f "${WT_ZIP}" ]]; then
  echo "ERROR: expected ${WT_ZIP} but it does not exist." >&2
  exit 1
fi

PYTHONPATH=src python -m route_rangers.cli.build_h3_vocab \
  --data_mode hf_zip \
  --hf_name "OpenTrace/WorldTrace" \
  --worldtrace_file "Trajectory.zip" \
  --worldtrace_local_path "${WT_ZIP}" \
  --res0 9 --res1 7 --res2 5 \
  --max_cells_l0 16384 \
  --max_cells_l1 4096 \
  --max_cells_l2 1024 \
  --output "data/h3_vocab_worldtrace_full_big.json"
