#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
export PYTHONPATH="/mnt/wangwai/tmp_pytorch3d_target:${PROJECT_ROOT}/src:${WORKSPACE_ROOT}/pi0.5/src${PYTHONPATH:+:$PYTHONPATH}"

RUN_ID="${CLAWVLA_ROBOTWIN_SEED_RUN_ID:-robotwin_valid_seeds_demo_clean_seed0}"
GPUS="${CLAWVLA_ROBOTWIN_SEED_GPUS:-0,1,2,3,4,5,6,7}"
LANES_PER_GPU="${CLAWVLA_ROBOTWIN_SEED_LANES_PER_GPU:-1}"
WORKERS="${CLAWVLA_ROBOTWIN_SEED_WORKERS:-}"

CMD=(
  /mnt/wangwai/miniconda3/envs/robotwin-py312/bin/python
  -m clawvla.scripts.robotwin_precompute_valid_seeds
  --run-id "${RUN_ID}"
  --task-config demo_clean
  --target-valid 100
  --seed 0
  --gpus "${GPUS}"
  --lanes-per-gpu "${LANES_PER_GPU}"
  --resume
)

if [[ -n "${WORKERS}" ]]; then
  CMD+=(--workers "${WORKERS}")
fi

"${CMD[@]}" "$@"
