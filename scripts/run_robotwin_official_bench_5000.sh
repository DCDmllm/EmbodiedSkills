#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
export PYTHONPATH="/mnt/wangwai/tmp_pytorch3d_target:${PROJECT_ROOT}/src:${WORKSPACE_ROOT}/pi0.5/src${PYTHONPATH:+:$PYTHONPATH}"

RUN_ID="${CLAWVLA_ROBOTWIN_RUN_ID:-robotwin_official_qwen3vl8b_pi05_demo_clean_5000}"
WORKERS="${CLAWVLA_ROBOTWIN_WORKERS:-1}"
VLLM_GPUS="${CLAWVLA_ROBOTWIN_VLLM_GPUS:-0,1}"
OPENPI_GPUS="${CLAWVLA_ROBOTWIN_OPENPI_GPUS:-6}"
ROBOTWIN_GPUS="${CLAWVLA_ROBOTWIN_ENV_GPUS:-7}"
SEED_CHECK_GPUS="${CLAWVLA_ROBOTWIN_SEED_CHECK_GPUS:-${ROBOTWIN_GPUS}}"
VALID_SEED_CACHE="${CLAWVLA_ROBOTWIN_VALID_SEED_CACHE:-/mnt/wangwai/vla/clawvla/runs/eval/robotwin_valid_seeds_demo_clean_seed0/valid_seeds}"
VLLM_PORT="${CLAWVLA_ROBOTWIN_VLLM_PORT:-18080}"
OPENPI_PORT_BASE="${CLAWVLA_ROBOTWIN_OPENPI_PORT_BASE:-9365}"

if [[ ! -d "${VALID_SEED_CACHE}" ]]; then
  echo "RoboTwin valid seed cache not found: ${VALID_SEED_CACHE}" >&2
  echo "Generate it with ${PROJECT_ROOT}/scripts/precompute_robotwin_valid_seeds.sh or set CLAWVLA_ROBOTWIN_VALID_SEED_CACHE." >&2
  exit 2
fi

/mnt/wangwai/miniconda3/envs/robotwin-py312/bin/python -m clawvla.scripts.robotwin_official_bench_eval \
  --run-id "${RUN_ID}" \
  --episodes-per-task 100 \
  --max-steps 200 \
  --workers "${WORKERS}" \
  --vllm-gpus "${VLLM_GPUS}" \
  --vllm-port "${VLLM_PORT}" \
  --tensor-parallel-size 2 \
  --openpi-gpus "${OPENPI_GPUS}" \
  --openpi-port-base "${OPENPI_PORT_BASE}" \
  --robotwin-gpus "${ROBOTWIN_GPUS}" \
  --seed-check-gpus "${SEED_CHECK_GPUS}" \
  --valid-seed-cache-dir "${VALID_SEED_CACHE}" \
  --keep-agent-logs none \
  --keep-result-json none \
  --keep-artifacts none \
  "$@"
