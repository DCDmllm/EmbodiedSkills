#!/usr/bin/env bash
set -euo pipefail

PROFILE="${CLAWVLA_RUN_PROFILE:-/mnt/linyutong/wangwai_mirror/vla/clawvla/configs/run_profiles/qwen3vl_pi05_vllm.json}"
PYTHON="${CLAWVLA_ROBOTWIN_PYTHON:-/mnt/wangwai/miniconda3/envs/robotwin-py312/bin/python}"

export PYTHONPATH="/mnt/wangwai/tmp_pytorch3d_target:/mnt/linyutong/wangwai_mirror/vla/clawvla/src"

exec "$PYTHON" -m clawvla.scripts.run_profile --profile "$PROFILE" "$@"
