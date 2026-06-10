#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DEFAULT_PYTHON="/mnt/wangwai/miniconda3/envs/verl-0.8-py310/bin/python"
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
  DEFAULT_PYTHON="python"
fi

CONFIG="${CLAWVLA_RL_CONFIG:-$REPO_ROOT/configs/rl/qwen3vl_pi05_grpo.yaml}"
PYTHON="${CLAWVLA_RL_PYTHON:-$DEFAULT_PYTHON}"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON" -m clawvla.rl.runner --config "$CONFIG" "$@"
