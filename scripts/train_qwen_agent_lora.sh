#!/usr/bin/env bash
set -euo pipefail

: "${LLAMA_FACTORY_ROOT:?Set LLAMA_FACTORY_ROOT to the LLaMA-Factory checkout}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export FORCE_TORCHRUN=1
export MASTER_PORT="${MASTER_PORT:-29644}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DISABLE_VERSION_CHECK=1
export PYTHONPATH="$LLAMA_FACTORY_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"
exec "$PYTHON_BIN" -m llamafactory.cli train configs/qwen/agent_skill_lora.yaml "$@"
