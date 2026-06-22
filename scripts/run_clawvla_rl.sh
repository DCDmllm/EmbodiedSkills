#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DEFAULT_PYTHON="$REPO_ROOT/.venv-openrlhf-py310-cu128/bin/python"
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
  DEFAULT_PYTHON="python"
fi

PYTHON="${CLAWVLA_RL_PYTHON:-$DEFAULT_PYTHON}"
PRESET="${CLAWVLA_RL_PRESET:-}"
CONFIG="${CLAWVLA_RL_CONFIG:-}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_clawvla_rl.sh [--preset NAME] [--config PATH] [runner args...]

Presets:
  robotwin-multitask   configs/rl/qwen3vl_pi05_multitask_1update.yaml
  robotwin-real5       configs/rl/qwen3vl_pi05_real_5step_1update.yaml
  robotwin-real1       configs/rl/qwen3vl_pi05_real_1update.yaml
  libero-multitask     configs/rl/qwen3vl_pi05_libero_multitask_1update.yaml
  libero-single        configs/rl/qwen3vl_pi05_libero_multitask_1update_single_gpu.yaml
  robocasa-rollout     configs/rl/qwen3vl_groot_robocasa_rollout_smoke.yaml
  robocasa-1update     configs/rl/qwen3vl_groot_robocasa_1update.yaml
  train-smoke          configs/rl/qwen3vl_pi05_train_smoke.yaml
  rollout-smoke        configs/rl/qwen3vl_pi05_rollout_smoke.yaml

Examples:
  scripts/run_clawvla_rl.sh --preset robotwin-real5 --mode train --run-id robotwin_smoke
  scripts/run_clawvla_rl.sh --preset libero-multitask --mode train --run-id libero_smoke
  scripts/run_clawvla_rl.sh --config configs/rl/qwen3vl_pi05_multitask_1update.yaml --mode dry-run

Environment:
  CLAWVLA_RL_PYTHON    Python used for the runner
  CLAWVLA_RL_CONFIG    Default config path
  CLAWVLA_RL_PRESET    Default preset name
EOF
}

preset_config() {
  case "$1" in
    robotwin-multitask) printf '%s\n' "$REPO_ROOT/configs/rl/qwen3vl_pi05_multitask_1update.yaml" ;;
    robotwin-real5) printf '%s\n' "$REPO_ROOT/configs/rl/qwen3vl_pi05_real_5step_1update.yaml" ;;
    robotwin-real1) printf '%s\n' "$REPO_ROOT/configs/rl/qwen3vl_pi05_real_1update.yaml" ;;
    libero-multitask) printf '%s\n' "$REPO_ROOT/configs/rl/qwen3vl_pi05_libero_multitask_1update.yaml" ;;
    libero-single) printf '%s\n' "$REPO_ROOT/configs/rl/qwen3vl_pi05_libero_multitask_1update_single_gpu.yaml" ;;
    robocasa-rollout) printf '%s\n' "$REPO_ROOT/configs/rl/qwen3vl_groot_robocasa_rollout_smoke.yaml" ;;
    robocasa-1update) printf '%s\n' "$REPO_ROOT/configs/rl/qwen3vl_groot_robocasa_1update.yaml" ;;
    train-smoke) printf '%s\n' "$REPO_ROOT/configs/rl/qwen3vl_pi05_train_smoke.yaml" ;;
    rollout-smoke) printf '%s\n' "$REPO_ROOT/configs/rl/qwen3vl_pi05_rollout_smoke.yaml" ;;
    "")
      printf '%s\n' "$REPO_ROOT/configs/rl/qwen3vl_pi05_multitask_1update.yaml"
      ;;
    *)
      printf 'Unknown RL preset: %s\n\n' "$1" >&2
      usage >&2
      return 2
      ;;
  esac
}

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --preset)
      if [[ $# -lt 2 ]]; then
        printf '%s\n' "--preset requires a value" >&2
        exit 2
      fi
      PRESET="$2"
      shift 2
      ;;
    --preset=*)
      PRESET="${1#*=}"
      shift
      ;;
    --config)
      if [[ $# -lt 2 ]]; then
        printf '%s\n' "--config requires a value" >&2
        exit 2
      fi
      CONFIG="$2"
      shift 2
      ;;
    --config=*)
      CONFIG="${1#*=}"
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  CONFIG="$(preset_config "$PRESET")"
elif [[ "$CONFIG" != /* ]]; then
  CONFIG="$REPO_ROOT/$CONFIG"
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON" -m clawvla.rl.openrlhf_runner --config "$CONFIG" "${ARGS[@]}"
