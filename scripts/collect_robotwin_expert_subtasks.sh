#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

SETTINGS="both"
SPLIT="train"
EPISODES_PER_TASK=50
EPISODES_CLEAN=""
EPISODES_RANDOMIZED=""
WORKERS=4
GPUS="0,1,2,3"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-python}"
MANAGER_PYTHON="${EMBODIEDSKILLS_PYTHON:-${ROBOTWIN_PYTHON}}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${WORKSPACE_ROOT}/RoboTwin}"
OUTPUT_ROOT="${EMBODIEDSKILLS_COLLECTION_ROOT:-${PROJECT_ROOT}/data/robotwin_experts}"
RUN_PREFIX="robotwin_expert_subtasks"
START_SEED_CLEAN=""
START_SEED_RANDOMIZED=""
EPISODE_CACHE_ROOT="${EMBODIEDSKILLS_EPISODE_CACHE_ROOT:-}"
MERGE_HDF5_MODE="stream"
RGB_INPUT_ORDER="rgb"
SAVE_VIDEO=0
KEEP_CACHE=0
DRY_RUN=0
TASK_NAMES=()
PYTHONPATH_PREFIXES=()
EXTRA_ARGS=()

usage() {
  sed -n '/^# Usage:/,/^# End usage/p' "$0" | sed '/^# End usage$/d; s/^# \{0,1\}//'
}

# Usage:
# Collect successful RoboTwin expert trajectories and trace every expert
# self.move(...) call as a frame-aligned raw subtask segment.
#
#   bash scripts/collect_robotwin_expert_subtasks.sh [options]
#
# Main options:
#   --settings clean|randomized|both  Environments to collect (default: both).
#   --episodes-per-task N             Successful episodes per task and setting (default: 50).
#   --episodes-clean N                Override the Clean episode count.
#   --episodes-randomized N           Override the Randomized episode count.
#   --workers N                       Parallel RoboTwin lanes (default: 4).
#   --gpus IDS                        Comma-separated CUDA ids (default: 0,1,2,3).
#   --repo-root PATH                  RoboTwin repository root.
#   --manager-python PATH             Python used for the manager process.
#   --robotwin-python PATH            Python used for RoboTwin worker processes.
#   --output-root PATH                Parent directory for collection runs.
#   --run-prefix NAME                 Stable output directory prefix.
#   --task-name NAME                  Restrict to a task; may be repeated.
#   --split train|val|custom          Dataset role written into metadata (default: train).
#   --start-seed-clean N              First Clean candidate seed.
#   --start-seed-randomized N         First Randomized candidate seed.
#   --episode-cache-root PATH         Optional local NVMe or /dev/shm frame cache.
#   --merge-hdf5-mode memory|stream   HDF5 merge mode (default: stream).
#   --rgb-input-order rgb|bgr         PKL image order (official RoboTwin: rgb).
#   --pythonpath-prefix PATH           Extra worker import path; may be repeated.
#   --save-video                      Save MP4 in addition to mandatory HDF5 images.
#   --keep-cache                      Keep RoboTwin per-frame PKL cache.
#   --dry-run                         Validate inputs and write plans without simulation.
#   --help                            Show this help.
#
# Any arguments after -- are passed to robotwin_collect_expert_subtasks.py.
# Re-running the same command resumes the same stable output directories.
# End usage

while [[ $# -gt 0 ]]; do
  case "$1" in
    --settings) SETTINGS="$2"; shift 2 ;;
    --episodes-per-task) EPISODES_PER_TASK="$2"; shift 2 ;;
    --episodes-clean) EPISODES_CLEAN="$2"; shift 2 ;;
    --episodes-randomized) EPISODES_RANDOMIZED="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --repo-root) ROBOTWIN_ROOT="$2"; shift 2 ;;
    --manager-python) MANAGER_PYTHON="$2"; shift 2 ;;
    --robotwin-python) ROBOTWIN_PYTHON="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --run-prefix) RUN_PREFIX="$2"; shift 2 ;;
    --task-name) TASK_NAMES+=("$2"); shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --start-seed-clean) START_SEED_CLEAN="$2"; shift 2 ;;
    --start-seed-randomized) START_SEED_RANDOMIZED="$2"; shift 2 ;;
    --episode-cache-root) EPISODE_CACHE_ROOT="$2"; shift 2 ;;
    --merge-hdf5-mode) MERGE_HDF5_MODE="$2"; shift 2 ;;
    --rgb-input-order) RGB_INPUT_ORDER="$2"; shift 2 ;;
    --pythonpath-prefix) PYTHONPATH_PREFIXES+=("$2"); shift 2 ;;
    --save-video) SAVE_VIDEO=1; shift ;;
    --keep-cache) KEEP_CACHE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) printf 'Unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${SETTINGS}" in
  clean|demo_clean) SETTINGS_TO_RUN=("demo_clean") ;;
  randomized|random|demo_randomized) SETTINGS_TO_RUN=("demo_randomized") ;;
  both) SETTINGS_TO_RUN=("demo_clean" "demo_randomized") ;;
  *) printf 'Invalid --settings value: %s\n' "${SETTINGS}" >&2; exit 2 ;;
esac

case "${SPLIT}" in
  train|val|custom) ;;
  *) printf 'Invalid --split value: %s\n' "${SPLIT}" >&2; exit 2 ;;
esac

default_seed() {
  local setting="$1"
  if [[ "${SPLIT}" == "train" ]]; then
    [[ "${setting}" == "demo_clean" ]] && printf '200000\n' || printf '400000\n'
  elif [[ "${SPLIT}" == "val" ]]; then
    [[ "${setting}" == "demo_clean" ]] && printf '300000\n' || printf '500000\n'
  else
    [[ "${setting}" == "demo_clean" ]] && printf '600000\n' || printf '700000\n'
  fi
}

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

collect_setting() {
  local setting="$1"
  local start_seed
  local episode_count="${EPISODES_PER_TASK}"
  if [[ "${setting}" == "demo_clean" && -n "${EPISODES_CLEAN}" ]]; then
    episode_count="${EPISODES_CLEAN}"
  elif [[ "${setting}" == "demo_randomized" && -n "${EPISODES_RANDOMIZED}" ]]; then
    episode_count="${EPISODES_RANDOMIZED}"
  fi
  local output_dir="${OUTPUT_ROOT}/${RUN_PREFIX}_${SPLIT}_${setting}_${episode_count}x"
  if [[ "${setting}" == "demo_clean" && -n "${START_SEED_CLEAN}" ]]; then
    start_seed="${START_SEED_CLEAN}"
  elif [[ "${setting}" == "demo_randomized" && -n "${START_SEED_RANDOMIZED}" ]]; then
    start_seed="${START_SEED_RANDOMIZED}"
  else
    start_seed="$(default_seed "${setting}")"
  fi

  local command=(
    "${MANAGER_PYTHON}" -m clawvla.scripts.robotwin_collect_expert_subtasks
    --repo-root "${ROBOTWIN_ROOT}"
    --task-config "${setting}"
    --split "${SPLIT}"
    --episodes-per-task "${episode_count}"
    --start-seed "${start_seed}"
    --workers "${WORKERS}"
    --gpus "${GPUS}"
    --robotwin-python "${ROBOTWIN_PYTHON}"
    --output-dir "${output_dir}"
    --episode-cache-root "${EPISODE_CACHE_ROOT}"
    --merge-hdf5-mode "${MERGE_HDF5_MODE}"
    --rgb-input-order "${RGB_INPUT_ORDER}"
  )

  local task_name
  for task_name in "${TASK_NAMES[@]}"; do
    command+=(--task-name "${task_name}")
  done
  local prefix
  for prefix in "${PYTHONPATH_PREFIXES[@]}"; do
    command+=(--pythonpath-prefix "${prefix}")
  done
  (( SAVE_VIDEO )) && command+=(--save-video) || command+=(--no-save-video)
  command+=(--no-polish-subgoals)
  if (( KEEP_CACHE )); then
    command+=(--keep-cache)
  fi
  if (( DRY_RUN )); then
    command+=(--dry-run)
  fi
  command+=("${EXTRA_ARGS[@]}")

  printf 'Collecting setting=%s output=%s start_seed=%s\n' "${setting}" "${output_dir}" "${start_seed}"
  "${command[@]}"
}

for setting in "${SETTINGS_TO_RUN[@]}"; do
  collect_setting "${setting}"
done
