#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/mjh/Projects/EmbodiedSkills}"
EVAL_ENV="${EVAL_ENV:-/data/mjh-conda/envs/robocerebra-openvla-eval}"
PI05_ENV="${PI05_ENV:-/data/mjh-conda/envs/openpi-torch-py312}"
OPENVLA_OFT="${OPENVLA_OFT:-/mnt/raid1/mjh/RoboTwin/RoboTwin/policy/openvla-oft}"
BENCH_ROOT="${BENCH_ROOT:-/mnt/raid1/mjh/datasets/RoboCerebraBench_case1}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/robocerebra_probe_logs/full_task_comparison}"
REPORT_PATH="${REPORT_PATH:-${REPO_ROOT}/outputs/robocerebra_full_task_model_comparison.md}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SEEDS="${SEEDS:-7 8 9}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}" "${REPO_ROOT}/outputs/robocerebra_probe_logs"

export CUDA_VISIBLE_DEVICES
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export LIBERO_CONFIG_PATH="${REPO_ROOT}/outputs/robocerebra_probe_logs/libero_config"
export PYTHONPATH="${OPENVLA_OFT}:${REPO_ROOT}/.deps/RoboCerebra_zip/LIBERO:${REPO_ROOT}/.deps/RoboCerebra_zip/evaluation:${REPO_ROOT}"

run_model() {
  local py_bin="$1"
  local model_name="$2"
  local model_kind="$3"
  local checkpoint="$4"
  local use_proprio="$5"
  local model_out="${OUT_ROOT}/${model_name}"
  local stdout_log="${OUT_ROOT}/${model_name}.stdout.log"

  echo "[RUN] ${model_name} (${model_kind})"
  set +e
  "${py_bin}" "${REPO_ROOT}/scripts/robocerebra_full_task_comparison.py" \
    --mode run-model \
    --model-name "${model_name}" \
    --model-kind "${model_kind}" \
    --checkpoint "${checkpoint}" \
    --bench-root "${BENCH_ROOT}" \
    --output-dir "${model_out}" \
    --seeds ${SEEDS} \
    ${use_proprio} \
    2>&1 | tee "${stdout_log}"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ ${status} -ne 0 ]]; then
    echo "[WARN] ${model_name} command exited with status ${status}; continuing."
  fi
}

run_model \
  "${PI05_ENV}/bin/python" \
  "pi05_robocerebra_lora_random_200ep_1kstep" \
  "pi05" \
  "${REPO_ROOT}/outputs/pi05_robocerebra_lora_random_200ep_1kstep/lora_params.pkl" \
  "--use-proprio"

run_model \
  "${EVAL_ENV}/bin/python" \
  "Yun5_OpenVLA-RoboCerebra-L1-Proprio-4000" \
  "openvla" \
  "/mnt/raid1/mjh/weights/robocerebra_openvla/Yun5_OpenVLA-RoboCerebra-L1-Proprio-4000" \
  "--use-proprio"

run_model \
  "${EVAL_ENV}/bin/python" \
  "Yun5_OpenVLA-RoboCerebra-Proprio-1" \
  "openvla" \
  "/mnt/raid1/mjh/weights/robocerebra_openvla/Yun5_OpenVLA-RoboCerebra-Proprio-1" \
  "--use-proprio"

run_model \
  "${EVAL_ENV}/bin/python" \
  "Yun5_OpenVLA-RoboCerebra-NO-Proprio-2000" \
  "openvla" \
  "/mnt/raid1/mjh/weights/robocerebra_openvla/Yun5_OpenVLA-RoboCerebra-NO-Proprio-2000" \
  "--no-use-proprio"

"${EVAL_ENV}/bin/python" "${REPO_ROOT}/scripts/robocerebra_full_task_comparison.py" \
  --mode report \
  --out-root "${OUT_ROOT}" \
  --report-path "${REPORT_PATH}"

echo "[DONE] report: ${REPORT_PATH}"
