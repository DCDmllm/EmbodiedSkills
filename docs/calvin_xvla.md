# CALVIN + X-VLA Integration Notes

This note describes the CALVIN rollout adapter and frozen X-VLA HTTP action backend. It is an integration and smoke-test
guide, not a benchmark claim.

## Process split

```text
OpenRLHF trainer (Python 3.10)
  -> ClawVLA policy proxy
  -> CALVIN rollout subprocess (calvin-py38)
  -> X-VLA action server (HTTP /act)
  -> CALVIN env.step(...)
```

The unified VLM policy owns observation grounding, task planning, verification, and recovery. X-VLA only generates the
bounded low-level action chunk.

## Environment

```bash
cd /path/to/EmbodiedSkills
export CLAWVLA_ROOT="$PWD"
conda activate calvin-py38
python -m pip install -r requirements/calvin-py38.txt
export PYTHONPATH="$CLAWVLA_ROOT/src:/mnt/wangwai/vla/CALVIN/calvin_env:/mnt/wangwai/vla/CALVIN/calvin_models"
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
```

The CALVIN subprocess runs on Python 3.8 through `PYTHONPATH`; do not install the repository with `pip install -e .` in
that environment. The CALVIN checkout supplies its simulator, task-oracle, Hydra, and dataset stack.

## Configuration

Core files:

```text
src/clawvla/envs/calvin.py
src/clawvla/action_backends/calvin.py
configs/calvin_xvla_enabled_probe.json
configs/rl/qwen3vl_calvin_xvla_1update.yaml
configs/rl/qwen3vl_calvin_xvla_1update_long_smoke.yaml
configs/rl/tasks/calvin_smoke.yaml
configs/rl/rewards/calvin.yaml
```

The checked-in probe expects:

```text
CALVIN repo          /mnt/wangwai/vla/CALVIN
validation dataset  /mnt/wangwai/vla/CALVIN/dataset/calvin_debug_dataset/validation
X-VLA endpoint      http://127.0.0.1:8000/act
```

These are lab-local paths. Update the JSON/YAML if the checkout, dataset, proxy, or server address differs.

## Observation and action contract

The adapter exposes the public CALVIN observations needed by the loop:

```text
rgb_static + rgb_gripper
20D calvin_proprio
task language and task-oracle status
```

The HTTP backend sends `image0`, `image1`, language instruction, proprioception, domain id, and requested step count.
With `serialization: json_numpy`, NumPy values are encoded by `json-numpy`.

The released X-VLA server may return 20 columns. ClawVLA consumes the first 10 columns as `calvin_ee_pose_10d`, applies
the configured gripper threshold, and executes at most the requested horizon. Non-finite, empty, or malformed responses
fail explicitly.

## Commands

Inspect the resolved OpenRLHF command without starting training:

```bash
./scripts/run_clawvla_rl.sh --preset calvin-xvla --mode dry-run --run-id calvin_xvla_check
./scripts/run_clawvla_rl.sh --preset calvin-long-smoke --mode dry-run --run-id calvin_xvla_long_check
```

The external X-VLA action server must be started separately and answer the configured `/act` endpoint before a real
rollout. The default config does not embed credentials.

## Current validation boundary

- Config loading, environment/action factories, reward registration, and OpenRLHF command wiring are covered by tests.
- Importing the ClawVLA CALVIN bridge in the current `calvin-py38` environment is verified.
- The one-update and long-smoke YAML files are integration presets; they do not establish benchmark success.
- Full-sequence CALVIN evaluation still requires a live X-VLA server and the matching validation dataset.
