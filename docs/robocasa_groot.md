# RoboCasa + GR00T Integration Notes

This note records the RoboCasa path that was added to ClawVLA. It is meant as an upload/review checklist, not a benchmark claim.

## Scope

The integration adds a real RoboCasa environment adapter and a frozen GR00T action backend:

```text
RoboCasa observation
-> ClawVLA observe/plan/preflight/execute loop
-> GR00T action backend
-> 12D robocasa_action
-> RoboCasa env.step(...)
```

The high-level VLM still owns vision grounding, scheduling, verification, and recovery. GR00T only emits low-level RoboCasa actions.

## Local Environment

Current local runtime uses a separate conda environment:

```bash
conda activate groot-py312
export PYTHONPATH=/mnt/wangwai/vla/clawvla/src:/mnt/wangwai/lerobot/src:/mnt/wangwai/RoboCasa:/mnt/wangwai/robosuite
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
export http_proxy=http://10.20.5.5:20171
export https_proxy=$http_proxy
```

The current local checkpoint path is:

```text
/mnt/wangwai/weights/robocasa/robocasa365_checkpoints/gr00t_n1-5/multitask_learning/checkpoint-120000
```

The main local config is:

```text
configs/robocasa_groot_enabled_probe.json
```

It contains machine-specific absolute paths and should be treated as a lab-local probe config unless path templating is added later.

## Action Schema

The GR00T checkpoint model config is:

```text
model_action_dim = 32
action_horizon = 16
max_action_dim = 32
```

RoboCasa execution uses 12 dimensions from the checkpoint metadata:

```text
base_motion              4
control_mode             1
end_effector_position    3
end_effector_rotation    3
gripper_close            1
total                   12
```

This is represented in ClawVLA as:

```json
"env_action_dim": 12,
"policy_kwargs": {
  "chunk_size": 16,
  "n_action_steps": 16,
  "max_action_dim": 32
}
```

Do not rename `env_action_dim` back to `action_dim`: that makes it look like the model head was changed to 12D. The model remains 32D; the environment action is 12D.

## Code Map

Core files:

```text
src/clawvla/envs/robocasa.py
src/clawvla/action_backends/groot.py
src/clawvla/scripts/groot_worker.py
src/clawvla/scripts/groot_inference_smoke.py
configs/robocasa_groot_enabled_probe.json
configs/rl/qwen3vl_groot_robocasa_rollout_smoke.yaml
configs/rl/qwen3vl_groot_robocasa_1update.yaml
configs/rl/rewards/robocasa.yaml
```

Shared loop/runtime changes:

```text
src/clawvla/envs/factory.py
src/clawvla/envs/__init__.py
src/clawvla/action_backends/factory.py
src/clawvla/action_backends/__init__.py
src/clawvla/components/motion.py
src/clawvla/components/safety.py
src/clawvla/components/vision.py
src/clawvla/scripts/run_loop.py
src/clawvla/rl/openrlhf_runner.py
src/clawvla/rl/rollout_worker.py
src/clawvla/rl/reward_registry.py
tests/test_rl_framework.py
```

## Semantics Policy

RoboCasa can expose object/fixture names from the real environment registry, but this is off by default.

Only these explicit flags enable it:

```text
environment.params.expose_environment_semantics
environment.metadata.expose_environment_semantics
environment.metadata.debug_expose_environment_semantics
```

Without those flags, localization ignores environment semantics. This avoids silently using privileged simulator metadata in normal runs.

## Artifact Layout

Images and raw observation summaries are written under `tmp_artifacts/` or the configured `environment.artifact_dir`.

Important paths:

```text
<artifact_prefix>/images/
<artifact_prefix>/raw_observation_summary.json
<artifact_prefix>/preflight_refresh/step_xxxx/images/
<artifact_prefix>/execute/after/images/
```

`tmp_artifacts/`, `tmp_runs/`, `runs/`, and checkpoints are git-ignored. Do not upload smoke output images or run archives unless explicitly needed for a report.

## Useful Commands

Light tests:

```bash
PYTHONPATH=/mnt/wangwai/vla/clawvla/src \
/mnt/wangwai/miniconda3/envs/robotwin-py312/bin/python \
-m pytest tests/test_rl_framework.py -q -k 'groot or robocasa or action_backend'
```

Direct loop smoke:

```bash
/mnt/wangwai/miniconda3/envs/groot-py312/bin/python \
-m clawvla.scripts.run_loop \
--config configs/robocasa_groot_enabled_probe.json \
--artifact-prefix robocasa_groot_smoke \
--max-steps 20
```

RL dry run:

```bash
./scripts/run_clawvla_rl.sh --preset robocasa-rollout --mode dry-run
```

One-update config:

```bash
./scripts/run_clawvla_rl.sh --preset robocasa-1update --mode dry-run
```

## Verified

- RoboCasa adapter captures three camera views and a 16D GR00T-compatible state vector.
- GR00T action spec reports `model_action_dim=32`, `env_action_dim=12`, and `horizon=16`.
- Action chunks are 12D `robocasa_action` commands.
- RoboCasa execution reports `action_effect`, including max action magnitude and before/after state delta.
- Execute-after images are stored under a separate `execute/after` prefix, avoiding before/after overwrite.
- Environment semantic hints are disabled by default and covered by tests.

## Not Claimed

- This does not claim strong RoboCasa task success. The tested GR00T checkpoint can move the robot but did not reliably solve grasp/place in the local smoke runs.
- This does not use fake actions, placeholder execution, or record-only rollouts.
- This does not require Docker.
