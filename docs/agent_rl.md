# ClawVLA Agent RL

This document describes the current RL training scaffold in this repository. It is intentionally explicit about what is verified and what is still incomplete, so uploaded code does not look more finished than it is.

## Scope

The RL path trains one unified VLM policy for `vision`, `scheduler`, `verifier`, and `recovery`. RoboTwin supplies the environment, and OpenPI/pi0.5 remains frozen as the low-level action backend.

The implementation is under:

```text
src/clawvla/rl/
src/clawvla/rewards/
configs/rl/
scripts/run_clawvla_rl.sh
tests/test_rl_framework.py
```

## Runtime Split

The training stack intentionally uses separate Python environments:

```text
verl-0.8-py310       verl / torch / vLLM / flash-attn training process
robotwin-py312       ClawVLA runtime and RoboTwin rollout process
openpi-torch-py312   OpenPI/pi0.5 worker process
```

The processes communicate through subprocesses, HTTP-compatible policy calls, and archived trajectory files. The environments are not mixed into one Python runtime.

## Data Path

During training, verl calls `ClawVLAAgentLoop`. One verl sample corresponds to one RoboTwin episode.

For each episode:

1. ClawVLA rollout runs the normal agent loop.
2. VLM calls are routed through the RL policy proxy or verl rollout backend.
3. Each policy call records messages, image references, parsed JSON, token IDs, multimodal payload, and failure state.
4. The adapter returns `prompt_ids`, `response_ids`, `response_mask`, optional `multi_modal_data`, and reward score to verl.

Training keeps the important invariants:

- Model-generated tokens have `response_mask=1`.
- Tool results, environment state, prompts, and skill outputs have `response_mask=0`.
- Image references used by the model must carry real training multimodal payload.
- Prompt/response overflow raises an error; silent truncation is not allowed.

## Reward

Reward registration is configured in `configs/rl/rewards/robotwin.yaml`.

Current dense RoboTwin specs cover these tasks:

```text
place_container_plate
stack_blocks_two
open_laptop
handover_mic
handover_block
press_stapler
click_bell
click_alarmclock
lift_pot
grab_roller
blocks_ranking_rgb
```

All other RoboTwin tasks are not fully registered yet. They should not be mapped as complete until a task-specific or family-specific dense spec exists. Unknown configured tasks fail in preflight through the reward registry.

Episode-level penalties are also explicit:

```text
incomplete_episode_penalty
invalid_decision_penalty
skill_failure_penalty
infra_failure_reward
```

Infrastructure failures are separated from bad trajectories. Bad model decisions are kept as trainable negative trajectories; service crashes and adapter corruption are treated as infra failures.

## Configs

Main config:

```text
configs/rl/qwen3vl_pi05_grpo.yaml
```

Useful smoke configs:

```text
configs/rl/qwen3vl_pi05_train_smoke.yaml
configs/rl/qwen3vl_pi05_rollout_smoke.yaml
configs/rl/qwen3vl_pi05_rollout_vllm_smoke.yaml
configs/rl/qwen3vl_pi05_rollout_real_smoke.yaml
configs/rl/qwen3vl_pi05_real_1update.yaml
configs/rl/qwen3vl_pi05_real_5step_1update.yaml
```

Cluster and task overlays:

```text
configs/rl/cluster/a100_8gpus.yaml
configs/rl/tasks/robotwin_train_small.yaml
configs/rl/rewards/robotwin.yaml
```

## Commands

Dry run:

```bash
cd /mnt/wangwai/vla/clawvla
./scripts/run_clawvla_rl.sh --config configs/rl/qwen3vl_pi05_grpo.yaml --mode dry-run
```

Mock rollout:

```bash
./scripts/run_clawvla_rl.sh \
  --config configs/rl/qwen3vl_pi05_rollout_smoke.yaml \
  --mode rollout-only \
  --policy-response '{"next_component":"vision","next_skill":"capture_views","stage":"observe","reason":"smoke"}'
```

One-update real multimodal startup check:

```bash
./scripts/run_clawvla_rl.sh \
  --config configs/rl/qwen3vl_pi05_real_5step_1update.yaml \
  --mode train \
  --run-id rl_real5_1update
```

Replay archived rewards:

```bash
./scripts/run_clawvla_rl.sh \
  --config configs/rl/qwen3vl_pi05_grpo.yaml \
  --mode replay-reward \
  --replay-path runs/rl/<run_id>/events.jsonl
```

Replay adapter masks:

```bash
./scripts/run_clawvla_rl.sh \
  --config configs/rl/qwen3vl_pi05_grpo.yaml \
  --mode replay-adapter \
  --replay-path runs/rl/<run_id>/events.jsonl
```

## Run Archive

Each run writes under:

```text
runs/rl/<run_id>/
```

Expected contents:

```text
resolved_config.yaml
manifest.json
preflight_report.json
events.jsonl
git_status.txt
git_diff.patch
logs/
trajectories/
rewards/
artifacts/
checkpoints/
env/
```

Generated run directories, Hydra outputs, logs, checkpoints, W&B output, and pycache are ignored by git.

## Tests

Focused local tests:

```bash
cd /mnt/wangwai/vla/clawvla
PYTHONPATH=src /mnt/wangwai/miniconda3/envs/robotwin-py312/bin/python -m pytest tests/test_rl_framework.py -q
```

The current tests cover config loading, reward registry failure behavior, policy proxy tracing, multimodal adapter payloads, response masks, terminal penalties, runtime env setup, and state placeholder rejection.

## Verified State

Verified:

- Real multimodal payload reaches the adapter.
- Response mask trains only model output tokens.
- Overflow is explicit, not silently truncated.
- A real five-step, one-update run completed with nonzero actor gradient.
- Generated files are archived into run directories and ignored by git.

Not yet fully verified:

- Full 25-step task completion through `motion.execute_action`.
- Dense reward coverage for all RoboTwin 2.0 tasks.
- Long multi-task training stability.
