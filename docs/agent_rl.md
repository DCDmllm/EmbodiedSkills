# ClawVLA Agent RL

This document describes the current RL training scaffold in this repository. It records the verified paths, the intended entrypoints, and the remaining validation work.

## Scope

The RL path trains one unified VLM policy for `vision`, `scheduler`, `verifier`, and `recovery`. RoboTwin and LIBERO can supply environment rollouts. OpenPI/pi0.5 remains frozen as the low-level action backend.

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
.venv-openrlhf-py310-cu128
                      OpenRLHF / torch / vLLM / DeepSpeed / flash-attn training process
robotwin-py312       ClawVLA runtime, RoboTwin rollout, and LIBERO rollout process
openpi-torch-py312   OpenPI/pi0.5 worker process
```

The processes communicate through subprocesses, HTTP-compatible policy calls, and archived trajectory files. The
environments are not mixed into one Python runtime.

## Data Path

During training, OpenRLHF calls `clawvla.rl.openrlhf_agent.AgentExecutor`. One OpenRLHF prompt corresponds to one
environment episode.

For each episode:

1. ClawVLA rollout runs the normal agent loop against the configured environment adapter.
2. VLM calls are routed through the RL policy proxy and OpenRLHF rollout backend.
3. Each policy call records messages, image references, parsed JSON, token IDs, multimodal payload, and failure state.
4. The adapter returns one training sample per policy call, with the real call prompt/images, response token range, and
   the episode reward.

Training keeps the important invariants:

- `action_ranges` covers only model-generated response tokens.
- Tool results, environment state, prompts, and skill outputs are context only; they are not trained as action tokens.
- Image references used by the model must carry real training multimodal payload.
- Prompt/response overflow raises an error; silent truncation is not allowed.
- GRPO grouping is by task/instruction/seed.
- Mixed text/multimodal training samples are aligned across data-parallel ranks before actor replay-buffer append.

## Reward

Reward registration is configured in:

```text
configs/rl/rewards/robotwin.yaml
configs/rl/rewards/libero.yaml
```

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

All 50 RoboTwin training tasks are mapped to the `robotwin` reward handler. Tasks without a dedicated dense spec currently
use the terminal/task-status baseline plus episode penalties. LIBERO object tasks use the same terminal penalty path plus the LIBERO reward registry hook.

Episode-level penalties are also explicit:

```text
incomplete_episode_penalty
invalid_decision_penalty
skill_failure_penalty
recoverable_preflight_penalty
infra_failure_reward
```

Infrastructure failures are separated from bad trajectories. Bad model decisions are kept as trainable negative trajectories; service crashes and adapter corruption are treated as infra failures.

## Configs

Main current OpenRLHF config:

```text
configs/rl/qwen3vl_pi05_multitask_1update.yaml
configs/rl/qwen3vl_pi05_libero_multitask_1update.yaml
```

Useful smoke configs:

```text
configs/rl/qwen3vl_pi05_train_smoke.yaml
configs/rl/qwen3vl_pi05_rollout_smoke.yaml
configs/rl/qwen3vl_pi05_rollout_vllm_smoke.yaml
configs/rl/qwen3vl_pi05_rollout_real_smoke.yaml
configs/rl/qwen3vl_pi05_real_1update.yaml
configs/rl/qwen3vl_pi05_real_5step_1update.yaml
configs/rl/qwen3vl_pi05_libero_multitask_1update_single_gpu.yaml
```

Cluster and task overlays:

```text
configs/rl/cluster/a100_8gpus.yaml
configs/rl/tasks/robotwin_train_small.yaml
configs/rl/tasks/libero_object_smoke.yaml
configs/rl/tasks/libero_object_all.yaml
configs/rl/rewards/robotwin.yaml
configs/rl/rewards/libero.yaml
```

## Commands

Dry run:

```bash
cd /mnt/wangwai/vla/clawvla
./scripts/run_clawvla_rl.sh --mode dry-run
./scripts/run_clawvla_rl.sh --preset libero-multitask --mode dry-run
clawvla-rl --preset robotwin-multitask --mode dry-run
```

One-update real multimodal startup checks:

```bash
./scripts/run_clawvla_rl.sh \
  --preset robotwin-real5 \
  --mode train \
  --run-id robotwin_rl_real5

./scripts/run_clawvla_rl.sh \
  --preset libero-multitask \
  --mode train \
  --run-id libero_rl_multitask
```

## Run Archive

Each run writes under:

```text
runs/rl/<run_id>/
```

Expected contents:

```text
resolved_config.yaml
logs/
artifacts/
checkpoints/
events.jsonl                # train mode, written after episodes start
trajectories/               # train mode episode result JSON
rewards/                    # train mode reward JSONL
```

Generated run directories, Hydra outputs, logs, checkpoints, W&B output, and pycache are ignored by git.

## Tests

Focused local tests:

```bash
cd /mnt/wangwai/vla/clawvla
PYTHONPATH=src /mnt/wangwai/miniconda3/envs/robotwin-py312/bin/python -m pytest tests/test_rl_framework.py -q
```

The current tests cover config loading, reward registry behavior, policy proxy tracing, multimodal adapter payloads, call-level OpenRLHF samples, terminal penalties, runtime env setup, state placeholder rejection, LIBERO config wiring, and OpenRLHF mixed-modality alignment.

## Verified State

Verified:

- Real multimodal payload reaches the adapter.
- `action_ranges` train only model output tokens.
- Overflow is explicit, not silently truncated.
- OpenRLHF dry-run generates the 50-task prompt dataset and command successfully.
- LIBERO `qwen3vl_pi05_libero_multitask_1update.yaml` completed one ZeRO-3 two-policy-GPU update with mixed text/multimodal samples.
- RobotWin `qwen3vl_pi05_real_5step_1update.yaml` completed one ZeRO-3 four-policy-GPU update with mixed text/multimodal samples.
- OpenRLHF runtime patches keep modality-compatible actor/ref forward batches and data-parallel replay-buffer order.
- The RobotWin five-step smoke had uniform negative rewards, so it validated infrastructure and synchronization only.
- Generated files are archived into run directories and ignored by git.

Remaining validation:

- Full 25-step task completion through `motion.execute_action`.
- Dense reward coverage for all RoboTwin 2.0 tasks.
- Long multi-task training stability.
