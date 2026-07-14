# ClawVLA Agent RL

This document describes the current RL training scaffold in this repository. It records the verified paths, the intended entrypoints, and the remaining validation work.

## Scope

The RL path trains one unified VLM policy for `vision`, `scheduler`, `verifier`, and `recovery`. RoboTwin, LIBERO, and RoboCasa can supply rollout observations. OpenPI/pi0.5 and GR00T remain frozen as low-level action backends.

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
groot-py312          RoboCasa rollout process and GR00T worker process
calvin-py38          CALVIN rollout process; calls the external X-VLA HTTP action server
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

完整的自然语言信用分配、奖励公式和50任务逐项映射见
[RoboTwin RL 奖励函数手册](robotwin_rl_reward_catalog.md)。

Reward registration is configured in:

```text
configs/rl/rewards/robotwin.yaml
configs/rl/rewards/libero.yaml
configs/rl/rewards/robocasa.yaml
configs/rl/rewards/calvin.yaml
```

All 50 configured RoboTwin training tasks now have an explicit dense reward spec. The specs are organized by physical
reward family:

```text
spatial / relative_place / collection_place
stack / stack_multi / ordering
articulation / cabinet_place
contact_press / tool_contact / scan
handover / dual_lift / container_lift
axis_lift / axis_away / shake / dump
```

The dense signals use simulator state such as actor poses, functional/contact points, gripper contact, articulation qpos,
task-private target/start fields, actor-pair contact, and collection member poses. Official terminal success always comes
from RoboTwin `check_success()`; an agent-loop `finished` result does not count as task success by itself. LIBERO object
tasks use the terminal penalty path plus the LIBERO reward registry hook. RoboCasa tasks use the RoboCasa task status and
GR00T action backend.

Episode-level penalties are also explicit:

```text
incomplete_episode_penalty
premature_finish_penalty
stalled_loop_penalty
invalid_decision_penalty
skill_failure_penalty
recoverable_preflight_penalty
infra_failure_reward
```

Infrastructure failures are separated from bad trajectories. Bad model decisions are kept as trainable negative trajectories; service crashes and adapter corruption are treated as infra failures.

## Configs

Main current OpenRLHF config:

```text
configs/rl/qwen3vl_pi05_online_seed_mix_grpo.yaml
configs/rl/qwen3vl_pi05_multitask_1update.yaml
configs/rl/qwen3vl_pi05_libero_multitask_1update.yaml
configs/rl/qwen3vl_groot_robocasa_1update.yaml
configs/rl/qwen3vl_calvin_xvla_1update.yaml
configs/rl/rynnbrain2b_pi05_real_1update.yaml
```

The RoboTwin online seed-mix config needs no offline grounding dataset. It combines exact expert task/seed/instruction
episodes (which have Planner references) with precomputed official valid seeds (grounding-only). Grounding-only groups
still receive the complete environment reward, while the Planner auxiliary score is masked. The current 60:40 mix expands
to 3727 prompts and 14908 rollouts at `rollout_n=4`.

Useful smoke configs:

```text
configs/rl/qwen3vl_pi05_train_smoke.yaml
configs/rl/qwen3vl_pi05_rollout_smoke.yaml
configs/rl/qwen3vl_pi05_rollout_vllm_smoke.yaml
configs/rl/qwen3vl_pi05_rollout_real_smoke.yaml
configs/rl/qwen3vl_pi05_real_1update.yaml
configs/rl/qwen3vl_pi05_real_5step_1update.yaml
configs/rl/qwen3vl_pi05_libero_multitask_1update_single_gpu.yaml
configs/rl/qwen3vl_groot_robocasa_rollout_smoke.yaml
configs/rl/qwen3vl_calvin_xvla_1update_long_smoke.yaml
configs/rl/rynnbrain2b_pi05_train_smoke.yaml
```

Cluster and task overlays:

```text
configs/rl/cluster/a100_8gpus.yaml
configs/rl/tasks/robotwin_train_small.yaml
configs/rl/tasks/libero_object_smoke.yaml
configs/rl/tasks/libero_object_all.yaml
configs/rl/rewards/robotwin.yaml
configs/rl/rewards/libero.yaml
configs/rl/rewards/robocasa.yaml
configs/rl/rewards/calvin.yaml
```

## Commands

Dry run:

```bash
cd /path/to/clawvla
./scripts/run_clawvla_rl.sh --mode dry-run
./scripts/run_clawvla_rl.sh --config configs/rl/qwen3vl_pi05_online_seed_mix_grpo.yaml --mode dry-run
./scripts/run_clawvla_rl.sh --preset libero-multitask --mode dry-run
./scripts/run_clawvla_rl.sh --preset robocasa-rollout --mode dry-run
./scripts/run_clawvla_rl.sh --preset calvin-xvla --mode dry-run
./scripts/run_clawvla_rl.sh --preset rynnbrain-train-smoke --mode dry-run
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

./scripts/run_clawvla_rl.sh \
  --preset robocasa-1update \
  --mode dry-run \
  --run-id robocasa_groot_1update
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
cd /path/to/clawvla
PYTHONPATH=src /mnt/wangwai/miniconda3/envs/robotwin-py312/bin/python -m pytest tests/test_rl_framework.py -q
```

If HTTP proxies are configured, add `NO_PROXY=127.0.0.1,localhost` (and the lowercase equivalent) so local PolicyProxy
requests are not routed through the cluster proxy.

The current tests cover config loading, reward registry behavior, policy proxy tracing, multimodal adapter payloads, call-level OpenRLHF samples, terminal penalties, runtime env setup, state placeholder rejection, LIBERO config wiring, RoboCasa/GR00T config wiring, and OpenRLHF mixed-modality alignment.

## Verified State

Verified:

- Real multimodal payload reaches the adapter.
- `action_ranges` train only model output tokens.
- Overflow is explicit, not silently truncated.
- OpenRLHF dry-run generates the 50-task prompt dataset and command successfully.
- LIBERO `qwen3vl_pi05_libero_multitask_1update.yaml` completed one ZeRO-3 two-policy-GPU update with mixed text/multimodal samples.
- RobotWin `qwen3vl_pi05_real_5step_1update.yaml` completed one ZeRO-3 four-policy-GPU update with mixed text/multimodal samples.
- RoboCasa + GR00T path reports model action dim 32, environment action dim 12, and real execute state/image deltas in smoke runs. Task success is not claimed.
- CALVIN environment/action factories, terminal reward, presets, and OpenRLHF config wiring are covered by local tests; a live X-VLA server is still required for real rollout validation.
- RynnBrain presets now use `openrlhf:` keys and their intended token/training limits are covered by a regression test.
- OpenRLHF runtime patches keep modality-compatible actor/ref forward batches and data-parallel replay-buffer order.
- The RobotWin five-step smoke had uniform negative rewards, so it validated infrastructure and synchronization only.
- Generated files are archived into run directories and ignored by git.

Remaining validation:

- Full 25-step task completion through `motion.execute_action`.
- Simulator calibration of dense reward thresholds across all 50 registered RoboTwin tasks.
- End-to-end CALVIN rollout with the matching live X-VLA server and validation dataset.
- Long multi-task training stability.
