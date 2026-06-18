# EmbodiedSkills 项目交接总览

本文档按当前仓库代码整理，面向熟悉 Python、VLM/VLA、RoboTwin、OpenRLHF/RL 基本概念的项目成员，说明 EmbodiedSkills 当前代码结构、运行方式、接口约束、扩展点和排错入口。

更细的章节：

- [Runtime 架构与执行循环](handoff/runtime_architecture.md)
- [组件、技能与数据接口](handoff/components_and_skills.md)
- [Agent RL 训练与奖励系统](handoff/rl_training_and_rewards.md)
- [扩展、测试与调试手册](handoff/extending_testing_debugging.md)
- [成员分工](handoff/member_roles.md)

## 当前定位

EmbodiedSkills 是一个面向 RoboTwin 操作任务的 embodied agent runtime。当前主线是：

```text
RoboTwin observation
  -> VLM visual grounding
  -> scheduler subgoal loop
  -> OpenPI/pi0.5 action chunk
  -> RoboTwin execute
  -> verifier / recovery
```

运行时由 scheduler 模型逐步选择控制动作或 skill；runtime 按阶段、黑板状态和 skill 前置条件校验并执行该决定。失败、不可用、异常和非法模型输出都会写入日志与 result，作为可追踪的运行状态。

## 代码目录

核心包在 `src/clawvla/`：

```text
src/clawvla/
  agent_loop.py              # agent 主循环，调 scheduler、校验 decision、执行 skill
  phase_policy.py            # 阶段顺序和每阶段允许的组件/skill
  runtime.py                 # AgentRuntime，skill 调用封装、异常显式化、history
  blackboard.py              # 组件共享状态、compact_context 给模型提示用
  model_calls.py             # 组件模型 JSON 调用统一入口
  models.py                  # local_hf / openai_compatible / azure_openai backend
  schema.py                  # Observation/Perception/TaskPlan/ActionChunk 等 dataclass schema
  components/                # vision/state/scheduler/safety/motion/verifier/recovery
  envs/                      # RoboTwin session 和 observation/action adapter
  action_backends/           # pi0.5/OpenPI action backend
  scripts/                   # 普通 agent、vLLM、OpenPI worker、smoke/probe 脚本
  rl/                        # OpenRLHF GRPO runner、policy proxy、trajectory adapter
  rewards/                   # RoboTwin reward snapshot / dense reward families
```

配置和入口：

```text
configs/robotwin_default.json
configs/robotwin_pi05_enabled_probe.json
configs/robotwin_pi05_worker_probe.json
configs/run_profiles/qwen3vl_pi05_vllm.json
configs/rl/
scripts/run_qwen3vl_pi05_agent.sh
scripts/run_clawvla_rl.sh
requirements/
tests/test_rl_framework.py
```

## 环境分工

当前代码按进程职责拆分为四类环境文件。

```text
robotwin-py312
  主 agent runtime、RoboTwin adapter、run_loop.py。
  requirements/robotwin-py312.txt 保持 RoboTwin/CUDA/SAPIEN 栈稳定，未在该文件中安装 torch。

vllm
  本地 Qwen3-VL OpenAI-compatible server。
  requirements/vllm.txt 包含 vllm、transformers、torch、qwen-vl-utils。

openpi-torch-py312
  pi0.5/OpenPI worker。
  requirements/openpi-torch-py312.txt 使用 torch + transformers==4.53.2，并通过 PYTHONPATH 引 RoboTwin 的 policy/pi05/src。

.venv-openrlhf-py310-cu128
  当前 Agent RL 训练环境，使用 OpenRLHF + vLLM + DeepSpeed + flash-attn。
```

普通 agent 运行通常是 `robotwin-py312 + vllm + openpi-torch-py312` 三个进程族配合。RL 训练当前由 OpenRLHF 环境启动，再通过子进程跑 `robotwin-py312` rollout 和 `openpi-torch-py312` worker。

## 普通 Agent 运行

推荐入口：

```bash
cd /mnt/wangwai/vla/clawvla
./scripts/run_qwen3vl_pi05_agent.sh \
  --instruction "place the container on the plate" \
  --artifact-prefix agent_qwen3vl_vllm_pi05 \
  --max-steps 50 \
  --gpus 5,6 \
  --run
```

先看展开命令：

```bash
./scripts/run_qwen3vl_pi05_agent.sh \
  --instruction "place the container on the plate" \
  --artifact-prefix agent_qwen3vl_vllm_pi05 \
  --max-steps 50 \
  --gpus 5,6 \
  --dry-run
```

这个入口读取 `configs/run_profiles/qwen3vl_pi05_vllm.json`，默认做这些事：

- 用 `vllm` 环境启动 Qwen3-VL OpenAI-compatible server。
- 生成临时 config，把 `vision/scheduler/verifier/recovery` 模型路由到本地 vLLM。
- 用 `robotwin-py312` 跑 `clawvla.scripts.run_loop`。
- 如果 action backend 配置为 worker mode 且 `auto_start` 为 true，启动 `openpi-torch-py312` 里的 `pi05_worker`。
- 运行结束后清理 vLLM 和 OpenPI worker。

主要输出：

```text
tmp_runs/<prefix>_vllm.log
tmp_runs/<prefix>_agent.log
tmp_runs/<prefix>_pi05_worker.log
tmp_runs/<prefix>_result.json
tmp_runs/<prefix>_vllm_config.json
tmp_artifacts/<prefix>/
```

`tmp_artifacts/<prefix>/` 下会有 observation 图片、depth、pointcloud、raw summary、执行后 observation、verify observation 等 artifact。`result.json` 保存 compact blackboard 和 loop step 结果，完整 stdout/stderr 在 agent log。

## 关键配置

### `configs/robotwin_default.json`

默认 RoboTwin 配置。action backend 是 pi05 但 `enabled=false`，更适合 inspect/smoke 或只看模型链路。

### `configs/robotwin_pi05_enabled_probe.json`

启用 pi05 action backend，OpenPI runtime mode 为 `subprocess`。每次 action chunk 通过子进程调用 `pi05_inference_smoke`。

### `configs/robotwin_pi05_worker_probe.json`

启用 pi05 action backend，OpenPI runtime mode 为 `worker`。正式 agent 入口使用它，避免每次 action 都重载 OpenPI 模型。

当前 RoboTwin 相关默认值：

```text
task_name: place_container_plate
task_config: demo_clean
camera_profile: Large_D435_Wide
planner_image_mode: current_rgb_4
static_camera_preset: selected_global_4
artifact_dir: /mnt/wangwai/vla/clawvla/tmp_artifacts
```

`Large_D435_Wide` 通过 RoboTwin `_camera_config.yml` 的 profile 应用到 head/wrist/static camera 配置里；EmbodiedSkills 运行时消费生成后的图像 artifact。

### `configs/run_profiles/qwen3vl_pi05_vllm.json`

普通 agent 推荐 profile。当前默认模型路径：

```text
/mnt/wangwai/weights/Qwen/Qwen3-VL-8B-Instruct
```

默认 `model-keys=vision,scheduler,verifier,recovery`，这几个 role 共用本地 vLLM 服务。`state/safety/motion` 当前不依赖模型做生成。

## 当前阶段顺序

阶段定义在 `src/clawvla/phase_policy.py`：

```text
observe -> plan -> preflight -> execute -> verify -> recover
```

各阶段语义：

- `observe`：采集观测、候选检测、source/target 绑定、更新 world_state。
- `plan`：用 scheduler model 生成完整 subgoal plan，选择当前 subgoal。
- `preflight`：执行运动前检查 observation/world/task/subgoal/camera/robot/action backend。
- `execute`：构造 motion goal、motion plan、调用 pi0.5/OpenPI 生成 action chunk、校验并执行。
- `verify`：执行后采集 fresh verify images，验证当前 subgoal。
- `recover`：只有 true failure 进入，生成具体 recovery patch，再回 observe/plan/preflight/finish。

更完整的阶段和 skill 说明见 [组件、技能与数据接口](handoff/components_and_skills.md)。

## Agent RL 运行

入口：

```bash
cd /mnt/wangwai/vla/clawvla
./scripts/run_clawvla_rl.sh --config configs/rl/qwen3vl_pi05_grpo.yaml --mode dry-run
```

真实 5-step 一次 update smoke：

```bash
./scripts/run_clawvla_rl.sh \
  --config configs/rl/qwen3vl_pi05_real_5step_1update.yaml \
  --mode train \
  --run-id rl_real5_1update
```

RL run 目录：

```text
runs/rl/<run_id>/
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

RL 细节见 [Agent RL 训练与奖励系统](handoff/rl_training_and_rewards.md)。

## Git 与生成文件

`.gitignore` 当前忽略：

```text
__pycache__/
*.py[cod]
tmp_runs/
tmp_artifacts/
runs/
checkpoints/
outputs/
wandb/
ray_results/
*.log
*.jsonl
*.pid
*.exit
*.ckpt / *.pt / *.pth / *.safetensors
```

需要提交的通常是：

- `src/clawvla/**`
- `configs/**`
- `requirements/**`
- `scripts/**`
- `docs/**`
- `tests/**`

提交时排除 run archive、checkpoint、临时 vLLM config、agent logs、artifact 图片和 pycache。

## 开发原则

当前代码整体遵循几个硬约束：

- 模型输出必须满足当前 skill 的 JSON schema；非法输出返回显式失败。
- unavailable/exception 作为显式失败路径记录。
- action chunk 执行前要检查 freshness、subgoal id、observation id、shape、finite 数值。
- OpenPI/pi0.5 的执行 prompt 来自当前 subgoal 的自然语言 `instruction`。
- verifier 判断当前 subgoal，不用全任务 success 替代 subgoal success。
- RL 只训练模型输出 token；工具结果、环境状态和上下文 token mask 为 0。
- RL 图文调用必须保留真实 multimodal payload；有 image ref 但没有训练 payload 会报错。
