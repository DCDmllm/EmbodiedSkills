# ClawVLA

ClawVLA 是一个 RoboTwin 操作 agent 运行时。当前主线是：

```text
RoboTwin observation -> VLM grounding -> scheduler subgoal loop -> OpenPI/pi0.5 action chunk -> RoboTwin execute -> verifier
```

它不是固定脚本顺序，而是 **scheduler 自由选 skill + runtime 检查前置条件**：

- scheduler 可以在当前阶段选择允许的 skill。
- 关键 skill 有硬前置，例如 `execute_action` 必须已有 fresh `action_chunk`。
- stale artifact 不会复用：新 observation 会让 overlay/motion artifact 失效，执行过的 action chunk 会标记 consumed。
- placeholder/unavailable/exception 都必须进入 terminal、agent log 和 result，不做静默兜底。
- source/target grounding 不做 label fallback；模型必须显式输出 candidate id。

## 当前架构

核心组件：

- `vision`：采集 RoboTwin 公开观测、生成候选、绑定 source/target、渲染 bbox overlay。
- `state`：把 perception 更新为 world state，维护 task object、stage 和简短历史。
- `scheduler`：选择下一步 skill，构建/推进 subgoal plan。
- `motion`：从当前 subgoal 构建 motion goal、motion plan、OpenPI/pi0.5 action chunk，并执行。
- `verifier`：动作后判断当前 subgoal 是否完成，以及下一步应该继续执行、重观察、重规划或恢复。
- `recovery`：失败后显式路由，不偷偷执行动作。

六个阶段：

```text
observe -> plan -> preflight -> execute -> verify -> recover
```

阶段不是必须线性只走一遍。一次 action chunk 执行完成后会进入 `verify`，verifier 再决定 `advance_subgoal`、`continue_execute`、`reobserve`、`replan`、`recover` 或 `finish`。

## 环境

项目现在实际用三套环境。

### 1. robotwin-py312

主 agent、RoboTwin 环境、wrapper 脚本运行在这里。

```bash
cd /mnt/wangwai/vla/clawvla
conda activate robotwin-py312
pip install -r requirements/robotwin-py312.txt
```

RoboTwin/CUDA/SAPIEN/PyTorch 依赖默认由现有 `robotwin-py312` 环境提供，requirements 里不强行装 torch，避免动 CUDA 栈。

### 2. vllm

本地 Qwen3-VL scheduler/vision/verifier/recovery 服务运行在这里。

```bash
conda activate vllm
pip install -r /mnt/wangwai/vla/clawvla/requirements/vllm.txt
```

当前 profile 使用模型：

```text
/mnt/wangwai/weights/Qwen/Qwen3-VL-8B-Instruct
```

### 3. openpi-torch-py312

pi0.5/OpenPI worker 运行在这里。

```bash
cd /mnt/wangwai/vla/clawvla
conda activate openpi-torch-py312
pip install -r requirements/openpi-torch-py312.txt
```

worker 通过 `PYTHONPATH` 引入 RoboTwin 自带 OpenPI 源码：

```text
/mnt/wangwai/RoboTwin/policy/pi05/src
```

## 配置

主配置：

```text
configs/robotwin_default.json
configs/robotwin_pi05_worker_probe.json
configs/run_profiles/qwen3vl_pi05_vllm.json
```

远程 OpenAI-compatible 配置不再写明文 key。需要远程模型时设置：

```bash
export OPENAI_COMPATIBLE_API_BASE_URL="http://host:port/v1"
export OPENAI_COMPATIBLE_API_KEY="..."
```

本地 vLLM 路径会在运行时生成临时 config，自动把 `vision/scheduler/verifier/recovery` 指到本地 vLLM 服务。

## 正式运行

推荐入口：

```bash
cd /mnt/wangwai/vla/clawvla
./scripts/run_qwen3vl_pi05_agent.sh \
  --instruction "place the container on the plate" \
  --artifact-prefix agent_subgoal_loop_25 \
  --max-steps 25 \
  --run
```

先看最终展开命令：

```bash
./scripts/run_qwen3vl_pi05_agent.sh \
  --instruction "place the container on the plate" \
  --artifact-prefix agent_subgoal_loop_25 \
  --max-steps 25 \
  --dry-run
```

profile 默认会：

- 在 `vllm` 环境启动 Qwen3-VL OpenAI-compatible server。
- 在 `robotwin-py312` 环境跑 agent loop。
- 在 `openpi-torch-py312` 环境启动常驻 pi0.5 worker。
- 进程结束时清理 vLLM 和 pi0.5 worker。

输出位置：

```text
tmp_runs/<prefix>_agent.log        # agent stdout/stderr 全量日志
tmp_runs/<prefix>_result.json      # 最终大 JSON
tmp_runs/<prefix>_vllm.log         # vLLM 服务日志
tmp_runs/<prefix>_pi05_worker.log  # pi0.5 worker 日志
tmp_artifacts/<prefix>/            # observation/action/overlay artifacts
```

terminal 只显示 Rich trace 和关键事件；完整 JSON 写入文件。

## Agent RL

RL 训练代码是项目的一部分，位于：

```text
src/clawvla/rl/          # OpenRLHF GRPO runner、agent executor、policy proxy、trajectory archive
src/clawvla/rewards/     # RoboTwin reward snapshot / dense reward
configs/rl/              # 训练、smoke、reward、cluster 配置
scripts/run_clawvla_rl.sh
```

核心约束：

- 训练一个统一 VLM policy；`vision/scheduler/verifier/recovery` 都走同一个 policy。
- OpenPI/pi0.5 冻结，只作为动作 backend。
- 训练样本保留真实图文输入；有 image ref 但没有 multimodal payload 会直接报错。
- loss mask 只覆盖模型输出 token；工具返回、环境状态、skill 结果不进 loss。
- 超长 prompt/response 不静默截断，配置不够会显式失败。
- 未配置 reward 的任务会在 preflight 失败，不做未知任务静默 fallback。

常用入口：

```bash
cd /mnt/wangwai/vla/clawvla
./scripts/run_clawvla_rl.sh --config configs/rl/qwen3vl_pi05_grpo.yaml --mode dry-run
./scripts/run_clawvla_rl.sh --config configs/rl/qwen3vl_pi05_real_5step_1update.yaml --mode train --run-id rl_real5_1update
```

更完整说明见 [docs/agent_rl.md](docs/agent_rl.md)。

## 脚本说明

主入口：

- `scripts/run_qwen3vl_pi05_agent.sh`：推荐正式入口，读取 run profile。
- `python -m clawvla.scripts.run_profile`：profile runner，可覆盖 instruction/max-steps/gpus 等。
- `python -m clawvla.scripts.run_loop_with_vllm`：手动启动本地 vLLM 并跑 agent。
- `python -m clawvla.scripts.run_loop`：只跑 agent loop，不负责启动 vLLM。

OpenPI/pi0.5：

- `python -m clawvla.scripts.pi05_worker`：常驻 pi0.5 worker。
- `python -m clawvla.scripts.pi05_backend_probe`：诊断 pi0.5 checkpoint/schema/adapter。
- `python -m clawvla.scripts.pi05_inference_smoke`：只跑 pi0.5 inference，不执行 RoboTwin。
- `python -m clawvla.scripts.robotwin_pi05_execute_once`：采集、推理并执行一次，用于端到端诊断。

轻量 smoke/probe：

- `python -m clawvla.scripts.inspect_stack`
- `python -m clawvla.scripts.artifact_smoke`
- `python -m clawvla.scripts.geometry_smoke`
- `python -m clawvla.scripts.robotwin_capture_once`
- `python -m clawvla.scripts.robotwin_execute_smoke`
- `python -m clawvla.scripts.round_once`

这些脚本目前都保留；它们用于定位环境、artifact、geometry、RoboTwin capture 或 action bridge 问题。

## 开发检查

不启动真实模型/环境的基础检查：

```bash
cd /mnt/wangwai/vla/clawvla
PYTHONPATH=src python -m clawvla.scripts.inspect_stack --config configs/robotwin_default.json
PYTHONPATH=src python -m clawvla.scripts.artifact_smoke --config configs/robotwin_default.json
PYTHONPATH=src python -m clawvla.scripts.geometry_smoke --config configs/robotwin_default.json
PYTHONPATH=src python -m clawvla.scripts.robotwin_execute_smoke
```

编译检查：

```bash
PYTHONPATH=src python -m compileall -q src/clawvla
```

## 当前注意点

- `preflight` 是正式执行前检查；不可用或检查失败必须显式返回失败，不写 placeholder 成功状态。
- `localize_task_objects` 必须显式产出顶层 `source_candidate_id` 和 `target_candidate_id`，不会根据 label 暗中补。
- `build_task_plan` 在模型输出空 subgoals 时会返回 `task_plan_invalid_model_output`，不会偷偷生成模板计划。
- vLLM profile 的 `--max-model-len` 当前为 `32768`，用于容纳四视角图像和较长 agent 上下文。
- `tmp_runs/`、`tmp_artifacts/`、`runs/`、`outputs/`、`checkpoints/`、`ray_results/`、`__pycache__/`、`.deps/` 和本地模型权重文件都是生成物或本机产物，不提交。
