# ClawVLA

ClawVLA 是一个面向 RoboTwin、LIBERO 和 RoboCasa 的多组件 VLA agent 运行时。当前主线是：

```text
environment observation -> VLM grounding -> scheduler subgoal loop -> frozen action backend -> environment execute -> verifier
```

运行时由 **scheduler 选择 skill + runtime 检查前置条件** 共同驱动：

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
- `motion`：从当前 subgoal 构建 motion goal、motion plan、动作 backend action chunk，并执行。
- `verifier`：动作后判断当前 subgoal 是否完成，以及下一步应该继续执行、重观察、重规划或恢复。
- `recovery`：失败后显式路由，不偷偷执行动作。

六个阶段：

```text
observe -> plan -> preflight -> execute -> verify -> recover
```

阶段可按反馈重复进入。一次 action chunk 执行完成后会进入 `verify`，verifier 再决定 `advance_subgoal`、`continue_execute`、`reobserve`、`replan`、`recover` 或 `finish`。

## 环境

项目按进程职责拆为六套环境，避免把互相冲突的 CUDA、仿真器和训练依赖装进同一个 Python：

| 环境 | 职责 | requirements |
| --- | --- | --- |
| `robotwin-py312` | 主 runtime、RoboTwin/LIBERO rollout、数据与评测工具 | `requirements/robotwin-py312.txt` |
| `vllm` | 本地 Qwen3-VL 服务 | `requirements/vllm.txt` |
| `openpi-torch-py312` | pi0.5/OpenPI worker | `requirements/openpi-torch-py312.txt` |
| `.venv-openrlhf-py310-cu128` | OpenRLHF/DeepSpeed/Ray 训练 | `requirements/openrlhf-py310-cu128.txt` |
| `groot-py312` | RoboCasa rollout 和 GR00T worker | 复用 RoboCasa/LeRobot 环境 |
| `calvin-py38` | CALVIN rollout 和 task oracle | `requirements/calvin-py38.txt` |

完整安装、路径布局、代理设置和验证命令见 [环境安装与进程分工](docs/environment_setup.md)。LIBERO 复用
`robotwin-py312`，不单独增加环境。

### 1. robotwin-py312

主 agent、RoboTwin 环境、wrapper 脚本运行在这里。

```bash
cd /mnt/wangwai/vla/clawvla
conda activate robotwin-py312
pip install -r requirements/robotwin-py312.txt
```

RoboTwin/CUDA/SAPIEN/PyTorch 依赖默认由现有 `robotwin-py312` 环境提供，requirements 里不强行装 torch，避免动 CUDA 栈。
专家子任务采集和合并工具直接使用的 `h5py`、OpenCV 已显式写入 requirements。

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
cd /mnt/linyutong/wangwai_mirror/vla/clawvla
conda activate openpi-torch-py312
pip install -r requirements/openpi-torch-py312.txt
```

正式 RoboTwin worker 通过 `PYTHONPATH` 引入 mirror 中与训练一致的 OpenPI 源码：

```text
/mnt/linyutong/wangwai_mirror/pi0.5/src
```

新的 PyTorch 直推路径还直接依赖 `sentencepiece`，已写入该环境 requirements。

### 4. openrlhf-py310-cu128

Agent RL trainer 使用仓库内的 Python 3.10 venv：

```bash
cd /mnt/wangwai/vla/clawvla
python3.10 -m venv .venv-openrlhf-py310-cu128
.venv-openrlhf-py310-cu128/bin/python -m pip install -r requirements/openrlhf-py310-cu128.txt
```

runner 通过 `PYTHONPATH` 引入 ClawVLA，不在 Python 3.10 环境执行 `pip install -e .`。

### 5. groot-py312

RoboCasa rollout 和 GR00T worker 运行在这里。

```bash
cd /mnt/wangwai/vla/clawvla
conda activate groot-py312
export PYTHONPATH=/mnt/wangwai/vla/clawvla/src:/mnt/wangwai/lerobot/src:/mnt/wangwai/RoboCasa:/mnt/wangwai/robosuite
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
```

当前本地 GR00T checkpoint：

```text
/mnt/wangwai/weights/robocasa/robocasa365_checkpoints/gr00t_n1-5/multitask_learning/checkpoint-120000
```

### 6. calvin-py38

CALVIN rollout 使用独立的 Python 3.8 环境，并通过 HTTP 调冻结的 X-VLA action server：

```bash
cd /mnt/wangwai/vla/clawvla
conda activate calvin-py38
python -m pip install -r requirements/calvin-py38.txt
export PYTHONPATH=/mnt/wangwai/vla/clawvla/src:/mnt/wangwai/vla/CALVIN/calvin_env:/mnt/wangwai/vla/CALVIN/calvin_models
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
```

不要在 `calvin-py38` 中执行 `pip install -e .`；项目包元数据要求 Python 3.12+，该子进程通过 `PYTHONPATH`
加载源码。完整说明见 [CALVIN + X-VLA 接入说明](docs/calvin_xvla.md)。

如果机器设置了 HTTP proxy，本地 vLLM、PolicyProxy、OpenPI 和 X-VLA 服务必须绕过代理：

```bash
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
```

## 配置

主配置：

```text
configs/robotwin_default.json
configs/robotwin_pi05_worker_probe.json
configs/robotwin_pi05_subtasks_25k.json
configs/libero_pi05_enabled_probe.json
configs/robocasa_groot_enabled_probe.json
configs/calvin_xvla_enabled_probe.json
configs/run_profiles/qwen3vl_pi05_vllm.json
configs/run_profiles/qwen3vl_pi05_libero_vllm.json
```

`robotwin_pi05_subtasks_25k.json` 是当前正式 RoboTwin 配置，加载验证 loss 最低的 25k checkpoint。
它使用 32-step action horizon、10 次 flow 去噪、256 token 上限，以及仅由训练 split 计算的归一化统计。

远程 OpenAI-compatible 配置不再写明文 key。需要远程模型时设置：

```bash
export OPENAI_COMPATIBLE_API_BASE_URL="http://host:port/v1"
export OPENAI_COMPATIBLE_API_KEY="..."
```

本地 vLLM 路径会在运行时生成临时 config，自动把 `vision/scheduler/verifier/recovery` 指到本地 vLLM 服务。

## 正式运行

推荐入口：

```bash
cd /mnt/linyutong/wangwai_mirror/vla/clawvla
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
- OpenPI/pi0.5 和 GR00T 冻结，只作为动作 backend。
- 训练样本保留真实图文输入；有 image ref 但没有 multimodal payload 会直接报错。
- `action_ranges` 只覆盖模型输出 token；工具返回、环境状态、skill 结果不作为 action token 训练。
- 超长 prompt/response 不静默截断，配置不够会显式失败。
- 50 个 RoboTwin 任务已在 `configs/rl/rewards/robotwin.yaml` 映射到 reward handler。
- LIBERO object tasks 已通过 `configs/rl/tasks/libero_object_*.yaml` 和 `configs/rl/rewards/libero.yaml` 接入同一套 RL runner。
- RoboCasa `PickPlaceCounterToCabinet` 已通过 `configs/robocasa_groot_enabled_probe.json`、`configs/rl/rewards/robocasa.yaml` 和 GR00T action backend 接入同一套 loop；GR00T 模型动作维度 32，RoboCasa 环境执行动作维度 12。
- CALVIN 已通过 `configs/calvin_xvla_enabled_probe.json`、`configs/rl/rewards/calvin.yaml` 和 X-VLA HTTP action backend 接入同一套 loop；外部 X-VLA server 仍由用户单独启动。
- OpenRLHF 多卡训练会对 mixed text/multimodal samples 做 modality-aligned replay 排列，保持 ZeRO-3 collective 顺序一致。

常用入口：

```bash
cd /mnt/wangwai/vla/clawvla
./scripts/run_clawvla_rl.sh --help
./scripts/run_clawvla_rl.sh --preset robotwin-multitask --mode dry-run
./scripts/run_clawvla_rl.sh --preset robotwin-real5 --mode train --run-id robotwin_rl_real5
./scripts/run_clawvla_rl.sh --preset libero-multitask --mode train --run-id libero_rl_multitask
./scripts/run_clawvla_rl.sh --preset robocasa-rollout --mode dry-run
./scripts/run_clawvla_rl.sh --preset robocasa-1update --mode dry-run
./scripts/run_clawvla_rl.sh --preset calvin-xvla --mode dry-run
./scripts/run_clawvla_rl.sh --preset rynnbrain-train-smoke --mode dry-run
clawvla-rl --preset libero-multitask --mode dry-run
```

更完整说明见 [docs/agent_rl.md](docs/agent_rl.md)。

已验证的真实 smoke：

- `configs/rl/qwen3vl_pi05_libero_multitask_1update.yaml`：2 张 policy GPU，LIBERO mixed text/multimodal GRPO 一次更新，checkpoint 正常写出。
- `configs/rl/qwen3vl_pi05_real_5step_1update.yaml`：4 张 policy GPU，RobotWin/OpenPI 5-step rollout + 一次更新，mixed modality path 正常完成。

## 脚本说明

主入口：

- `scripts/run_qwen3vl_pi05_agent.sh`：推荐正式入口，读取 run profile。
- `scripts/run_clawvla_rl.sh`：RL 入口，支持 `--preset robotwin-real5`、`--preset libero-multitask` 等短名。
- `python -m clawvla.scripts.run_profile`：profile runner，可覆盖 instruction/max-steps/gpus 等。
- `python -m clawvla.scripts.run_loop_with_vllm`：手动启动本地 vLLM 并跑 agent。
- `python -m clawvla.scripts.run_loop`：只跑 agent loop，不负责启动 vLLM。

OpenPI/pi0.5：

- `python -m clawvla.scripts.pi05_worker`：常驻 pi0.5 worker。
- `python -m clawvla.scripts.pi05_backend_probe`：诊断 pi0.5 checkpoint/schema/adapter。
- `python -m clawvla.scripts.pi05_inference_smoke`：只跑 pi0.5 inference，不执行 RoboTwin。
- `python -m clawvla.scripts.robotwin_pi05_execute_once`：采集、推理并执行一次，用于端到端诊断。
- `python -m clawvla.scripts.libero_pi05_execute_once`：LIBERO 采集、pi0.5 推理、7D action 执行诊断。
- `python -m clawvla.scripts.pi05_libero_action_smoke`：LIBERO action adapter 轻量 smoke。

GR00T / RoboCasa：

- `configs/robocasa_groot_enabled_probe.json`：RoboCasa + GR00T 本地 probe 配置，默认任务 `robocasa/PickPlaceCounterToCabinet`。
- `python -m clawvla.scripts.groot_worker --config configs/robocasa_groot_enabled_probe.json --load-policy`：常驻 GR00T worker。
- `python -m clawvla.scripts.groot_inference_smoke --config configs/robocasa_groot_enabled_probe.json --artifact-dir <artifact_dir> --prompt "move to the bottle"`：只测 GR00T action backend，不执行环境。

完整整理见 [docs/robocasa_groot.md](docs/robocasa_groot.md)。

CALVIN / X-VLA：

- `configs/calvin_xvla_enabled_probe.json`：CALVIN validation 环境和 X-VLA `/act` endpoint 配置。
- `python -m clawvla.rl.openrlhf_runner --preset calvin-xvla --mode dry-run`：展开 CALVIN one-update 配置。

完整整理见 [docs/calvin_xvla.md](docs/calvin_xvla.md)。

轻量 smoke/probe：

- `python -m clawvla.scripts.inspect_stack`
- `python -m clawvla.scripts.artifact_smoke`
- `python -m clawvla.scripts.geometry_smoke`
- `python -m clawvla.scripts.robotwin_capture_once`
- `python -m clawvla.scripts.libero_capture_once`
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

单元测试（显式绕过本机 HTTP proxy）：

```bash
NO_PROXY=127.0.0.1,localhost PYTHONPATH=src \
  /mnt/wangwai/miniconda3/envs/robotwin-py312/bin/python -m pytest -q
```

## 当前注意点

- `preflight` 是正式执行前检查；不可用或检查失败必须显式返回失败，不写 placeholder 成功状态。
- `localize_task_objects` 必须显式产出顶层 `source_candidate_id` 和 `target_candidate_id`，不会根据 label 暗中补。
- `build_task_plan` 在模型输出空 subgoals 时会返回 `task_plan_invalid_model_output`，不会偷偷生成模板计划。
- vLLM profile 的 `--max-model-len` 当前为 `32768`，用于容纳四视角图像和较长 agent 上下文。
- `tmp_runs/`、`tmp_artifacts/`、`runs/`、`outputs/`、`checkpoints/`、`ray_results/`、`__pycache__/`、`.deps/` 和本地模型权重文件都是生成物或本机产物，不提交。
