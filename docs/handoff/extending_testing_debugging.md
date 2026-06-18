# 扩展、测试与调试手册

本文说明如何在当前 EmbodiedSkills 里加 skill、加组件、接新 action backend、加 reward，以及常见调试路径。

## 添加新 Skill

当前 skill 注册集中在 `src/clawvla/components/*` 和 `src/clawvla/skills/builtin.py`。

### 1. 选择归属组件

已有组件：

```text
vision
state
scheduler
safety
motion
verifier
recovery
```

如果新能力属于已有阶段语义，优先放到已有组件里。只有边界清楚、状态和模型 role 都不同的时候，再新建组件。

### 2. 写 handler

示例结构：

```python
def my_skill(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    payload = request.payload

    # 显式检查输入
    value = blackboard.read("some_key")
    if value is None:
        return unavailable("my_skill_unavailable", "missing_some_key", {})

    # 可选模型调用
    if context.has_model and payload.get("use_model", True):
        raw = call_component_json(
            context,
            instruction="...",
            payload={
                "blackboard": blackboard.compact_context(),
                "required_schema": {...},
            },
            image_paths=payload.get("image_paths"),
            render_format=payload.get("render_format", "json"),
        )
        # validate raw

    # 写 blackboard
    blackboard.write("my_output", output, event_type="component.my_skill")
    return ok("my_skill_done", {"my_output": to_dict(output)})
```

原则：

- handler 返回明确状态。缺输入、缺模型、非法输出都返回 `unavailable(...)` 或明确 `SkillResult(success=False, ...)`。
- 如果调用模型，必须给 `required_schema`，并验证关键字段。
- 如果写入会让旧 artifact 失效，调用 `mark_motion_artifacts_stale()` 或 `mark_grounding_overlay_stale()`。
- 写 blackboard 时给 event_type，方便日志和 result 追踪。
- 输出必须能 JSON 化；dataclass 优先实现/使用 `to_dict()`。

### 3. 注册 skill

在对应 component 文件的 `register_xxx_skills(registry)` 中添加：

```python
register_skill(
    registry,
    "component_name",
    "my_skill",
    "Human-readable description.",
    my_skill,
    requires_model=True,
)
```

如果是新组件，还要：

1. 新建 `src/clawvla/components/my_component.py`。
2. 在 `src/clawvla/skills/builtin.py` import 并调用 `register_my_component_skills(registry)`。
3. 在 agent config 的 `components` 和 `models` 中加入对应配置。

### 4. 加入阶段策略

修改 `src/clawvla/phase_policy.py` 的 `DEFAULT_ALLOWED_SKILLS`。

只把 skill 加到语义正确的阶段。例如：

- 需要当前 observation 的视觉分析：`observe`。
- 执行动作前检查：`preflight`。
- 生成/校验/执行 action chunk：`execute`。
- 执行后判断：`verify`。
- 真失败修复：`recover`。

### 5. 加入 runtime 前置条件

如果 skill 需要特定 blackboard 状态，在 `AgentLoop._skill_prerequisite_error()` 中加检查。

runtime 需要校验明显无效调用，prompt 说明用于帮助模型选择正确顺序。

### 6. 更新 scheduler prompt/schema

如果新 skill 改变阶段语义，需要更新：

- `scheduler._scheduler_instruction(loop_mode=True)`
- 必要时更新 `_loop_schema()` 或相关 required schema。

说明要具体到：

- 什么时候选它。
- payload 需要哪些字段。
- 成功后会产生什么 blackboard artifact。
- 失败时下一步应该如何处理。

### 7. 加测试

至少加：

- handler 单测：缺输入失败、正常输入成功。
- AgentLoop 校验：阶段不对或前置缺失时被拒绝。
- scheduler payload/allowed skills 是否暴露正确。
- 如果涉及模型输出：非法 JSON/schema 输出是否显式失败。

测试文件当前集中在 `tests/test_rl_framework.py`。

## 添加新 Reward

RoboTwin 环境/物体/爪子接触接口速查见 [RoboTwin 奖励接口速查](robotwin_reward_interfaces.md)。

### 新 RoboTwin task 复用已有 family

如果任务属于现有 family：

1. 打开 `src/clawvla/rewards/robotwin_reward.py`。
2. 在 `TASK_REWARD_SPECS` 增加：

```python
"new_task_name": RewardSpec(
    task_name="new_task_name",
    family="pick_place",
    source="object_actor_name",
    target="target_actor_name",
    metadata={
        "release_xy_threshold": 0.05,
        "release_z_threshold": 0.035,
        ...
    },
)
```

3. 在 `configs/rl/rewards/robotwin.yaml` 的 `reward.task_map` 加：

```yaml
new_task_name: robotwin
```

4. 在具体训练 config 里确认 `rollout.task_name` 和 `reward.task_map` 一致。

### 新 reward family

新增 family 时：

1. 加 `_new_family_reward(before, after, spec, step_cost)`。
2. 在 `compute_robotwin_reward()` 中分发：

```python
if spec.family == "new_family":
    return _new_family_reward(before, after, spec, step_cost)
```

3. 设计 snapshot 需要的 actor/articulation 字段。
4. 必要时扩展 `RewardSpec` 字段，但优先把任务特有阈值放到 `metadata`。
5. 输出 `RewardResult` 必须包含：

```python
reward: float
events: dict[str, bool]
metrics: dict[str, float | None]
milestones: dict[str, bool]
reason: str
family: str
task_name: str
```

### 自定义外部 reward handler

`RewardRegistry` 支持 `module:callable`。

示例：

```python
from clawvla.rl.reward_registry import RewardHandler, RewardRegistry

def register_my_rewards(registry: RewardRegistry) -> None:
    registry.register(
        RewardHandler(
            name="my_reward",
            snapshot=my_snapshot,
            compute=my_compute,
            finalize=my_finalize,
        )
    )
```

配置：

```yaml
reward:
  registry:
    - my_package.my_rewards:register_my_rewards
  task_map:
    my_task: my_reward
```

`snapshot(env, blackboard)` 和 `compute(before, after, context)` 不应吞异常。RL reward tracker 会把 snapshot/compute failure 写到 JSONL 和 terminal。

## 接新 Action Backend

当前 action backend 协议在 `src/clawvla/action_backends/base.py`：

```python
class ActionBackend(Protocol):
    name: str

    def build_action_chunk(
        self,
        motion_goal: MotionGoal | None,
        world_state: WorldState | None,
        observation: ObservationBundle | None,
        request: dict[str, Any],
    ) -> ActionBackendResult:
        ...
```

返回：

```python
ActionBackendResult:
  success: bool
  status: str
  action_chunk: ActionChunk | None
  metadata: dict
  errors: list[str]
```

新增 backend：

1. 新建 `src/clawvla/action_backends/my_backend.py`。
2. 实现 `build_action_chunk()`。
3. 在 `src/clawvla/action_backends/factory.py` 中按 `metadata.action_backend.type` 分发。
4. 确保输出 action chunk 能通过 `motion._validate_action_chunk_report()`：
   - qpos: 每个 command 14 维。
   - ee: 每个 command 16 维。
   - commands 非空、finite。
5. 在 config `metadata.action_backend` 加配置。
6. 增加 probe/smoke 脚本或测试。

backend 遇到缺 prompt、缺图像、缺状态时返回 `success=False` 和明确错误。

## 接新模型服务

如果兼容 OpenAI chat completions：

1. 在 config model 中设置：

```json
{
  "backend": "openai_compatible",
  "model": "served-model-name",
  "api_base_url": "http://host:port/v1",
  "api_key": "...",
  "max_new_tokens": 2048,
  "temperature": 0.0
}
```

2. 或使用 env：

```json
"api_base_url_env": "OPENAI_COMPATIBLE_API_BASE_URL",
"api_key_env": "OPENAI_COMPATIBLE_API_KEY"
```

如果本地 HF：

```json
{
  "backend": "local_hf",
  "model": "/path/to/model",
  "torch_dtype": "bfloat16",
  "device_map": "auto"
}
```

当前 `ModelRuntime` 的 local path 使用 `AutoProcessor` 和 `AutoModelForImageTextToText`，适合图文模型，但不负责高吞吐 serving。正式多卡本地服务建议用 vLLM profile。

## 接新 RoboTwin task / embodiment / camera

### task

改 config：

```json
"robotwin": {
  "task_name": "new_task",
  "task_config": "demo_clean",
  "seed": 0,
  "now_ep_num": 0
}
```

`RoboTwinSession.instantiate_task()` 会 import：

```text
envs.<task_name>
```

并要求模块中有同名 class。

### task_config

`prepare_task_args()` 读取：

```text
<repo_root>/task_config/<task_config>.yml
<repo_root>/task_config/_embodiment_config.yml
<repo_root>/task_config/_camera_config.yml
```

然后写：

```text
collect_data=False
save_data=False
eval_video_log=False
need_plan=False
```

### embodiment

RoboTwin task config 里的 `embodiment` 可以是：

- 长度 1：左右机械臂同一个 embodiment，`dual_arm_embodied=True`。
- 长度 3：左右机械臂分别指定，并带 `embodiment_dis`，`dual_arm_embodied=False`。

运行时会从 `_embodiment_config.yml` 中解析 file_path，并读取左右 `config.yml`。

### camera profile

`RobotwinConfig.camera_profile` 会传给 `apply_camera_profile()`：

- 设置 `camera.head_camera_type`。
- 设置 `camera.wrist_camera_type`。
- 对 left/right embodiment config 里的 `head_camera` 和 `front_camera` static camera 设置 type。

当前配置使用：

```text
Large_D435_Wide
```

preflight 当前期望 RGB artifact 分辨率：

```text
960x540
```

如果更换 camera profile，要同步检查 `safety._camera_status()` 的期望分辨率是否仍正确。

## 测试

当前主要测试文件：

```text
tests/test_rl_framework.py
```

测试覆盖方向：

- RL config load。
- reward registry 未配置任务失败。
- policy proxy tracing。
- OpenRLHF `action_ranges` 只训练模型输出 token。
- multimodal payload 顺序和缺失报错。
- terminal reward penalty。
- runtime environment env 写入。
- RoboTwin camera profile 应用。
- VLA prompt 必须来自 subgoal instruction。
- pi05 prompt 不接受 request prompt 兜底。
- task plan schema/validation。
- place verify release condition。
- preflight pass/block。
- localization contract。
- blackboard compact。
- scheduler 无模型显式失败。
- stage advance 限制。
- preflight observation refresh。
- verify image capture。
- verifier canonical next_action。
- recovery patch。
- verify observation 清理和 archived images。
- stale action chunk block。
- placeholder perception 拒绝。

常用命令：

```bash
cd /mnt/wangwai/vla/clawvla
PYTHONPATH=src /mnt/wangwai/miniconda3/envs/robotwin-py312/bin/python -m pytest tests/test_rl_framework.py -q
```

只跑某个测试：

```bash
PYTHONPATH=src /mnt/wangwai/miniconda3/envs/robotwin-py312/bin/python -m pytest \
  tests/test_rl_framework.py::test_pi05_prompt_requires_motion_plan_vla_prompt -q
```

静态编译检查：

```bash
PYTHONPATH=src /mnt/wangwai/miniconda3/envs/robotwin-py312/bin/python -m compileall -q src/clawvla
```

## Smoke / Probe 脚本

### inspect config/skills

```bash
PYTHONPATH=src python -m clawvla.scripts.inspect_stack \
  --config configs/robotwin_default.json
```

输出组件、注册 skill、model config、RoboTwin config、compact runtime。

### artifact smoke

```bash
PYTHONPATH=src python -m clawvla.scripts.artifact_smoke \
  --config configs/robotwin_default.json \
  --artifact-prefix smoke
```

不启动 RoboTwin，用 fake raw observation 测 artifact 写入和 observation normalization。

### geometry smoke

```bash
PYTHONPATH=src python -m clawvla.scripts.geometry_smoke \
  --config configs/robotwin_default.json
```

测试 depth/bbox 到 metric geometry 的路径。

### RoboTwin capture once

```bash
PYTHONPATH=src python -m clawvla.scripts.robotwin_capture_once \
  --config configs/robotwin_pi05_worker_probe.json \
  --instruction "place the container on the plate" \
  --artifact-prefix capture_once
```

用于确认 SAPIEN/RoboTwin 渲染、camera profile、artifact 写入是否正常。

### pi05 backend probe

```bash
PYTHONPATH=src python -m clawvla.scripts.pi05_backend_probe \
  --config configs/robotwin_pi05_worker_probe.json
```

检查 pi05 checkpoint、norm stats、OpenPI source、adapter summary。

### pi05 inference smoke

```bash
PYTHONPATH=src:/mnt/wangwai/RoboTwin/policy/pi05/src \
python -m clawvla.scripts.pi05_inference_smoke \
  --config configs/robotwin_pi05_enabled_probe.json \
  --artifact-dir tmp_artifacts/<prefix>/... \
  --prompt "place the container on the plate" \
  --num-steps 2 \
  --horizon 10
```

只跑 OpenPI inference，不执行 RoboTwin。

## 常见问题定位

### 1. `observation_unavailable: ... find a rendering device`

通常是 RoboTwin/SAPIEN 渲染环境问题，OpenPI worker 不影响 rendering device。检查：

- 运行环境是否有可用 GPU/EGL/Vulkan。
- `runtime_environment.env` 是否设置了：

```text
__EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

- 当前进程的 `CUDA_VISIBLE_DEVICES`。
- 是否能单独跑 `robotwin_capture_once`。

### 2. scheduler 一直重复某个 skill

看：

```text
tmp_runs/<prefix>_agent.log
tmp_runs/<prefix>_result.json
```

重点查：

- `clawvla_loop_decision`
- `clawvla_decision_blocked`
- `clawvla_skill_finish`
- result 里的 `blackboard.recent_loop_history`
- `last_scheduler_error`
- `last_skill_exception`
- `runtime_state.next_required_decision`

重复超过 `allow_same_decision_repeats` 会变成 `invalid_decision`，并写入 loop history。

### 3. preflight 失败

看 `preflight_report.errors`。常见：

```text
missing_task_plan / missing_current_subgoal
stale_perception / stale_world_state
camera_<name>_missing_rgb_path
camera_<name>_unexpected_resolution
missing_14d_robot_state
robotwin_env_unavailable
action_backend_disabled
openpi_worker_unreachable
```

视觉/相机类错误可以在 preflight 内调用 `vision.refresh_preflight_observation`。task/subgoal/binding 错误一般回 `plan`。backend/worker 错误一般先看 OpenPI worker 日志。

### 4. action chunk 校验失败

看 `last_action_validation_report`：

```text
missing_action_chunk
invalid_action_type
stale_action_chunk
consumed_action_chunk
action_chunk_subgoal_mismatch
action_chunk_observation_mismatch
empty_action_commands
unsupported_action_type
invalid_action_command_indexes
```

如果是 stale/consumed，从最早 stale artifact 重新 build/plan/emit。

### 5. verify 没有进入下一步

verify 正常顺序：

```text
capture_verify_views -> verify_progress -> advance_subgoal 或 repair_stage_transition
```

如果 `verify_progress` 返回：

```text
next_action=continue_execute
```

runtime 应通过 `scheduler.repair_stage_transition target_stage=preflight` 回到 preflight，再进入 execute；verify 阶段不直接调 motion。

如果缺 verify images，`verify_progress` 会返回 `verification_unavailable`。

### 6. OpenPI worker 问题

看：

```text
tmp_runs/<prefix>_pi05_worker.log
```

以及 preflight 的 action_backend check：

```text
worker.ok
worker.mode
worker.host
worker.port
worker.reason
```

worker health 请求是 socket JSON：

```json
{"op":"health"}
```

正常返回：

```json
{"status":"ok","backend":"pi05_worker"}
```

### 7. RL 训练长度报错

如果报：

```text
OpenRLHF policy prompt exceeds configured max_length
```

说明单次 policy call 超过配置长度。当前代码把超长样本视为配置错误，因为截断会破坏训练数据。处理方式：

- 提高对应长度。
- 减少 rollout max_steps。
- 减少单次模型输出长度。
- 检查 scheduler 是否产生冗长 JSON 或重复循环。

### 8. RL multimodal payload 报错

如果报：

```text
policy call used image refs but did not carry training multi_modal_data
OpenRLHF image policy call did not carry mm_train_inputs
```

说明某次图文 policy call 只记录了 image ref，没有通过训练 backend 的 multimodal processor 生成训练 payload。训练路径需要修 proxy/backend，保留真实图文 payload。

## 文档维护规则

改以下文件时，建议同步更新交接文档：

```text
src/clawvla/phase_policy.py
src/clawvla/agent_loop.py
src/clawvla/components/*.py
src/clawvla/action_backends/*.py
src/clawvla/envs/*.py
src/clawvla/rl/*.py
src/clawvla/rewards/*.py
configs/**/*.json
configs/**/*.yaml
scripts/*.sh
requirements/*.txt
```

尤其是：

- 新增/删除 skill。
- 改阶段顺序或阶段允许 skill。
- 改 scheduler/verifier/recovery schema。
- 改 OpenPI 输入输出映射。
- 改 reward task_map 或 reward family。
- 改 run archive 结构。
