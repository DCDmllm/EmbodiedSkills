# Runtime 架构与执行循环

本文说明普通 EmbodiedSkills agent 的运行路径：配置加载、runtime 初始化、scheduler 决策、skill 执行、阶段推进、日志和 artifact。

## 入口关系

推荐入口是 shell 脚本：

```text
scripts/run_qwen3vl_pi05_agent.sh
  -> python -m clawvla.scripts.run_profile
    -> python -m clawvla.scripts.run_loop_with_vllm
      -> vLLM OpenAI-compatible server
      -> python -m clawvla.scripts.run_loop
        -> AgentRuntime
        -> AgentLoop
        -> components / env adapter / action backend
```

如果已经有外部 OpenAI-compatible 模型服务，也可以绕过 `run_loop_with_vllm`，直接使用 `run_loop.py` 和一个模型 API 配好的 config。

## 配置加载

`src/clawvla/config.py` 定义：

- `ModelConfig`
- `ComponentConfig`
- `RobotwinConfig`
- `RuntimeEnvironment`
- `AgentConfig`

`load_config(path)` 从 JSON 读出 `AgentConfig`。当前普通 agent 配置格式为 JSON。

模型 backend 枚举：

```text
none
local_hf
openai_compatible
azure_openai
```

`ModelConfig` 常用字段：

```text
backend
model
api_base_url / api_base_url_env
api_key / api_key_env
api_version
max_new_tokens
temperature
request_timeout
reasoning_effort
enable_thinking
stream
device_map
torch_dtype
metadata
```

`ComponentConfig.skills` 是配置展示字段；组件实际能运行哪些 skill 由代码里的 `SkillRegistry` 决定，阶段里能选择哪些 skill 由 `PhasePolicy` 决定。

## Runtime 初始化

`AgentRuntime(config)` 做这些事：

1. 调 `build_component_registry(config)` 创建组件。
2. 创建 `Blackboard(task_instruction=config.task.get("instruction"))`。
3. 调 `build_action_backend(config)` 创建 action backend，并写入 blackboard key `action_backend`。
4. 初始化 `history`。

`run_loop.py` 会再写入：

```text
env_adapter        RoboTwinAdapter(config.robotwin)
run_robotwin       bool(args.run)
artifact_prefix    args.artifact_prefix
task_instruction   args.instruction
rl_reward_tracker  可选，仅 RL rollout 时安装
```

## ComponentRegistry 与 SkillRegistry

`src/clawvla/components/factory.py`：

- `build_skill_registry()` 注册所有内置 skill。
- `build_model_runtimes(config)` 为每个 model key 创建 `ModelRuntime`。
- `build_component_registry(config)` 只注册 config 中 `enabled=true` 的 component。

`Component.run_skill()` 会把 `SkillRequest` 和 `SkillContext` 交给实际 skill handler。

`SkillContext` 包含：

```python
component_name: str
blackboard: Blackboard
model_runtime: ModelRuntime | None
```

`context.has_model` 用来判断该组件是否有可用模型。

## Blackboard

`Blackboard` 是组件之间的共享状态。常见 key：

```text
stage
observation
verify_observation
perception
world_state
grounding_overlay
task_plan
current_subgoal
motion_goal
motion_plan
action_chunk
action_backend_result
execution_report
last_safety_report
preflight_report
last_action_validation_report
last_verification_report
last_resolved_verification_report
last_recovery_directive
last_retry_request
loop_history
last_loop_decision
last_scheduler_decision
last_skill_exception
last_perception_error
last_localization_error
last_stage_transition
```

`Blackboard.write(key, value, event_type=...)` 会写 value，并在有 event_type 时追加事件。

`Blackboard.compact_context()` 会生成给模型看的压缩上下文，保留：

- task instruction
- 当前 stage
- world_state 的候选摘要
- task_plan/current_subgoal
- motion_state 摘要
- 最近 loop history，最多 20 条
- 最近异常、perception/localization error
- safety/preflight/action validation/verification/recovery 摘要

注意：`last_verification_report` 只在 `verify` 或 `recover` 阶段作为 active report 出现在 compact context。其他阶段会只显示 `inactive_verification_report_present`，避免旧 verify 文本误导后续阶段。

## AgentLoop 主循环

`src/clawvla/agent_loop.py` 的 `AgentLoop.run()` 是核心。

每一步大致流程：

1. 从 blackboard 读当前 `stage`，没有就用 `initial_stage`。
2. 调 `_choose_decision(stage)`，内部运行 `scheduler.choose_next_skill`。
3. 把 scheduler 输出解析成 `LoopDecision`。
4. 根据 `control` 分支：
   - `finish_run`：结束。
   - `advance_stage`：只推进到默认下一阶段。
   - `run_skill`：校验并执行具体 skill。
5. 执行 skill 前后调用 RL reward tracker hook。
6. skill 成功后做必要的 post update，例如执行成功后进入 verify。
7. 写 `loop_history`。
8. 到 `max_steps` 后返回 `max_steps_reached` 或 `max_steps_reached_with_failures`。

`AgentLoopConfig`：

```text
max_steps: int = 12
initial_stage: str = "observe"
stop_on_skill_error: bool = False
allow_same_decision_repeats: int = 3
scheduler_payload: dict = {}
```

默认 `stop_on_skill_error=False`，skill 失败会记录并反馈给 scheduler。重复完全相同 decision 超过限制会变成 `invalid_decision`。

## LoopDecision schema

定义在 `src/clawvla/loop_types.py`：

```python
control: "run_skill" | "advance_stage" | "finish_run"
stage: str | None
next_component: str | None
next_skill: str | None
payload: dict
reason: str
narration: str | None
state_summary: str | None
expected_result: str | None
budget_steps: int | None
metadata: dict
```

当前约束：

- `run_skill` 必须有 `next_component` 和 `next_skill`。
- `run_skill.stage` 必须等于当前 blackboard stage。
- `advance_stage` 只推进默认下一阶段，schema 中不携带目标 stage。
- 跨阶段修复跳转用 `scheduler.repair_stage_transition`，payload 里需要 `target_stage` 和 `reason`。
- `finish_run` 用于任务结束或明确停止。

## 阶段顺序和允许 skill

定义在 `src/clawvla/phase_policy.py`：

```text
observe
plan
preflight
execute
verify
recover
```

每个阶段有一个 component -> skills map。AgentLoop 会把当前阶段可见的 skills 传给 scheduler，同时运行时还会在 `_validate_run_skill_decision()` 里做硬校验。

当前阶段可选范围：

```text
observe:
  vision: capture_views, perceive_scene, localize_task_objects,
          lift_depth_cluster, lift_geometry, bind_arm, estimate_uncertainty
  state: update_world_state, summarize_state

plan:
  scheduler: build_task_plan, select_current_subgoal, advance_subgoal,
             allocate_budget, repair_stage_transition
  state: summarize_state

preflight:
  vision: refresh_preflight_observation
  safety: preflight_action
  scheduler: repair_stage_transition

execute:
  motion: build_motion_goal, plan_motion, emit_action_chunk,
          validate_action_chunk, execute_action
  state: summarize_state
  scheduler: repair_stage_transition

verify:
  verifier: verify_progress
  state: update_world_state, summarize_state
  vision: capture_verify_views
  scheduler: advance_subgoal, repair_stage_transition

recover:
  recovery: decide_recovery, build_retry_request
  scheduler: repair_stage_transition
  vision: capture_views, perceive_scene, localize_task_objects, estimate_uncertainty
  state: summarize_state
```

## Runtime decision 校验

`AgentLoop._validate_run_skill_decision()` 会检查：

- control 是否支持。
- component/skill 是否存在。
- scheduler 内部 skill `choose_next_skill` 不对外暴露。
- run_skill 的 stage 是否等于当前 stage。
- skill 是否在当前阶段允许列表中。
- verify 阶段如果已经有 verification report，只允许按 report 的 next_action 调用 `advance_subgoal` 或 `repair_stage_transition`。
- `repair_stage_transition` 是否有合法 target 和 reason。
- `refresh_preflight_observation` 是否只在 preflight 且确实有视觉/相机类 preflight error。
- skill-specific prerequisites。
- `motion.emit_action_chunk` 的 payload 必须有 `horizon`，范围是 10 到 32。

skill-specific prerequisites 包括：

- perceive/localize/uncertainty 需要 observation。
- capture_verify_views 需要成功 execution_report。
- build_task_plan 需要 world_state ready。
- select_current_subgoal 需要 task_plan。
- advance_subgoal 需要成功 verify report。
- motion 相关 skill 需要 preflight 已通过。
- plan_motion 需要 fresh motion_goal。
- emit_action_chunk 需要 fresh motion_plan。
- execute_action 需要 fresh action_chunk。
- verify_progress 需要 fresh verify_observation 和 verify images。
- recovery 需要 failure report 或 recovery directive。

这些失败都会成为 `invalid_decision` 或 skill failure 写进 loop history。

## 阶段推进

默认推进只由 `advance_stage` 完成：

```text
observe -> plan
plan -> preflight
preflight -> execute
execute -> verify
verify -> recover
recover 无默认下一阶段
```

特殊情况：

- `motion.execute_action` 成功返回 `action_executed` 后，AgentLoop 会把 stage 写成 `verify`。
- `scheduler.advance_subgoal` 成功后，如果还有下一个 subgoal，会把 stage 写成 `preflight`。
- `scheduler.advance_subgoal` 发现 task plan 完成时，返回 `task_plan_complete`，后续 loop 可 finish。
- `scheduler.repair_stage_transition` 可以显式进入 `observe`、`plan`、`preflight`、`recover`。

## ModelRuntime

`src/clawvla/models.py` 支持：

- local Hugging Face：`AutoProcessor` + `AutoModelForImageTextToText`
- OpenAI-compatible：`openai.OpenAI(...).chat.completions.create`
- Azure OpenAI

组件调用模型统一走 `call_component_json()`：

1. 把 payload 渲染成 JSON 或 XML。
2. 拼上 skill instruction 和“只返回一个 JSON object”的统一约束。
3. 把 `image_paths` 转成 image content。
4. 调 `model_runtime.generate_text()`。
5. 记录 `model.call` 和 `model.output` 事件。
6. 从 raw text 中抽最后一个 JSON object。

OpenAI-compatible 后端会把本地图片转成 data URL；vLLM 本地服务也按 OpenAI chat completions 接口接收。

## RoboTwin Adapter

`RoboTwinAdapter` 在 `src/clawvla/envs/robotwin.py`：

- `capture_views(**kwargs)`：
  - 如果 `setup=True`，先调用 `RoboTwinSession.setup()`。
  - 调 `task_env.get_obs()`。
  - 把 raw observation 转成 `ObservationBundle`。
  - 写 RGB/depth/pointcloud artifacts。
  - 保存 `last_observation`。

- `execute_action(action_chunk)`：
  - 遍历 `action_chunk.commands`。
  - 调 `task_env.take_action(command, action_type=action_chunk.action_type)`。
  - 执行后 `get_obs()` 和 `check_success()`。
  - 保存执行后 observation。
  - 返回 execution report。

`RoboTwinSession` 在 `src/clawvla/envs/robotwin_session.py`：

- 动态 import `envs.<task_name>`。
- 从 RoboTwin 的 `task_config/<task_config>.yml`、`_embodiment_config.yml`、`_camera_config.yml` 组合 setup args。
- `apply_camera_profile()` 把 camera profile 写入 head/wrist/static camera config。
- `apply_embodiment_files()` 根据 embodiment 配置写左右机械臂文件路径。

## Action Backend

`build_action_backend(config)` 默认返回 `Pi05ActionBackend`。

`Pi05ActionBackend.build_action_chunk()` 主要路径：

1. 检查 backend enabled、pretrained_path 存在。
2. `diagnose()` 判断 checkpoint 格式。
3. OpenPI checkpoint：
   - worker/subprocess mode：走独立进程。
   - direct mode：当前进程加载 torch OpenPI runtime。
4. 输出 `ActionBackendResult`，里面包含 `ActionChunk`。

OpenPI 输入：

```text
prompt: motion_plan.vla_prompt
images:
  base_0_rgb       <- head_camera
  left_wrist_0_rgb <- left_camera
  right_wrist_0_rgb<- right_camera
state: 14D RoboTwin joint_action vector
```

OpenPI 输出经过 decode/unnormalize/absolute action/encode 后，生成 RoboTwin `qpos` action chunk。当前 qpos 维度检查是 14，ee 维度检查是 16。

`_resolve_prompt()` 明确要求 `motion_plan.vla_prompt`；缺失时抛错。

## 日志与结果

普通运行的 stdout/stderr 里有两类输出：

- JSON 事件：`clawvla_skill_start`、`clawvla_skill_finish`、`clawvla_loop_decision`、`clawvla_decision_blocked`、`clawvla_status_notice` 等。
- 人类可读 trace：scheduler/skill/openpi/execute 等短行。

`run_loop_with_vllm.py` 会用 `TerminalRenderer` 把 JSON 事件渲染成 Rich panel，同时完整输出写到：

```text
tmp_runs/<prefix>_agent.log
```

最终 result 写到：

```text
tmp_runs/<prefix>_result.json
```

result 里有：

```text
loop
blackboard.compact_context()
history_length
model_calls 最近 32 条 model.call/model.output
```
