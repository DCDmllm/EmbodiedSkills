# 组件、技能与数据接口

本文按当前 `src/clawvla/components/` 和 schema 说明每个组件、skill 输入输出、blackboard 写入，以及添加新 skill 时要遵守的接口。

## 通用 Skill 接口

所有 skill handler 形态：

```python
def some_skill(request: SkillRequest, context: SkillContext) -> SkillResult:
    ...
```

`SkillRequest`：

```python
component: str
skill: str
payload: dict
request_id: str
stage: str | None
budget_steps: int | None
metadata: dict
```

`SkillContext`：

```python
component_name: str
blackboard: Blackboard
model_runtime: ModelRuntime | None
has_model: bool
```

`SkillResult`：

```python
success: bool
status: str
output: dict
request_id: str | None
component: str | None
skill: str | None
errors: list[str]
metadata: dict
```

推荐返回工具：

- `ok(status, output)`：成功，写 status notice。
- `unavailable(status, reason, output)`：显式不可用/失败，`success=False`，reason 进入 errors。

handler 内部异常需要显式进入结果。未捕获异常会被 `AgentRuntime.run_skill()` 转成：

```text
status: skill_exception
output.exception: exception_type / message / traceback_tail
errors: [short exception]
```

## 主要 Schema

### ObservationBundle

来自 RoboTwin raw observation：

```python
observation_id: str
task_instruction: str | None
camera_views: dict[str, CameraView]
robot_arms: dict[str, RobotArmState]
pointcloud_ref: str | None
raw: dict
metadata: dict
```

当前常见 camera name：

```text
head_camera
front_camera
left_camera
right_camera
```

### CameraView

```python
name: str
rgb_path: str | None
depth_path: str | None
mask_path: str | None
intrinsics: list[float] | None
extrinsics: list[float] | None
metadata: dict
```

### PerceptionResult

```python
observation_id: str | None
candidates: list[SceneCandidate]
source_candidate_id: str | None
target_candidate_id: str | None
arm_binding: dict[str, str]
uncertainty: dict
geometry_summary: dict
metadata: dict
```

`source_candidate_id` / `target_candidate_id` 是 top-level 字段。当前 binding skill 是 `vision.localize_task_objects`。

### SceneCandidate

```python
candidate_id: str
label: str | None
role_hypotheses: dict[str, float]
bbox_by_view: dict[str, list[float]]
mask_ref_by_view: dict[str, str]
evidence: dict
metric_geometry: MetricGeometry
support: dict
visibility: str
confidence: float
status: str
metadata: dict
```

当前 VLM prompt 允许 bbox 为空；`render_grounding_overlay` 只有在已有 bbox 时可用。

### WorldState

```python
world_state_id: str
task_instruction: str | None
scene_summary: str
geometry_summary: dict
candidates: list[SceneCandidate]
robot_arms: dict[str, RobotArmState]
relations: list[WorldRelation]
source_candidate_id: str | None
target_candidate_id: str | None
stage: str
needs_reobserve: bool
uncertainty_reasons: list[str]
metadata: dict
```

`state.update_world_state` 不创建 placeholder perception，也不从 label 猜 source/target。它只把已有 `PerceptionResult` 写成 world state。

### TaskPlan / Subgoal

```python
TaskPlan:
  task: str | None
  subgoals: list[Subgoal]
  current_subgoal_id: str | None
  status: str
  metadata: dict

Subgoal:
  subgoal_id: str
  type: str
  instruction: str | None
  source_candidate_id: str | None
  target_candidate_id: str | None
  status: str
  completion_criteria: dict
  metadata: dict
```

`Subgoal.instruction` 是发送给 pi0.5/OpenPI 的短 horizon 自然语言命令。motion 层会直接消费这个字段。

### MotionGoal / ActionChunk / SafetyReport / VerificationReport

```python
MotionGoal:
  skill
  source_candidate_id
  target_candidate_id
  acting_arm
  motion_hint
  target_pose
  constraints
  metadata

ActionChunk:
  action_type: "qpos" | "ee" | "unavailable" | ...
  commands: list[list[float]]
  control_horizon: int | None
  metadata: dict

SafetyReport:
  allowed: bool
  status: str
  checks: dict
  clipped_motion_goal: MotionGoal | None
  errors: list[str]
  metadata: dict

VerificationReport:
  success: bool
  partial_progress: bool
  failure_type: str | None
  progress_score: float | None
  residuals: dict
  should_reobserve: bool
  notes: list[str]
  metadata: dict
```

## Vision 组件

注册函数：`register_vision_skills()`

### `capture_views`

作用：

- 从 `env_adapter.capture_views()` 获取 observation。
- 如果 payload 有 `setup=True`，RoboTwin adapter 会 setup task。
- 新 observation 会标记 grounding overlay 和 motion artifacts stale。
- 写 blackboard `observation`。

失败：

- env 报错会返回 `observation_unavailable`，并写 `last_observation_error`。

### `capture_verify_views`

作用：

- 只在 `verify` 阶段用于执行后验证。
- 要求 `execution_report.status == "action_executed"`。
- 调 env.capture_views 采集 fresh verify observation。
- 写 `verify_observation` 和 `last_verify_capture`。

它不做 perceive/localize/plan，也不改 current subgoal。

### `perceive_scene`

作用：

- 用 VLM 检测 task-relevant visual candidates。
- 输出 top-level `candidates` 和 `uncertainty`。
- 不绑定 source/target。

当前 required schema：

```json
{
  "candidates": [
    {
      "candidate_id": "C1",
      "label": "short visual label or null",
      "visibility": "yes|partial|uncertain",
      "confidence": 0.0,
      "status": "short status"
    }
  ],
  "uncertainty": {"needs_reobserve": false, "reasons": []}
}
```

如果模型输出没有 `candidates`：

- 有 existing perception 时，保留 existing perception 并返回成功状态 `scene_perception_preserved_existing_after_invalid_model_output`。
- 没有 existing perception 时，返回 `scene_perception_invalid_model_output`。

这条保留逻辑用于保护已有可用视觉状态；新一轮无效 VLM 输出会写 `last_perception_error`。

### `localize_task_objects`

作用：

- 绑定 source/target 到 candidates。
- 输出 top-level `candidates`、`source_candidate_id`、`target_candidate_id`、`uncertainty`。
- bbox 可为空；该 skill 只做 candidate 绑定。

当前 required schema：

```json
{
  "candidates": [
    {
      "candidate_id": "C1",
      "label": "object label",
      "visibility": "yes|partial|uncertain",
      "confidence": 0.0,
      "status": "semantic object status"
    }
  ],
  "source_candidate_id": "candidate id or null",
  "target_candidate_id": "candidate id or null",
  "uncertainty": {"needs_reobserve": false, "reasons": []}
}
```

contract checks：

- source id 必须存在。
- target id 必须存在。
- source/target 必须在 candidates 中。
- source/target 需要绑定到不同 candidate。
- source/target candidate 的 `visibility` 需要是可见状态。

失败会写 `last_localization_error` 并返回 `localization_invalid_model_output`。

### `refresh_preflight_observation`

作用：

- 只处理 preflight 的视觉/相机类失败。
- 内部顺序执行：
  1. `capture_views`
  2. `perceive_scene`
  3. `localize_task_objects`
  4. `state.update_world_state`
- 成功后标记 motion artifacts stale，stage 仍为 `preflight`。

允许的 preflight errors：

```text
stale_perception
stale_world_state
world_state_requires_reobserve
missing_observation
missing_observation_id
camera_*
```

这是 preflight 内部刷新当前 visual state 的技能。

### `render_grounding_overlay`

只有在已有 bbox 时渲染 overlay。没有 bbox 会返回 `grounding_overlay_unavailable`。

### `lift_depth_cluster` / `lift_geometry`

根据 bbox + depth 或 pointcloud_ref 计算 metric geometry。它是可选证据；没有 metric geometry 时可以走 image-grounded VLA 执行。

`lift_geometry` 是 `lift_depth_cluster` 的 alias。

### `bind_arm`

把 payload 中的 arm binding 写入 `perception.arm_binding`。

### `estimate_uncertainty`

有模型时调用 VLM 判断 visual state 是否可靠；没有模型时只基于候选是否为空做简单 uncertainty。

## State 组件

### `update_world_state`

要求：

- `blackboard.perception` 必须是 `PerceptionResult`。
- candidates 非空。
- perception metadata 需要排除 placeholder。

写入：

```text
world_state
stage
```

`source_candidate_id` 和 `target_candidate_id` 直接来自 perception top-level 字段，不做 label fallback。

### `summarize_state`

把 `blackboard.compact_context()` 写入 `state_summary`。

## Scheduler 组件

### `choose_next_skill`

这是 agent loop 内部调用的 scheduler model skill。它读：

```text
blackboard.compact_context()
current_observation_images
loop_mode
current_stage
stage_order
allowed_skills
runtime_state
required_schema
```

输出 `LoopDecision`。如果 `control=run_skill` 缺 `next_component` 或 `next_skill`，返回 `scheduler_invalid_model_output`。

如果 scheduler model 不可用，返回 `scheduler_model_unavailable`，不选择默认 skill。

### `build_task_plan`

要求 world_state 有 source/target。调用 scheduler model 生成完整 ordered subgoal plan。

模型输出必须满足：

- subgoals 非空。
- subgoal_id 唯一。
- current_subgoal_id 在 subgoals 中。
- 每个 subgoal 有非空 `instruction`。
- 每个 subgoal 有 `completion_criteria.natural_language`。
- completion criteria 需要是具体条件文本。
- 至少有 subgoal 引用 source id。
- target id 存在且不同于 source 时，至少有 subgoal 引用 target id。

如果 scheduler model 不可用，返回 `task_plan_unavailable`，不生成模板 plan。

### `select_current_subgoal`

从 task_plan 选择 current subgoal，优先 current id，否则第一个 pending/running。没有可选 subgoal 时，task_plan 标记 succeeded。

### `advance_subgoal`

只在当前 subgoal 被成功 verify 后推进。

要求：

- task_plan 存在。
- current_subgoal 存在。
- last_verification_report 存在且 `success=True`。
- verification metadata 中的 `current_subgoal_id` 等于当前 subgoal id。

如果还有 pending subgoal：

- 当前 subgoal 标记 succeeded。
- 下一个 subgoal 标记 running。
- 清 preflight/safety report。
- stage 写为 `preflight`。
- motion artifacts stale。

如果没有下一个 subgoal：

- task_plan.status = succeeded。
- current_subgoal = None。
- archive 并清 active verification。
- 返回 `task_plan_complete`。

### `allocate_budget`

把 budget clamp 到 1 到 20，写 `last_budget_steps`。当前它只是记录预算，不直接控制 OpenPI horizon。OpenPI horizon 由 `motion.emit_action_chunk` payload 的 `horizon` 控制。

### `repair_stage_transition`

显式修复阶段跳转。payload 必须有：

```json
{
  "target_stage": "observe|plan|preflight|recover",
  "reason": "..."
}
```

行为：

- target `plan`：清 task_plan/current_subgoal，motion artifacts stale。
- target `observe`：清 observation/perception/world_state/grounding_overlay，并归档 reobserve request。
- target `preflight`：motion artifacts stale。
- target `recover`：进入 recover。
- 离开 verify 到 observe/plan/preflight 时，归档并清 active verification。

## Safety 组件

### `preflight_action`

每次运动前必须跑。它会先 mark motion artifacts stale，再构造 SafetyReport。

检查项：

```text
task_state:
  observation/perception/world_state/task_plan/current_subgoal 是否存在
  current_subgoal 是否匹配 task_plan.current_subgoal_id
  world_state.needs_reobserve

observation_freshness:
  perception.observation_id 是否等于 current observation
  world_state.metadata.observation_id 是否等于 current observation

object_binding:
  source/target candidate 是否存在
  source/target 是否相同
  target_required 对 transport/place/release 为 true
  source/target label 和 visibility

camera_inputs:
  head_camera/front_camera/left_camera/right_camera 是否有可读 RGB 文件
  当前期望分辨率 960x540
  OpenPI 需要 head/left/right

robot_state:
  14D joint action vector 是否存在且 finite

robotwin_env:
  env_adapter/task_env/last_observation 是否存在

action_backend:
  backend 是否存在、enabled、pretrained_path 是否存在、OpenPI worker health 是否 ok
```

成功：

```text
allowed=True
status=preflight_passed
```

失败：

```text
allowed=False
status=preflight_failed
errors=[...]
```

report 写入 `last_safety_report` 和 `preflight_report`。

## Motion 组件

### `build_motion_goal`

从 `current_subgoal` 和 `world_state` 建 MotionGoal。优先使用 payload 中的 source/target，其次 current_subgoal，再其次 world_state。

target handle：

- 有 metric position：`target_type=metric_pose`
- 无 metric 但有 source/target candidate：`target_type=image_grounded`
- 都没有：`missing_visual_target`

当前主路径是 image-grounded，通过 OpenPI/pi0.5 执行。

### `plan_motion`

如果 target handle 是 image_grounded：

- 从 current_subgoal.instruction 得到 `vla_prompt`。
- 写 `motion_plan`，status `image_grounded_motion_plan_built`。
- image paths 来自当前 observation。

如果没有可执行 handle，写 `motion_plan_unavailable`。

### `emit_action_chunk`

要求 motion_plan fresh。payload 必须有 `horizon`，AgentLoop 限制 10 到 32；当前正式 pi0.5 checkpoint 的
action horizon 也是 32。

行为：

- 调 `action_backend.build_action_chunk(...)`。
- 给 action chunk metadata stamp：
  - subgoal_id
  - observation_id
  - consumed=False
  - stale=False
- 写 `action_backend_result` 和 `action_chunk`。

失败返回具体 backend status，例如 `pi05_unavailable`、`action_chunk_unavailable`。

### `validate_action_chunk`

检查：

- action_chunk 存在。
- action_type 有效，取值排除 None/unavailable/noop。
- chunk 有效，状态排除 stale/consumed。
- chunk subgoal_id 等于 current subgoal。
- chunk observation_id 等于 current observation。
- commands 非空。
- qpos 维度 14，ee 维度 16。
- 每个 command 是 list、维度正确、全 finite。

写 `last_action_validation_report`。

### `execute_action`

执行前再次跑同一套 action chunk validation。成功后：

- 调 env.execute_action。
- 写 `execution_report`。
- 如果 env 有 last_observation，写回 `observation`。
- mark grounding overlay stale。
- mark action chunk consumed。
- mark motion artifacts stale。
- 返回 `action_executed`。

如果 env 或 action chunk 不可用，返回显式 failure。

## Verifier 组件

### `verify_progress`

要求：

- verifier model 可用。
- payload image_paths 非空。
- 当前 subgoal 存在。
- execution_report 存在。

它只验证当前 subgoal，不把全任务 success 当 subgoal success。

模型 required schema：

```json
{
  "subgoal_success": false,
  "task_success": false,
  "partial_progress": false,
  "failure_type": "none|not_done|observation_stale|execution_failed|ambiguous|other",
  "progress_score": 0.0,
  "should_reobserve": false,
  "next_action": "advance_subgoal|continue_execute|reobserve|recover|finish",
  "notes": ["short evidence note"]
}
```

`_canonical_next_action()` 会规范化：

```text
subgoal_success=True           -> advance_subgoal
should_reobserve=True          -> reobserve
failure_type observation_stale -> reobserve
failure_type ambiguous         -> reobserve
failure_type not_done/none     -> continue_execute
其他 failure_type              -> recover
```

report 写入 `last_verification_report`。

## Recovery 组件

### `decide_recovery`

只处理 verifier next_action=recover 的 true failure。

要求：

- verification/execution failure report 存在。
- 如果 verification 存在，必须 `success=False` 且 metadata.next_action 是 `recover`。
- recovery model 可用。

模型 required schema：

```json
{
  "recoverable": true,
  "failure_diagnosis": "short concrete diagnosis",
  "patch_type": "retry_current_subgoal|replace_current_subgoal|insert_recovery_subgoal|replan|reobserve|abort",
  "next_stage": "preflight|plan|observe|finish",
  "repaired_subgoal": {
    "subgoal_id": "...",
    "type": "...",
    "instruction": "...",
    "source_candidate_id": "...",
    "target_candidate_id": "...",
    "status": "pending",
    "completion_criteria": {"natural_language": "..."}
  },
  "notes": ["..."]
}
```

patch_type 和 next_stage 必须匹配：

```text
retry_current_subgoal    -> preflight
replace_current_subgoal  -> preflight
insert_recovery_subgoal  -> preflight
replan                   -> plan
reobserve                -> observe
abort                    -> finish
```

### `build_retry_request`

应用 recovery directive：

- retry/replace：patch 当前 subgoal。
- insert：在当前 subgoal 前插入 recovery subgoal。
- preflight retry：清 preflight/safety report，motion artifacts stale。
- 写 `last_retry_request`。

后续 scheduler 再通过 `repair_stage_transition` 进入对应 stage。
