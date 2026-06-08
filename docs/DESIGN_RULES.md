# ClawVLA Design Rules

这些规则优先级高于旧 `agentvla` 里的实现习惯。

## 1. 旧代码只参考，不搬运

旧项目长期迭代后有大量实验分支、调试逻辑、重复判断和临时策略。ClawVLA 可以参考：

- RoboTwin 环境约定；
- 已经踩过的视觉/左右臂/pointcloud 问题；
- 重要的输入输出字段；
- 可验证的失败案例。

ClawVLA 不直接搬运：

- 大型 rollout 脚本；
- task-specific 状态机；
- prompt 里堆叠的临时规则；
- privilege/debug 逻辑；
- 与当前架构无关的历史兼容分支。

## 2. 小文件和清晰职责

原则：

- 文件长度允许自然波动，不做机械行数限制。
- 如果一个文件开始同时承担多种职责，应优先按组件、后端、schema 或工具函数拆分。
- 禁止再次出现几千行的 `robotwin_rollout_safe.py` 式入口文件。
- CLI 只做参数解析和调用 runtime，不承载核心业务逻辑。
- 每个 skill 的实现放在所属组件模块里，不放到一个全局大文件。

推荐目录形态：

```text
components/
  vision.py
  state.py
  scheduler.py
  safety.py
  motion.py
  verifier.py
  recovery.py
skills/
  base.py
  builtin.py
```

其中 `skills/builtin.py` 只负责注册，不写大段实现。

## 3. World State 重新设计

旧 world state 只作为经验参考，ClawVLA 重新设计状态表达。

默认主线使用 JSON-like typed dataclass，因为：

- 容易被 Python 代码消费；
- 方便日志、回放、schema 校验；
- 方便传给 OpenAI-compatible 接口；
- 方便后续落盘成 JSONL。

XML 可以作为 prompt rendering 的一种视图，但不作为内部主状态格式。组件配置里的 `prompt_format` 可以选择 `json` 或 `xml`。

内部状态建议分层：

- `ObservationBundle`：当前公开观测。
- `PerceptionResult`：视觉候选和 grounding。
- `WorldState`：跨步骤稳定状态。
- `SchedulerDecision`：下一步调度决策。
- `MotionGoal`：运控目标。
- `SafetyReport`：执行前检查结果。
- `VerificationReport`：执行后 residual 和成败判断。

## 4. 状态词条原则

World state 里只保留下游要用的词条，不塞全量调试信息。

建议词条：

- `task_instruction`
- `scene_summary`
- `stage`
- `candidates`
- `source_candidate_id`
- `target_candidate_id`
- `robot_arms`
- `relations`
- `needs_reobserve`
- `uncertainty_reasons`
- `last_motion_goal`
- `last_safety_report`
- `last_verification_report`

候选物体建议词条：

- `candidate_id`
- `label`
- `role_hypotheses`
- `bbox_by_view`
- `mask_ref_by_view`
- `evidence`
- `metric_geometry`
- `support`
- `visibility`
- `confidence`
- `status`

`metric_geometry` 是可选块。只有存在 depth / 标定 / 点云等可用 metric evidence 时才填充：

- `available`
- `source`
- `position_3d`
- `extent_3d`
- `pointcloud_ref`
- `pointcloud_local`
- `geometry_views`
- `support_gap`
- `quality`

下游组件必须先检查 `metric_geometry.available` 或具体字段是否存在，不能把 3D 当作基础输入。

调试大对象放 artifact 文件或 `metadata` 引用，不直接塞进核心状态。

当前 ClawVLA 的最小状态链路是：

```text
ObservationBundle -> PerceptionResult -> WorldState -> SchedulerDecision
```

这条链路必须保持轻量，图片、点云、mask 等大对象应通过路径或 artifact id 引用。

## 8. Artifact Policy

RGB、depth、mask、pointcloud、完整 raw observation 都不直接进入 `WorldState`。

推荐做法：

- RGB 保存为 PNG，`CameraView.rgb_path` 引用。
- depth 保存为 NPY，`CameraView.depth_path` 引用。
- raw observation 只保存 summary 到 JSON。
- pointcloud / mask 后续单独加 artifact writer，不塞进 `metadata`。
- 每一步的 artifact prefix 由 runtime 或 env adapter 提供，例如 `episode_000/step_003/start`。

## 9. RoboTwin Session Boundary

`RoboTwinSession` 只负责：

- import task class；
- 准备 `setup_demo()` 参数；
- 调用 `setup_demo()`；
- 调用 `get_obs()`。

它不负责：

- rollout loop；
- scheduler；
- motion/action 构造；
- verifier；
- recovery；
- benchmark success 判断。

这些逻辑必须留在对应 component 或 runtime 中。

## 5. 组件边界

- `vision` 只回答看到了什么、任务指哪个、几何在哪里、是否不确定。
- `state` 只维护事实状态和短历史，不决定执行动作。
- `scheduler` 只决定下一步调哪个组件/skill，不直接生成低层动作。
- `safety` 必须能硬拒绝或裁剪 motion，不依赖模型自觉。
- `motion` 只把 motion goal 变成 plan/action，不做任务级策略。
- `verifier` 只判断发生了什么，不直接接管恢复流程。
- `recovery` 如果启用，只输出恢复 directive，不偷偷执行动作。

## 6. 模型使用原则

每个组件可以绑定不同模型，但模型只做它擅长的事情：

- 视觉/grounding 可以用 VLM。
- 调度可以用 LLM，但必须输出结构化 JSON。
- 安全和几何优先用工具/规则。
- 运控优先用控制器/策略模型。
- verifier 可以用 VLM + residual 工具混合。

所有模型输出必须经过 parser 和 validator，再进入 blackboard。

## 7. Prompt Rendering

内部状态和日志保留结构化 Python/dataclass/JSON-like 对象。

给模型看的上下文由 rendering 层生成：

- `json`：默认格式，适合 schema、日志和回放。
- `xml`：可用于长上下文分区、减少字段边界歧义。

组件不能手写一大段混杂 prompt 和状态拼接逻辑；应通过统一 rendering/model call helper 生成模型输入。
