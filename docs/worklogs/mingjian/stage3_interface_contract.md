# 阶段 3：Planner、Verifier 与 CALVIN oracle 接口契约（待核心负责人确认）

日期：2026-07-17
实现分支：`mingjian/calvin-stage3-integration`

## 1. 权威顺序

```text
CALVIN task oracle success
        ↓ 最高优先级，立即终止
Visual Verifier progress / failure
        ↓ oracle 未成功时，决定继续、重观察或恢复
Planner subgoal state
        ↓ 组织动作意图，不得覆盖 oracle 终局
```

## 2. Planner → X‑VLA

对 `calvin_http`：

|字段|契约|
|---|---|
|计划子目标数|恰好 1|
|执行文本|blackboard 中的 CALVIN 官方任务语言原样传递|
|候选绑定|可选，不得阻塞执行|
|模型原始计划|保留在 metadata，供训练与诊断|
|执行计划|由 backend contract 归一化，不因大小写或同义改写直接拒绝|

对其他 action backend：行为保持原样；未知 backend 继续采用保守的 candidate-binding 规则。

## 3. Verifier

Verifier 只在 `execution_report.success != true` 时工作：

- 使用最新 post-action 图像判断可见进度；
- 可以建议 `continue_execute`、`reobserve`、`replan` 或 `recover`；
- 不得把 oracle 已成功的全任务重新判回失败；
- CALVIN 原子任务的验证对象是完整官方任务，不是 Planner 曾提出但未执行的拆分动作。

## 4. Environment / oracle → AgentLoop

`execute_action` 的最小返回契约：

```json
{
  "status": "action_executed",
  "success": true,
  "done": true,
  "observation": {}
}
```

其中：

- `success=true` 表示 task oracle 已确认完整任务成功；
- `done=true` 可由环境终止、oracle 成功或步数上限产生；
- 只有 `success=true` 触发“成功收车”；单独的 `done=true` 不得冒充成功；
- 若 execution report 未携带成功值，AgentLoop 可回读 `env_adapter.task_status().success`；
- 成功终局必须写为 `finished / environment_oracle_success`，并同步 task plan/subgoal 状态。

## 5. 错误分类

|类别|示例|计入环境/HTTP 异常|
|---|---|---:|
|任务未完成|固定步预算耗尽、oracle 一直为 false|否|
|环境故障|reset/step/render 异常|是|
|HTTP 故障|超时、非 2xx、响应结构错误|是|
|模型动作无效|空动作、维度错误、NaN/Inf|是|
|受控 Agent 预算结束|`max_steps_reached`|否|

## 6. 回归验收

- 完整 Agent 至少一条 `finished / environment_oracle_success`；
- oracle 成功后不得再有 scheduler、verifier 或 localization 调用；
- 同一入口可生成成功轨迹与预算耗尽型失败轨迹；
- 固定 seed 0–9 的环境/X‑VLA/oracle 回归，环境/HTTP 异常率为 0；
- 所有单元与集成测试通过。

## 7. 签署

- [ ] Planner 核心负责人：姓名 / 日期 / 结论
- [ ] Verifier 核心负责人：姓名 / 日期 / 结论
- [ ] CALVIN oracle / Runtime 核心负责人：姓名 / 日期 / 结论

如负责人要求调整，请在本文件下追加评审意见，不要只在聊天消息中口头确认，以便接口变化可追溯。
