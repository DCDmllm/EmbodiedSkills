# ClawVLA 当前计划与进度

本文档记录当前主线状态。历史调试日志和旧阻塞点不再放在这里，避免和实际代码状态冲突。

## 目标

ClawVLA 的目标是把 RoboTwin 操作流程整理成可调度的多 skill agent：

```text
observe -> plan -> preflight -> execute -> verify -> recover
```

当前策略不是固定脚本顺序，而是：

- scheduler 自由选择当前阶段允许的 skill。
- runtime 做 skill 前置条件和 stage exit 检查。
- 所有 placeholder/unavailable/exception 必须显式进入 terminal、log、result。
- 不做静默 label fallback，不替模型偷偷补 source/target。
- 现阶段不加复杂限幅、安全拒绝或恢复策略，优先把完整闭环跑通。

## 已完成

- 项目骨架、组件注册、skill registry、blackboard、配置加载。
- RoboTwin observation adapter 和 artifact writer。
- `vision.capture_views/perceive_scene/localize_task_objects/ground_task_objects/render_grounding_overlay`。
- `TaskPlan/Subgoal/GroundingOverlay` 核心状态对象。
- scheduler narration/state_summary/expected_result trace 输出。
- prerequisite-driven gating：缺 artifact 时 block，而不是静默跳过。
- stale 规则：新 observation 失效 overlay/motion artifacts；执行后 action chunk consumed。
- Rich terminal trace；完整 JSON 只写 `tmp_runs/<prefix>_result.json`。
- 本地 vLLM wrapper：启动、ready 检查、tee agent log、结束清理。
- pi0.5/OpenPI worker 常驻生命周期：agent 启动时加载，进程结束时清理。
- OpenPI/pi0.5 action chunk 接入到 `motion.emit_action_chunk`。
- RoboTwin `ActionChunk -> take_action` 执行入口。

## 当前主线

正式入口：

```bash
cd /mnt/wangwai/vla/clawvla
./scripts/run_qwen3vl_pi05_agent.sh \
  --instruction "place the container on the plate" \
  --artifact-prefix agent_subgoal_loop_25 \
  --max-steps 25 \
  --run
```

运行时会生成：

```text
tmp_runs/<prefix>_agent.log
tmp_runs/<prefix>_result.json
tmp_runs/<prefix>_vllm.log
tmp_runs/<prefix>_pi05_worker.log
tmp_artifacts/<prefix>/
```

## 关键约束

- `ground_task_objects` 必须输出有效 `source_candidate_id` 和 `target_candidate_id`。
- `build_task_plan` 必须输出非空 `subgoals`；空输出会显式失败。
- `emit_action_chunk` 需要 fresh `motion_plan`。
- `execute_action` 需要 fresh non-empty `action_chunk`。
- `verify_progress` 需要 `current_subgoal` 和 `execution_report`。
- bbox overlay 只给 VLM 侧使用；OpenPI/VLA 继续吃 raw image。

## 待做

- 跑一轮更长的 real smoke，确认 verifier 在 16k context 下不再被 vLLM 拒绝。
- 压缩 verifier 输入，避免长期依赖更长 context。
- 补更系统的 mock loop tests：正常完成、verify fail、blocked decision。
- 清理 pi0.5/OpenPI worker 的诊断输出，把核心 action transform 对齐结论写成固定检查项。
- 等主流程稳定后，再补真正的 safety/recovery 策略。
