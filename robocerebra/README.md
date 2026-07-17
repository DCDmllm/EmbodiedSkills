# RoboCerebra

这里放 EmbodiedSkills 接入 RoboCerebra 的轻量代码、数据说明和阶段交接信息。

当前主线使用 LeRobot 版数据，而不是 raw RoboCerebra HDF5 全量包：

```text
lerobot/robocerebra_unified
```

我们的解释方式是：LeRobot 里的每个 `episode_index` 已经是一个 short-horizon subgoal episode，可以直接作为 VLA 训练样本：

- prompt：LeRobot `task_text`
- state：RoboCerebra raw 8D state
- action：RoboCerebra raw 7D action
- image：front video + wrist video
- boundary：LeRobot metadata 里的 episode frame range

第一阶段状态：

- 代码和文档已整理进 PR。
- LeRobot 数据下载、索引、norm stats、pi0.5/OpenPI batch、LoRA trainer、inference server 和 baseline runner 都已经有初版。
- 官方 `Ideal/case1` 原生基线已经能跑通并输出日志/视频/summary。
- 现有模型还不能稳定完成任务；当前结果应理解为“评测链路跑通，模型能力不足或 schema/训练还需继续诊断”。

主要文档：

- [handoff.md](handoff.md)：数据、训练、rollout 和下一步交接。
- [baseline.md](baseline.md)：原生基线评测方法和当前结果摘要。
- [samples/robocerebra_subgoals_sample.jsonl](samples/robocerebra_subgoals_sample.jsonl)：小样例 subgoal records。

大数据、decoded PNG、rollout 视频、checkpoint 和本地 `outputs/` artifact 不进仓库。
