# 铭健阶段 4 CALVIN 工程报告：给五步长跑铺好同一条跑道

首次成稿：2026-07-17；矩阵与门禁更新：2026-07-20

工作分支：`mingjian/calvin-stage4-eval`

## 结论

阶段 4 的**评测基础设施、首条真实五步试跑和开发期 10 条固定序列矩阵已经完成**。正式 ABC→D 评测尚未开始，因此当前状态是：

- 评测器、清单、断点续跑、JSON/CSV 和指标：**PASS**；
- 官方 1000 序列池协议：**PASS**；
- X-VLA baseline 首条真实五步序列：**5/5 PASS**；
- ClawVLA Agent 首条五步序列：**5/5 PASS**，oracle 拒绝误判后仍能继续执行并干净结束；
- 10 条固定序列 X-VLA baseline：**7/10 完整成功，平均完成 3.9/5**；
- 10 条固定序列 ClawVLA Agent：**7/10 完整成功，平均完成 3.9/5**；
- 两个 runner 的环境/HTTP 异常均为 **0**，失败集中在同三条序列和同类低层任务；
- 完整 `task_ABC_D` 正式评测：归档文件已出现于本机，但仍 **WAITING FOR CHECKSUM / UNZIP / PROTOCOL FREEZE**。

形象地说，阶段 3 证明机器人能跑完一圈短跑；阶段 4 先修了一条带五个接力区的跑道。每个接力区都不重置场景，上一棒推开的抽屉、拿起的方块和点亮的灯会原样留给下一棒。计时器不仅记“赢没赢”，还会记跑了多少环境步、发了多少动作块、在哪里卡住、有没有提前冲线。

## 这次做了什么

### 1. 新增统一序列评测入口

新增 `src/clawvla/scripts/calvin_sequence_eval.py`，支持：

- 从 JSON manifest 读取固定序列与 seed；
- 用同一任务、同一 seed 运行原生 X-VLA baseline 或完整 Agent；
- `--runner baseline`、`--runner agent`、序列筛选、数量限制和 dry-run；
- JSONL 逐条落盘、断点续跑、汇总 JSON 和 CSV；
- 完成 1/2/3/4/5 步的比例、平均完成长度和整序列成功率；
- 平均环境步、平均动作块、stall、过早 finish、失败类型和环境/HTTP 异常数。

结果逐条写入后再更新汇总，像列车每到一站就盖一个章；即使中途机器重启，`--resume` 也能从未完成的站继续，不用推倒重跑。

### 2. 同一 CALVIN 场景跨子任务持续推进

`CalvinAdapter.advance_sequence_subtask()` 会在 oracle 确认当前子任务成功后：

- 保留同一个物理环境，不调用 reset；
- 把当前环境信息设为下一子任务的 oracle 起点；
- 切换官方 subtask 和 canonical language；
- 清理上一子任务的 done/reward 状态。

这保证五步序列不是五张互不相干的照片，而是一段连续录像。

### 3. 固定官方 1000 序列池

真实 dry-run 暴露了一个协议陷阱：`get_sequences(N)` 的结果不是前缀稳定的。用 `N=3` 取“第 2 条”和用 `N=10` 取“第 2 条”，可能得到完全不同的任务。它像同一个座位号随着车厢长度变化，坐到的乘客也变了。

官方 `evaluate_policy.py` 明确使用：

```text
NUM_SEQUENCES = 1000
```

因此现在：

- manifest 明确写入 `official_sequence_pool_size = 1000`；
- adapter 按固定 1000 池生成并按 index 取序列；
- 配置、运行计划、结果和 observation metadata 都记录 pool size；
- manifest 冻结了该池前 10 条序列的 initial state 与五个 subtask；
- 若运行时解析结果与清单漂移，动作发出前立即报 `calvin_official_sequence_drift`。

### 4. Planner 原子契约增加安全网

第一条 Agent 真实试跑中，第 1 步 `open_drawer` 成功；第 2 步 `lift_pink_block_table` 的语言是 `grasp and lift the pink block`。Qwen 尝试拆成 grasp、lift 两段，输出又因长度截断，连续五次解析失败后触发 stall。

原先原子契约在“模型 JSON 成功解析之后”才生效，像安全网装在坑底。现在仍先保留 Planner 的建议机会；若模型 JSON 截断或解析失败，backend contract 会立刻生成唯一子目标，并把官方任务语言逐字交给 X-VLA，同时记录 `model_proposal_error`，不再重复撞同一堵墙。

### 5. CALVIN oracle 真正掌握终局裁决权

第二条 Agent 试跑越过了 Planner 截断，但视觉 Verifier 在 oracle 仍为 false 时误判成功，Agent 随后 `task_plan_complete`，形成可复现的 `premature_finish`。

现在对声明 `completion_authority = environment_oracle` 的 backend：

- Verifier 的 `advance_subgoal` 或 `finish` 只是一条建议；
- oracle 明确为 false 时，Agent 拒绝结束，恢复当前原子子目标；
- 流程返回 preflight，继续生成动作；
- 只有 `env.task_status().success == true` 才能给当前 CALVIN 子任务盖章。

## 首条真实五步 baseline

固定条件：

```text
sequence = official_004
pool size = 1000
seed = 0
runner = baseline
X-VLA checkpoint = 2toINF/X-VLA-Calvin-ABC_D
horizon = 20
inference steps = 10
```

序列：

```text
open_drawer
-> lift_pink_block_table
-> place_in_slider
-> turn_on_lightbulb
-> rotate_blue_block_left
```

结果：

|子任务|成功|环境步|动作块|
|---|---:|---:|---:|
|open_drawer|是|62|4|
|lift_pink_block_table|是|99|5|
|place_in_slider|是|30|2|
|turn_on_lightbulb|是|79|4|
|rotate_blue_block_left|是|86|5|
|合计|5/5|356|20|

整序列成功率为 100%，长度 1–5 完成率均为 100%，环境/HTTP 异常为 0。这里只是一条开发样例，不能外推为正式 benchmark 成绩。

机器可读证据：

```text
tmp_runs/stage4_sequence_eval/baseline_official_004_pool1000_debug/
```

该目录还保存了 125 个图像、深度和 observation 摘要文件，可继续制作关键帧或视频。

## Agent 失败案例与修复链

第一次 Agent 对照：

```text
完成长度 = 1/5
失败类型 = stalled_loop
原因 = repeated_failed_skill:scheduler.build_task_plan:count=5
环境/HTTP 异常 = 0
```

证据：

```text
tmp_runs/stage4_sequence_eval/agent_official_004_pool1000_debug/
```

增加 Planner 原子兜底后的第二次对照：

```text
完成长度 = 1/5
失败类型 = premature_finish
原因 = task_plan_complete（但 CALVIN oracle 仍为 false）
环境/HTTP 异常 = 0
```

证据：

```text
tmp_runs/stage4_sequence_eval/agent_official_004_atomic_fallback_debug/
```

这两条失败不是废片：第一条证明卡死保护器能收住无限重试，第二条准确抓到 Verifier 越权。两者都被评测器分进独立失败桶，正好验证阶段 4 的诊断指标。

完成两层修复后的第三次真实对照：

|子任务|成功|环境步|动作块|Agent 循环步|
|---|---:|---:|---:|---:|
|open_drawer|是|62|4|37|
|lift_pink_block_table|是|119|6|56|
|place_in_slider|是|29|2|19|
|turn_on_lightbulb|是|78|4|37|
|rotate_blue_block_left|是|88|5|46|
|合计|5/5|376|21|195|

最终完成长度为 5，长度 1–5 完成率均为 100%；stall、premature finish、环境/HTTP 异常均为 0。第 2 个子任务中，视觉 Verifier 曾在 oracle 为 false 时建议结束，保护逻辑把当前原子子目标重新扶回跑道；再执行一个动作块后，环境 oracle 才确认成功。最后系统在第 5 个子任务的第 5 个动作块后由 oracle 干净结束，没有继续绕圈。

机器可读证据：

```text
tmp_runs/stage4_sequence_eval/agent_official_004_oracle_guard_debug/
```

Agent 比 baseline 多用 20 个环境步和 1 个动作块（376/21 对 356/20）。这只是同一 seed 的一条开发样例，用来证明评测链路、修复和终止协议可以工作，不能外推为总体性能结论。

## 10 条固定序列开发矩阵

2026-07-19 至 2026-07-20，使用固定官方 1000 序列池的 `official_000`–`official_009`、固定 seed 0 和同一 X-VLA checkpoint，分别跑完 baseline 与 Agent。该矩阵仍使用 debug 数据中的 scene-D 仿真配置，目的在于验证工程稳定性、长序列生命周期和失败归因，不是正式 ABC→D benchmark。

|指标|X-VLA baseline|ClawVLA Agent|
|---|---:|---:|
|序列数|10|10|
|完整成功|7/10|7/10|
|完整序列成功率|70%|70%|
|平均完成长度|3.9/5|3.9/5|
|长度 1/2/3/4/5 完成率|90% / 80% / 80% / 70% / 70%|90% / 80% / 80% / 70% / 70%|
|平均环境步|314.0|335.1|
|平均动作块|17.3|18.5|
|环境或 HTTP 异常|0|0|
|premature finish|0|0|
|失败序列|`001`、`008`、`009`|`001`、`008`、`009`|

三条失败的完成长度分别为 `0/5`、`3/5` 和 `1/5`。baseline 在单子任务 120 环境步预算耗尽后记为 `task_failure`；Agent 在 80 个循环步耗尽后记为 `stalled_loop`。具体失败点一致：

- `official_001`：第 1 步 `turn_off_led`；
- `official_008`：完成前三步后卡在 `lift_pink_block_slider`；
- `official_009`：完成 `push_red_block_left` 后卡在 `lift_blue_block_table`。

这组结果说明 Agent 的 Planner/oracle 保护没有降低完整序列成功数，也没有引入环境、HTTP 或过早结束故障；当前三条失败更接近冻结 X-VLA 在 LED 和 block lifting 上的低层能力/预算边界。不过两个 runner 使用的停止预算口径不同，因此这里不能把平均步数差异解释成严格的性能优劣，正式比较必须在 ABC→D 数据和统一 protocol 下重跑。

机器可读证据：

```text
tmp_runs/stage4_sequence_eval/stage4_debug_pilot10_baseline/
tmp_runs/stage4_sequence_eval/stage4_debug_pilot10_agent_oracle_consistent/
```

## 三条失败序列的小规模归因实验

2026-07-20 对 `official_001`、`official_008`、`official_009` 做了两组 planner-free X-VLA baseline 诊断。实验保持 checkpoint、seed、horizon、inference steps、manifest 和 debug scene-D 数据不变，只改变单子任务最大环境步数。这样可以先隔离低层策略和预算，不把 Qwen Planner 的生成差异混入归因。

第一组按主矩阵原协议复跑，每个子任务最多 120 环境步：

|序列|主矩阵完成长度|原协议复跑|复跑失败点|
|---|---:|---:|---|
|`official_001`|0/5|0/5|`turn_off_led`，120 步耗尽|
|`official_008`|3/5|3/5|`lift_pink_block_slider`，120 步耗尽|
|`official_009`|1/5|1/5|`lift_blue_block_table`，120 步耗尽|

三条失败位置和完成长度完全复现，环境/HTTP 异常为 0，说明主矩阵的三条失败不是偶发服务故障。

第二组只把单子任务最大环境步数从 120 增至 240，结果如下：

|序列|诊断结果|关键证据|归因|
|---|---:|---|---|
|`official_001`|5/5|`turn_off_led` 在第 182 步成功，后续四步全部完成|预算敏感；原 120 步不足|
|`official_008`|5/5|`lift_pink_block_slider` 在第 108 步成功，整序列完成|存在轨迹波动；并非稳定不可完成|
|`official_009`|1/5|`lift_blue_block_table` 执行 240 步仍失败|稳定低层能力/状态恢复缺口|

`official_008` 在原协议复跑中 120 步仍失败，而加预算诊断中第 108 步成功，表明 X-VLA 推理/仿真轨迹并非严格确定；不能把一次成功简单归结为“只差固定的若干步”。`official_009` 在 120 与 240 步末帧中都表现为机械臂停在蓝色方块附近但未形成 oracle 认可的抬起状态，继续重复动作没有带来任务进展，更符合低层抓取能力或失败后恢复能力不足，而不是 HTTP、oracle 或 Agent 控制流故障。

机器可读证据和末帧位于：

```text
tmp_runs/stage4_sequence_eval/stage4_attribution_baseline_repeat_budget120/
tmp_runs/stage4_sequence_eval/stage4_attribution_baseline_extended_budget240/
```

这项诊断不替换固定协议的 `7/10` 主结果，也不据此把开发成功率改写为 `9/10`。合理结论是：原三条失败中，两条对预算或轨迹波动敏感，一条是稳定的低层失败；Stage 4 工程链路无需为了追求 `10/10` 继续调参。正式成功率必须在 ABC→D 数据、统一预算和冻结 protocol 下重新评测。

## 使用命令

先检查 10 条序列、两个 runner 共 20 个 job：

```bash
PYTHONPATH=src python -m clawvla.scripts.calvin_sequence_eval \
  --config tmp_artifacts/tmp_runs/stage3_planner_episode_008_vllm_config.json \
  --manifest configs/calvin/stage4_official_pilot.json \
  --output-dir tmp_runs/stage4_sequence_eval \
  --run-id stage4_dry_run \
  --runner baseline --runner agent --dry-run
```

真实运行时清空代理，保留 localhost 直连，并使用 `--resume` 断点续跑：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  PYTHONPATH=/home2/gmj/Agent_skill/EmbodiedSkills/src:/home2/gmj/CALVIN/calvin_env:/home2/gmj/CALVIN/calvin_models \
  PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d \
  /home2/gmj/miniconda3/envs/calvin-py38/bin/python \
  -m clawvla.scripts.calvin_sequence_eval \
  --config tmp_artifacts/tmp_runs/stage3_planner_episode_008_vllm_config.json \
  --manifest configs/calvin/stage4_official_pilot.json \
  --output-dir tmp_runs/stage4_sequence_eval \
  --run-id stage4_debug_pilot10_baseline \
  --runner baseline --resume
```

Agent 矩阵使用同一 manifest 和配置，将 `--run-id` 改为 `stage4_debug_pilot10_agent_oracle_consistent`、`--runner` 改为 `agent`。分开落盘可避免恢复运行时两个 runner 的产物和失败归因混在同一目录。

## 测试门禁

本阶段新增和修改覆盖：

- manifest schema、固定池和 10 条五步序列；
- baseline/Agent 运行计划；
- 单任务覆盖项清除与 pool size 注入；
- 跨子任务保持物理场景与 oracle 重定位；
- 官方序列漂移拦截；
- 长度 1–5、环境步、动作块、stall、过早 finish 指标；
- JSONL 断点续跑与 CSV；
- Planner JSON 截断后的原子兜底；
- oracle false 时拒绝视觉完成。

阶段 4 首条五步试跑完成时的门禁为 `221 passed, 1 skipped`。2026-07-20 在当前工作树和轻量 Python 3.12 环境重新核验：

```text
全量轻量门禁（清空 localhost 代理）：210 passed, 13 skipped
CALVIN 专项：38 passed, 1 skipped
git diff --check：PASS
```

13 个 skip 来自轻量环境未安装的 Torch/OpenRLHF/外部实现或显式关闭的真实 CALVIN integration，并不等于真实 backend 已被当前轻量门禁覆盖。若不清空 `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY`，两个 localhost PolicyProxy 测试会返回 HTTP 502；这属于已知启动环境要求，所有正式命令必须保留 `NO_PROXY=127.0.0.1,localhost`。最终提交前还需要在适用的带 Torch/OpenRLHF 环境补跑专项门禁，并再次执行 `compileall`、敏感信息和生成物审计。

## 一周工作后的提交建议

以 2026-07-16 至 2026-07-20 这一周为范围，最合适的代码提交点是：**Stage 4 开发矩阵闭环后提交一个可审查的工程 checkpoint；不等待正式 ABC→D benchmark、Stage 5 Planner SFT 或 Stage 6 GRPO。**

理由如下：

- Stage 0–2 已给出环境、真实 `/act`、真实 `env.step()` 和 oracle 证据；
- Stage 3 已打通单任务 Agent，并有 10-seed 回归；
- Stage 4 已补齐序列评测器、官方 1000 池、跨子任务生命周期、断点续跑和 10 序列双 runner pilot；
- 当前剩余的 ABC→D 下载验收和正式评测主要是数据/实验边界，不应让已完成的工程 diff 长期悬空；
- Stage 5/6 会引入数据构建、训练配置和 optimizer 链路，风险与审查主题明显不同，适合另起提交。

这里的“适合提交”不等于“可以直接合并”。提交前必须满足：核心负责人签署阶段 3 接口契约；排除 `source/`、`tmp_runs/`、`tmp_artifacts/`、模型和数据；更新依赖状态与本报告；全量及专项测试达到约定门禁；完成敏感信息、大文件和本机路径审计。建议将这一周的工作按审查主题拆成 2–3 个连续 commit，而不是把所有修改压成一个大提交：

1. CALVIN adapter/backend、oracle 生命周期与 direct-VLA contract；
2. capture/action/baseline/sequence 评测入口、manifest 与回归测试；
3. 工作日志、环境说明和阶段报告。

完成上述门禁后，该分支可以作为 **Stage 4 engineering checkpoint / PASS WITH RISK** 提交评审；正式 ABC→D benchmark 和训练结果继续留在后续阶段，不写进本次提交结论。

## 还没有完成的部分

1. 核心负责人评审并签署阶段 3 Planner/Verifier/oracle 接口契约。
2. 将 `official_009 / lift_blue_block_table` 保留为稳定低层失败案例，并按任务族补齐其他代表成功/失败案例索引。
3. 对本机 `task_ABC_D.zip` 完成官方 SHA256 校验和解压；在此之前不能把“文件存在”写成“数据就绪”。
4. 团队冻结正式 checkpoint、sequence 列表、随机性和规模后，按官方 protocol 跑正式 ABC→D；当前 debug dataset pilot 不冒充正式成绩。
5. 完成本周工程 checkpoint 的分 commit 审计、测试和评审；不提交虚拟环境、运行产物、模型或数据。
6. 只有阶段 4 多任务 pilot 稳定复现且团队确认需要训练，才进入阶段 5 Planner SFT。
