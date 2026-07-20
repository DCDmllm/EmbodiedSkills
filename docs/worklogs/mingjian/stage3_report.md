# 铭健阶段 3 CALVIN 工程报告：绿旗一举，Agent 立即收车

日期：2026-07-17

工作分支：`mingjian/calvin-stage3-integration`
基线分支：`mingjian/calvin-stage1`

任务流程图见：[阶段 3 CALVIN 任务流程可视化](stage3_task_flow_visualization.md)。

## 结论

阶段 3 的**工程硬门槛已经通过**：

- CALVIN oracle 一旦报告成功，Agent 在同一个 `execute_action` 步骤内结束，不再进入 verify 绕圈；
- `calvin_http` 被明确标成图像+语言 direct VLA，动作后不再强制重新做候选框定位；
- Planner 即使仍提出“抓取→移动”等拆分，backend contract 也会把执行计划收束为一个原子子目标，并把 CALVIN 官方语言原样送给 X‑VLA；
- 成功与受控失败都能用同一入口命令稳定生成机器可读轨迹；
- 固定 seed `0–9` 共 10 轮全部成功，环境/HTTP 异常为 0；
- 全量测试 `211 passed, 1 skipped`（最终数字以本分支最后一次门禁为准）。

形象地说，旧系统像三位裁判各吹各的哨：Planner 还在检查“手有没有抓稳”，定位器又说“我没看到候选框”，而 CALVIN 终点裁判早已举起绿旗。现在规则被排成一条清晰的优先级：X‑VLA 听官方整句指令，视觉 verifier 负责途中观察，CALVIN oracle 一旦判胜，主循环立刻刹车、记分、收车。

仍有一个**组织流程门槛**不能由本分支代签：核心负责人需确认 [阶段 3 接口契约](stage3_interface_contract.md)。因此本报告区分为：

- 工程实现与测试：**PASS**；
- 正式阶段关闭：**WAITING FOR OWNER SIGN-OFF**；
- 完整 ABC→D 官方 benchmark：仍等待 `task_ABC_D.zip` 下载、校验与解压，不能用 debug scene-D 结果冒充正式分数。

## 修改内容

### 1. Planner 原子任务直通，但不再用脆弱的字符串拒绝

`CalvinHttpActionBackend` 现在公开自己的 planning contract：

```text
mode                          = atomic_instruction_passthrough
max_subgoals                  = 1
candidate_bindings_required   = false
completion_authority          = environment_oracle
```

Qwen Planner 仍然参与规划，其原始建议会被保存在 `model_proposed_task_plan` 中，方便训练和诊断；真正交给 X‑VLA 的执行计划则被 backend contract 收束为：

```text
S1: push the sliding door to the left side
```

这不是“看见大小写不同就退单”的逐字校验，而是执行适配层主动使用 blackboard 中的官方原句。真实 run 001 中，Qwen 原始提出了 2 个子目标，最终执行计划仍稳定变成 1 个官方原子任务。

### 2. Direct VLA 不再被候选框卡住

`calvin_http.requires_candidate_bindings = False` 后：

- 初始 bootstrap 在拿到静态相机和夹爪相机图像后即可进入 plan；
- preflight 仍检查图像、机器人状态、环境和 action backend；
- perception、world state、bbox 和 candidate id 只作为可选诊断信息；
- post-action observation 更新后，下一轮 preflight 直接基于新图像检查，不再调用容易返回空列表的 localization。

真实成功轨迹中的 `localize_task_objects` 调用次数为 **0**。

### 3. Oracle 成功成为最高优先级终止信号

每次 `motion.execute_action` 成功返回后，AgentLoop 立即读取：

```text
execution_report.status == action_executed
execution_report.success == true
```

若报告缺失，还可读取 `env_adapter.task_status().success`。一旦 oracle 成功，循环立即：

- 返回 `loop.status = finished`；
- 写入 `reason = environment_oracle_success`；
- 将 task plan 与当前 subgoal 标为 `succeeded`；
- 不再发起下一次 scheduler、localization 或 verifier 调用。

同时修正了 `task_status` 的一致性：oracle 成功时 `done` 必须为 true，即使底层 Gym 的原始 done 仍为 false。

### 4. 动作 horizon 合约保持严格

CALVIN backend 继续执行此前完成的上限修复：

```text
execution_horizon = min(requested_horizon, configured_horizon)
```

当前固定口径为：调用方最多请求 32、CALVIN 配置 20、实际每块最多执行 20；X‑VLA 的 `steps=10` 仍表示 flow-matching 推理步数，不是动作行数。

### 5. 固定 seed 回归入口

`run_loop` 新增 `--seed`，CALVIN reset 会消费并记录该 seed。`calvin_xvla_baseline` 新增 `--seeds 0,1,...`，可用一个命令跑完固定矩阵，避免每轮手工改 JSON。

## 真实完整 Agent 成功证据

主要结果：

```text
tmp_artifacts/tmp_runs/stage3_contract_real_001_result.json
tmp_artifacts/tmp_runs/stage3_fixed_seed_0_result.json
```

run 001：

|指标|结果|
|---|---:|
|Loop 终局|`finished / execute`|
|终止原因|`environment_oracle_success`|
|CALVIN 环境步|61|
|动作块|20 + 20 + 20 + 1|
|Oracle success|true|
|Task plan 状态|succeeded|
|当前 subgoal 状态|succeeded|
|Localization 调用|0|

第四个动作块执行第 1 步时 oracle 判胜，Agent 同一步结束；这正是旧 run 008 缺少的“干净收车”。

## 成功与受控失败复现

成功轨迹使用统一入口：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  PYTHONPATH=/home2/gmj/Agent_skill/EmbodiedSkills/src:/home2/gmj/CALVIN/calvin_env:/home2/gmj/CALVIN/calvin_models \
  PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d \
  /home2/gmj/miniconda3/envs/calvin-py38/bin/python -m clawvla.scripts.run_loop \
  --config tmp_artifacts/tmp_runs/stage3_planner_episode_008_vllm_config.json \
  --instruction 'push the sliding door to the left side' \
  --artifact-prefix stage3_fixed_seed_0 \
  --initial-stage observe --max-steps 52 --seed 0 --run --initial-observe
```

受控失败只把同一入口的 `--max-steps` 改为 `1`，稳定得到：

```text
loop.status = max_steps_reached
loop.reason = max_steps=1
task_status.success = false
```

证据：`tmp_artifacts/tmp_runs/stage3_contract_failure_budget_001_result.json`。这是预算耗尽型任务失败，不是环境或 HTTP 故障。

## 10 个固定 seed 工程回归

命令：

```bash
/home2/gmj/miniconda3/envs/calvin-py38/bin/python \
  -m clawvla.scripts.calvin_xvla_baseline \
  --config tmp_artifacts/tmp_runs/stage3_planner_episode_008_vllm_config.json \
  --instruction 'push the sliding door to the left side' \
  --artifact-prefix stage3_fixed_seed_matrix_0_9 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --max-env-steps 100 --horizon 20 --inference-steps 10
```

结果：

|Seed|0|1|2|3|4|5|6|7|8|9|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|成功|✓|✓|✓|✓|✓|✓|✓|✓|✓|✓|
|环境步|62|62|62|61|62|62|61|62|61|61|

汇总：`10/10` 成功，成功率 `100%`，环境/HTTP 异常 `0`。机器可读证据：

```text
tmp_artifacts/calvin/stage3_fixed_seed_matrix_0_9/baseline_report.json
```

## 测试门禁

固定清空 localhost 代理后运行：

```text
208 passed, 1 skipped   # 核心修改后的首轮全量门禁
```

新增 seed CLI、baseline seed 矩阵和 task_status 一致性测试后的最终全量门禁为：

```text
211 passed, 1 skipped
```

`compileall` 与 `git diff --check` 均通过。

## 尚需负责人完成的一步

核心负责人只需评审并签署 [阶段 3 接口契约](stage3_interface_contract.md)，重点确认三句话：

1. CALVIN 官方任务是 backend-owned 的单原子执行指令；
2. visual Verifier 只在 oracle 尚未成功时提供进度建议；
3. CALVIN oracle success 是全任务最高优先级终止信号。

签署后，本阶段即可从“工程 PASS、等待确认”正式关闭为 PASS。
