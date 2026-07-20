# 铭健阶段 2 CALVIN / X-VLA 真实链路验收报告：从“插头能对上”到“机械臂真的动起来”

日期：2026-07-16。

工作分支：`mingjian/calvin-stage1`。基线为 `ee48a785b329b5ceb99d24801eaf0a2f5914c94f`，叠加当前可审查的 M0–M2 worktree diff。

## 阶段结论

阶段 2 工程验收完成：**PASS**。

如果把前两个阶段比作铺设轨道和安装信号灯，阶段 2 就是第一次让真车上轨：CALVIN 场景真实 reset，两台相机真实拍图，X-VLA 真模型接收请求并给出动作，动作真正在仿真器里推动机械臂，最终由 task oracle 而不是模型自己宣布成败。

完整证据如下：

- CALVIN 连续 3 次 reset/capture 全部通过；每次都有相同的 20D proprio、两路 RGB/depth 图像和真实 task oracle 初始状态。
- 发布版 X-VLA checkpoint 在 1 张 RTX A6000 上加载成功，10/10 次 `/act` 请求全部满足协议。
- 一条真实 action chunk 已进入 `CALVIN env.step()`，保存了动作前后状态；第一条短 chunk 后 oracle 仍为 false，说明系统没有把“动了一下”误报成“任务完成”。
- 不接 Planner、只用 X-VLA 的 `move_slider_left` baseline 在官方客户端契约下第 62 步成功。
- 使用同一固定初始状态再跑 3 次，3/3 成功，分别用 61、61、62 个环境步。
- 环境和 HTTP 基础设施错误数为 0。

边界同样明确：这是使用 debug 数据中 scene-D 仿真配置得到的工程验收，不是官方 1000-sequence ABC→D benchmark。正式 benchmark 在数据和 protocol 冻结前仍为 **BLOCK**。试车成功证明方向盘、发动机和刹车已经协同工作，但不等于正式拉力赛成绩已经产生。

## 可复现身份：给每个关键部件刻上钢印

|项目|实际值|
|---|---|
|CALVIN 源码|`/home2/gmj/CALVIN`，`fa03f01f19c65920e18cf37398a9ce859274af76`|
|CALVIN env 子模块|`1431a46bd36bde5903fb6345e68b5ccc30def666`|
|X-VLA 源码|`/home2/gmj/X-VLA`，`6bc2513f5f1cbec715cc668b414392a6cae5c671`|
|Hugging Face checkpoint revision|`d76710ee314ee1fa8506f421664c989b40bae415`|
|Checkpoint 路径|`/home2/gmj/models/X-VLA-Calvin-ABC_D`|
|`model.safetensors` SHA256|`98813bb2063fa62f8ffb21696546005b61defca6c0724adeab524c8f82f84bd4`|
|CALVIN Python|`/home2/gmj/miniconda3/envs/calvin-py38/bin/python`|
|X-VLA Python|`/home2/gmj/miniconda3/envs/xvla-stable/bin/python`|
|X-VLA GPU|物理 GPU 0，RTX A6000；服务运行时约占 4.0 GiB|
|服务地址|`127.0.0.1:8000`；验收后已干净停止|

记录这些信息不是文书工作，而是防止日后出现“同名模型、不同权重”“同一命令、不同环境”的幽灵差异。只要 commit、hash 和 Python 路径对不上，就不能声称在复现同一次实验。

## 交付的代码入口

- `src/clawvla/scripts/calvin_capture_once.py`：反复 reset/capture，检查 static/gripper artifact、有限 20D proprio、reset 差异，并保存初始 oracle info。它像启动前的绕车检查，确认镜头、仪表和初始位置都正常。
- `src/clawvla/scripts/calvin_xvla_action_smoke.py`：重复调用真实 `/act`，统计延迟、shape 和有限值，并通过显式 `--execute` 决定是否真正执行。默认先看发动机转速，不会一拧钥匙就冲出车库。
- `src/clawvla/scripts/calvin_xvla_baseline.py`：运行不含 Planner 的 X-VLA baseline episode，按有界 chunk 执行动作，以 task oracle 停止，并输出失败原因和机器可读报告。
- `src/clawvla/action_backends/calvin.py`：把 X-VLA flow sampling steps 与 execution horizon 分开，记录原始响应 shape，再消费上游 CALVIN 客户端定义的前 10 列。
- 公共和本机 probe 默认值已改为 canonical task annotation、execution horizon 20、inference steps 10，并记录 checkpoint 身份与 hash。

## 真实服务如何解开动作协议

checkpoint 配置明确写着：

```text
num_actions=30
real_action_dim=20
action_mode=ee6d
```

真实服务返回 `30 x 20`，即使请求里是 `steps=10`。源码检查加真实响应共同确认了每个字段的含义：

- `steps=10`：flow-matching 的推理迭代次数，不是 action 行数；
- execution horizon `20`：执行返回数组的前 20 行，与官方 CALVIN 客户端一致；
- 每行 `[0:10]`：absolute XYZ、rotation-6D 和 gripper scalar；
- domain id：`2`；
- 图像映射：`image0=static`，`image1=gripper`；
- gripper 映射：scalar `< 0.8` 时 open `1`，否则 close `-1`。

阶段 1 曾提出“响应行数超过 execution horizon 就拒绝”，真实服务证明这把尺子量错了对象：上游模型天生固定吐出 30 行，客户端负责有界消费。最终实现不再把正常响应挡在门外，同时记录原始 shape，避免悄无声息地吞掉协议变化。

## 命令与现场证据

### 第一站：连续 3 次 reset/capture

```bash
python -m clawvla.scripts.calvin_capture_once \
  --config configs/calvin_xvla_mingjian.local.json \
  --repeat 3 \
  --artifact-prefix stage2/capture_contract
```

结果：

```text
status：calvin_capture_contract_passed
相机：gripper, static
proprio 维度：20
三次 reset 的 proprio 最大绝对差：0.0
```

证据文件：`tmp_artifacts/calvin/stage2/capture_contract/capture_report.json`。

这一步证明同一个初始状态连续三次都能稳稳站回起跑线，两台相机不是空壳路径，20D 状态也不是偶尔缺一列的松动插头。

### 第二站：真实 `/act` 连续请求 10 次

```bash
python -m clawvla.scripts.calvin_xvla_action_smoke \
  --config configs/calvin_xvla_mingjian.local.json \
  --requests 10 \
  --artifact-prefix stage2/xvla_action_smoke_official_contract
```

结果：

```text
合约通过：10/10
原始响应 shape：[30, 20]
有界 action chunk：[20, 10]
延迟 min/max/mean：0.198 / 0.260 / 0.209 秒
HTTP 错误：0
```

证据文件：`tmp_artifacts/calvin/stage2/xvla_action_smoke_official_contract/action_smoke_report.json`。

10 次连续成功的意义不是“模型任务成功率 100%”，而是接口像一根拧紧的水管：没有漏包、没有 NaN、没有偶发少维，也没有请求绕进代理。

### 第三站：让真实动作进入环境

同一个 action-smoke 入口使用 `--execute`，共请求 10 次，horizon 10、inference steps 10。结果：

```text
请求合约通过：10/10
执行状态：action_executed
第一条 chunk 后 oracle success：false
```

oracle 为 false 是正确结果，不是失败：一条短 chunk 只是机械臂迈出第一步，系统没有因为“轮子转了”就宣称“已经到终点”。报告保存了实际命令、状态差、动作后图像和 oracle info：`tmp_artifacts/calvin/stage2/xvla_action_smoke/action_smoke_report.json`。

### 第四站：不接 Planner 的 baseline 完整试车

第一次诊断保留了旧设置：通用短指令和 10 行 chunk。它没有 HTTP 或环境异常，但跑满 720 个 simulator step 后仍因预算耗尽失败。该失败被完整保留：

`tmp_artifacts/calvin/stage2/baseline_full_budget/baseline_report.json`。

这条失败轨迹非常有价值。它告诉我们“车没到终点”并不是发动机熄火，而是驾驶口令和换挡节奏没有对齐官方设置。

对齐上游后，使用 canonical language 和 20 行 execution horizon：

```text
success：true
task-oracle 成功步：62
chunk 数：4
环境/HTTP 错误：0
```

证据文件：`tmp_artifacts/calvin/stage2/baseline_official_client_settings/baseline_report.json`。

最后使用更新后的默认配置，对同一固定初始状态连续复跑：

```text
episode：3
成功：3
环境步数：61, 61, 62
环境/HTTP 错误：0
```

证据文件：`tmp_artifacts/calvin/stage2/baseline_fixed_state_3runs/baseline_report.json`。

单次成功可能是“刚好这次顺利”，3/3 且步数接近则说明链路已经具有基本可重复性。所有 runtime evidence 都被 Git 有意忽略；进入代码评审的是源码、测试和本报告，不包括图像、日志或模型权重。

## 尚未跨越的边界与已知提示

- debug 数据只用于工程 smoke，不报告官方 CALVIN score，也不宣称 generalization 成绩。
- 完整 `task_ABC_D` 尚未获取；正式 sequence evaluation 属于后续阶段。
- CALVIN 使用的旧 Gym 包会打印上游维护提示，这是已知外部依赖提示。
- CALVIN 的 `PlayTableSimEnv.__del__` 会再次调用 `close()`。adapter 现在在显式关闭后标记 Bullet client 已释放，防止解释器退出时再次 disconnect。
- Qwen3-VL/OpenRLHF 资源仍缺失，但它们不影响阶段 2 的 planner-free X-VLA baseline。

最终判断很清楚：真实 CALVIN/X-VLA 工程链路已经完成点火、挂挡、行驶和重复试车；正式 benchmark 与训练线仍停在另一道有明确标识的闸门后面。
