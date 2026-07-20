# 铭健阶段 1 测试与发布闸门报告：给整条流水线装上仪表盘

日期：2026-07-16。

工作分支：`mingjian/calvin-stage1`，基于 `ee48a785b329b5ceb99d24801eaf0a2f5914c94f`。

## 阶段结论

阶段 1 已完成。轻量 Python 门禁、CALVIN 合约测试和真实 reset/capture integration 全部通过。最初的本机基线是 `157 passed, 9 failed, 12 skipped`；修复后，Python 3.12 轻量环境连续两次得到完全一致的 `191 passed, 0 failed, 13 skipped`。

发布判断：**PASS WITH RISK**。

这里的含义不是“所有外部世界都已准备完毕”，而是“当前代码像一辆通过出厂检测的车，可以进入下一工位；仪表盘同时清楚亮着尚未接入 OpenRLHF 与正式 ABC→D 数据的提示灯”。真实 X-VLA action 已在阶段 2 通过，GRPO 和正式 benchmark 不属于本闸门的放行范围。

## 九项初始失败如何被真正解决

没有把失败测试整批删除，也没有靠反复重跑把红灯碰成绿灯。每个问题都被换成可移植实现或明确外部边界：

- CALVIN/LIBERO artifact 目录改用 `${PROJECT_ROOT}`；JSON loader 会递归展开 `${PROJECT_ROOT}`、`${WORKSPACE_ROOT}` 和普通环境变量。
- CALVIN/LIBERO RL 的 base-config 和 cwd 从仓库根目录解析，不再绑死旧路径 `/mnt/wangwai/vla/clawvla`。
- LIBERO checkpoint diagnosis 测试在临时目录中构造最小 LeRobot checkpoint 合约，不再偷偷读取另一台机器上的模型。
- OpenRLHF 命令构造测试隔离掉当前测试目标之外的生产 planner 数据集。
- camera-profile 单元测试使用合成相机配置；当外部 RoboTwin checkout 不存在时，源码审计给出带原因的显式 skip。

这一步像把测试台上的延长线换成标准插座：不是为了让某台机器“恰好能跑”，而是让同一份代码换工位后仍知道应该接在哪里。

## 新增的 CALVIN 回归防护网

`tests/test_calvin_contracts.py` 覆盖：

- X-VLA 响应缺失、为空、维度不足、含 NaN/Inf、不是 JSON object、超时，以及固定模型 horizon 的有界前缀处理；
- 精确 20D proprio 和 static/gripper 双 RGB 要求；
- 10D position/rotation/gripper 转换，包括 `0.8` 阈值；
- oracle success、simulator done、episode 步数耗尽三种停止条件；
- oracle 必须比较 reset info 与 current info；
- close 后清除动态状态，防止跨 episode 污染；
- sequence/subtask 越界和 annotation 缺失时显式报错；
- CALVIN reward 精确数值 `-0.05`、`9.95`、`1.95`；
- `finish_run` 不能越权替代官方 oracle success；
- 可移植配置展开、`run_environment` 优先级和 localhost proxy 绕过；
- 一条带 marker 的真实 reset/capture integration 路径。

测试捕获并推动修复了两个重要缺陷：

1. 最初误以为 X-VLA 响应行数等于请求 `steps`。阶段 2 真实验证说明，`steps` 控制 flow sampling，checkpoint 固定返回 30 行；backend 现在先记录原始形状，再取明确的有界前缀。
2. `CalvinAdapter.close()` 曾保留上一 episode 的 observation、reward、done、info 和 step count。现在关门时会把这些动态状态全部清空，不让上一场戏的布景混进下一场。

## 命令与结果

### Python 3.12 轻量快速门禁

```text
Python：/home2/gmj/Agent_skill/.venv-embodiedskills-test-py312/bin/python
环境：NO_PROXY=no_proxy=127.0.0.1,localhost；PYTHONPATH=src
第 1 次：191 passed, 13 skipped in 2.41s
第 2 次：191 passed, 13 skipped in 2.39s
结论：稳定 PASS
```

`git diff --check` 与 `python -m compileall -q src/clawvla tests` 同样通过。连续两次相同结果很重要：闸门不能像接触不良的灯泡，拍一下亮、再拍一下灭。

### CALVIN 合约门禁

```text
测试目标：tests/test_calvin_contracts.py
结果：25 passed, 1 skipped
```

唯一默认 skip 是真实 integration test。随后通过显式环境变量打开并单独执行，结果为 PASS。

### 真实 CALVIN reset/capture

```text
外层 pytest Python：Python 3.12 测试环境
CALVIN 子进程：/home2/gmj/miniconda3/envs/calvin-py38/bin/python
配置：configs/calvin_xvla_mingjian.local.json（Git 忽略的本机 override）
数据：/home2/gmj/CALVIN/dataset/calvin_debug_dataset/validation
结果：1 passed
验证内容：reset、EGL、static/gripper 图像、20D proprio、live env 绑定和 CALVIN task-status backend
```

第一次 integration 调用本身返回码为 0，但解析器误把 PyBullet 最后的 `Destroy EGL OpenGL window.` 当成结果行，而真正的 JSON 证据在前面。修复后，解析器会精确寻找结构化结果。这像从一叠收据中按编号找发票，而不是顺手拿最底下一张纸。

### 带 Torch 的补充门禁

```text
Python：/home2/gmj/miniconda3/envs/gr00t-n1/bin/python（3.10.20）
Torch：2.5.1+cu124
目标：tests/test_rl_framework.py tests/test_openrlhf_experience_integration.py
结果：158 passed, 3 skipped in 3.34s
```

这次运行覆盖了轻量环境因没有 Torch 而跳过的普通测试。剩余 3 个 skip 精确对应尚未安装的 `openrlhf` 包及其 `Experience` 实现。

### RoboTwin 源码合约审计

后续出现的 `/home2/gmj/RoboTwin` checkout 让原本跳过的 50-task 审计真正跑了起来。它发现 3 个任务的 `check_success()` 读取了只在专家 `play_once()` 中赋值的字段：普通 Agent rollout 还没走专家轨迹就可能踩到未初始化变量。

最小修复把相同的 arm 选择逻辑，以及 cabinet 的初始 `origin_z`，提前放进 actor setup。50-task 审计随后通过。这 3 项仍保留在 RoboTwin checkout 的独立、可审查 diff 中，不混入 EmbodiedSkills commit。

## Skip 与风险账本

轻量门禁的 13 个 skip 都有明确去向：

- 1 个真实 CALVIN integration：已单独启用并通过；
- 12 个 Torch 依赖测试：已在 `gr00t-n1` 中补跑；其中只有 3 个进一步依赖缺失的 OpenRLHF。

阶段 1 剩余风险：

|风险|影响|负责人及下一步|
|---|---|---|
|OpenRLHF 环境/包缺失|3 项真实 `Experience` split/alignment integration 无法对生产库执行|王崴/雨桐提供验证环境；铭健在 GRPO 前复跑|

阶段 2 之后仍存在的边界：

- 缺少正式 CALVIN ABC→D 数据；
- 缺少 Qwen3-VL 与 OpenRLHF 训练环境。

因此，阶段 1 代码可以带着已写明的风险进入合并评审；阶段 2 的真实 X-VLA action 与固定状态工程 baseline 也已有实证。仍禁止把这些结果写成官方 CALVIN benchmark 或 GRPO 训练结论。
