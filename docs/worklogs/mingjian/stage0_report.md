# 铭健阶段 0 验收报告：先把实验室的地基画清楚

日期：2026-07-16。

## 阶段结论

阶段 0 已完成。这个阶段没有急着让机械臂挥动，而是先把仓库、数据、Python 环境、GPU、端口和外部依赖逐件摆上工作台，贴好标签并通电检查。现在本机已经具备：CALVIN 源码、debug 数据、Python 3.8 桥接环境、可移植本机配置、X-VLA 服务环境、带 hash 的 checkpoint，以及可展开的 RL dry-run 接线。

验收结论：

- **M0–M2 工程工作：PASS。**
- **正式 ABC→D benchmark 与 GRPO：BLOCK。** 前者缺少正式数据和冻结口径，后者缺少 Qwen3-VL/OpenRLHF 资源。

这意味着“试车间已经通电、工具齐全，可以进入下一道工序”，不意味着“论文中的正式比赛已经跑完”。

## 命令与结果

### 项目初始测试基线

```text
环境：/home2/gmj/Agent_skill/.venv-embodiedskills-test-py312
命令：NO_PROXY=127.0.0.1,localhost PYTHONPATH=src python -m pytest -q --tb=no
结果：157 passed, 9 failed, 12 skipped
```

第一次运行时，本机 PolicyProxy 还出现了 HTTP 502。原因不是服务代码坏了，而是全局代理把发往 localhost 的请求“带出门绕了一圈”。补上 `NO_PROXY`/`no_proxy` 后，这类失败消失。

剩余 9 项失败全部被逐项点名，没有藏进“整体通过”四个字里：

- CALVIN 与 LIBERO artifact 仍指向 `/mnt/wangwai` 的旧机器路径；
- 缺少 LIBERO π0.5 checkpoint；
- 缺少 `/mnt/wangwai` 下的 LIBERO base config；
- 缺少在线 seed-mix 测试需要的 RoboTwin expert split/seed 数据；
- 缺少 RoboTwin checkout/camera profile 资产。

这些问题随后成为阶段 1 的可移植性和依赖治理任务。阶段 0 的作用正像医生第一次检查：先把每个异常指标写清楚，而不是把体温计藏起来宣布健康。

### 本机 CALVIN 配置检查

- Agent 配置解析：PASS。
- 使用本机 artifact 路径构造 adapter：PASS。
- RL 配置继承和变量展开：PASS。
- stack inspection：PASS。
- task context 与 oracle 创建：PASS。
- canonical task language 解析为 `push the sliding door to the left side`：PASS。

这组结果证明配置文件不再只是纸面路线图：路径能落到本机，adapter 能建立，任务语言和 oracle 也能接上。

### RL dry-run

```text
Run ID：mingjian_stage0_dryrun
运行目录：runs/rl/mingjian_stage0_dryrun
Prompt 数：1
Rollout group：2
Policy GPU：0,1
Environment GPU 字段：2
W&B：本机配置中关闭
```

dry-run 成功生成命令，但当时有意保留了几个尚不存在的目的地：

- `/home2/gmj/Agent_skill/EmbodiedSkills/.venv-openrlhf-py310-cu128/bin/python`
- `/home2/gmj/models/Qwen3-VL-8B-Instruct`
- 当时尚未准备好的 X-VLA server/checkpoint

所以 dry-run 证明的是“电路图可以展开、线头接到了正确位置”，不是“训练电流已经真正流过 optimizer”。后续阶段 2 已补齐并验证 X-VLA 部分；OpenRLHF/Qwen 仍属于后续训练阻塞项。

## 阶段 0 闸门清单

- [x] 记录仓库 commit 与 dirty 状态。
- [x] 记录 GPU、驱动、磁盘与 Python 环境。
- [x] 获取 CALVIN 与 X-VLA 上游源码。
- [x] 获取 CALVIN debug 数据并校验 checksum。
- [x] 建立 Python 3.12 轻量测试环境。
- [x] 建立 Python 3.8 CALVIN 环境。
- [x] 验证 ClawVLA bridge、CALVIN sequence、simulator 与 oracle 导入。
- [x] 记录本机代理绕过和空闲服务端口。
- [x] 创建并解析 Git 忽略的本机 Agent/RL 配置。
- [x] 完成 RL dry-run，并解释所有缺失路径。
- [x] 建立 X-VLA 环境并启动过真实服务。
- [x] 下载 X-VLA checkpoint 并记录 hash。
- [ ] 获取正式 ABC→D 数据。
- [ ] 获取 Qwen3-VL/OpenRLHF 训练资源。

未勾选项是明确摆在路障牌上的后续边界，不是被藏起来的阶段 0 失败。X-VLA 的完整实证和 checkpoint 身份分别记录在 `stage2_report.md` 与 `environment_inventory.md`。
