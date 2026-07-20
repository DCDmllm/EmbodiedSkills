# 铭健阶段 0 环境清单

最后核验：2026-07-16（Asia/Shanghai）。

阶段 0 做的不是“安装几个包”这么简单，而是给实验室做一次彻底盘点：仓库在哪一层货架、数据有多少箱、哪把钥匙能启动哪台机器、GPU 和端口是否空闲。只有先把地基画成图纸，后面的失败才不会全部被模糊地归咎于“模型不行”。

## 仓库基线

|仓库|路径|分支|Commit|状态|
|---|---|---|---|---|
|EmbodiedSkills|`/home2/gmj/Agent_skill/EmbodiedSkills`|`main`|`ee48a785b329b5ceb99d24801eaf0a2f5914c94f`|创建工作日志前干净；本机 local 配置受 Git 忽略|
|CALVIN|`/home2/gmj/CALVIN`|`main`|`fa03f01f19c65920e18cf37398a9ce859274af76`|干净；`calvin_env` 子模块为 `1431a46bd36bde5903fb6345e68b5ccc30def666`|
|X-VLA|`/home2/gmj/X-VLA`|`main`|`6bc2513f5f1cbec715cc668b414392a6cae5c671`|干净|

上游来源：

- CALVIN：<https://github.com/mees/calvin>
- X-VLA：<https://github.com/2toinf/X-VLA>
- CALVIN X-VLA checkpoint：<https://huggingface.co/2toINF/X-VLA-Calvin-ABC_D>

## 主机资源

|项目|实际值|
|---|---|
|主机与内核|`admin123-Super-Server`，Linux `6.17.0-35-generic`，x86_64|
|GPU|3 × NVIDIA RTX A6000|
|单卡显存|49,140 MiB|
|NVIDIA 驱动|`580.126.20`|
|shell 中的 `nvcc`|未找到；不能只根据驱动版本臆测 toolkit 版本|
|`/home2` 存储|总计 22 TiB，已用 19 TiB，可用 2.7 TiB，盘点时使用率 88%|
|根分区|总计 915 GiB，盘点时可用 595 GiB|

盘点当时 GPU 0 约占用 1.8 GiB，GPU 1–2 各约 15 MiB。这只是拍照瞬间的仪表读数，不代表永久预留；任何长实验启动前都必须重新看表。

## Python 环境

### 轻量项目测试环境

```text
路径：/home2/gmj/Agent_skill/.venv-embodiedskills-test-py312
Python：3.12.3
用途：ClawVLA 编译、配置解析和依赖较轻的单元测试
安装方式：editable /home2/gmj/Agent_skill/EmbodiedSkills + pytest
```

这套环境像电工的万用表：启动快，适合先测线路通断，但它不假装自己是 RoboTwin、OpenRLHF、CALVIN 或 X-VLA 的完整发动机舱。

### CALVIN rollout 环境

```text
Conda 名称：calvin-py38
Python：/home2/gmj/miniconda3/envs/calvin-py38/bin/python
Python 版本：3.8.20
用途：CALVIN simulator、task oracle 和 ClawVLA CALVIN 子进程
```

已验证导入：

- `CalvinAdapter` 与 `CalvinHttpActionBackend`：通过。
- 使用 OmegaConf 读取 debug dataset merged config：通过。
- `get_sequences(...)`：通过。
- `calvin_env.envs.play_table_env.get_env`：通过。
- `move_slider_left` task context/oracle 初始化：通过。

安装兼容说明：

- 官方 `install.sh` 指向当前镜像不可用的 `cmake==3.18.4`，本机改用 `cmake==3.18.4.post1`。
- CALVIN 的旧 `use_2to3` 包要求 setuptools 低于 58，本环境使用 `setuptools==57.5.0`。
- 原始 `pyhash==0.9.3` 无法用当前主机编译器构建；`pyhash2==0.9.4` 提供相同的 `pyhash` 导入模块，作为兼容桥使用。
- `torch==1.13.1+cu117` 与 `torchvision==0.14.1+cu117` 只安装在 `calvin-py38`，不与 X-VLA 混用。
- `pip check` 仍报告 `MulticoreTSNE`、`plotly`、`pyhash`、`sentence-transformers` 和 `wandb` metadata 缺失。它们属于训练或可视化依赖，当前 adapter 评测必经的导入已经通过。`pyhash` 警告仍存在，是因为兼容发行包名为 `pyhash2`。

### X-VLA 服务环境

```text
Conda 名称：xvla-stable
Python：/home2/gmj/miniconda3/envs/xvla-stable/bin/python
Python 版本：3.10.20
PyTorch：2.5.1+cu124
Transformers：4.45.2
PEFT：0.17.1
FastAPI：0.115.6
Uvicorn：0.34.3
json-numpy：2.1.0
NumPy：1.26.4
```

它与 `calvin-py38` 完全隔离，已经成功导入 X-VLA model/processor，在一张 RTX A6000 上加载发布 checkpoint，并真实处理 `/act` 请求。PyTorch 版本比上游参考的 2.1 新，因此上述“实测能工作”的版本组合就是复现实验时应保存的机器铭牌。

### 仍未准备的资源

- OpenRLHF trainer 环境 `.venv-openrlhf-py310-cu128`：缺失。
- 本机 dry-run 使用的 Qwen3-VL 路径：缺失。

这两项会挡住后续 GRPO，但不影响 M0–M2 的 planner-free CALVIN/X-VLA 验证。

## CALVIN 数据

```text
数据集：/home2/gmj/CALVIN/dataset/calvin_debug_dataset
归档：/home2/gmj/CALVIN/dataset/calvin_debug_dataset.zip
归档 SHA256：c66d09147e2c806b244f18ea7d61e388d4dac11f828929779437f728d03e1204
Training episode 文件：2771
Validation episode 文件：1675
Training merged config：存在
Validation merged config：存在
```

debug 数据只负责 reset、capture 和接口 smoke。发布的 X-VLA CALVIN checkpoint 与参考客户端面向 ABC→D，并使用 `ABC_D/validation`。因此正式评测还需要匹配的 `task_ABC_D` 数据和明确冻结的 protocol。这里的边界就像“试车场”和“正式赛道”之间的围栏：可以验证车能开，但不能混写成绩。

## X-VLA checkpoint

```text
仓库 revision：d76710ee314ee1fa8506f421664c989b40bae415
路径：/home2/gmj/models/X-VLA-Calvin-ABC_D
模型文件：model.safetensors（3,519,068,172 bytes）
模型 SHA256：98813bb2063fa62f8ffb21696546005b61defca6c0724adeab524c8f82f84bd4
Checkpoint 配置：num_actions=30，real_action_dim=20，action_mode=ee6d
```

hash 是这台发动机的“钢印号码”。报告模型结果时必须同时写出它，避免两份同名 checkpoint 实际内容不同却被放在同一张对比表里。

## 代理状态

首次盘点时，全局 HTTP/HTTPS proxy 已设置，但 `NO_PROXY` 与 `no_proxy` 为空。本机 PolicyProxy 请求因此绕到代理，返回 HTTP 502。把两个绕过变量都设为 `127.0.0.1,localhost` 后，失败消失。

所有本机 CALVIN 配置都显式设置这两个变量。不得把代理凭据写入日志，也不得把个人代理 URL 复制进团队公共配置。

## 仅限本机的配置

以下文件通过 `*.local.*` 被 Git 有意忽略：

```text
configs/calvin_xvla_mingjian.local.json
configs/rl/qwen3vl_calvin_xvla_mingjian.local.yaml
```

它们把 artifact 和源码路径解析到 `/home2/gmj`，设置 localhost 代理绕过，将 policy GPU 映射到 0–1、environment GPU 映射到 2，并关闭 W&B。它们像本机的接线图：保证这台机器能复现，但不会被提交成所有人的默认布线。
