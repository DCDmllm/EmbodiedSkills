# 铭健 CALVIN 依赖状态表

最后核验：2026-07-16。

把整条实验链想成一列火车：源码是车体，环境是轨道，数据是沿途地图，checkpoint 是发动机，HTTP 服务则是把动力传到车轮的传动轴。下表逐项标明哪些已经装好，哪些仍会让后续列车停在站台。

|依赖项|状态|证据或路径|负责人及下一步|阻塞范围|
|---|---|---|---|---|
|EmbodiedSkills 源码|就绪（READY）|`/home2/gmj/Agent_skill/EmbodiedSkills`，基线 commit `ee48a785...`|铭健：持续记录实际 commit 与 dirty diff|无|
|CALVIN 源码及子模块|就绪（READY）|`/home2/gmj/CALVIN`，commit `fa03f01f...`|铭健：smoke 期间不随意修改上游源码|无|
|CALVIN debug 数据|工程 smoke 就绪|2771 个 training episode、1675 个 validation episode；归档 SHA256 已记录|铭健：只用于工程试车|正式 ABC→D 结论|
|CALVIN ABC→D 数据|缺失（MISSING）|X-VLA 参考入口要求 `ABC_D/validation`|铭健与王崴：确认正式口径后下载 `task_ABC_D`|官方序列评测|
|`calvin-py38` 桥接环境|就绪，存在已知 metadata 警告|bridge、sequence、simulator、oracle 导入全部通过|铭健：保留兼容安装说明|reset/capture 已不受导入阻塞|
|CALVIN 真实 reset/EGL|工程 smoke 就绪|阶段 1 integration 已验证真实 reset、static/gripper 图像、20D proprio 和 task status|铭健：环境变化后复跑带 marker 的测试|capture-only smoke 无阻塞|
|X-VLA 源码|就绪（READY）|`/home2/gmj/X-VLA`，commit `6bc2513f...`|铭健：服务实现以参考代码为准|无|
|X-VLA Python 3.10 环境|就绪（READY）|`/home2/gmj/miniconda3/envs/xvla-stable`；精确版本见环境清单|铭健：保存版本账本|阶段 2 无阻塞|
|X-VLA CALVIN checkpoint|就绪（READY）|`/home2/gmj/models/X-VLA-Calvin-ABC_D`；SHA256 `98813bb...84bd4`|铭健：每份 baseline 报告保留 hash|阶段 2 无阻塞|
|X-VLA `/act` 服务|已验证，测试后已停止|真实服务在 `127.0.0.1:8000` 处理全部阶段 2 请求，验收后干净退出|复跑时使用端口表中的已验证命令|阶段 2 无阻塞|
|Qwen3-VL policy checkpoint|缺失（MISSING）|本机 dry-run 占位路径 `/home2/gmj/models/Qwen3-VL-8B-Instruct` 不存在|王崴/雨桐确认共享权重；铭健记录 hash 与路径|完整 Agent loop 与 GRPO|
|OpenRLHF trainer 环境|缺失（MISSING）|仓库内 `.venv-openrlhf-py310-cu128` 不存在|王崴/雨桐提供验证过的环境或安装配方|one-update GRPO|
|W&B 凭据|当前阶段不需要|变量未设置，本机配置关闭 W&B|仅在批准训练时启用|在线实验记录|
|本机代理绕过|本机配置已就绪|`NO_PROXY=no_proxy=127.0.0.1,localhost`|每个 launcher 必须保留|可靠的 localhost HTTP|

## 尚未跨越的正式评测边界

当前 EmbodiedSkills CALVIN probe 使用 `calvin_debug_dataset`，而发布的 X-VLA checkpoint 面向 CALVIN ABC→D。正式比较成功率之前，团队还需要冻结三件事：

1. debug 数据只负责接口 smoke；
2. `task_ABC_D` 用于 X-VLA baseline 和正式序列评测；
3. sequence 列表、reset 行为和序列数量严格继承哪个参考入口。

阶段 2 使用 debug 数据中的 scene-D 仿真配置完成工程验证和固定状态 baseline。它证明“发动机已经能带动车轮”，但没有证明“整列火车已经跑完官方赛程”，因此不能写成官方 1000-sequence benchmark 结果。
