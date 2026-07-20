# 铭健 CALVIN 关键决策日志

这份日志像实验室门口的白板：不仅写“最后怎么做”，还写清“为什么这样做”。这样换一台机器、换一个人接手时，不必重新踩一遍同样的坑。

## 2026-07-16——把三套运行环境分开，避免把不同规格的齿轮硬塞进同一个变速箱

- CALVIN rollout 固定使用 Python 3.8 / torch 1.13.1。
- X-VLA 服务使用隔离的 Python 3.10 环境。本机实测可用的组合是 torch 2.5.1+cu124；它比上游参考的 torch 2.1 更新，因此精确版本已经写入环境清单。
- ClawVLA 与 X-VLA 只通过本机 HTTP 通信，不合并两边互相冲突的依赖。
- OpenRLHF/Qwen 训练是第三套环境，不混入 CALVIN rollout 环境。

原因：三个仓库分别锁定了不同的 Python、PyTorch 和 CUDA 用户态依赖。强行混装就像把三种电压的设备接到同一个插排上，也许暂时亮灯，真正运行时却很难判断是谁烧坏了谁。进程隔离后，问题可以沿 HTTP 边界准确定位。

## 2026-07-16——debug 数据只做工程试车，不冒充正式比赛赛道

- `/home2/gmj/CALVIN/dataset/calvin_debug_dataset` 可用于 reset、图像采集和动作协议 smoke。
- 它不能用于宣称 X-VLA ABC→D 正式成绩，也不能代替 CALVIN 官方对比。
- 正式评测必须等待 `task_ABC_D` 和冻结后的 protocol manifest。

原因：X-VLA checkpoint 与参考客户端明确面向 ABC→D，而当前 EmbodiedSkills probe 指向 debug 数据。debug 数据像厂区里的短测试跑道，适合确认方向盘、刹车和仪表盘是否工作，但不能拿圈速去冒充正式赛道成绩。

## 2026-07-16——用真实服务解开 X-VLA 动作协议，而不是根据数组外形猜含义

- static RGB 对应 `image0`，gripper RGB 对应 `image1`。
- domain id 为 `2`；flow-matching 采样使用请求字段 `steps=10`。
- checkpoint 无论 `steps` 为多少，都固定返回 `30 x 20`。EmbodiedSkills 只执行一个有界的行前缀，并取每行前 10 个值。
- 官方客户端每次执行前 20 行，因此公共 CALVIN probe 的默认 execution horizon 改为 20，inference steps 保持 10。
- gripper 映射保持 `1 if action[9] < 0.8 else -1`，与 X-VLA CALVIN 参考客户端一致。

原因：先从参考客户端核对源码，再用真实 checkpoint 现场验证。旧设置“通用短指令 + 10 行动作”跑满 720 步仍失败；对齐官方的“canonical instruction + 20 行动作”后，第 62 步成功，之后固定初始状态又连续 3/3 成功。这个对照像把两把外形相似的钥匙插进同一把锁：只有实测才能知道齿纹是否真的匹配。

## 2026-07-16——本机路径放进 Git 忽略配置，不把个人工位写成全实验室默认地址

- 本机路径写入 Git 忽略的 `*.local.*` 文件。
- 公共配置中的可移植性问题作为阶段 1 的代码与测试任务单独解决。

原因：环境盘点期间直接改公共生产路径，会把“这台机器如何运行”和“团队默认配置应该是什么”混成一件事。local override 像个人转接头：解决本机接线，但不改变墙上的公共插座。

## 2026-07-16——使用 `pyhash2` 作为旧 CALVIN 依赖的兼容桥

- `pyhash==0.9.3` 和 `MulticoreTSNE` 无法在当前主机工具链中编译。
- `pyhash2==0.9.4` 提供同名的 `pyhash` 导入模块，使 CALVIN 评测工具可以加载。
- X-VLA rollout 不使用 `MulticoreTSNE`，因此不把它拖入关键运行路径。

原因：目标是修通评测必经的桥，而不是为了安装一块不经过的旧路牌去改上游源码或污染环境。

## GitHub 最终交付规则

- 日期：2026-07-16。
- 决策：阶段执行期间保留本地、可审查的实现 diff。请求范围完成后，先跑全量测试和仓库卫生检查，再创建最终 commit，并在明确授权后推送至 `https://github.com/DCDmllm/EmbodiedSkills.git`。
- push 前检查：核对 branch/status/staged diff；运行适用的全量测试和 `git diff --check`；扫描凭据、本机临时路径、异常大文件和生成产物。
- commit 范围：只包含源码、公共配置、requirements、脚本、文档和测试。排除 runs、logs、JSONL 产物、checkpoint、模型权重、仿真图像、缓存和 `*.local.*`。
- 安全边界：保留其他成员已有改动，不使用破坏性 worktree 清理命令。

## 阶段 1 CALVIN 合约决策

- 日期：2026-07-16。
- X-VLA horizon：阶段 1 曾提议拒绝所有“返回行数大于 execution horizon”的响应，阶段 2 真实验证证明这个假设不成立。`steps` 是采样迭代次数，checkpoint 的 `num_actions=30` 决定返回行数。现在 backend 会记录原始响应形状，再截取明确的有界前缀。
- episode 生命周期：`CalvinAdapter.close()` 关闭仿真器后清空所有动态 episode 状态。静态任务上下文可以重载，但 observation、oracle start info、reward、done 和 step count 绝不能像上一位乘客遗落的行李一样进入下一条轨迹。
- integration 门禁：真实 CALVIN reset/capture 保留 `calvin_integration` marker 和显式环境变量。默认 CI 给出有原因的 skip；配置齐全的机器通过指定 CALVIN Python 执行同一项测试。
- 外部源码审计：缺少 OpenRLHF 或 RoboTwin 源码时，只允许对应外部实现测试显式 skip；纯单元测试继续使用合成 fixture，不能随之跳过。

## 工作分支

- 日期：2026-07-16。
- 决策：在 `mingjian/calvin-stage1` 上继续 CALVIN 与测试工作，该分支基于 `ee48a785b329b5ceb99d24801eaf0a2f5914c94f`。
- 阶段 0/1 已存在的未提交 worktree 原样带到新分支，不移动 `main` 或 `origin/main` 的 commit 指针。
- commit 与 push 在工作完成、最终审计通过并获得相应授权前保持推迟。
