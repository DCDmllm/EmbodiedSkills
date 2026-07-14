# ClawVLA RoboTwin RL 奖励函数手册

本文记录当前 ClawVLA 使用 OpenRLHF/GRPO 训练统一 VLM 时，RoboTwin 50 类任务的奖励来源、奖励公式、任务映射，以及模型输出的自然语言如何获得 RL 信用。

对应实现：

- `src/clawvla/rewards/robotwin_reward.py`：仿真状态 snapshot 与 dense reward。
- `src/clawvla/rl/reward_tracker.py`：在每次 `motion.execute_action` 前后采样状态并计算奖励。
- `src/clawvla/rl/rollout_worker.py`：汇总 dense reward、terminal penalty 和官方成功状态。
- `src/clawvla/rl/openrlhf_agent.py`：把完整 episode reward 写到每次 VLM 调用的训练样本。
- `src/clawvla/rl/openrlhf_runtime_patches.py`：按 task/instruction/seed 做 GRPO 分组和 advantage。

## 1. 总体奖励链路

```text
VLM 看见当前 prompt / 图片
  ↓
输出 JSON，其中包含视觉判断、下一技能、子任务自然语言、验证或恢复决策
  ↓
ClawVLA 根据 JSON 调用组件
  ↓
Planner 生成的子任务自然语言原样进入 π0.5
  ↓
冻结的 π0.5 输出 32 步动作，RoboTwin 执行动作
  ↓
读取真实仿真状态：pose / contact / qpos / gripper / functional point
  ↓
计算本次 execute_action 的 dense reward
  ↓
整局结束后叠加官方 success 与 episode penalty
  ↓
总 episode reward 写给本局每一次 VLM policy call
  ↓
同 task + instruction + seed 的多条 rollout 做 GRPO 组内归一化
```

主奖励来自真实物理状态和官方成功，不使用关键词打分。唯一的文字辅助项是首次 `build_task_plan` 的
组内 expert-plan 语义相似度；它只调整 Planner 调用自身，权重远低于物理成功。

## 2. “自然语言描述”如何获得奖励

### 2.1 训练的不是一句孤立文本

统一 VLM 同时承担以下调用：

| 角色 | 典型输出 | 对奖励的影响 |
| --- | --- | --- |
| vision | 场景候选物、source/target 绑定、置信度 | 绑定正确才可能规划和执行正确对象；绑定错误通常导致无进展、skill failure 或最终失败。 |
| scheduler/planner | 下一组件、完整 task plan、每条 subgoal instruction | subgoal instruction 会原样交给 π0.5，是自然语言影响动作结果的最直接路径。 |
| verifier | 当前子任务是否完成、继续/重试/推进 | 过早推进会造成后续状态错误；过早结束不能绕过官方 `check_success()`。 |
| recovery | 重试、重新观察、修补子目标 | 合理恢复可继续获得后续成功奖励；无效循环会积累 step cost 和失败惩罚。 |

每次 VLM 调用单独成为一个训练样本：

```text
该次真实 prompt + 该次真实 images -> 该次完整 JSON response
```

`action_ranges` 只覆盖模型生成的 response token。prompt、图片、工具结果、黑板状态不是被优化的输出 token。

### 2.2 自然语言没有关键词奖励

例如模型输出：

```text
Use the left arm to grasp the brown bottle and the right arm to grasp the green bottle.
```

不会因为单独包含 `left arm`、`grasp` 或颜色词自动加分。主要信用链路是：

1. 这句话进入 π0.5。
2. π0.5 是否真的让正确夹爪接触并抓住两个瓶子。
3. 瓶子是否抬升并靠近各自目标位置。
4. RoboTwin 的 actor pose、真实 contact 和最终 `check_success()` 是否改善。
5. 上述物理变化产生 dense reward 和 success bonus。

相反，一句语言流畅但对象、左右臂或目标位置错误的指令，会因为物理状态无进展而低分；其完整 plan
通常也会在顺序感知的 expert-plan 语义对齐中低于正确 rollout。

### 2.3 整局信用如何分配给多次文字输出

设一局包含 12 次 VLM 调用，总环境奖励为 `R_episode`。episode-level GRPO 会给12次调用写入相同的
环境 advantage，但每次只训练自己的 response token：

```text
call_0 response tokens ← R_episode
call_1 response tokens ← R_episode
...
call_11 response tokens ← R_episode
```

同一个 `task_name + instruction + seed` 通常采样4条 rollout。设4条回报为：

```text
[-2.0, 0.5, 4.0, 11.2]
```

GRPO 对这4个 episode 做组内标准化。高于组均值的轨迹获得正 advantage，低于组均值的轨迹获得负 advantage；同一 episode 的所有 policy call 共享该 episode advantage。

这部分仍属于粗粒度 credit assignment。首次 `build_task_plan` 另有局部 `0.2 * A_plan`，因此 Planner
输出可以在同组 rollout 间获得更细的相对信号；其余调用仍只共享环境 advantage。

### 2.4 典型结果

| 模型输出行为 | 最终训练信号 |
| --- | --- |
| JSON 格式正确，子任务清楚，π0.5 产生正确接触和位移 | dense reward 上升，成功时再加官方 success bonus。 |
| JSON 无效或选择不存在的组件/技能 | 通常形成 `invalid_decision` 或 skill failure，并进入 episode penalty。 |
| 子任务文字过于模糊，π0.5 没有移动正确物体 | 只有 `-step_cost`，还可能叠加 incomplete/skill failure。 |
| Planner 提前输出 finish，但 `check_success()` 为假 | 仍记 incomplete，不允许通过文字宣称成功。 |
| Verifier 正确发现未完成并触发恢复 | 会暂时增加步骤成本，但若后续成功，总回报可能明显更高。 |
| 一直保持抓取姿态但没有新进展 | 新奖励族的 contact/grasp milestone 只首次给分，不能靠静止反复刷分。 |

### 2.5 已采集子任务文本怎么利用

已有数据位于 `runs/data/robotwin_expert_subtasks_train_50x50_merged`，可按 episode 顺序把 canonical
segment instruction 重建成完整 plan。主训练不要求预先制作 grounding JSON：这批 expert episode 的 seed
直接作为普通在线 RL prompt，所有 VLM 调用照常接收真实环境奖励，只有首次 Planner 调用再获得 plan 辅助分。

当前已启用首次 `build_task_plan` 的顺序感知语义辅助分。它把预测 subgoals 与同任务 expert episode 的
canonical/paraphrase 序列做单调对齐，未匹配步骤自然降分。分数只在相同 task/instruction/seed 的 GRPO
组内标准化，并以 `0.2` 权重加到 Planner 调用自身；vision/verifier/recovery 调用不接收这项 advantage。
参考计划按 task + seed 精确匹配。没有参考计划的官方有效 seed 直接 mask，不按零分处理；它们仍通过在线
图片、环境动作和物理 reward 训练 grounding/verification/recovery。官方成功 `+20` 仍是主信号。

官方有效 seed 由 `clawvla.scripts.robotwin_precompute_valid_seeds` 通过 expert `plan_success + check_success()`
筛出。`100000+` 的 50 × 100 缓存保留给 official eval；训练另采 `300000+` 的独立缓存，避免 seed
泄漏。当前缓存为 49 类 × 30 加 `put_object_cabinet` 21 条，共 1491 条且覆盖全部 50 类。
`qwen3vl_pi05_online_seed_mix_grpo.yaml` 将 2236 个 train-split expert-plan seed 与
1491 个 grounding-only seed 按约 60:40 混合，得到 3727 个 prompt；`rollout_n=4` 时一轮对应 14908 条 rollout。

```bash
cd /path/to/clawvla
./scripts/run_clawvla_rl.sh \
  --config configs/rl/qwen3vl_pi05_online_seed_mix_grpo.yaml \
  --mode dry-run \
  --run-id robotwin_online_seed_mix_check
```

`build_robotwin_planner_sft` 和对应 SFT 配置仍保留为可选的 Planner warm-start，但不再是在线 GRPO 的前置条件。

当前 8 卡在线配置按 policy/OpenPI/RoboTwin = 4/2/2 分配，环境 lane 会在两张后端卡间轮转。GRPO
使用 group size 4、temperature 1.0、全参学习率 1e-6、KL 0.001；vLLM utilization 0.7，tool loop
最多 150 次，另有 5 次重复失败与 6 次无进展动作的提前 stall 保护。
PI0.5 action horizon 默认 32；Planner 可省略该字段，若显式指定则范围为 15–32，短 horizon 仅用于纠错。
周期 checkpoint 每 10 个外层 GRPO 更新保存一次，并保留全部历史 checkpoint。

正式在线配置会在训练启动时常驻 2 个 OpenPI worker（GPU4/5）和 4 个 RoboTwin lane worker
（GPU6/7 轮转）。OpenPI 的冻结 π0.5 权重只加载一次；RoboTwin Python/SAPIEN lane 进程不重启，
但每条轨迹仍会关闭旧 task env 并按新的 task/seed 创建干净场景。环境 `task_status` 自身抛异常时，
episode 会被标成 `infra_failure` 并禁止进入 GRPO，而不是错误地当作策略失败扣分。

## 3. 奖励在什么时刻计算

`RuntimeRewardTracker` 只包围真实动作执行：

```text
before motion.execute_action
  snapshot_before

执行 π0.5 action chunk

after motion.execute_action
  snapshot_after
  reward(snapshot_before, snapshot_after)
```

因此 dense reward 的时间粒度是一次 action chunk，而不是每个物理 substep。没有执行动作的纯文字调用不会立即产生 dense reward，但会通过完整 episode reward 接受训练。

## 4. 仿真中实际读取的信号

| 信号 | RoboTwin 接口/字段 | 用途 |
| --- | --- | --- |
| 官方成功 | `task_env.check_success()` | 唯一正式 terminal success。 |
| 物体位置/姿态 | `actor.get_pose().p/q` | 距离、方向、抬升、堆叠和放置。 |
| 语义功能点 | `get_functional_point(i)` | 托盘槽位、扫描头、容器中心、物体功能中心。 |
| 语义接触点 | `get_contact_point(i)` | 按钮、把手、抓取点、锤击点。 |
| 真实夹爪接触 | `get_gripper_actor_contact_position(actor_name)` | 判断是否真的碰到/抓住目标，而不是只看夹爪闭合。 |
| 真实物体接触 | `check_actors_contact(name_a, name_b)` | 锤子碰块、罐子进入篮子等。 |
| 夹爪状态 | `is_*_gripper_open/close()`、gripper value | 抓取、释放和双臂协作。 |
| TCP/EE | `get_left/right_tcp_pose()` | 接近功能点、持物同向运动和扫描几何。 |
| 关节状态 | `get_qpos()`、`get_qlimits()` | laptop、microwave、switch、cabinet。 |
| 动态集合 | `bread`、`bottles`、`sphere_lst` | 多物体装入容器和倾倒计数。 |
| 任务私有字段 | `target_pose`、`start_height`、`origin_z`、`arm_tag` | 与每个任务的 `check_success()` 对齐。 |

图片不用于计算物理 reward。图片只作为 VLM 输入；reward 使用仿真内部真值，避免视觉估计误差污染奖励。

## 5. 全局 episode 奖励

默认配置：

| 项目 | 数值 | 触发条件 |
| --- | ---: | --- |
| `step_cost` | `-0.05` | 每次成功进入某个物理 reward family 的 `execute_action` 计算。 |
| `incomplete_episode_penalty` | `-1.0` | RoboTwin 官方 `task_status.success=False`。Agent loop 的 `finished` 不能取消它。 |
| `premature_finish_penalty` | `-4.0` | Agent 主动 `finished` 且官方失败；与 incomplete `-1` 叠加后共 `-5.0`。 |
| `stalled_loop_penalty` | `-8.0` | 5次相同失败或6次成功动作无进展；替代 incomplete，叠加在此前 dense reward 上。 |
| `invalid_decision_penalty` | `-2.0` | 非法组件、技能、stage 或控制决策。 |
| `skill_failure_penalty` | `-1.0` | 非 invalid、非可恢复 preflight 的失败技能。 |
| `recoverable_preflight_penalty` | `-0.1` | stale perception/world state 等可恢复刷新。 |
| infra failure | 不训练 | 服务崩溃、adapter损坏、结果文件缺失等不作为坏策略样本。 |

所有 dense family 都保留官方成功奖励：

| 项目 | 数值 |
| --- | ---: |
| 本 episode 首次官方 success | `+20.0` |
| 已成功后继续动作，或破坏后再次成功 | `+0.0` success bonus；仍计算 step cost 与物理进展/退步 |

Terminal record 本身不再额外重复加成功分，只负责 penalty 与官方完成状态。
`task_success` 是只增不减的 episode milestone，因此成功→破坏→再成功也不能重复领取 `+20.0`。

## 6. 奖励族公式

以下 `new` 表示该 milestone 在此前 action chunk 中从未出现。连续 shaping 使用状态势能差，例如
`scale * (distance_before - distance_after)`。它正负对称且不逐步裁剪，因此任何完整往返的 shaping
严格相消；再计入每个 chunk 的 step cost 后，往返行为为净负回报。

### 6.1 `spatial`

适用：普通放置、双物体目标、固定目标姿态、功能点对齐。

```text
reward = -step_cost
       + 0.2                           if new real gripper contact
       + 0.6                           if new grasp(contact + closed)
       + 2 * (distance_before-distance_after)
       + (orientation_error_before-orientation_error_after)
       + 4 * held_object_height_delta  if grasped
       + 2.0                           if goal first becomes satisfied
       + 1.0                           if that goal also requires release
       + official success bonus
```

一个任务可包含多个 goal。每个 goal 独立记录 contact、grasp、distance、height、orientation、release 和 satisfied milestone。

### 6.2 `pick_place`（原有专用族）

适用：`place_container_plate`。

| 条件 | 奖励 |
| --- | ---: |
| 首次 gripper与source接触 | `+0.4` |
| 首次 grasp | `+1.0` |
| 首次 held+lifted+与TCP同向移动 | `+1.0` |
| carried后首次在目标附近释放 | `+3.0` |
| 持物时source-target距离势能差 | `distance_before-distance_after` |
| 本episode首次官方成功 | `+20` |

### 6.3 `relative_place`

适用：放在另一个物体左边/右边。

```text
new grasp                         +0.5
到理想间距的误差势能差             2 * (error_before-error_after)
首次满足方向、距离、侧向对齐和释放  +3.0
```

`new grasp` 由 episode milestone 去重，松开后重新抓取不会再次获得 `+0.5`。没有强制要求必须搬运，
因为官方任务只关心最终相对位置，推动也可能是合法策略；抓取只是帮助探索的弱提示。距离项是保守势能差，
靠近后原路或换步长远离都无法净赚 shaping。

### 6.4 `collection_place`

适用：数量动态的 bread/bottle 集合。

```text
所有物体距离目标的势能差             sum(distance_before-distance_after)
有效区域内物体数量变化               1.5 * (count_after-count_before)
首次全部进入目标                    +3.0
```

### 6.5 `container_lift`

适用：先把物体放进篮子，再把装载后的篮子抬起。

```text
source-container 距离势能差          2 * (distance_before-distance_after)
首次真实接触且source进入container    +2.0
首次source和container都抬升且容器正  +3.0
```

容器正立通过容器局部 y 轴与世界 z 轴的点积判断，与官方成功逻辑一致。

### 6.6 `stack` / `stack_multi`

`stack` 是两块积木专用旧族；`stack_multi` 支持三块积木和任意顺序碗堆叠。

`stack_multi`：

```text
堆叠链总误差势能差                   2 * (error_before-error_after)
每个首次达到 xy/z 对齐的相邻pair      +1.5
首次全部pair对齐且双爪释放             +3.0
```

碗任务先按当前 z 高度排序，再检查相邻层；积木任务使用固定 block1→block2→block3 顺序。

### 6.7 `ordering`

适用：RGB/尺寸排序。

| 条件 | 奖励 |
| --- | ---: |
| 首次物体pair对齐 | `+1.5` |
| 首次x顺序正确且对齐 | `+2.0` |
| 首次顺序正确、对齐且释放 | `+3.0` |

### 6.8 `articulation`

适用：laptop、microwave、switch。

```text
首次与关节物体真实接触                +contact_bonus
qpos ratio 势能差                      delta * progress_scale（反向变化等额扣分）
首次达到目标 qpos ratio                +target_bonus
```

目标 ratio 分别来自各任务官方成功阈值。

### 6.9 `cabinet_place`

适用：打开 cabinet 后把物体放进去。

```text
cabinet qpos ratio 势能差              3 * delta
object 到 cabinet functional point势能差 2 * (distance_before-distance_after)
首次物体靠近目标且释放                +3.0
```

### 6.10 `contact_press`

适用：stapler、bell、alarm clock。

```text
TCP 到指定 contact point 的距离势能差  distance_before-distance_after
首次发生真实接触且夹爪闭合             +2.0
```

最终成功仍由任务自己的 `stage_success_tag/check_success()` 判断。

### 6.11 `tool_contact`

适用：hammer击打block。

```text
首次真实抓住工具                      +0.6
工具functional point距离势能差          3 * (distance_before-distance_after)
首次对齐并发生hammer-block真实接触      +4.0
```

### 6.12 `handover`

适用：microphone/block交接。

```text
首次左臂抓住                          +1.0
首次右臂抓住                          +1.0
首次抓持时抬升                         +0.8
首次完成双臂转移或目标释放              +4.0
```

### 6.13 `dual_lift`

适用：pot、roller。

```text
首次双侧接触且双爪闭合                 +1.2
首次受控抬升                           +2.0
首次达到目标高度且双爪闭合             +3.0
```

### 6.14 `axis_lift` / `axis_away`

`axis_lift`：根据任务选择的方向，把瓶子移到左/右侧并抬高。

```text
首次抓住                              +0.6
有符号轴向位置势能差                   2 * delta
抓持高度势能差                         4 * delta_z
首次同时达到侧向阈值和高度阈值         +3.0
```

`axis_away`：playing card 的 `abs(x)` 向桌边阈值推进，首次越过阈值并释放 `+3.0`。

### 6.15 `dump`

适用：把小桌面垃圾桶里的球倒入大垃圾桶。

```text
目标高度区间内sphere数量变化           1.0 * count_delta
抓持container高度势能差                3 * delta_z
首次container达到倾倒高度             +1.5
首次所有sphere进入目标高度区间         +4.0
```

### 6.16 `scan`

适用：scanner扫描object。

1. 从 scanner functional point 的四元数旋转局部 `[0,0,-1]` 得到扫描射线。
2. 计算object到射线的垂直误差与沿射线深度。

```text
首次scanner和object都被双臂持有        +0.8
扫描线误差势能差                       4 * (error_before-error_after)
首次线误差<0.025且深度在(0,0.07)       +4.0
```

### 6.17 `shake`

适用：竖直/水平摇瓶。

```text
首次真实持瓶                          +0.5
首次持瓶达到目标高度                  +0.8
前3个产生足够目标轴位移的chunk          每次+1.0
```

水平任务检查 x 轴位移，竖直任务检查 z 轴位移。只有实际运动可获得 shake 增量，且整局最多3次；
之后继续摇动只有 step cost，不能靠未完成任务时无限摇动累积分数。

## 7. 50类任务逐项映射

| # | RoboTwin task | 奖励族 | 主要对象/目标 | 当前 dense 条件与阈值 |
| ---: | --- | --- | --- | --- |
| 1 | `adjust_bottle` | axis_lift | bottle、qpose_tag | 按qpose_tag向左/右越过`|x|>0.15`，功能点高度`>0.9`。 |
| 2 | `beat_block_hammer` | tool_contact | hammer FP0 → block FP1 | xy误差`<0.02`并发生真实hammer-block接触。 |
| 3 | `blocks_ranking_rgb` | ordering | block1/2/3 | pair差`x<0.13,y<0.03`，x顺序1<2<3并释放。 |
| 4 | `blocks_ranking_size` | ordering | block1/2/3 | 与RGB排序相同的几何条件，actor编号对应大→小。 |
| 5 | `click_alarmclock` | contact_press | alarm CP0 | TCP接近；真实接触点xy/z约`0.03`内。 |
| 6 | `click_bell` | contact_press | bell CP0 | TCP接近；真实接触点xy`0.025`、z`0.03`内。 |
| 7 | `dump_bin_bigbin` | dump | deskbin、sphere_lst、dustbin | deskbin高度`>=1.0`；每个sphere高度进入`[0.13,0.25]`。 |
| 8 | `grab_roller` | dual_lift | roller | 双爪闭合、双侧接触、roller高度`>0.8`。 |
| 9 | `handover_block` | handover | box → target_box FP1 | box与目标xy`<0.03`、z`<0.01`，释放。 |
| 10 | `handover_mic` | handover | microphone | 两臂接触历史、接收臂闭合、原臂打开、高度`>0.92`并越过侧边。 |
| 11 | `hanging_mug` | spatial | mug FP0 → rack pose/FP0中点 | xy`<0.02`、高度`>0.86`并释放。 |
| 12 | `lift_pot` | dual_lift | pot CP0/CP1 | 双侧抓取、pot高度`>0.82`；官方成功还检查TCP距抓点`<0.03`和朝向。 |
| 13 | `move_can_pot` | spatial | can → task.target_pose | target pose距离进展，xy约`0.04`、z约`0.03`并释放；官方成功补充相对侧和朝向。 |
| 14 | `move_pillbottle_pad` | spatial | pillbottle → pad | xy`<0.03`并释放。 |
| 15 | `move_playingcard_away` | axis_away | playingcards | `abs(x)>0.23`并双爪释放。 |
| 16 | `move_stapler_pad` | spatial | stapler → pad | xy`<0.025`、z`<0.02`、绝对四元数分量近似均匀并释放。 |
| 17 | `open_laptop` | articulation | laptop | qpos ratio`>=0.4`；接触点和TCP进展，官方成功要求TCP距CP1`<0.1`。 |
| 18 | `open_microwave` | articulation | microwave | qpos ratio`>=0.6`。 |
| 19 | `pick_diverse_bottles` | spatial×2 | bottle1/2 → left/right target_pose | 两瓶xy各`<0.1`、高度各`>0.89`。 |
| 20 | `pick_dual_bottles` | spatial×2 | bottle1/2 → left/right target_pose | 与diverse bottles相同，分别记录两瓶进展。 |
| 21 | `place_a2b_left` | relative_place | object在target_object左侧 | 距离`(0.08,0.2)`、左侧关系、y差`<0.05`并释放。 |
| 22 | `place_a2b_right` | relative_place | object在target_object右侧 | 距离`(0.08,0.2)`、右侧关系、y差`<0.05`并释放。 |
| 23 | `place_bread_basket` | collection_place | bread列表 → breadbasket | 每块bread xy`<0.05`，高度`>0.73`并释放。 |
| 24 | `place_bread_skillet` | spatial | bread → skillet FP0 | xy`<0.035`，高度由官方成功最终确认。 |
| 25 | `place_burger_fries` | spatial×2 | hamburg→tray FP0；fries→FP1 | 两个xy距离分别`<0.08`并释放。 |
| 26 | `place_can_basket` | container_lift | can进入basket后抬起 | actor真实接触、距离`<0.15`；两者相对起点抬升`>0.02`且basket正立。 |
| 27 | `place_cans_plasticbox` | spatial×2 | object1/2 → plasticbox FP0/FP1最近点 | 两个物体距最近槽位xy`<0.04`并释放。 |
| 28 | `place_container_plate` | pick_place | container → plate | xy`<0.05`、z`<0.03`并释放；含抓取、搬运和距离进展。 |
| 29 | `place_dual_shoes` | spatial×2 | left/right shoe → shoebox两槽 | xy`<0.05`、目标四元数误差约`<0.14`并释放。 |
| 30 | `place_empty_cup` | spatial | cup FP0 → coaster FP0 | xy`<0.035`、z`<0.015`并释放。 |
| 31 | `place_fan` | spatial | fan → target_pose | 3D距离`<0.04`、目标四元数误差`<0.1`并释放。 |
| 32 | `place_mouse_pad` | spatial | mouse → target | xy约`<0.02`，用官方四元数乘积形式检查平放并释放。 |
| 33 | `place_object_basket` | container_lift | object进入basket后抬起 | 与can-basket同族：真实接触、距离、两者抬升、basket正立。 |
| 34 | `place_object_scale` | spatial | object → scale FP0 | xy`<0.035`并释放。 |
| 35 | `place_object_stand` | spatial | object → displaystand | xy`<0.03`并释放。 |
| 36 | `place_phone_stand` | spatial | phone FP0 → stand FP0 | 3D阈值约`0.045`并释放。 |
| 37 | `place_shoe` | spatial | shoe → 固定目标 | xy`<0.05`、目标四元数误差`<0.14`并释放。 |
| 38 | `press_stapler` | contact_press | stapler CP2 | TCP接近和真实接触，目标附近约`0.03`。 |
| 39 | `put_bottles_dustbin` | collection_place | bottles列表 → `[-0.45,0]` | xy分轴阈值`[0.221,0.325]`，z在`[0.2,0.7]`。 |
| 40 | `put_object_cabinet` | cabinet_place | cabinet qpos；object→cabinet FP0 | 奖励开柜进展、xy`<0.05`、物体释放；官方成功补充相对初始高度范围。 |
| 41 | `rotate_qrcode` | spatial | qrcode姿态 | 目标四元数`[0.707,0.707,0,0]`误差`<0.1`、高度`<0.75`并释放。 |
| 42 | `scan_object` | scan | scanner FP0射线 → object | object到扫描线误差`<0.025`、深度`(0,0.07)`且双臂均持物。 |
| 43 | `shake_bottle` | shake-z | bottle | 持瓶高度`>0.8`；chunk的z位移`>=0.025`才给shake增量，整局最多3次。 |
| 44 | `shake_bottle_horizontally` | shake-x | bottle | 持瓶高度`>0.8`；chunk的x位移`>=0.025`才给shake增量，整局最多3次。 |
| 45 | `stack_blocks_three` | stack_multi | block1→2→3 | 相邻xy`<0.025`、z offset`0.05±0.012`，全部释放。 |
| 46 | `stack_blocks_two` | stack | block2 → block1 | xy`<0.035`、目标z offset约`0.05`，含抓取/搬运/堆叠。 |
| 47 | `stack_bowls_three` | stack_multi | bowl1/2/3按高度排序 | 相邻xy`<0.04`，z offsets约`[0.03,0.04]`，全部释放。 |
| 48 | `stack_bowls_two` | stack_multi | bowl1/2按高度排序 | xy`<0.04`、z offset约`0.03`，全部释放。 |
| 49 | `stamp_seal` | spatial | seal → target | xy`<0.01`并释放。 |
| 50 | `turn_switch` | articulation | switch | qpos ratio接近上限，dense目标`>=0.95`；官方成功为距上限`<0.05`。 |

## 8. 防奖励作弊规则

| 风险 | 当前处理 |
| --- | --- |
| 模型直接宣称任务完成 | terminal success 必须来自 RoboTwin `check_success()`。 |
| 官方成功后破坏再恢复 | `task_success` 是只增不减的 episode milestone，`+20` 整局只能领取一次。 |
| 原地保持抓取/目标状态反复刷分 | 所有绝对状态 bonus 采用milestone，只在首次达到时给分。 |
| 靠近—远离或打开—关闭往返刷分 | 距离、姿态、关节、计数和高度 shaping 使用未逐步裁剪的正负对称势能差；完整循环严格相消。 |
| 抓住抬高、松手降低后重新抓取 | 首次抓取后继续计算高度势能，松手下降也会等额扣回；适用于 spatial、axis_lift 和 dump。 |
| 失败时持续摇瓶刷动作分 | shake增量整局最多发3次，之后继续摇动只产生step cost。 |
| 相同失败skill持续重试 | 完全相同失败累计5次后以`stalled_loop`结束，触发序列不再逐次重复扣skill failure。 |
| action执行成功但无物理进展 | 同一subgoal连续6个action chunk没有新milestone或正向物理reward后结束。 |
| stall后保留已完成进展 | 不覆盖整局分数；在此前dense reward上追加`-8`，保持GRPO组内排序。 |
| 通过无效动作拖延 | 每个action chunk有`-0.05` step cost。 |
| 官方失败时提前finish省步骤 | incomplete `-1` 再叠加 premature finish `-4`，合计 `-5`。 |
| 反复输出非法JSON或非法技能 | invalid decision和skill failure按episode累计惩罚。 |
| 服务崩溃被当成坏策略 | infra failure从训练样本中剔除。 |
| 只优化文字格式、不完成物理任务 | Planner语义项只有`0.2`组内权重；官方成功`+20`和物理进展仍主导。 |

旧的 `pick_place/stack/articulation/handover/contact_press/dual_lift/ordering` family 也已统一改成首次触发制。

## 9. Reward audit 建议

正式RL前，每类任务至少检查一条成功专家轨迹和一条失败轨迹：

1. 输出每次 action chunk 的 reward、events、metrics、milestones。
2. 成功轨迹的累计奖励应明显高于失败轨迹。
3. 接近目标时距离指标应单调改善或总体改善。
4. 抓取、放置、堆叠等离散 milestone 不应无意义重复。
5. 四条同seed rollout必须有足够reward方差，否则GRPO advantage会接近0。
6. 如果某个dense proxy与官方 `check_success()` 冲突，以官方成功为准并修正proxy。
