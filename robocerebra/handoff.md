# RoboCerebra 交接文档

这是 RoboCerebra subgoal-level VLA 数据、pi0.5/OpenPI 训练和原生 benchmark 接入的精简交接文档。第一阶段核心结论是：**代码链路和基线评测已经跑通，但现有模型还不能稳定完成官方任务**。

## 范围

VLA 训练默认使用 LeRobot 版数据：

```text
lerobot/robocerebra_unified
```

默认不要下载或使用 raw RoboCerebra 137GB HDF5 全量包。raw metadata 对 planning/subgoal language 检查有价值，但可训练的 VLA 主线使用 LeRobot parquet 和 mp4 shards。

## 数据解释

LeRobot 版本已经按 short-horizon skill 做了 episode 级切分。当前适配方式是把每个 `episode_index` 当成一个 VLA subgoal episode。

重要字段：

- `episode_index`
- `task_index`
- `task_text`
- `dataset_from_index`
- `dataset_to_index`
- `num_frames`
- front video path
- wrist video path
- raw action shape `[7]`
- raw state shape `[8]`

当前全量索引路径：

```text
outputs/robocerebra_lerobot_full_index.jsonl
```

开发时使用的本地全量 LeRobot 数据路径：

```text
/mnt/raid1/mjh/datasets/robocerebra_lerobot_unified
```

已验证的数据规模：

- episodes：`6660`
- frames：`571116`
- fps: `20`
- 本地 parquet/mp4 占用约 `1.5GB`
- action rows 与 episode frame range 对齐
- front/wrist video 使用 PyAV/libdav1d 抽查解码通过

## VLA Schema

pi0.5/OpenPI batch schema：

- prompt：LeRobot `task_text`
- `image.base_0_rgb`: front image
- `image.left_wrist_0_rgb`: wrist image
- `image.right_wrist_0_rgb`: zero image, mask false
- raw state：8D，pad 到 model dim 32
- raw action：7D，pad 到 model action dim 32
- action horizon：32
- `action_mask`: valid action chunk steps

必须使用 RoboCerebra 自己的 norm stats：

```text
outputs/openpi_assets/robocerebra_unified_full/norm_stats.json
```

不要复用 RoboTwin norm stats。

全量 norm stats 完整性：

- valid frames：`571116`
- state NaN/Inf：`0 / 0`
- action NaN/Inf：`0 / 0`
- 没有 near-zero std action dim
- raw gripper action 是 dim 6，近似二值 `0/1`

## 关键脚本

数据和 metadata：

```text
scripts/inspect_robocerebra_metadata.py
scripts/check_robocerebra_alignment.py
scripts/download_robocerebra_lerobot_full.py
scripts/build_robocerebra_lerobot_full_index.py
scripts/compute_robocerebra_norm_stats.py
scripts/export_robocerebra_lerobot_vla.py
scripts/export_robocerebra_subgoals.py
scripts/export_robocerebra_visual_sample.py
```

pi0.5/OpenPI：

```text
scripts/openpi_robocerebra_config.py
scripts/pi05_robocerebra_inference_server.py
scripts/robocerebra_full_task_comparison.py
scripts/train_pi05_robocerebra_lora_minimal.py
scripts/train_pi05_robocerebra_lora_random.py
scripts/train_pi05_robocerebra_lora_full_multigpu.py
```

原生 baseline：

```text
scripts/run_robocerebra_full_task_comparison.sh
scripts/robocerebra_full_task_comparison.py
```

baseline 细节见 [baseline.md](baseline.md)。

## 第一阶段交付状态

已完成：

- RoboCerebra LeRobot 数据下载、索引、metadata inspection。
- LeRobot episode 到 subgoal-level VLA sample 的解释和导出。
- full-data norm stats 计算。
- pi0.5/OpenPI batch smoke、LoRA trainer、full-data 6GPU trainer 初版。
- OpenVLA 社区 checkpoint 原生 RoboCerebra rollout 对比。
- 日志、summary、rollout video 输出链路。

未完成或仍需继续：

- 现有模型在 `Ideal/case1` full-task direct prompt 下还不能成功完成任务。
- pi0.5 rollout 需要 inference service 方式解耦 OpenPI 环境和 RoboCerebra eval 环境。
- 当前更有意义的评测指标应先看 first GT subgoal 的接近、接触、闭爪、位移和 lift，而不是直接追 full-task success。

一句话状态：**第一阶段已经能跑通 RoboCerebra benchmark 和基线链路，但模型成功率还没有做好。**

## 重要训练修复

第一次 full-data pi0.5 LoRA 暴露了一个 schema/loss bug：

- RoboCerebra action 是 raw 7D。
- pi0.5 action dim 是 32。
- 旧 trainer 把 action dim `7:32` pad 为 0 后，仍然在 32 维上平均 loss。
- 这会让真实 7D action 只占未加权 action-dim loss 的 `7/32 = 21.875%`，gripper 只占 `1/32 = 3.125%`。

现在 full-data trainer 计算 per-dimension flow loss，并使用：

- horizon `action_mask`
- 真实 7D action weights
- padded dims weight `0`
- default raw action weights `1,1,1,1,1,1,4`

这样 gripper dim 在 LoRA 训练中有足够权重。

## Graspfix 500-Step Result

Checkpoint：

```text
outputs/pi05_robocerebra_lora_full_6gpu_graspfix_500step/lora_params.pkl
```

第一个 GT subgoal：

```text
Pick up cream cheese from coffee table
```

Seed 7 chunk sweep，max steps 300：

| execute_chunk_len | min EEF-to-cream | gripper contact | lift | cream displacement |
|---:|---:|---|---:|---:|
| 16 | 0.0477 | true | 0.0000 | 0.0522 |
| 8 | 0.0530 | true | 0.0000 | 0.0529 |
| 4 | 0.0221 | false | 0.0000 | 0.0521 |
| 1 | 0.0129 | true | 0.0000 | 0.0552 |

chunk length 1 在距离/contact 上最好，但每个 env step 都调用 pi0.5，速度慢。chunk length 16 是更现实的多 seed 设置，并且已经能看到 contact/displacement。

chunk 16 多 seed 结果：

| seed | closest object | min EEF-to-cream | gripper contact | lift | cream displacement |
|---:|---|---:|---|---:|---:|
| 7 | cream_cheese_1 | 0.0477 | true | 0.0000 | 0.0522 |
| 8 | cream_cheese_1 | 0.0396 | true | 0.0000 | 0.0534 |
| 9 | cream_cheese_1 | 0.0416 | false | 0.0000 | 0.0521 |

解释：

- 模型已经不只是接近，seed 7/8 出现 contact。
- 三个 seed 里目标物体都有约 5cm 位移。
- 但模型仍不能稳定 grasp/lift，也还不能完成 pick subgoal。

这说明可以进入分阶段 continuation training，但 rollout 仍应聚焦 first-subgoal contact、grasp、lift，而不是直接用 full-task success 做主指标。

## 当前长训

夜间任务：

```text
tmux session: robocerebra_graspfix_cont3k
output dir: outputs/pi05_robocerebra_lora_full_6gpu_graspfix_cont3k_from500
init: outputs/pi05_robocerebra_lora_full_6gpu_graspfix_500step/lora_params.pkl
```

配置：

- 6 GPUs
- global batch size 6
- 3000 continuation steps
- 每 500 step save/eval
- phase-balanced sampler
- raw action weights `1,1,1,1,1,1,4`
- padded action weight `0`

启动检查：

- step 2 loss: `0.3798`
- step 3 loss: `0.1902`
- step 4 loss: `0.6102`
- max GPU memory: about `36.21GB`
- no NaN at startup
- steady step time 约 `36.5s`

3000 continuation steps 预计约 31 小时，加上 eval/checkpoint overhead。

## 下一道门

checkpoint 出来后，先评测 first GT subgoal，再考虑 full-task：

1. `lora_params_step_000500.pkl`
2. `lora_params_step_001000.pkl`
3. final `lora_params.pkl`

常规 seed 7/8/9 对比使用 `execute_chunk_len=16`；如果模型看起来有希望，再对 seed 7 复查 chunk 1。

对比指标：

- success
- completed subgoals
- min EEF-to-cream distance
- closest object
- gripper contact
- first close distance
- cream displacement
- lift height
- NaN/Inf
- rollout video
