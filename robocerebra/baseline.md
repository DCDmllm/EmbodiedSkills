# RoboCerebra 原生基线评测

本文档记录第一阶段 RoboCerebra 原生基线的评测范围和当前结果。重点是确认 benchmark、环境、模型接口、日志和视频链路能跑通；现阶段不是证明模型已经能高成功率完成长程任务。

## 评测任务

使用官方 RoboCerebra benchmark：

```text
Ideal/case1
```

完整任务指令：

```text
Organize all the food boxes into the white storage box.
```

使用原始 `goal.json` 的 6 个 ordered subgoals：

1. pick `cream_cheese_1`
2. put `cream_cheese_1` in `white_storage_box_1_bottom_side`
3. pick `popcorn_1`
4. put `popcorn_1` in `white_storage_box_1_right_side`
5. pick `butter_1`
6. put `butter_1` in `white_storage_box_1_left_side`

约束：

- seed：`7/8/9`
- 使用相同 init scene。
- 使用相同相机输入和 success checker。
- 不使用 Planner 拆解。
- 不测试同义改写。
- 每次 policy call 都给 full-task instruction。

## 一键运行脚本

主脚本：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_robocerebra_full_task_comparison.sh
```

常用环境变量：

```bash
REPO_ROOT=/home/mjh/Projects/EmbodiedSkills
BENCH_ROOT=/mnt/raid1/mjh/datasets/RoboCerebraBench_case1
OPENVLA_OFT=/mnt/raid1/mjh/RoboTwin/RoboTwin/policy/openvla-oft
EVAL_ENV=/data/mjh-conda/envs/robocerebra-openvla-eval
PI05_ENV=/data/mjh-conda/envs/openpi-torch-py312
OUT_ROOT=outputs/robocerebra_probe_logs/full_task_comparison
REPORT_PATH=outputs/robocerebra_full_task_model_comparison.md
```

输出：

```text
outputs/robocerebra_probe_logs/full_task_comparison/<model>/seed_<seed>/
  summary.json
  step_log.jsonl
  rollout.mp4

outputs/robocerebra_full_task_model_comparison.md
```

## 当前模型

已接入：

- `Yun5/OpenVLA-RoboCerebra-L1-Proprio-4000`
- `Yun5/OpenVLA-RoboCerebra-Proprio-1`
- `Yun5/OpenVLA-RoboCerebra-NO-Proprio-2000`
- `pi05_robocerebra_lora_random_200ep_1kstep`

pi0.5 注意事项：

- pi0.5 official base + LoRA 需要 OpenPI/JAX 环境。
- RoboCerebra rollout 需要 RoboCerebra/LIBERO/robosuite 环境。
- 当前机器上这两套依赖分布在不同 conda env，所以直接同进程加载会失败。
- 已新增 `scripts/pi05_robocerebra_inference_server.py`，下一步应通过本地 inference service/IPC 方式把 pi0.5 接进 RoboCerebra eval。

## 当前结果摘要

完整任务 `Ideal/case1`，seed `7/8/9`：

| 模型 | 完成 rollout | full-task success | 平均完成 subgoal | 主要失败 |
|---|---:|---:|---:|---|
| `Yun5_OpenVLA-RoboCerebra-L1-Proprio-4000` | 3/3 | 0/3 | 0.00/6 | 没有接近第一个目标物体 |
| `Yun5_OpenVLA-RoboCerebra-Proprio-1` | 3/3 | 0/3 | 0.00/6 | 没有接近第一个目标物体 |
| `Yun5_OpenVLA-RoboCerebra-NO-Proprio-2000` | 3/3 | 0/3 | 0.00/6 | 没有接近第一个目标物体 |
| `pi05_robocerebra_lora_random_200ep_1kstep` | 0/3 | 0/3 | 0.00/6 | 同进程 backend load failed |

结论：

- 第一阶段已经能跑通 RoboCerebra 原生 benchmark、OpenVLA checkpoint 加载、rollout、success checker、summary、step log 和 video。
- 现有 OpenVLA 社区 checkpoint 在这个 full-task direct prompt 设置下没有完成第一个 subgoal。
- pi0.5 不是模型评测失败，而是需要 service 化解耦环境后再跑正式 rollout。
- 因此当前状态是“benchmark 代码和基线链路可运行”，不是“模型已经能成功完成 RoboCerebra 长程任务”。

## 下一步

优先级建议：

1. 完成 pi0.5 inference service/client，把 OpenPI 环境和 RoboCerebra eval 环境解耦。
2. 先评测第一个 GT subgoal，而不是直接看 full-task success。
3. 对 pi0.5 记录接近、接触、闭爪距离、物体位移、lift height。
4. 再决定是否继续 full-data LoRA 长训。
