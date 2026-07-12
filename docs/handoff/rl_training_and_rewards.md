# Agent RL 训练与奖励系统

RoboTwin 50类任务的逐项奖励表和“自然语言如何获得奖励”的完整说明见
[RoboTwin RL 奖励函数手册](../robotwin_rl_reward_catalog.md)。

当前主训练路径是 OpenRLHF。本文说明当前 `src/clawvla/rl/`、`src/clawvla/rewards/` 和 `configs/rl/` 的结构。

普通 agent runtime 见 [Runtime 架构与执行循环](runtime_architecture.md)。

## 训练目标

当前 RL 训练路径训练一个统一 VLM policy。这个 policy 同时承担：

```text
vision
scheduler
verifier
recovery
```

OpenPI/pi0.5、GR00T 和 X-VLA 不参与 RL 更新，只作为冻结 action backend。真实交互环境可以是 RoboTwin、
LIBERO、RoboCasa 或 CALVIN，由 `environment` 配置和对应 reward handler 选择。

一条 OpenRLHF prompt 对应一个 environment episode。episode 中的每次 VLM 调用都通过 policy proxy 路由到当前 actor rollout server，同时记录 token、图像 payload 和 JSON 输出。episode 结束后按 policy call 展开：每次 call 生成一条训练样本，样本使用真实执行时该 call 看到的 prompt/images，并把整条 episode 的最终 reward 写到该 call。

## RL 配置结构

定义在 `src/clawvla/rl/config.py`：

```python
RLConfig:
  name
  run_id
  trainer: EnvCommandConfig
  robotwin: EnvCommandConfig
  environment: EnvCommandConfig
  openpi: EnvCommandConfig
  policy: PolicyConfig
  rollout: RolloutConfig
  reward: RewardConfig
  openrlhf: OpenRLHFConfig
  cluster: ClusterConfig
  logging: LoggingConfig
  checkpoint: CheckpointConfig
  metadata
```

YAML 支持 `extends`，父子配置通过 deep update 合并。

### EnvCommandConfig

```python
python: str
env: dict[str, str]
unset_env: list[str]
cwd: str
```

用于 trainer/robotwin/openpi 三类进程。

### PolicyConfig

```python
model_path
served_model_name
roles
max_new_tokens
temperature
request_timeout
proxy_host
proxy_port
external_base_url
api_key
```

`roles` 当前默认：

```text
vision, scheduler, verifier, recovery
```

rollout worker 会把普通 agent config 中这些 role 的模型改成：

```text
model: <served_model_name>:<role>
api_base_url: policy proxy base url
api_key: config.policy.api_key
```

### RolloutConfig

```python
base_config
instruction
initial_stage
max_steps
run_robotwin
episodes
group_size
artifact_prefix
task_name
seeds
episode_timeout_s
openpi_port_base
start_openpi_worker
```

如果配置了 `rollout.tasks`，每个 task/seed 会生成一行 OpenRLHF prompt dataset；否则沿用单任务 `task_name/instruction/episodes/seeds`。`group_size` 和 `openrlhf.rollout_n` 共同决定每个 prompt 的 rollout samples 数；`max_steps` 是 agent loop 最大步数，action horizon 由 motion payload 控制。

### RewardConfig

```python
registry
task_map
step_cost
incomplete_episode_penalty
premature_finish_penalty
invalid_decision_penalty
skill_failure_penalty
recoverable_preflight_penalty
infra_failure_reward
```

`task_map` 必须显式把 task name 映射到 reward handler。当前 `configs/rl/rewards/robotwin.yaml` 已把 50 个 RoboTwin task 映射到 `robotwin` handler。

### OpenRLHFConfig

当前主要参数：

```python
algorithm = "grpo"
train_mode = "full" | "lora"
dtype = "bfloat16"
flash_attention = True
tensor_parallel_size
rollout_n
total_epochs
total_training_steps
max_prompt_length
max_response_length
max_model_len
max_num_batched_tokens
max_num_seqs
gpu_memory_utilization
gradient_checkpointing
use_remove_padding
learning_rate
actor_ppo_mini_batch_size
actor_ppo_micro_batch_size_per_gpu
actor_ppo_max_token_len_per_gpu
rollout_log_prob_micro_batch_size_per_gpu
ref_log_prob_micro_batch_size_per_gpu
fsdp_param_offload
fsdp_optimizer_offload
```

当前 OpenRLHF 训练命令在 `openrlhf_runner._openrlhf_train_command()` 中生成，关键设置：

```text
python -m openrlhf.cli.train_ppo_ray
--train.agent_func_path src/clawvla/rl/openrlhf_agent.py
--rollout.n_samples_per_prompt <group_size>
--algo.advantage.estimator group_norm
--ds.zero_stage 3
--ds.attn_implementation flash_attention_2
--actor.gradient_checkpointing_enable
```

OpenRLHF runtime patches 会把一个 episode 返回的多条 call-level samples 展平，并按 `task_name + instruction + seed` 做组内 advantage。

### ClusterConfig

```python
gpus
policy_gpus
openpi_gpus
robotwin_gpus
ray_num_workers
```

训练进程设置：

```text
CUDA_VISIBLE_DEVICES = ",".join(policy_gpus)
```

rollout worker 写临时 agent config 时，会把 OpenPI worker 的 `cuda_visible_devices` 改成 `openpi_gpus`。RoboTwin rollout 子进程环境会设置 `CUDA_VISIBLE_DEVICES=robotwin_gpus`。

## 入口模式

入口脚本：

```bash
./scripts/run_clawvla_rl.sh --config <yaml> --mode <mode>
```

当前 OpenRLHF 入口支持模式：

```text
dry-run
train
```

### dry-run

只生成 resolved config、OpenRLHF prompt dataset，并打印将要执行的 OpenRLHF 命令。检查：

- prompt dataset 是否能按 rollout task/seed 展开。
- OpenRLHF 命令里的 GPU、DeepSpeed、vLLM、FA2 参数是否符合配置。

### train

生成 OpenRLHF prompt dataset、resolved config，然后用 trainer python 启动：

```text
python -m openrlhf.cli.train_ppo_ray ...
```

OpenRLHF 通过 `AgentExecutor` 回调真实 RoboTwin episode。

## Run Archive

`openrlhf_runner` 会创建：

```text
runs/rl/<run_id>/
  logs/
  artifacts/
  checkpoints/
  resolved_config.yaml
```

其中：

- `artifacts/openrlhf_prompts.jsonl` 是 OpenRLHF prompt dataset。
- `logs/openrlhf_train.log` 是 OpenRLHF train 模式日志。
- `checkpoints/` 是 OpenRLHF checkpoint 输出目录。
- `resolved_config.yaml` 是合并 `extends` 后的训练配置。

train 模式里，每个真实 RoboTwin episode 还会由 rollout worker 写入：

```text
runs/rl/<run_id>/
  events.jsonl
  logs/<episode>_agent.log
  trajectories/<episode>_result.json
  rewards/<episode>_rewards.jsonl
  artifacts/<episode>/
```

`events.jsonl` 是 RL 层统一事件流，包括 policy proxy、episode start/finish、subprocess start/finish 和 episode records。

## Policy Proxy

`src/clawvla/rl/policy_proxy.py` 实现 OpenAI-compatible proxy：

```text
GET  /v1/models
POST /v1/chat/completions
```

每次 chat completion 会生成 `PolicyCallTrace`：

```python
call_id
role
model
messages
image_refs
raw_text
parsed_json
prompt_ids
response_ids
response_logprobs
status
error
started_at
ended_at
metadata
```

当前 backend：

- `StaticPolicyBackend`：固定回复，用于 smoke。
- `OpenAIForwardBackend`：转发到外部 OpenAI-compatible policy。
- `_OpenRLHFPolicyBackend`：训练时转到 OpenRLHF/vLLM 当前 actor rollout server。

proxy 会记录：

- compact messages
- image refs
- raw text
- parsed JSON 或 JSON parse error
- token ids/logprobs
- multimodal payload 计数

## OpenRLHF AgentExecutor

`src/clawvla/rl/openrlhf_agent.py` 实现：

```python
class AgentExecutor(AgentExecutorBase)
```

一条 OpenRLHF prompt 的流程：

1. 创建 `TrajectoryWriter(run_dir/events.jsonl)`。
2. 创建 `PolicyProxy(port=0)`，backend 是 `_OpenRLHFPolicyBackend`。
3. 在子线程里调用 `run_rollout_episode()`。
4. 收集 proxy.calls，写 episode。
5. 调 `build_policy_call_adapter()` 对每次 policy call 单独建训练样本。
6. 每条样本写入 `observation_tokens`、`action_ranges`、`mm_train_inputs`、`reward/scores` 和 group metadata。
7. 返回 `list[dict]` 给 OpenRLHF。

## 轨迹 Adapter 与 Action Ranges

`src/clawvla/rl/trajectory.py`

`build_policy_call_adapter(policy_call)` 是底层 helper：

- 每次 policy call 单独保留自己的 `prompt_ids` 和 `response_ids`。
- helper 内部的 `response_mask` 长度等于该 call 的 response，全部为 1。
- OpenRLHF 正式训练样本使用 `action_ranges`，只覆盖 `response_ids` 对应范围。
- OpenRLHF 路径没有 logprobs 时交给框架重新算。

OpenRLHF 训练样本实际由 `openrlhf_agent._episode_to_call_samples()` 生成，关键字段是：

```python
{
  "observation_tokens": prompt_ids + response_ids,
  "action_ranges": [(len(prompt_ids), len(prompt_ids) + len(response_ids))],
  "images": [...],
  "mm_train_inputs": ...,
  "reward": episode_reward,
  "scores": episode_reward,
  "extra_logs": {
    "clawvla_group_uid": ...,
    "clawvla_episode_uid": ...,
    "clawvla_call_index": ...,
    "clawvla_policy_calls": ...
  }
}
```

如果某次 call 有 `image_refs` 但没有训练所需的 multimodal payload，且 `require_multimodal_payload=True`，会直接抛错。也就是说，训练输入必须保留真实图像 tensor/payload，不允许只记录图片路径再做纯文本训练。

## Rollout Worker

`src/clawvla/rl/rollout_worker.py`

`run_rollout_episode()`：

1. 创建 `EpisodeRecord`。
2. 分配 OpenPI port。
3. 写临时 agent config：
   - task instruction
   - task name
   - seed
   - artifact_dir
   - VLM role -> policy proxy
   - OpenPI worker port/GPU
4. 用 `robotwin.python` 启动：

```text
python -m clawvla.scripts.run_loop
```

5. 设置环境：

```text
OPENAI_COMPATIBLE_API_KEY
OPENAI_COMPATIBLE_API_BASE_URL
CLAWVLA_RL_REWARD_JSONL
CLAWVLA_RL_TASK_NAME
CLAWVLA_RL_STEP_COST
CUDA_VISIBLE_DEVICES for robotwin_gpus
```

6. 解析 result JSON 得到 skill_calls。
7. 读取 reward JSONL。
8. 追加 episode terminal reward。

如果 agent 子进程非 0 退出或 result JSON 缺失，episode status 变成 `infra_failure`。

## Episode Reward

episode reward 来源：

1. `RuntimeRewardTracker` 在每次 `motion.execute_action` 前后 snapshot，写 step reward。
2. rollout worker 追加 terminal reward：
   - incomplete episode penalty
   - invalid decision penalty
   - skill failure penalty
   - recoverable preflight penalty
3. 如果 episode.status 是 `finished` 且没有 step reward，初始 reward_score 会是 0.0；最终仍会追加 terminal reward。

`openrlhf_agent._episode_reward()`：

- 如果 episode 是 `infra_failure`，抛错并排除在 policy update 之外。
- 如果 episode.reward_score 存在，直接用 archived score。
- 否则根据 invalid decision / skill failure 计算 fallback penalty。

## Reward Registry

`src/clawvla/rl/reward_registry.py`

接口：

```python
RewardHandler:
  name: str
  snapshot(env, blackboard) -> Any
  compute(before, after, context) -> Any
  finalize(EpisodeRecord) -> RewardRecord | None
```

`build_reward_registry(import_paths, task_map)`：

1. import 每个 `module:callable`。
2. callable 接收 `RewardRegistry` 并注册 handler。
3. 把 task name 映射到 handler name。

内置：

```text
clawvla.rl.reward_registry:register_builtin_robotwin
```

它注册 handler name `robotwin`：

- snapshot：从 live RoboTwin task_env 调 `snapshot_robotwin_task()`。
- compute：调 `compute_robotwin_reward()`。

RoboTwin 可用的任务环境、actor、gripper 和接触接口见 [RoboTwin 奖励接口速查](robotwin_reward_interfaces.md)。

`RewardRegistry.handler_for_task(task_name)` 对未配置 task 抛 `KeyError`。当前 50 任务配置已经显式映射；新增 task 时必须同步更新 `configs/rl/rewards/robotwin.yaml`。

## RoboTwin Reward Families

`src/clawvla/rewards/robotwin_reward.py`

当前内置 specs：

```text
place_container_plate  family=pick_place
stack_blocks_two       family=stack
open_laptop            family=articulation
handover_mic           family=handover
handover_block         family=handover
press_stapler          family=contact_press
click_bell             family=contact_press
click_alarmclock       family=contact_press
lift_pot               family=dual_lift
grab_roller            family=dual_lift
blocks_ranking_rgb     family=ordering
```

### Snapshot 内容

`snapshot_robotwin_task(task_env, task_name)` 保存：

```python
RewardSnapshot:
  task_name
  success: task_env.check_success()
  actors: actor pose / contact points / functional points / gripper contact positions
  grippers: left/right open/closed/value
  articulations: qpos / qpos_ratio
  metadata:
    take_action_cnt
    eval_success
    tcp
    ee
```

actor 名字来自 `RewardSpec`：

- pick_place：source、target
- stack：top、base
- articulation：articulated
- handover/dual_lift/contact_press：object 和可选 target
- ordering：ordered list

### pick_place

以 `place_container_plate` 为例，事件：

```text
source_contact
source_grasped        gripper-object contact + any gripper closed
source_lifted         source z 相对 before 增加超过 lift_margin
source_carried        已 grasp/lift 且物体运动和 TCP 运动方向一致
near_target           source/target xy 距离小于阈值
released_near_target  carried_seen + near_target + z_aligned + both grippers open
task_success          after.success
```

奖励由 step cost、contact/grasp/carry/release bonus、task success bonus 和 carried 后距离进展组成。

### stack

类似 pick_place，但目标是 top block 对齐 base block，并在 z_offset 附近稳定释放。

事件：

```text
top_contact
top_grasped
top_lifted
top_carried
xy_aligned
stacked_and_released
task_success
```

### articulation

用于 `open_laptop`：

```text
articulation_contact
joint_changed_positive
target_open_ratio_reached
task_success
```

使用 articulation qpos ratio 计算 progress。

### handover

用于 `handover_mic`、`handover_block`：

```text
object_contact
left_grasp
right_grasp
lifted_while_held
handover_or_target_release
task_success
```

既支持双臂 handover，也支持 release 到 target 的任务。

### contact_press

用于 `press_stapler`、`click_bell`、`click_alarmclock`：

```text
approached_press_point
pressed_with_closed_gripper
task_success
```

通过目标 contact/function point 到 TCP 的距离进展和接触闭合判定。

### dual_lift

用于 `lift_pot`、`grab_roller`：

```text
bilateral_contact
both_grippers_closed
controlled_lift
height_reached
task_success
```

要求双侧接触、双夹爪闭合和高度进展。

### ordering

用于 `blocks_ranking_rgb`：

```text
objects_aligned
x_order_correct
released_order
task_success
```

检查 ordered 物体的 x 顺序和 y 对齐。

### 扩展物理奖励族

当前还包括 `spatial`、`relative_place`、`collection_place`、`container_lift`、`cabinet_place`、
`stack_multi`、`tool_contact`、`dump`、`scan`、`shake`、`axis_lift` 和 `axis_away`。这些 family 共同覆盖
50个 RoboTwin 训练任务，并使用 actor pose、functional/contact point、真实接触、qpos、集合成员和任务私有目标字段。

### terminal_only

`compute_robotwin_reward()` 内部只为未登记的新任务保留 terminal-only 防御性路径。当前50个训练任务均不走该路径。
episode terminal success 只认环境 `task_status.success`（RoboTwin 即 `check_success()`），不能由 Agent loop 的
`finished` 代替。

## 添加新 RoboTwin 奖励

最小路径：

1. 在 `TASK_REWARD_SPECS` 加 `RewardSpec`。
2. 如果已有 family 能覆盖，只填 actor 名、阈值、bonus。
3. 如果是新 family，加 `_new_family_reward()` 并在 `compute_robotwin_reward()` 分发。
4. 在 `configs/rl/rewards/robotwin.yaml` 和训练 config 的 `reward.task_map` 里映射 task。
5. 加测试，至少覆盖：
   - snapshot 缺 actor 时不崩。
   - 关键事件 true/false。
   - terminal success bonus。
   - task_map 未配置时报错。

如果 reward 需要 RoboTwin task_env 的新接口，优先通过 snapshot helper 显式读取，并让缺接口时返回空/None metric。`compute()` 只消费 snapshot 结果，不直接假设某个 actor 一定存在。

## 常用命令

Dry run：

```bash
./scripts/run_clawvla_rl.sh \
  --config configs/rl/qwen3vl_pi05_multitask_1update.yaml \
  --mode dry-run
```

真实 50 任务 one-update smoke：

```bash
./scripts/run_clawvla_rl.sh \
  --config configs/rl/qwen3vl_pi05_multitask_1update.yaml \
  --mode train \
  --run-id openrlhf_multitask_1update
```
