# RL 轨迹训练改造清单

本文档记录 ClawVLA agent RL 的正式训练目标和实现清单。目标是让训练输入与真实执行输入完全一致，并按完整任务轨迹计算奖励和 GRPO 分组。

## 目标标准

- 每次模型调用独立成为训练样本：`prompt_k + images_k -> response_k`。
- 训练时模型不能看到真实执行时没有看到的其他调用 prompt/response。
- 工具结果、环境状态、skill 返回、图片路径等只能作为后续 prompt 的输入，不能作为模型输出 token 训练。
- 如果某次模型调用使用图片，训练样本必须携带真实多模态 payload；缺失时直接报错。
- 一个完整 episode 的所有 policy call 都应该进入训练 batch。
- episode 奖励来自整条任务轨迹，写入每个 policy call 的训练样本。
- GRPO 分组只能在同一个任务实例内做：同一个 `task_name + instruction + seed/initial_state` 的多条 rollout 共用同一个 `uid`，每条 rollout 使用不同 `traj_uid`。
- 不同 RoboTwin 任务不能混在同一个 GRPO group 里。

## 实施清单

- [x] 去掉正式训练路径里的整局串联 adapter。
- [x] 增加 per-call adapter：每个 `PolicyCallTrace` 独立转换为一个训练样本。
- [x] 每个 per-call 样本使用真实 `prompt_ids`、`response_ids`、`response_logprobs`、`multi_modal_data`、`mm_processor_kwargs`。
- [x] 每个 OpenRLHF per-call 样本的 `action_ranges` 只覆盖模型真实输出。
- [x] 对 prompt 长度、response 长度、logprob 长度做显式校验。
- [x] 对图片引用但缺少训练 payload 的情况显式报错。
- [x] `OpenRLHF AgentExecutor` 返回 `list[dict]`，一条 sample 对应一次模型调用。
- [x] `extra_logs` 写入 `clawvla_group_uid`、`clawvla_episode_uid`、`clawvla_call_index`、`clawvla_policy_calls`。
- [x] episode 奖励写入每个 call sample 的 `reward/scores`。
- [x] 更新测试，覆盖 per-call 样本、图片 payload、长度校验、训练 loop 不再调用整局串联 adapter。
- [x] 跑 `tests/test_rl_framework.py`。
- [x] 跑 RL dry-run。

## 后续增强

- 根据真实训练日志继续确认 `task_name + instruction + seed` 分组在长训里保持稳定。
- 如果 OpenRLHF 后续版本改变 agent output schema，需要同步更新 `openrlhf_runtime_patches.py`，不能退回整局串联。
- 评估是否引入 GiGPO 式 step-level credit assignment；默认先使用 episode-level GRPO。
