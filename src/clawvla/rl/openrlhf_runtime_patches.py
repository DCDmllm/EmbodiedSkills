from __future__ import annotations

import functools
import sys
from typing import Any


def apply_openrlhf_runtime_patches() -> None:
    """Patch OpenRLHF so one environment episode may emit many call-level samples."""
    _patch_rollout_actor_flatten_agent_outputs()
    _patch_samples_generator_flatten_agent_outputs()
    _patch_samples_generator_preserve_clawvla_rollout_batches()
    _patch_clawvla_group_advantages()


def _flatten_agent_outputs(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        flattened: list[dict[str, Any]] = []
        for item in value:
            flattened.extend(_flatten_agent_outputs(item))
        return flattened
    raise TypeError(f"OpenRLHF ClawVLA agent output must be dict or nested list[dict], got {type(value).__name__}")


def _patch_rollout_actor_flatten_agent_outputs() -> None:
    module = sys.modules.get("openrlhf.trainer.ray.vllm_engine")
    if module is None:
        return

    cls = getattr(module, "RolloutRayActor", None)
    if cls is None or getattr(cls, "_clawvla_flatten_agent_outputs", False):
        return

    original_generate_responses = cls.generate_responses

    @functools.wraps(original_generate_responses)
    async def generate_responses(self, *args, **kwargs):
        responses = await original_generate_responses(self, *args, **kwargs)
        return _flatten_agent_outputs(responses)

    cls.generate_responses = generate_responses
    cls._clawvla_flatten_agent_outputs = True


def _patch_samples_generator_flatten_agent_outputs() -> None:
    module = sys.modules.get("openrlhf.trainer.ppo_utils.samples_generator")
    if module is None:
        return

    cls = getattr(module, "SamplesGenerator", None)
    if cls is None or getattr(cls, "_clawvla_flatten_agent_outputs", False):
        return

    def _generate_vllm(self, dataloader_iter, num_prompts: int, dynamic_filtering, **generate_kwargs):
        prompts_consumed = 0
        accepted_experiences = []

        prompts, labels, images, exhausted = module._collect_prompt_batch(dataloader_iter, num_prompts)
        if not prompts:
            return [], prompts_consumed, True

        target_num_prompts = len(prompts)
        pending_refs = self._dispatch_prompts_to_vllm(prompts, labels, images=images, **generate_kwargs)
        prompts_consumed += target_num_prompts

        pbar = module.tqdm(range(target_num_prompts), desc="Generate samples")

        while pending_refs:
            ready_refs, pending_refs = module.ray.wait(pending_refs, num_returns=1, timeout=10.0)
            for ref in ready_refs:
                responses = _flatten_agent_outputs(module.ray.get(ref))
                experiences = [
                    self._process_response_into_experience(response, **generate_kwargs) for response in responses
                ]

                if dynamic_filtering and all(e.scores is not None for e in experiences):
                    scores = [e.scores[0].item() for e in experiences]
                    avg_reward = sum(scores) / len(scores)
                    min_r, max_r = self.args.algo.dynamic_filtering_range
                    if not (min_r < avg_reward < max_r):
                        module.logger.info(
                            "Filtered out: "
                            f"avg_reward={avg_reward:.2f}, "
                            f"threshold=({min_r:.2f}, {max_r:.2f}), "
                            f"scores={[f'{s:.2f}' for s in scores]}"
                        )
                        experiences = []

                if experiences:
                    accepted_experiences.extend(experiences)
                    pbar.set_postfix({"prompts_consumed": prompts_consumed})
                    pbar.update()
                elif dynamic_filtering:
                    new_prompts, new_labels, new_images, exhausted = module._collect_prompt_batch(dataloader_iter, 1)
                    prompts_consumed += len(new_prompts)
                    if exhausted and not new_prompts:
                        for remaining_ref in pending_refs:
                            module.ray.cancel(remaining_ref)
                        return [], prompts_consumed, True
                    if new_prompts:
                        new_refs = self._dispatch_prompts_to_vllm(
                            new_prompts, new_labels, images=new_images, **generate_kwargs
                        )
                        pending_refs.extend(new_refs)

        return accepted_experiences, prompts_consumed, exhausted

    cls._generate_vllm = _generate_vllm
    cls._clawvla_flatten_agent_outputs = True


def _patch_samples_generator_preserve_clawvla_rollout_batches() -> None:
    module = sys.modules.get("openrlhf.trainer.ppo_utils.samples_generator")
    if module is None:
        return

    cls = getattr(module, "SamplesGenerator", None)
    if cls is None or getattr(cls, "_clawvla_preserve_rollout_batches", False):
        return

    def generate_samples(self, **generate_kwargs):
        if getattr(self, "_clawvla_pending_exhausted", False):
            self._clawvla_pending_exhausted = False
            return [], None, 0, True

        if getattr(self, "_dataloader_iter", None) is None:
            self._dataloader_iter = iter(self.prompts_dataloader)
            self._sample_buffer = []

        buffered = list(getattr(self, "_sample_buffer", []))
        if buffered:
            self._sample_buffer = []
            return buffered, None, 0, False

        if self._dataloader_iter is None:
            return [], None, 0, True

        prompts_consumed = 0
        filter_pass_rate = None
        if self.args.vllm.enable_sleep:
            module.batch_vllm_engine_call(self.vllm_engines, "wake_up")

        try:
            gen_batch_size = (
                getattr(self.args.rollout, "vllm_generate_batch_size", None) or self.args.rollout.batch_size
            )
            experiences, prompts_consumed, dataloader_exhausted = self._generate_vllm(
                dataloader_iter=self._dataloader_iter,
                num_prompts=gen_batch_size,
                dynamic_filtering=self.args.algo.dynamic_filtering_enable,
                **generate_kwargs,
            )
        finally:
            if self.args.vllm.enable_sleep:
                module.batch_vllm_engine_call(self.vllm_engines, "sleep")

        if self.args.algo.dynamic_filtering_enable and prompts_consumed:
            filter_pass_rate = len(experiences) / prompts_consumed * 100

        if dataloader_exhausted:
            self._dataloader_iter = None
            module.logger.info("Prompt dataloader is exhausted.")

        if not experiences:
            return [], filter_pass_rate, prompts_consumed, self._dataloader_iter is None

        if dataloader_exhausted:
            self._clawvla_pending_exhausted = True

        return experiences, filter_pass_rate, prompts_consumed, False

    cls.generate_samples = generate_samples
    cls._clawvla_preserve_rollout_batches = True


def _patch_clawvla_group_advantages() -> None:
    module = sys.modules.get("openrlhf.trainer.ppo_utils.experience_maker")
    if module is None:
        return

    cls = getattr(module, "RemoteExperienceMaker", None)
    if cls is None or getattr(cls, "_clawvla_group_advantage_patch", False):
        return

    original_compute = cls.compute_advantages_and_returns

    @functools.wraps(original_compute)
    def compute_advantages_and_returns(self, experiences):
        if not _has_clawvla_group_metadata(experiences):
            return original_compute(self, experiences)
        return _compute_clawvla_group_advantages(self, experiences, module)

    cls.compute_advantages_and_returns = compute_advantages_and_returns
    cls._clawvla_group_advantage_patch = True


def _has_clawvla_group_metadata(experiences: list[Any]) -> bool:
    if not experiences:
        return False
    return all(
        isinstance(getattr(experience, "info", None), dict)
        and "clawvla_group_uid" in experience.info
        and "clawvla_episode_uid" in experience.info
        for experience in experiences
    )


def _compute_clawvla_group_advantages(self: Any, experiences: list[Any], module: Any) -> list[Any]:
    import torch

    module.apply_length_penalties(experiences, self.strategy.args)

    args = self.strategy.args
    exp_len = [int(experience.rewards.reshape(-1).numel()) for experience in experiences]
    raw_rewards = torch.cat([experience.rewards.reshape(-1) for experience in experiences], dim=0).float()
    group_uids = torch.cat([_info_vector(experience, "clawvla_group_uid") for experience in experiences], dim=0)
    episode_uids = torch.cat([_info_vector(experience, "clawvla_episode_uid") for experience in experiences], dim=0)

    shaped_rewards = _shape_episode_rewards(
        raw_rewards=raw_rewards,
        group_uids=group_uids,
        episode_uids=episode_uids,
        estimator=args.algo.advantage.estimator,
    ).to(raw_rewards.device)
    reward_chunks = shaped_rewards.split(exp_len)

    for experience, reward in zip(experiences, reward_chunks, strict=True):
        reward = module.compute_reward(
            reward,
            self.kl_ctl.value,
            experience.kl,
            action_mask=experience.action_mask,
            reward_clip_range=args.reward.clip_range,
        )

        if self.advantage_estimator == "gae":
            experience.advantages, experience.returns = self.get_advantages_and_returns(
                experience.values,
                reward,
                experience.action_mask,
                args.algo.advantage.gamma,
                args.algo.advantage.lambd,
            )
        elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"]:
            if args.algo.advantage.gamma != 1.0 and self.advantage_estimator in [
                "rloo",
                "reinforce_baseline",
                "group_norm",
                "dr_grpo",
            ]:
                args.algo.advantage.gamma = 1.0
            experience.returns = self.get_cumulative_returns(reward, experience.action_mask, args.algo.advantage.gamma)
            experience.advantages = experience.returns.clone()
        else:
            raise ValueError(f"Unknown advantage_estimator {self.advantage_estimator}")

        experience.info["return"] = reward.sum(dim=-1)
        experience.kl = None

    if args.algo.advantage.estimator in ["gae", "reinforce", "reinforce_baseline"]:
        all_advantages = torch.cat([exp.advantages.flatten() for exp in experiences], dim=0).float()
        all_action_masks = torch.cat([exp.action_mask.flatten() for exp in experiences], dim=0).float()
        num_actions = all_action_masks.sum()
        if num_actions > 0:
            mean = (all_advantages * all_action_masks).sum() / num_actions
            if args.algo.advantage.no_std_norm:
                rstd = 1
            else:
                var = ((all_advantages - mean).pow(2) * all_action_masks).sum() / num_actions
                rstd = torch.rsqrt(var + 1e-8)
            for exp in experiences:
                exp.advantages = (exp.advantages - mean) * rstd

    return experiences


def _info_vector(experience: Any, key: str):
    value = experience.info[key]
    if hasattr(value, "detach"):
        return value.detach().reshape(-1).cpu().long()
    raise TypeError(f"OpenRLHF ClawVLA info field must be tensor-like: {key}={type(value).__name__}")


def _shape_episode_rewards(*, raw_rewards, group_uids, episode_uids, estimator: str):
    import torch

    shaped = torch.empty_like(raw_rewards)
    group_to_episode_rewards: dict[int, dict[int, float]] = {}
    for reward, group_uid, episode_uid in zip(raw_rewards, group_uids, episode_uids, strict=True):
        group_key = int(group_uid.item())
        episode_key = int(episode_uid.item())
        episode_rewards = group_to_episode_rewards.setdefault(group_key, {})
        reward_value = float(reward.item())
        if episode_key in episode_rewards and abs(episode_rewards[episode_key] - reward_value) > 1e-6:
            raise ValueError(
                "OpenRLHF ClawVLA call samples from the same episode disagree on reward: "
                f"group_uid={group_key} episode_uid={episode_key} "
                f"left={episode_rewards[episode_key]} right={reward_value}"
            )
        episode_rewards[episode_key] = reward_value

    shaped_by_episode: dict[tuple[int, int], float] = {}
    for group_key, episode_rewards in group_to_episode_rewards.items():
        episode_keys = list(episode_rewards)
        values = torch.tensor([episode_rewards[key] for key in episode_keys], dtype=raw_rewards.dtype)
        if estimator == "rloo" and values.numel() > 1:
            baseline = (values.sum() - values) / (values.numel() - 1)
            normalized = values - baseline
        elif estimator in ["reinforce_baseline", "dr_grpo"]:
            normalized = values - values.mean()
        elif estimator == "group_norm":
            std = values.std(unbiased=True) if values.numel() > 1 else values.new_tensor(0.0)
            normalized = (values - values.mean()) / (std + 1e-9)
        else:
            normalized = values
        for episode_key, value in zip(episode_keys, normalized, strict=True):
            shaped_by_episode[(group_key, episode_key)] = float(value.item())

    for index, (group_uid, episode_uid) in enumerate(zip(group_uids, episode_uids, strict=True)):
        shaped[index] = shaped_by_episode[(int(group_uid.item()), int(episode_uid.item()))]
    return shaped
