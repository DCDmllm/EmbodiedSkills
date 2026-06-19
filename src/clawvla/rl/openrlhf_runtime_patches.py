from __future__ import annotations

import functools
import logging
import sys
from typing import Any

_PATCH_LOGGER: Any | None = None


def apply_openrlhf_runtime_patches() -> None:
    """Patch OpenRLHF so one environment episode may emit many call-level samples."""
    _patch_rollout_actor_flatten_agent_outputs()
    _patch_samples_generator_flatten_agent_outputs()
    _patch_samples_generator_preserve_clawvla_rollout_batches()
    _patch_clawvla_modality_bucketed_experience_forward()
    _patch_clawvla_training_modality_alignment()
    _patch_clawvla_actor_training_shuffle()
    _patch_clawvla_group_advantages()


def _patch_logger() -> Any:
    global _PATCH_LOGGER
    if _PATCH_LOGGER is not None:
        return _PATCH_LOGGER
    try:
        from openrlhf.utils.logging_utils import init_logger

        _PATCH_LOGGER = init_logger(__name__)
    except Exception:
        logging.basicConfig(level=logging.INFO)
        _PATCH_LOGGER = logging.getLogger(__name__)
    return _PATCH_LOGGER


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


def _patch_clawvla_modality_bucketed_experience_forward() -> None:
    module = sys.modules.get("openrlhf.trainer.ppo_utils.experience_maker")
    if module is None:
        return

    cls = getattr(module, "RemoteExperienceMaker", None)
    if cls is None or getattr(cls, "_clawvla_modality_bucketed_forward_patch", False):
        return

    original_make_experience = cls.make_experience

    @functools.wraps(original_make_experience)
    def make_experience(self, samples_list):
        buckets = _split_indices_by_modality(samples_list)
        non_empty_buckets = [indices for indices in buckets.values() if indices]
        if len(non_empty_buckets) <= 1:
            return original_make_experience(self, samples_list)
        return _make_experience_with_modality_buckets(self, samples_list, buckets, module)

    cls.make_experience = make_experience
    cls._clawvla_modality_bucketed_forward_patch = True


def _make_experience_with_modality_buckets(self: Any, samples_list: list[Any], buckets: dict[str, list[int]], module: Any):
    start_time = module.time.time()
    module.logger.info(
        "ClawVLA bucketed mixed-modality experience forward: "
        f"text={len(buckets['text'])} multimodal={len(buckets['multimodal'])}"
    )
    module.logger.info(f"Starting experience making with {sum([len(s.sequences) for s in samples_list])} samples")

    args = self.strategy.args
    device = "cpu"
    duplicate_factor = args.ds.ring_attn_size * args.ds.tensor_parallel_size
    dummy_ref = module.ray.put([[None]] * (len(samples_list) * duplicate_factor))

    sequences_list = [s.sequences for s in samples_list]
    attention_mask_list = [s.attention_mask for s in samples_list]
    action_mask_list = [s.action_mask for s in samples_list]
    forward_kwargs = dict(sequences=sequences_list, action_mask=action_mask_list, attention_mask=attention_mask_list)

    use_reward_model = samples_list[0].rewards is None
    if use_reward_model:
        if self.reward_model_group is None:
            raise ValueError("reward_model_group is required when rewards are not precomputed")
        r_refs = self._dispatch_forward(
            self.reward_model_group,
            args.train.colocate_all,
            sequences=sequences_list,
            attention_mask=attention_mask_list,
            pad_sequence=[True] * len(samples_list),
        )
    else:
        r_refs = None

    action_log_probs_list = _dispatch_forward_by_modality(
        self,
        module=module,
        group=self.actor_model_group,
        sync_condition=args.train.colocate_all or args.train.colocate_actor_ref,
        samples_list=samples_list,
        buckets=buckets,
        duplicate_factor=duplicate_factor,
        base_forward_kwargs=forward_kwargs,
        result_name="actor_action_log_probs",
    )

    if self.critic_model_group is not None:
        if args.train.colocate_critic_reward and r_refs is not None:
            module.ray.get(r_refs)
            module.ray.get(self.reward_model_group.async_run_method(method_name="empty_cache"))

        value_ref = self._dispatch_forward(
            self.critic_model_group,
            args.train.colocate_all or args.train.colocate_critic_reward,
            **forward_kwargs,
        )
    else:
        value_ref = dummy_ref

    if self.initial_model_group is not None:
        base_action_log_probs_list = _dispatch_forward_by_modality(
            self,
            module=module,
            group=self.initial_model_group,
            sync_condition=args.train.colocate_all or args.train.colocate_actor_ref,
            samples_list=samples_list,
            buckets=buckets,
            duplicate_factor=duplicate_factor,
            base_forward_kwargs=forward_kwargs,
            result_name="reference_action_log_probs",
        )
    else:
        base_action_log_probs_ref = dummy_ref
        base_action_log_probs_list = self._flatten_results(base_action_log_probs_ref, duplicate_factor)

    value_list = self._flatten_results(value_ref, duplicate_factor)

    if use_reward_model:
        rewards_list = self._flatten_results(r_refs, duplicate_factor)
        for i, samples in enumerate(samples_list):
            samples.rewards = rewards_list[i]
            samples.info["reward"] = rewards_list[i]

    if not (
        len(samples_list) == len(action_log_probs_list) == len(base_action_log_probs_list) == len(value_list)
    ):
        raise RuntimeError(
            "OpenRLHF ClawVLA bucketed forward result length mismatch: "
            f"samples={len(samples_list)} action={len(action_log_probs_list)} "
            f"reference={len(base_action_log_probs_list)} value={len(value_list)}"
        )

    for samples, action_log_probs, base_action_log_probs, value in zip(
        samples_list, action_log_probs_list, base_action_log_probs_list, value_list, strict=True
    ):
        if (self.initial_model_group is not None) and (not args.algo.kl.use_loss):
            kl = module.compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                kl_estimator=self.strategy.args.algo.kl.estimator,
            )
            logprobs_diff = action_log_probs.float() - base_action_log_probs.float()
        else:
            kl = module.torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)
            logprobs_diff = module.torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)
        kl_mean = module.masked_mean(kl, samples.action_mask, dim=-1)
        logprobs_diff_mean = module.masked_mean(logprobs_diff, samples.action_mask, dim=-1)

        if not args.algo.kl.use_loss:
            base_action_log_probs = None

        samples.action_log_probs = action_log_probs
        samples.base_action_log_probs = base_action_log_probs
        samples.values = value
        samples.kl = kl
        samples.info["kl"] = kl_mean
        samples.info["logprobs_diff"] = logprobs_diff_mean

    duration = module.time.time() - start_time
    time_str = str(module.timedelta(seconds=duration)).split(".")[0]
    module.logger.info(f"Experience making completed in {time_str}")
    return samples_list


def _dispatch_forward_by_modality(
    self: Any,
    *,
    module: Any,
    group: Any,
    sync_condition: bool,
    samples_list: list[Any],
    buckets: dict[str, list[int]],
    duplicate_factor: int,
    base_forward_kwargs: dict[str, list[Any]],
    result_name: str,
) -> list[Any]:
    results: list[Any] = [None] * len(samples_list)
    for bucket_name in ("text", "multimodal"):
        indices = buckets[bucket_name]
        if not indices:
            continue
        if bucket_name == "multimodal":
            _validate_multimodal_bucket_payload(samples_list, indices)
        dispatch_indices = _pad_indices_for_actor_group(group, indices)
        kwargs = _select_forward_kwargs(base_forward_kwargs, dispatch_indices)
        if bucket_name == "multimodal":
            kwargs["mm_train_inputs_list"] = [samples_list[index].mm_train_inputs for index in dispatch_indices]

        refs = self._dispatch_forward(group, sync_condition, **kwargs)
        bucket_results = self._flatten_results(refs, duplicate_factor)
        if len(bucket_results) != len(dispatch_indices):
            raise RuntimeError(
                "OpenRLHF ClawVLA bucketed forward returned unexpected result count: "
                f"name={result_name} bucket={bucket_name} results={len(bucket_results)} "
                f"dispatched={len(dispatch_indices)} original={len(indices)}"
            )
        _scatter_by_indices(results, indices, bucket_results[: len(indices)], result_name)

    missing = [index for index, value in enumerate(results) if value is None]
    if missing:
        raise RuntimeError(f"OpenRLHF ClawVLA bucketed forward left missing {result_name}: indices={missing}")
    return results


def _select_forward_kwargs(base_forward_kwargs: dict[str, list[Any]], indices: list[int]) -> dict[str, list[Any]]:
    return {key: [value[index] for index in indices] for key, value in base_forward_kwargs.items()}


def _scatter_by_indices(results: list[Any], indices: list[int], values: list[Any], result_name: str) -> None:
    if len(indices) != len(values):
        raise RuntimeError(
            "OpenRLHF ClawVLA bucketed scatter length mismatch: "
            f"name={result_name} indices={len(indices)} values={len(values)}"
        )
    for index, value in zip(indices, values, strict=True):
        if results[index] is not None:
            raise RuntimeError(f"OpenRLHF ClawVLA duplicate bucket result for {result_name}: index={index}")
        results[index] = value


def _pad_indices_for_actor_group(group: Any, indices: list[int]) -> list[int]:
    min_size = _minimum_batch_size_for_actor_group(group)
    if len(indices) >= min_size:
        return list(indices)
    if not indices:
        raise ValueError("OpenRLHF ClawVLA cannot pad an empty modality bucket.")
    return list(indices) + [indices[-1]] * (min_size - len(indices))


def _minimum_batch_size_for_actor_group(group: Any) -> int:
    actor_handlers = getattr(group, "_actor_handlers", None)
    duplicate_actors = int(getattr(group, "duplicate_actors", 1) or 1)
    if not actor_handlers:
        return 1
    return max(1, len(actor_handlers) // duplicate_actors)


def _validate_multimodal_bucket_payload(samples_list: list[Any], indices: list[int]) -> None:
    missing = [
        index
        for index in indices
        if not _contains_multimodal_train_inputs(getattr(samples_list[index], "mm_train_inputs", None))
    ]
    if missing:
        raise ValueError(
            "OpenRLHF ClawVLA multimodal call samples are missing mm_train_inputs: "
            f"indices={missing}"
        )


def _split_indices_by_modality(samples_list: list[Any]) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = {"text": [], "multimodal": []}
    for index, sample in enumerate(samples_list):
        key = "multimodal" if _experience_has_multimodal_payload(sample) else "text"
        buckets[key].append(index)
    return buckets


def _experience_has_multimodal_payload(experience: Any) -> bool:
    info = getattr(experience, "info", None)
    if isinstance(info, dict) and "clawvla_has_images" in info:
        return _tensor_like_truthy(info["clawvla_has_images"])
    if _contains_multimodal_train_inputs(getattr(experience, "mm_train_inputs", None)):
        return True
    return _contains_images(getattr(experience, "images", None))


def _contains_multimodal_train_inputs(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return any(_contains_multimodal_train_inputs(item) for item in value)
    return True


def _contains_images(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return any(_contains_images(item) for item in value)
    return bool(value)


def _tensor_like_truthy(value: Any) -> bool:
    if hasattr(value, "detach"):
        tensor = value.detach()
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        if hasattr(tensor, "reshape"):
            tensor = tensor.reshape(-1)
        if hasattr(tensor, "numel") and int(tensor.numel()) == 0:
            return False
        if hasattr(tensor, "any"):
            return bool(tensor.bool().any().item())
    return bool(value)


def _patch_clawvla_training_modality_alignment() -> None:
    module = sys.modules.get("openrlhf.trainer.ray.launcher")
    if module is None:
        return

    cls = getattr(module, "RayActorGroup", None)
    if cls is None or getattr(cls, "_clawvla_training_modality_alignment_patch", False):
        return

    original_async_run_method_batch = cls.async_run_method_batch

    @functools.wraps(original_async_run_method_batch)
    def async_run_method_batch(self, method_name, **kwargs):
        if method_name == "append" and "experience" in kwargs:
            effective_actors = _effective_actor_count(self)
            aligned_kwargs, stats = _align_batched_kwargs_by_modality(kwargs, effective_actors)
            if stats is not None:
                _patch_logger().info(
                    "ClawVLA training modality alignment: "
                    f"dp={stats['dp']} text={stats['text']} multimodal={stats['multimodal']} "
                    f"local_steps={stats['local_steps']}"
                )
                kwargs = aligned_kwargs
        return original_async_run_method_batch(self, method_name, **kwargs)

    cls.async_run_method_batch = async_run_method_batch
    cls._clawvla_training_modality_alignment_patch = True


def _patch_clawvla_actor_training_shuffle() -> None:
    module = sys.modules.get("openrlhf.trainer.ray.ppo_actor")
    if module is None:
        return

    cls = getattr(module, "ActorPPOTrainer", None)
    if cls is None or getattr(cls, "_clawvla_disable_mixed_modality_shuffle_patch", False):
        return

    original_ppo_train = cls.ppo_train

    @functools.wraps(original_ppo_train)
    def ppo_train(self, kl_ctl: float):
        replay_buffer = getattr(self, "replay_buffer", None)
        if not _replay_buffer_has_mixed_modalities(replay_buffer):
            return original_ppo_train(self, kl_ctl)

        original_dataloader = module.DataLoader

        def data_loader(dataset, *args, **kwargs):
            if dataset is replay_buffer and kwargs.get("shuffle"):
                kwargs["shuffle"] = False
            return original_dataloader(dataset, *args, **kwargs)

        _patch_logger().info("ClawVLA disabled actor replay-buffer shuffle for mixed modalities.")
        module.DataLoader = data_loader
        try:
            return original_ppo_train(self, kl_ctl)
        finally:
            module.DataLoader = original_dataloader

    cls.ppo_train = ppo_train
    cls._clawvla_disable_mixed_modality_shuffle_patch = True


def _align_batched_kwargs_by_modality(
    kwargs: dict[str, Any], effective_actors: int
) -> tuple[dict[str, Any], dict[str, int] | None]:
    experiences = kwargs.get("experience")
    if not isinstance(experiences, list):
        return kwargs, None

    aligned_indices, stats = _align_experience_indices_by_modality(experiences, effective_actors)
    if stats is None:
        return kwargs, None

    aligned_kwargs: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, list) and len(value) == len(experiences):
            aligned_kwargs[key] = [value[index] for index in aligned_indices]
        elif isinstance(value, tuple) and len(value) == len(experiences):
            aligned_kwargs[key] = tuple(value[index] for index in aligned_indices)
        else:
            aligned_kwargs[key] = value
    return aligned_kwargs, stats


def _align_experience_indices_by_modality(
    experiences: list[Any], effective_actors: int
) -> tuple[list[int], dict[str, int] | None]:
    if effective_actors <= 1 or not experiences:
        return list(range(len(experiences))), None

    buckets = _split_indices_by_modality(experiences)
    if not buckets["text"] or not buckets["multimodal"]:
        return list(range(len(experiences))), None

    total = len(experiences)
    if total % effective_actors != 0:
        raise RuntimeError(
            "OpenRLHF ClawVLA mixed-modality training batch cannot be evenly split across actor ranks: "
            f"total={total} effective_actors={effective_actors}. "
            "Increase rollout samples or change task mix so every actor rank receives the same number of samples."
        )

    uneven = {
        name: len(indices)
        for name, indices in buckets.items()
        if indices and len(indices) % effective_actors != 0
    }
    if uneven:
        raise RuntimeError(
            "OpenRLHF ClawVLA mixed-modality training batch cannot align modalities across actor ranks: "
            f"effective_actors={effective_actors} counts={uneven}. "
            "Each non-empty modality bucket must be divisible by the actor data-parallel size."
        )

    rank_indices: list[list[int]] = [[] for _ in range(effective_actors)]
    for bucket_name in ("text", "multimodal"):
        indices = buckets[bucket_name]
        for start in range(0, len(indices), effective_actors):
            column = indices[start : start + effective_actors]
            for rank, index in enumerate(column):
                rank_indices[rank].append(index)

    local_steps = total // effective_actors
    if any(len(chunk) != local_steps for chunk in rank_indices):
        raise RuntimeError(
            "OpenRLHF ClawVLA modality alignment produced uneven actor chunks: "
            f"local_steps={local_steps} chunk_lengths={[len(chunk) for chunk in rank_indices]}"
        )

    for local_step in range(local_steps):
        modalities = {
            _experience_modality(experiences[rank_indices[rank][local_step]]) for rank in range(effective_actors)
        }
        if len(modalities) != 1:
            raise RuntimeError(
                "OpenRLHF ClawVLA modality alignment failed internal consistency check: "
                f"local_step={local_step} modalities={sorted(modalities)}"
            )

    aligned_indices = [index for chunk in rank_indices for index in chunk]
    if sorted(aligned_indices) != list(range(total)):
        raise RuntimeError("OpenRLHF ClawVLA modality alignment lost or duplicated training samples.")

    stats = {
        "dp": effective_actors,
        "text": len(buckets["text"]),
        "multimodal": len(buckets["multimodal"]),
        "local_steps": local_steps,
    }
    return aligned_indices, stats


def _effective_actor_count(group: Any) -> int:
    actor_handlers = getattr(group, "_actor_handlers", None)
    duplicate_actors = int(getattr(group, "duplicate_actors", 1) or 1)
    if not actor_handlers:
        return 1
    return max(1, len(actor_handlers) // duplicate_actors)


def _replay_buffer_has_mixed_modalities(replay_buffer: Any) -> bool:
    items = getattr(replay_buffer, "items", None)
    if not isinstance(items, list) or len(items) < 2:
        return False
    modalities = {_experience_modality(item) for item in items}
    return "text" in modalities and "multimodal" in modalities


def _experience_modality(experience: Any) -> str:
    return "multimodal" if _experience_has_multimodal_payload(experience) else "text"


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
