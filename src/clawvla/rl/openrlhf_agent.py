from __future__ import annotations

import asyncio
from copy import deepcopy
import os
from pathlib import Path
import re
import zlib
from typing import Any

from openrlhf.utils.agent import AgentExecutorBase
from openrlhf.utils.vlm_utils import process_prompt_with_images

from clawvla.rl.config import load_rl_config
from clawvla.rl.policy_proxy import PolicyBackend, PolicyGeneration, PolicyProxy
from clawvla.rl.planner_similarity import (
    is_build_task_plan_call,
    load_planner_reference_index,
    score_predicted_plan,
)
from clawvla.rl.rollout_worker import run_rollout_episode
from clawvla.rl.trajectory import TrajectoryWriter, build_policy_call_adapter


class AgentExecutor(AgentExecutorBase):
    """OpenRLHF executor that turns one RoboTwin episode into call-level samples."""

    def __init__(self) -> None:
        config_path = os.environ.get("CLAWVLA_OPENRLHF_RL_CONFIG")
        if not config_path:
            raise RuntimeError("CLAWVLA_OPENRLHF_RL_CONFIG is required for ClawVLA OpenRLHF training.")
        self.rl_config = load_rl_config(config_path)
        run_dir = os.environ.get("CLAWVLA_OPENRLHF_RUN_DIR")
        self.run_dir = Path(run_dir or Path(self.rl_config.logging.run_root) / self.rl_config.resolved_run_id())

    async def execute(self, prompt, label, sampling_params, max_length: int, hf_tokenizer, llm_engine, images=None):
        metadata = _parse_label(label)
        episode_index = int(metadata.get("index", 0) or 0)
        seed_value = metadata.get("seed")
        seed = int(seed_value if seed_value is not None else episode_index)
        task_name = str(metadata.get("task_name") or self.rl_config.rollout.task_name)
        instruction = str(metadata.get("instruction") or self.rl_config.rollout.instruction)
        task_params = metadata.get("params") if isinstance(metadata.get("params"), dict) else {}
        writer = TrajectoryWriter(self.run_dir / "events.jsonl")
        backend = _OpenRLHFPolicyBackend(
            sampling_params=sampling_params,
            max_length=max_length,
            hf_processor=hf_tokenizer,
            llm_engine=llm_engine,
            event_loop=asyncio.get_running_loop(),
        )
        proxy = PolicyProxy(host=self.rl_config.policy.proxy_host, port=0, backend=backend, trajectory_writer=writer)
        proxy.start()
        try:
            episode = await asyncio.to_thread(
                run_rollout_episode,
                self.rl_config,
                run_dir=self.run_dir,
                episode_index=episode_index,
                seed=seed,
                policy_base_url=proxy.base_url,
                task_name=task_name,
                instruction=instruction,
                task_params=task_params,
            )
        finally:
            proxy.stop()

        episode.policy_calls = list(proxy.calls)
        writer.write_episode(episode)
        reward_score = _episode_reward(
            episode,
            invalid_decision_penalty=self.rl_config.reward.invalid_decision_penalty,
            skill_failure_penalty=self.rl_config.reward.skill_failure_penalty,
        )
        planner_scores = _planner_call_scores(
            episode,
            self.rl_config.planner_aux,
            enabled_for_sample=bool(metadata.get("planner_reference_available", False)),
        )
        return _episode_to_call_samples(
            prompt=prompt,
            label=label,
            episode=episode,
            reward_score=reward_score,
            group_uid=_stable_uid(f"{episode.task_name}:{episode.instruction}:{episode.seed}"),
            planner_scores=planner_scores,
            planner_advantage_weight=float(self.rl_config.planner_aux.advantage_weight),
            task_index=int(metadata.get("task_index", 0) or 0),
            source=str(metadata.get("source") or "configured"),
        )


class _OpenRLHFPolicyBackend(PolicyBackend):
    def __init__(
        self,
        *,
        sampling_params: Any,
        max_length: int,
        hf_processor: Any,
        llm_engine: Any,
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.sampling_params = sampling_params
        self.max_length = max_length
        self.hf_processor = hf_processor
        self.llm_engine = llm_engine
        self.event_loop = event_loop

    def generate(self, request: dict[str, Any], trace: Any) -> PolicyGeneration:
        future = asyncio.run_coroutine_threadsafe(self._generate_async(request, trace), self.event_loop)
        return future.result(timeout=float(request.get("timeout") or 1800.0))

    async def _generate_async(self, request: dict[str, Any], trace: Any) -> PolicyGeneration:
        prompt_text = _messages_to_prompt_text(self.hf_processor, request.get("messages") or [])
        prompt_ids, mm_train_inputs, pil_images = process_prompt_with_images(
            self.hf_processor,
            prompt_text,
            trace.image_refs,
        )
        max_new_tokens = _request_max_tokens(request)
        available_tokens = self.max_length - len(prompt_ids)
        if available_tokens <= 0:
            raise ValueError(
                "OpenRLHF policy prompt exceeds configured max_length: "
                f"prompt_tokens={len(prompt_ids)} max_length={self.max_length}"
            )

        params = deepcopy(self.sampling_params)
        params.max_tokens = max(1, min(max_new_tokens or available_tokens, available_tokens))
        if request.get("temperature") is not None:
            params.temperature = float(request["temperature"])
        if request.get("top_p") is not None:
            params.top_p = float(request["top_p"])

        mm_data = {"image": pil_images} if pil_images else None
        output = await self.llm_engine.generate(prompt_ids, params, multi_modal_data=mm_data)
        completion = output.outputs[0]
        response_ids = list(completion.token_ids)
        response_logprobs = _extract_logprobs(completion, response_ids)
        trace._clawvla_openrlhf_mm_train_inputs = mm_train_inputs
        return PolicyGeneration(
            text=str(completion.text),
            prompt_ids=list(prompt_ids),
            response_ids=response_ids,
            response_logprobs=response_logprobs,
            multi_modal_data={"images": pil_images} if pil_images else {},
            metadata={"openrlhf_request_id": getattr(output, "request_id", None)},
        )


def _episode_to_call_samples(
    *,
    prompt: str,
    label: str,
    episode: Any,
    reward_score: float,
    group_uid: int,
    planner_scores: dict[str, float] | None = None,
    planner_advantage_weight: float = 0.0,
    task_index: int = 0,
    source: str = "configured",
) -> list[dict[str, Any]]:
    if episode.status == "infra_failure":
        raise RuntimeError(f"Episode infra failure must not be used for policy update: {episode.errors}")
    if not episode.policy_calls:
        raise ValueError(f"OpenRLHF episode produced no policy calls: episode_id={episode.episode_id}")

    episode_uid = _stable_uid(episode.episode_id)
    policy_calls = len(episode.policy_calls)
    samples = []
    planner_scores = dict(planner_scores or {})
    episode_logs = _episode_extra_logs(
        episode,
        reward_score=reward_score,
        task_index=task_index,
        source=source,
    )
    for call_index, call in enumerate(episode.policy_calls):
        adapter = build_policy_call_adapter(call, require_multimodal_payload=True)
        prompt_ids = adapter["prompt_ids"]
        response_ids = adapter["response_ids"]
        mm_train_inputs = getattr(call, "_clawvla_openrlhf_mm_train_inputs", None)
        if call.image_refs and mm_train_inputs is None:
            raise ValueError(
                "OpenRLHF image policy call did not carry mm_train_inputs: "
                f"episode_id={episode.episode_id} call_id={call.call_id} role={call.role}"
            )
        rollout_log_probs = None
        if call.response_logprobs:
            if len(call.response_logprobs) != len(response_ids):
                raise ValueError(
                    "OpenRLHF policy call response_logprobs length mismatch: "
                    f"episode_id={episode.episode_id} call_id={call.call_id} "
                    f"response_ids={len(response_ids)} response_logprobs={len(call.response_logprobs)}"
                )
            rollout_log_probs = [0.0] * len(prompt_ids) + list(call.response_logprobs)
        samples.append(
            {
                "prompt": f"{prompt}\n[episode={episode.episode_id} call={call_index}]",
                "label": label,
                "images": list(call.image_refs) or None,
                "mm_train_inputs": mm_train_inputs,
                "reward": float(reward_score),
                "scores": float(reward_score),
                "observation_tokens": prompt_ids + response_ids,
                "action_ranges": [(len(prompt_ids), len(prompt_ids) + len(response_ids))],
                "rollout_log_probs": rollout_log_probs,
                "extra_logs": {
                    **episode_logs,
                    "clawvla_group_uid": int(group_uid),
                    "clawvla_episode_uid": int(episode_uid),
                    "clawvla_call_index": int(call_index),
                    "clawvla_policy_calls": int(policy_calls),
                    "clawvla_has_images": int(bool(call.image_refs)),
                    "clawvla_image_count": int(len(call.image_refs)),
                    "clawvla_is_build_task_plan": int(call.call_id in planner_scores),
                    "clawvla_plan_score_available": int(call.call_id in planner_scores),
                    "clawvla_plan_score": float(planner_scores.get(call.call_id, 0.0)),
                    "clawvla_plan_advantage_weight": float(planner_advantage_weight),
                },
            }
        )
    if not samples:
        raise ValueError(f"OpenRLHF episode produced no trainable policy calls: episode_id={episode.episode_id}")
    return samples


def _episode_extra_logs(
    episode: Any,
    *,
    reward_score: float,
    task_index: int,
    source: str,
) -> dict[str, float | int]:
    """Build numeric, call-repeated metadata for episode-deduplicated logging."""
    terminal = next(
        (reward for reward in reversed(episode.rewards) if reward.family == "episode_terminal"),
        None,
    )
    terminal_metrics = dict(getattr(terminal, "metrics", {}) or {})
    terminal_events = dict(getattr(terminal, "events", {}) or {})
    official_success = bool(
        terminal_events.get(
            "official_task_success",
            episode.metadata.get("official_task_success", False),
        )
    )
    family_rewards: dict[str, float] = {}
    for reward in episode.rewards:
        family = _safe_metric_component(str(reward.family or "unclassified"))
        family_rewards[family] = family_rewards.get(family, 0.0) + float(reward.reward)

    terminal_reward = family_rewards.get("episode_terminal", 0.0)
    logs: dict[str, float | int] = {
        "clawvla_task_index": int(task_index),
        "clawvla_official_task_success": int(official_success),
        "clawvla_task_status_available": int(bool(episode.metadata.get("task_status_available", False))),
        "clawvla_episode_incomplete": int(terminal_events.get("episode_incomplete", not official_success)),
        "clawvla_premature_finish": int(terminal_events.get("premature_finish", False)),
        "clawvla_stalled_loop": int(terminal_events.get("stalled_loop", episode.status == "stalled_loop")),
        "clawvla_budget_exhausted": int(
            terminal_events.get(
                "budget_exhausted",
                episode.status in {"max_steps_reached", "max_steps_reached_with_failures"},
            )
        ),
        "clawvla_invalid_decisions": float(terminal_metrics.get("invalid_decisions", 0.0) or 0.0),
        "clawvla_failed_skills": float(terminal_metrics.get("failed_skills", 0.0) or 0.0),
        "clawvla_recoverable_preflight_failures": float(
            terminal_metrics.get("recoverable_preflight_failures", 0.0) or 0.0
        ),
        "clawvla_skill_calls": float(terminal_metrics.get("skill_calls", len(episode.skill_calls)) or 0.0),
        "clawvla_episode_reward": float(reward_score),
        "clawvla_dense_reward": float(reward_score) - terminal_reward,
        "clawvla_terminal_reward": terminal_reward,
        "clawvla_source_expert_subgoals": int(source == "expert_subgoals"),
        "clawvla_source_official_valid_grounding": int(source == "official_valid_grounding"),
    }
    logs.update({f"clawvla_reward_family__{family}": value for family, value in family_rewards.items()})
    return logs


def _safe_metric_component(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip()).strip("_").lower()
    return normalized or "unknown"


def _planner_call_scores(
    episode: Any,
    planner_aux: Any,
    *,
    enabled_for_sample: bool,
) -> dict[str, float]:
    if not enabled_for_sample or not bool(getattr(planner_aux, "enabled", False)):
        return {}
    references = load_planner_reference_index(
        str(planner_aux.dataset_root),
        str(planner_aux.repair_ledger),
        str(planner_aux.split_manifest),
        str(planner_aux.split_name),
        int(planner_aux.max_reference_plans_per_task),
    ).get(str(episode.task_name), ())
    references = tuple(reference for reference in references if int(reference.seed) == int(episode.seed))
    if not references:
        return {}
    for call in episode.policy_calls:
        if is_build_task_plan_call(call):
            score = score_predicted_plan(getattr(call, "parsed_json", None), references)
            call.metadata["planner_reference_available"] = True
            call.metadata["planner_semantic_score"] = float(score)
            return {str(call.call_id): float(score)}
    return {}


def _messages_to_prompt_text(processor: Any, messages: list[dict[str, Any]]) -> str:
    converted = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"OpenRLHF policy message must be a dict: index={message_index} type={type(message).__name__}")
        role = str(message.get("role", "user"))
        content = message.get("content")
        if isinstance(content, list):
            converted_content = []
            for item_index, item in enumerate(content):
                if not isinstance(item, dict):
                    raise TypeError(
                        "OpenRLHF policy message content item must be a dict: "
                        f"message_index={message_index} item_index={item_index} type={type(item).__name__}"
                    )
                if item.get("type") == "text":
                    converted_content.append({"type": "text", "text": str(item.get("text", ""))})
                elif item.get("type") == "image_url":
                    converted_content.append({"type": "image"})
                else:
                    raise ValueError(
                        "Unsupported OpenRLHF policy message content item type: "
                        f"message_index={message_index} item_index={item_index} type={item.get('type')!r}"
                    )
            converted.append({"role": role, "content": converted_content})
        else:
            converted.append({"role": role, "content": str(content if content is not None else "")})

    if hasattr(processor, "apply_chat_template"):
        return str(processor.apply_chat_template(converted, tokenize=False, add_generation_prompt=True))
    return "\n".join(str(item.get("content", "")) for item in converted)


def _extract_logprobs(completion: Any, response_ids: list[int]) -> list[float]:
    logprobs = getattr(completion, "logprobs", None)
    if not logprobs:
        return []
    if len(logprobs) != len(response_ids):
        raise ValueError(
            "vLLM returned response logprobs with unexpected length: "
            f"response_ids={len(response_ids)} logprobs={len(logprobs)}"
        )
    values = []
    for index, token_id in enumerate(response_ids):
        item = logprobs[index]
        if isinstance(item, dict) and token_id in item:
            values.append(float(item[token_id].logprob))
        else:
            raise ValueError(f"vLLM logprobs missing sampled token: index={index} token_id={token_id}")
    return values


def _request_max_tokens(request: dict[str, Any]) -> int | None:
    value = request.get("max_tokens", request.get("max_completion_tokens"))
    if value is None:
        return None
    return int(value)


def _parse_label(label: Any) -> dict[str, Any]:
    if isinstance(label, dict):
        return label
    if not label:
        raise ValueError("OpenRLHF ClawVLA label is required.")
    import json

    try:
        payload = json.loads(str(label))
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenRLHF ClawVLA label must be valid JSON: {label!r}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"OpenRLHF ClawVLA label must decode to a JSON object, got {type(payload).__name__}")
    return payload


def _episode_reward(episode: Any, *, invalid_decision_penalty: float, skill_failure_penalty: float) -> float:
    if episode.status == "infra_failure":
        raise RuntimeError(f"Episode infra failure must not be used for policy update: {episode.errors}")
    if episode.reward_score is not None:
        return float(episode.reward_score)
    reward = 0.0
    for skill in episode.skill_calls:
        if skill.status == "invalid_decision":
            reward += invalid_decision_penalty
        elif not skill.success:
            reward += skill_failure_penalty
    return reward


def _stable_uid(value: str) -> int:
    return int(zlib.crc32(value.encode("utf-8")) & 0x7FFFFFFF)
