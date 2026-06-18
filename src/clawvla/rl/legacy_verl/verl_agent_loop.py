from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..config import load_rl_config
from ..policy_proxy import PolicyBackend, PolicyGeneration, PolicyProxy
from ..rollout_worker import run_rollout_episode
from ..trajectory import TrajectoryWriter, build_policy_call_adapters

try:
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
    from verl.utils.profiler import simple_timer
except Exception:  # pragma: no cover - exercised in non-verl env preflight.
    AgentLoopBase = object  # type: ignore[assignment,misc]
    AgentLoopMetrics = None  # type: ignore[assignment]
    AgentLoopOutput = None  # type: ignore[assignment]

    def register(name: str):  # type: ignore[no-redef]
        def decorator(cls: type) -> type:
            cls._clawvla_verl_register_error = f"verl_import_unavailable_for_agent_loop:{name}"
            return cls

        return decorator


class VerlServerManagerBackend:
    def __init__(self, *, agent_loop: Any, sampling_params: dict[str, Any], event_loop: asyncio.AbstractEventLoop):
        self.agent_loop = agent_loop
        self.sampling_params = sampling_params
        self.event_loop = event_loop

    def generate(self, request: dict[str, Any], trace: Any) -> PolicyGeneration:
        future = asyncio.run_coroutine_threadsafe(self._generate_async(request), self.event_loop)
        return future.result(timeout=float(request.get("timeout") or 1800.0))

    async def _generate_async(self, request: dict[str, Any]) -> PolicyGeneration:
        messages = _openai_to_verl_messages(request.get("messages") or [])
        multi_modal_data = await self.agent_loop.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self.agent_loop._get_mm_processor_kwargs(audios)
        prompt_ids = await self.agent_loop.apply_chat_template(
            messages,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        output = await self.agent_loop.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=self.sampling_params,
            image_data=images,
            video_data=videos,
            audio_data=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        token_ids = list(output.token_ids)
        response_length = int(
            getattr(self.agent_loop.rollout_config, "response_length", len(token_ids)) or len(token_ids)
        )
        _validate_single_generation_length(len(token_ids), response_length)
        text = self.agent_loop.tokenizer.decode(token_ids, skip_special_tokens=True)
        log_probs = list(output.log_probs) if getattr(output, "log_probs", None) else []
        output_multi_modal_data = {
            key: value
            for key, value in {
                "images": images,
                "videos": videos,
                "audios": audios,
            }.items()
            if value is not None
        }
        return PolicyGeneration(
            text=text,
            prompt_ids=list(prompt_ids),
            response_ids=token_ids,
            response_logprobs=log_probs,
            multi_modal_data=output_multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            metadata={"num_preempted": getattr(output, "num_preempted", None)},
        )


class ClawVLAAgentLoop(AgentLoopBase):  # type: ignore[misc,valid-type]
    def __init__(self, *args: Any, rl_config_path: str, run_dir: str | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.rl_config_path = rl_config_path
        self.rl_config = load_rl_config(rl_config_path)
        self.run_dir = Path(run_dir or Path(self.rl_config.logging.run_root) / self.rl_config.resolved_run_id())

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> Any:
        if AgentLoopOutput is None or AgentLoopMetrics is None:
            raise RuntimeError("verl_import_unavailable")
        metrics: dict[str, Any] = {
            "generate_sequences": 0.0,
            "tool_calls": 0.0,
            "compute_score": 0.0,
            "num_preempted": -1,
        }
        writer = TrajectoryWriter(self.run_dir / "events.jsonl")
        backend: PolicyBackend = VerlServerManagerBackend(
            agent_loop=self,
            sampling_params=sampling_params,
            event_loop=asyncio.get_running_loop(),
        )
        proxy = PolicyProxy(host=self.rl_config.policy.proxy_host, port=0, backend=backend, trajectory_writer=writer)
        proxy.start()
        try:
            seed = _seed_from_kwargs(kwargs, self.rl_config.rollout.seeds)
            episode_index = int(kwargs.get("index", 0) or 0)
            with simple_timer("tool_calls", metrics):
                episode = await asyncio.to_thread(
                    run_rollout_episode,
                    self.rl_config,
                    run_dir=self.run_dir,
                    episode_index=episode_index,
                    seed=seed,
                    policy_base_url=proxy.base_url,
                )
        finally:
            proxy.stop()
        episode.policy_calls = list(proxy.calls)
        writer.write_episode(episode)
        reward_score = _episode_reward(
            episode,
            self.rl_config.reward.invalid_decision_penalty,
            self.rl_config.reward.skill_failure_penalty,
        )
        metrics["num_preempted"] = max(
            [int(call.metadata.get("num_preempted") or -1) for call in episode.policy_calls] or [-1]
        )
        outputs = _build_policy_call_outputs(
            episode=episode,
            reward_score=reward_score,
            metrics=metrics,
            prompt_length=int(self.rollout_config.prompt_length),
            response_length=int(self.rollout_config.response_length),
            agent_loop_output_cls=AgentLoopOutput,
            agent_loop_metrics_cls=AgentLoopMetrics,
            kwargs=kwargs,
        )
        return outputs


def _openai_to_verl_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            converted_content = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    converted_content.append({"type": "text", "text": str(item.get("text", ""))})
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url") if isinstance(item.get("image_url"), dict) else {}
                    converted_content.append({"type": "image", "image": str(image_url.get("url") or "")})
            converted.append({"role": str(message.get("role", "user")), "content": converted_content})
        else:
            converted.append(
                {"role": str(message.get("role", "user")), "content": str(content if content is not None else "")}
            )
    return converted


def _seed_from_kwargs(kwargs: dict[str, Any], seeds: list[int]) -> int:
    extra_info = kwargs.get("extra_info")
    if isinstance(extra_info, dict) and extra_info.get("seed") is not None:
        return int(extra_info["seed"])
    index = int(kwargs.get("index", 0) or 0)
    if seeds:
        return int(seeds[index % len(seeds)])
    return index


def _validate_adapter_lengths(
    adapter: dict[str, Any],
    *,
    prompt_length: int,
    response_length: int,
    episode_id: str,
) -> None:
    prompt_count = len(adapter["prompt_ids"])
    response_count = len(adapter["response_ids"])
    mask_count = len(adapter["response_mask"])
    logprob_count = len(adapter["response_logprobs"])
    if response_count != mask_count:
        raise ValueError(
            "agent adapter response_ids/response_mask length mismatch: "
            f"episode_id={episode_id} response_ids={response_count} response_mask={mask_count}"
        )
    if response_count != logprob_count:
        raise ValueError(
            "agent adapter response_ids/response_logprobs length mismatch: "
            f"episode_id={episode_id} response_ids={response_count} response_logprobs={logprob_count}"
        )
    if prompt_count > prompt_length:
        raise ValueError(
            "agent adapter prompt exceeds configured verl prompt length: "
            f"episode_id={episode_id} prompt_ids={prompt_count} max_prompt_length={prompt_length}"
        )
    if response_count > response_length:
        raise ValueError(
            "agent adapter response exceeds configured verl response length: "
            f"episode_id={episode_id} response_ids={response_count} max_response_length={response_length}. "
            "Increase max_response_length or reduce rollout steps; silent truncation would corrupt training data."
        )


def _build_policy_call_outputs(
    *,
    episode: Any,
    reward_score: float,
    metrics: dict[str, Any],
    prompt_length: int,
    response_length: int,
    agent_loop_output_cls: Any,
    agent_loop_metrics_cls: Any,
    kwargs: dict[str, Any],
) -> list[Any]:
    adapters = build_policy_call_adapters(episode.policy_calls, require_multimodal_payload=True)
    if len(adapters) != len(episode.policy_calls):
        raise ValueError(
            "policy-call adapter count mismatch: "
            f"episode_id={episode.episode_id} calls={len(episode.policy_calls)} adapters={len(adapters)}"
        )

    reward_extra_info = _reward_extra_info(episode)
    outputs = []
    for call_index, (call, adapter) in enumerate(zip(episode.policy_calls, adapters, strict=True)):
        _validate_adapter_lengths(
            adapter,
            prompt_length=prompt_length,
            response_length=response_length,
            episode_id=episode.episode_id,
        )
        outputs.append(
            agent_loop_output_cls(
                prompt_ids=adapter["prompt_ids"],
                response_ids=adapter["response_ids"],
                response_mask=adapter["response_mask"],
                response_logprobs=adapter["response_logprobs"],
                multi_modal_data=adapter["multi_modal_data"] or None,
                mm_processor_kwargs=adapter["mm_processor_kwargs"] or None,
                reward_score=reward_score,
                num_turns=len(episode.policy_calls) * 2,
                metrics=agent_loop_metrics_cls(**metrics),
                extra_fields={
                    "episode_id": episode.episode_id,
                    "episode_status": episode.status,
                    "task_name": episode.task_name,
                    "instruction": episode.instruction,
                    "seed": episode.seed,
                    "uid": _string_or_none(kwargs.get("uid")),
                    "session_id": _string_or_none(kwargs.get("session_id")),
                    "traj_uid": episode.episode_id,
                    "trajectory_group_id": _trajectory_group_id(episode, kwargs),
                    "call_index": call_index,
                    "call_id": call.call_id,
                    "role": call.role,
                    "policy_calls": len(episode.policy_calls),
                    "reward_extra_info": {
                        **reward_extra_info,
                        "call_index": call_index,
                        "call_id": call.call_id,
                        "role": call.role,
                    },
                },
            )
        )
    if not outputs:
        raise ValueError(f"agent episode produced no trainable policy calls: episode_id={episode.episode_id}")
    return outputs


def _reward_extra_info(episode: Any) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "episode_status": episode.status,
        "task_name": episode.task_name,
        "seed": episode.seed,
        "skill_failures": sum(1 for item in episode.skill_calls if not item.success),
        "policy_calls": len(episode.policy_calls),
    }


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _trajectory_group_id(episode: Any, kwargs: dict[str, Any]) -> str:
    uid = kwargs.get("uid")
    if uid is not None:
        return str(uid)
    return f"{episode.task_name}:{episode.instruction}:{episode.seed}"


def _validate_single_generation_length(token_count: int, response_length: int) -> None:
    if token_count > response_length:
        raise ValueError(
            "single policy generation exceeds configured verl response length: "
            f"token_ids={token_count} max_response_length={response_length}. "
            "Increase max_response_length; silent truncation would corrupt JSON actions and training data."
        )


def _episode_reward(episode: Any, invalid_decision_penalty: float, skill_failure_penalty: float) -> float:
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
