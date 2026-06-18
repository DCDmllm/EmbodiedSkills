from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4


@dataclass
class PolicyCallTrace:
    call_id: str
    role: str | None
    model: str
    messages: list[dict[str, Any]]
    image_refs: list[str] = field(default_factory=list)
    raw_text: str | None = None
    parsed_json: dict[str, Any] | None = None
    prompt_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    status: str = "pending"
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        *,
        role: str | None,
        model: str,
        messages: list[dict[str, Any]],
        image_refs: list[str],
    ) -> "PolicyCallTrace":
        return cls(
            call_id=f"pcall_{uuid4().hex[:12]}",
            role=role,
            model=model,
            messages=messages,
            image_refs=image_refs,
        )

    def finish(self, *, raw_text: str | None, status: str, error: str | None = None) -> None:
        self.raw_text = raw_text
        self.status = status
        self.error = error
        self.ended_at = time.time()


@dataclass
class SkillCallTrace:
    step_index: int | None
    stage: str | None
    component: str
    skill: str
    status: str
    success: bool
    errors: list[str] = field(default_factory=list)
    output_keys: list[str] = field(default_factory=list)


@dataclass
class RewardRecord:
    step_index: int | None
    task_name: str
    reward: float
    family: str | None = None
    reason: str = ""
    events: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float | None] = field(default_factory=dict)
    milestones: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeRecord:
    episode_id: str
    task_name: str
    instruction: str
    seed: int | None = None
    status: str = "created"
    reward_score: float | None = None
    policy_calls: list[PolicyCallTrace] = field(default_factory=list)
    skill_calls: list[SkillCallTrace] = field(default_factory=list)
    rewards: list[RewardRecord] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, *, task_name: str, instruction: str, seed: int | None = None) -> "EpisodeRecord":
        return cls(episode_id=f"ep_{uuid4().hex[:12]}", task_name=task_name, instruction=instruction, seed=seed)

    def total_reward(self) -> float:
        if self.reward_score is not None:
            return float(self.reward_score)
        return float(sum(item.reward for item in self.rewards))


class TrajectoryWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_event(self, event: str, payload: dict[str, Any]) -> None:
        record = {"event": event, "time": time.time(), **_jsonable(payload)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def write_episode(self, episode: EpisodeRecord) -> None:
        self.write_event("clawvla_rl_episode", {"episode": asdict(episode)})


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def build_policy_call_adapter(
    policy_call: PolicyCallTrace,
    *,
    require_multimodal_payload: bool = True,
) -> dict[str, Any]:
    if not policy_call.prompt_ids:
        raise ValueError(
            "policy call cannot be used for training without prompt_ids: "
            f"call_id={policy_call.call_id} role={policy_call.role}"
        )
    if not policy_call.response_ids:
        raise ValueError(
            "policy call cannot be used for training without response_ids: "
            f"call_id={policy_call.call_id} role={policy_call.role} status={policy_call.status}"
        )

    response_logprobs = list(policy_call.response_logprobs)
    if not response_logprobs:
        response_logprobs = [0.0] * len(policy_call.response_ids)

    multi_modal_data = _single_multi_modal_data(
        policy_call,
        require_payload=require_multimodal_payload,
    )
    mm_processor_kwargs = _single_mm_processor_kwargs(policy_call)
    return {
        "prompt_ids": list(policy_call.prompt_ids),
        "response_ids": list(policy_call.response_ids),
        "response_mask": [1] * len(policy_call.response_ids),
        "response_logprobs": response_logprobs,
        "multi_modal_data": multi_modal_data,
        "mm_processor_kwargs": mm_processor_kwargs,
    }


def build_policy_call_adapters(
    policy_calls: list[PolicyCallTrace],
    *,
    require_multimodal_payload: bool = True,
) -> list[dict[str, Any]]:
    if not policy_calls:
        raise ValueError("Cannot build policy-call adapters without policy calls.")
    return [
        build_policy_call_adapter(call, require_multimodal_payload=require_multimodal_payload)
        for call in policy_calls
    ]


def _single_multi_modal_data(
    policy_call: PolicyCallTrace,
    *,
    require_payload: bool,
) -> dict[str, Any]:
    multi_modal_data = getattr(policy_call, "_clawvla_multi_modal_data", None)
    if policy_call.image_refs and not multi_modal_data and require_payload:
        raise ValueError(
            "policy call used image refs but did not carry training multi_modal_data: "
            f"call_id={policy_call.call_id} role={policy_call.role} image_refs={len(policy_call.image_refs)}"
        )
    if multi_modal_data is None:
        return {}
    if not isinstance(multi_modal_data, dict):
        raise ValueError(
            "policy call carried invalid multi_modal_data: "
            f"call_id={policy_call.call_id} role={policy_call.role} type={type(multi_modal_data).__name__}"
        )
    return {str(key): value for key, value in multi_modal_data.items() if value is not None}


def _single_mm_processor_kwargs(policy_call: PolicyCallTrace) -> dict[str, Any]:
    kwargs = getattr(policy_call, "_clawvla_mm_processor_kwargs", None)
    if kwargs is None:
        return {}
    if not isinstance(kwargs, dict):
        raise ValueError(
            "policy call carried invalid mm_processor_kwargs: "
            f"call_id={policy_call.call_id} role={policy_call.role} type={type(kwargs).__name__}"
        )
    return dict(kwargs)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
