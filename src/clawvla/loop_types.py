from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


RUN_SKILL = "run_skill"
ADVANCE_STAGE = "advance_stage"
FINISH_RUN = "finish_run"
MIN_ACTION_HORIZON = 10
MAX_ACTION_HORIZON = 32


@dataclass
class LoopDecision:
    control: str = RUN_SKILL
    stage: str | None = None
    next_component: str | None = None
    next_skill: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    narration: str | None = None
    state_summary: str | None = None
    expected_result: str | None = None
    budget_steps: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LoopDecision":
        control = str(payload.get("control") or RUN_SKILL)
        return cls(
            control=control,
            stage=str(payload["stage"]) if payload.get("stage") is not None else None,
            next_component=str(payload["next_component"]) if payload.get("next_component") is not None else None,
            next_skill=str(payload["next_skill"]) if payload.get("next_skill") is not None else None,
            payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
            reason=str(payload.get("reason", "")),
            narration=str(payload["narration"]) if payload.get("narration") is not None else None,
            state_summary=str(payload["state_summary"]) if payload.get("state_summary") is not None else None,
            expected_result=str(payload["expected_result"]) if payload.get("expected_result") is not None else None,
            budget_steps=int(payload["budget_steps"]) if payload.get("budget_steps") is not None else None,
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoopStepRecord:
    step_index: int
    stage_before: str
    decision: LoopDecision
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoopRunResult:
    status: str
    final_stage: str
    steps: list[LoopStepRecord] = field(default_factory=list)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "final_stage": self.final_stage,
            "steps": [step.to_dict() for step in self.steps],
            "reason": self.reason,
        }
