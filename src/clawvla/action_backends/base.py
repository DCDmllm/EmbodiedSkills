from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from ..schema import ActionChunk, MotionGoal, ObservationBundle, WorldState


@dataclass
class ActionBackendResult:
    success: bool
    status: str
    action_chunk: ActionChunk | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.action_chunk is not None:
            payload["action_chunk"] = self.action_chunk.to_dict()
        return payload


class ActionBackend(Protocol):
    name: str

    def build_action_chunk(
        self,
        motion_goal: MotionGoal | None,
        world_state: WorldState | None,
        observation: ObservationBundle | None,
        request: dict[str, Any],
    ) -> ActionBackendResult:
        raise NotImplementedError
