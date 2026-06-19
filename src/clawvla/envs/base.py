from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..schema import ActionChunk, ObservationBundle


class RobotEnvAdapter(ABC):
    @abstractmethod
    def capture_views(self, **kwargs) -> ObservationBundle:
        raise NotImplementedError

    @abstractmethod
    def execute_action(self, action_chunk: ActionChunk | None) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError

    def status(self) -> dict[str, Any]:
        return {
            "backend": self.metadata().get("backend"),
            "ready": False,
            "needs_setup": True,
            "last_observation_present": getattr(self, "last_observation", None) is not None,
            "live_env_bound": False,
        }

    def preflight_spec(self) -> dict[str, Any]:
        return {
            "backend": self.metadata().get("backend"),
            "required_cameras": [],
            "expected_resolution": None,
            "state": {"required": False, "dim": None, "source": None},
            "action": {"required": False, "types": {}},
        }

    def task_status(self) -> dict[str, Any]:
        return {
            "backend": self.metadata().get("backend"),
            "task_name": self.metadata().get("task_name"),
            "success": None,
            "done": None,
            "step_count": None,
        }
