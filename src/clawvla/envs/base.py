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
