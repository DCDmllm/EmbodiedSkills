from __future__ import annotations

from typing import Any

from .robotwin import RoboTwinAdapter


class RmbenchAdapter(RoboTwinAdapter):
    """RMBench adapter using its RoboTwin-compatible runtime contract."""

    @staticmethod
    def _with_backend(payload: dict[str, Any]) -> dict[str, Any]:
        return {**payload, "backend": "rmbench"}

    def _capture_metadata(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return self._with_backend(super()._capture_metadata(kwargs))

    def execute_action(self, action_chunk):
        return self._with_backend(super().execute_action(action_chunk))

    def metadata(self) -> dict[str, Any]:
        return self._with_backend(super().metadata())

    def status(self) -> dict[str, Any]:
        return self._with_backend(super().status())

    def preflight_spec(self) -> dict[str, Any]:
        return self._with_backend(super().preflight_spec())

    def task_status(self) -> dict[str, Any]:
        return self._with_backend(super().task_status())
