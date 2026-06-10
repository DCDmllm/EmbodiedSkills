from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable

from .trajectory import EpisodeRecord, RewardRecord


SnapshotFn = Callable[[Any, Any], Any]
ComputeFn = Callable[[Any, Any, dict[str, Any]], Any]
FinalizeFn = Callable[[EpisodeRecord], RewardRecord]


@dataclass
class RewardHandler:
    name: str
    snapshot: SnapshotFn
    compute: ComputeFn
    finalize: FinalizeFn | None = None


class RewardRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, RewardHandler] = {}
        self._task_map: dict[str, str] = {}

    def register(self, handler: RewardHandler) -> None:
        if handler.name in self._handlers:
            raise ValueError(f"Reward handler already registered: {handler.name}")
        self._handlers[handler.name] = handler

    def map_task(self, task_name: str, handler_name: str) -> None:
        if handler_name not in self._handlers:
            raise ValueError(f"Cannot map task {task_name} to unknown reward handler: {handler_name}")
        self._task_map[task_name] = handler_name

    def handler_for_task(self, task_name: str) -> RewardHandler:
        handler_name = self._task_map.get(task_name)
        if not handler_name:
            raise KeyError(f"No reward handler configured for task: {task_name}")
        return self._handlers[handler_name]

    def configured_tasks(self) -> list[str]:
        return sorted(self._task_map)


def build_reward_registry(import_paths: list[str], task_map: dict[str, str]) -> RewardRegistry:
    registry = RewardRegistry()
    for import_path in import_paths:
        func = _load_callable(import_path)
        func(registry)
    for task_name, handler_name in task_map.items():
        registry.map_task(task_name, handler_name)
    return registry


def register_builtin_robotwin(registry: RewardRegistry) -> None:
    from clawvla.rewards.robotwin_reward import compute_robotwin_reward, snapshot_robotwin_task

    def snapshot(env: Any, blackboard: Any) -> Any:
        task_name = _task_name(env, blackboard)
        task_env = _task_env(env)
        if task_env is None:
            raise RuntimeError("RoboTwin reward snapshot requires a live task_env.")
        return snapshot_robotwin_task(task_env, task_name=task_name)

    def compute(before: Any, after: Any, context: dict[str, Any]) -> Any:
        return compute_robotwin_reward(
            before,
            after,
            task_name=context.get("task_name"),
            step_cost=float(context.get("step_cost", 0.05)),
        )

    registry.register(RewardHandler("robotwin", snapshot=snapshot, compute=compute))


def reward_record_from_result(step_index: int | None, result: Any) -> RewardRecord:
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    return RewardRecord(
        step_index=step_index,
        task_name=str(payload.get("task_name") or ""),
        reward=float(payload.get("reward", 0.0)),
        family=payload.get("family"),
        reason=str(payload.get("reason") or ""),
        events=dict(payload.get("events") or {}),
        metrics=dict(payload.get("metrics") or {}),
        milestones=dict(payload.get("milestones") or {}),
    )


def _load_callable(import_path: str) -> Callable[[RewardRegistry], None]:
    module_name, _, attr = import_path.partition(":")
    if not module_name or not attr:
        raise ValueError(f"Reward registry import must be module:callable, got: {import_path}")
    module = import_module(module_name)
    func = getattr(module, attr)
    if not callable(func):
        raise TypeError(f"Reward registry target is not callable: {import_path}")
    return func


def _task_env(env: Any) -> Any:
    if env is None:
        return None
    session = getattr(env, "session", None)
    if session is not None and getattr(session, "task_env", None) is not None:
        return session.task_env
    if getattr(env, "bound_task_env", None) is not None:
        return env.bound_task_env
    return env if hasattr(env, "check_success") else None


def _task_name(env: Any, blackboard: Any) -> str | None:
    if blackboard is not None:
        task_name = getattr(getattr(blackboard, "values", {}), "get", lambda *_: None)("reward_task_name")
        if task_name:
            return str(task_name)
    config = getattr(env, "config", None)
    if config is not None and getattr(config, "task_name", None):
        return str(config.task_name)
    task_env = _task_env(env)
    return str(getattr(task_env, "task_name", "") or task_env.__class__.__name__) if task_env is not None else None
