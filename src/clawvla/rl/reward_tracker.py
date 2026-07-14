from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

from clawvla.notices import emit_runtime_event, emit_status_notice
from clawvla.rl.reward_registry import build_reward_registry


class RuntimeRewardTracker:
    """Optional runtime hook installed by RL rollout workers.

    This hook is deliberately explicit: snapshot/compute failures are emitted as
    events and written to JSONL; they are not converted into successful rewards.
    """

    def __init__(
        self,
        *,
        task_name: str,
        output_path: str | Path,
        step_cost: float = 0.05,
        registry_imports: list[str] | None = None,
        task_map: dict[str, str] | None = None,
    ):
        self.task_name = task_name
        self.output_path = Path(output_path)
        self.step_cost = step_cost
        imports = registry_imports or ["clawvla.rl.reward_registry:register_builtin_robotwin"]
        mapping = dict(task_map or {task_name: "robotwin"})
        self.registry = build_reward_registry(imports, mapping)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.touch(exist_ok=True)
        self._before: dict[int, Any] = {}
        self._milestones: dict[str, bool] = {}
        self.last_reward_payload: dict[str, Any] | None = None
        self.last_progress: bool | None = None

    def before_skill(self, *, blackboard: Any, component: str, skill: str, step_index: int | None) -> None:
        if component != "motion" or skill != "execute_action":
            return
        if step_index is None:
            self._write(
                "reward_snapshot_skipped",
                {"reason": "missing_loop_step", "component": component, "skill": skill},
            )
            return
        try:
            snapshot = self._snapshot(blackboard)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            self._write("reward_snapshot_failed", {"step_index": step_index, "phase": "before", "reason": reason})
            emit_status_notice(
                "reward_snapshot_failed",
                success=False,
                source="rl.reward_tracker",
                reason=reason,
                always=True,
            )
            return
        self._before[int(step_index)] = snapshot
        self._write("reward_snapshot", {"step_index": step_index, "phase": "before", "snapshot": _jsonable(snapshot)})

    def after_skill(self, *, blackboard: Any, component: str, skill: str, step_index: int | None, result: Any) -> None:
        if component != "motion" or skill != "execute_action":
            return
        if step_index is None:
            self._write(
                "reward_compute_skipped",
                {"reason": "missing_loop_step", "component": component, "skill": skill},
            )
            return
        before = self._before.pop(int(step_index), None)
        if before is None:
            self._write("reward_compute_skipped", {"step_index": step_index, "reason": "missing_before_snapshot"})
            return
        try:
            after = self._snapshot(blackboard)
            handler = self.registry.handler_for_task(self.task_name)
            reward = handler.compute(
                before,
                after,
                {"task_name": self.task_name, "step_cost": self.step_cost, "blackboard": blackboard},
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            self._write("reward_compute_failed", {"step_index": step_index, "reason": reason})
            emit_status_notice(
                "reward_compute_failed",
                success=False,
                source="rl.reward_tracker",
                reason=reason,
                always=True,
            )
            return
        reward_payload = reward.to_dict() if hasattr(reward, "to_dict") else dict(reward)
        previous_milestones = dict(self._milestones)
        self._milestones.update(dict(reward_payload.get("milestones") or {}))
        new_milestone = any(
            bool(value) and not bool(previous_milestones.get(key))
            for key, value in dict(reward_payload.get("milestones") or {}).items()
        )
        reward_value = float(reward_payload.get("reward", -self.step_cost))
        self.last_progress = bool(new_milestone or reward_value > -float(self.step_cost) + 1e-6)
        self.last_reward_payload = dict(reward_payload)
        blackboard.write(
            "rl_last_reward_progress",
            {
                "step_index": step_index,
                "progress": self.last_progress,
                "new_milestone": new_milestone,
                "reward": reward_value,
            },
            event_type="rl.reward_progress",
        )
        payload = {
            "step_index": step_index,
            "reward": reward_payload,
            "skill_status": getattr(result, "status", None),
        }
        self._write("reward_record", payload)
        emit_runtime_event("clawvla_rl_reward_record", payload)

    def _snapshot(self, blackboard: Any) -> Any:
        env = blackboard.read("env_adapter")
        handler = self.registry.handler_for_task(self.task_name)
        snapshot = handler.snapshot(env, blackboard)
        if hasattr(snapshot, "metadata") and isinstance(snapshot.metadata, dict):
            snapshot.metadata["reward_milestones"] = dict(self._milestones)
        return snapshot

    def _write(self, event: str, payload: dict[str, Any]) -> None:
        record = {"event": f"clawvla_rl_{event}", "time": time.time(), **_jsonable(payload)}
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


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
