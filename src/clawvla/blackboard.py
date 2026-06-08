from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BlackboardEvent:
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)


class Blackboard:
    """Shared state store for component-to-component coordination."""

    def __init__(self, task_instruction: str | None = None):
        self.task_instruction = task_instruction
        self.values: dict[str, Any] = {}
        self.events: list[BlackboardEvent] = []

    def write(self, key: str, value: Any, event_type: str | None = None) -> None:
        self.values[key] = value
        if event_type:
            self.append_event(event_type, {"key": key})

    def read(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def append_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        self.events.append(BlackboardEvent(event_type=event_type, payload=dict(payload or {})))

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_instruction": self.task_instruction,
            "values": deepcopy(self.values),
            "events": [event.__dict__ for event in self.events],
        }

    def compact_context(self) -> dict[str, Any]:
        world_state = self.values.get("world_state")
        return {
            "task_instruction": self.task_instruction,
            "stage": self.values.get("stage"),
            "world_state": _compact_world_state(world_state),
            "task_plan": _compact_task_plan(self.values.get("task_plan")),
            "current_subgoal": _to_dict(self.values.get("current_subgoal")),
            "grounding_overlay": _compact_grounding_overlay(self.values.get("grounding_overlay")),
            "motion_state": _compact_motion_state(self.values),
            "last_scheduler_decision": _to_dict(self.values.get("last_scheduler_decision")),
            "recent_loop_history": _compact_loop_history(self.values.get("loop_history")),
            "last_skill_exception": self.values.get("last_skill_exception"),
            "last_perception_error": self.values.get("last_perception_error"),
            "last_localization_error": self.values.get("last_localization_error"),
            "last_grounding_error": self.values.get("last_grounding_error"),
            "bootstrap_observe_failures": self.values.get("bootstrap_observe_failures"),
            "last_safety_report": _to_dict(self.values.get("last_safety_report")),
            "last_verification_report": _to_dict(self.values.get("last_verification_report")),
            "last_recovery_directive": _to_dict(self.values.get("last_recovery_directive")),
            "last_retry_request": _to_dict(self.values.get("last_retry_request")),
            "event_count": len(self.events),
        }


def _to_dict(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _compact_world_state(value: Any) -> Any:
    if value is None:
        return None
    if not hasattr(value, "to_dict"):
        return value
    payload = value.to_dict()
    candidates = payload.get("candidates", [])
    if isinstance(candidates, list):
        payload["candidates"] = [
            {
                "candidate_id": item.get("candidate_id"),
                "label": item.get("label"),
                "bbox_by_view": item.get("bbox_by_view"),
                "mask_ref_by_view": item.get("mask_ref_by_view"),
                "visibility": item.get("visibility"),
                "confidence": item.get("confidence"),
                "status": item.get("status"),
                "metric_geometry": _compact_metric_geometry(item.get("metric_geometry")),
                "evidence": item.get("evidence"),
                "role_hypotheses": item.get("role_hypotheses"),
            }
            for item in candidates
            if isinstance(item, dict)
        ][:12]
    return payload


def _compact_task_plan(value: Any) -> Any:
    if value is None:
        return None
    payload = _to_dict(value)
    if not isinstance(payload, dict):
        return payload
    subgoals = payload.get("subgoals")
    if isinstance(subgoals, list):
        payload["subgoals"] = [
            {
                "subgoal_id": item.get("subgoal_id"),
                "type": item.get("type"),
                "source_candidate_id": item.get("source_candidate_id"),
                "target_candidate_id": item.get("target_candidate_id"),
                "status": item.get("status"),
                "completion_criteria": item.get("completion_criteria"),
            }
            for item in subgoals
            if isinstance(item, dict)
        ][:12]
    return payload


def _compact_grounding_overlay(value: Any) -> Any:
    if value is None:
        return None
    payload = _to_dict(value)
    if not isinstance(payload, dict):
        return payload
    return {
        "observation_id": payload.get("observation_id"),
        "image_paths": payload.get("image_paths"),
        "object_refs": payload.get("object_refs"),
        "stale": payload.get("stale"),
        "metadata": payload.get("metadata"),
    }


def _compact_motion_state(values: dict[str, Any]) -> dict[str, Any]:
    action_chunk = values.get("action_chunk")
    action_payload = _to_dict(action_chunk)
    commands = action_payload.get("commands") if isinstance(action_payload, dict) else []
    return {
        "motion_goal_present": values.get("motion_goal") is not None,
        "motion_plan_status": (values.get("motion_plan") or {}).get("status")
        if isinstance(values.get("motion_plan"), dict)
        else getattr(values.get("motion_plan"), "status", None),
        "action_chunk_present": action_chunk is not None,
        "action_chunk_type": action_payload.get("action_type") if isinstance(action_payload, dict) else None,
        "action_chunk_command_count": len(commands or []),
        "action_chunk_consumed": (action_payload.get("metadata") or {}).get("consumed")
        if isinstance(action_payload, dict) and isinstance(action_payload.get("metadata"), dict)
        else None,
        "execution_report_status": (values.get("execution_report") or {}).get("status")
        if isinstance(values.get("execution_report"), dict)
        else None,
    }


def _compact_metric_geometry(value: Any) -> Any:
    if not isinstance(value, dict):
        return None
    return {
        "available": value.get("available"),
        "source": value.get("source"),
        "position_3d": value.get("position_3d"),
        "extent_3d": value.get("extent_3d"),
        "support_gap": value.get("support_gap"),
        "pointcloud_ref": value.get("pointcloud_ref"),
        "point_count": (value.get("pointcloud_local") or {}).get("point_count")
        if isinstance(value.get("pointcloud_local"), dict)
        else None,
        "quality": value.get("quality"),
    }


def _compact_loop_history(value: Any) -> Any:
    if not isinstance(value, list):
        return []
    compact = []
    for item in value[-8:]:
        if hasattr(item, "to_dict"):
            item = item.to_dict()
        if not isinstance(item, dict):
            continue
        decision = item.get("decision") if isinstance(item.get("decision"), dict) else {}
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        output = result.get("output") if isinstance(result.get("output"), dict) else {}
        compact.append(
            {
                "step_index": item.get("step_index"),
                "stage_before": item.get("stage_before"),
                "control": decision.get("control"),
                "next_component": decision.get("next_component"),
                "next_skill": decision.get("next_skill"),
                "decision_stage": decision.get("stage"),
                "narration": decision.get("narration"),
                "state_summary": decision.get("state_summary"),
                "expected_result": decision.get("expected_result"),
                "status": item.get("status"),
                "error": item.get("error"),
                "skill_status": result.get("status"),
                "success": result.get("success"),
                "errors": result.get("errors"),
                "reason": _compact_result_reason(output),
            }
        )
    return compact


def _compact_result_reason(output: dict[str, Any]) -> Any:
    if output.get("reason"):
        return output.get("reason")
    exception = output.get("exception")
    if isinstance(exception, dict):
        return exception.get("message")
    return None
