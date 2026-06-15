from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


MAX_CONTEXT_TEXT_CHARS = 120
MAX_CONTEXT_LIST_ITEMS = 8
MAX_CONTEXT_LOOP_HISTORY = 20


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
        stage = self.values.get("stage")
        active_verification_report = (
            self.values.get("last_verification_report")
            if stage in {None, "verify", "recover"}
            else None
        )
        return {
            "task_instruction": self.task_instruction,
            "stage": stage,
            "world_state": _compact_world_state(world_state),
            "task_plan": _compact_task_plan(self.values.get("task_plan")),
            "current_subgoal": _to_dict(self.values.get("current_subgoal")),
            "grounding_overlay": _compact_grounding_overlay(self.values.get("grounding_overlay")),
            "motion_state": _compact_motion_state(self.values),
            "last_scheduler_decision": _to_dict(self.values.get("last_scheduler_decision")),
            "recent_loop_history": _compact_loop_history(self.values.get("loop_history")),
            "last_skill_exception": _compact_exception(self.values.get("last_skill_exception")),
            "last_perception_error": _compact_error_payload(self.values.get("last_perception_error")),
            "last_localization_error": _compact_error_payload(self.values.get("last_localization_error")),
            "bootstrap_observe_failures": self.values.get("bootstrap_observe_failures"),
            "last_safety_report": _compact_safety_report(self.values.get("last_safety_report")),
            "preflight_report": _compact_safety_report(self.values.get("preflight_report")),
            "last_action_validation_report": _compact_action_validation_report(
                self.values.get("last_action_validation_report")
            ),
            "last_verification_report": _compact_verification_report(active_verification_report),
            "inactive_verification_report_present": (
                active_verification_report is None and self.values.get("last_verification_report") is not None
            ),
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
                "visibility": item.get("visibility"),
                "confidence": item.get("confidence"),
                "status": item.get("status"),
                "metric_geometry": _compact_metric_geometry(item.get("metric_geometry")),
                "evidence_keys": sorted(item.get("evidence", {}).keys()) if isinstance(item.get("evidence"), dict) else [],
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
                "instruction": item.get("instruction"),
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
    for item in value[-MAX_CONTEXT_LOOP_HISTORY:]:
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
                "narration": _short_text(decision.get("narration")),
                "state_summary": _short_text(decision.get("state_summary")),
                "expected_result": _short_text(decision.get("expected_result")),
                "status": item.get("status"),
                "error": _short_text(item.get("error")),
                "skill_status": result.get("status"),
                "success": result.get("success"),
                "errors": _compact_error_list(result.get("errors")),
                "reason": _compact_result_reason(output),
                "result_summary": _compact_result_summary(output),
            }
        )
    return compact


def _compact_result_summary(output: dict[str, Any]) -> Any:
    for key in ("perception", "world_state"):
        payload = _to_dict(output.get(key))
        if isinstance(payload, dict):
            candidates = payload.get("candidates")
            return {
                "object": key,
                "observation_id": payload.get("observation_id")
                or (payload.get("metadata") or {}).get("observation_id")
                if isinstance(payload.get("metadata"), dict)
                else payload.get("observation_id"),
                "source_candidate_id": payload.get("source_candidate_id"),
                "target_candidate_id": payload.get("target_candidate_id"),
                "needs_reobserve": payload.get("needs_reobserve"),
                "candidate_count": len(candidates) if isinstance(candidates, list) else None,
            }
    geometry = output.get("geometry_summary")
    if isinstance(geometry, dict):
        return {
            "object": "geometry_summary",
            "lifted_candidates": geometry.get("lifted_candidates"),
            "candidate_count": geometry.get("candidate_count"),
            "reason": _short_text(geometry.get("reason")),
        }
    return None


def _compact_result_reason(output: dict[str, Any]) -> Any:
    if output.get("reason"):
        return _short_text(output.get("reason"))
    exception = output.get("exception")
    if isinstance(exception, dict):
        return _short_text(exception.get("message"))
    return None


def _compact_error_payload(value: Any) -> Any:
    payload = _to_dict(value)
    if not isinstance(payload, dict):
        return _short_text(payload)
    compact = {}
    for key, item in payload.items():
        if key in {"raw_preview", "existing_perception", "candidate_summaries"}:
            compact[key] = _short_text(item)
        elif key in {"errors", "reasons"}:
            compact[key] = _compact_error_list(item)
        else:
            compact[key] = _short_text(item) if isinstance(item, str) else item
    return compact


def _compact_exception(value: Any) -> Any:
    payload = _to_dict(value)
    if not isinstance(payload, dict):
        return _short_text(payload)
    return {
        "exception_type": payload.get("exception_type"),
        "message": _short_text(payload.get("message")),
        "traceback_available": bool(payload.get("traceback")),
    }


def _compact_safety_report(value: Any) -> Any:
    payload = _to_dict(value)
    if not isinstance(payload, dict):
        return payload
    checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
    return {
        "allowed": payload.get("allowed"),
        "status": payload.get("status"),
        "errors": _compact_error_list(payload.get("errors")),
        "checks": {
            "task_state": _compact_named_check(checks.get("task_state")),
            "observation_freshness": _compact_named_check(checks.get("observation_freshness")),
            "object_binding": _compact_object_binding_check(checks.get("object_binding")),
            "camera_inputs": _compact_camera_check(checks.get("camera_inputs")),
            "robot_state": _compact_named_check(checks.get("robot_state")),
            "robotwin_env": _compact_named_check(checks.get("robotwin_env")),
            "action_backend": _compact_action_backend_check(checks.get("action_backend")),
        },
        "metadata": {
            "next_stage": (payload.get("metadata") or {}).get("next_stage")
            if isinstance(payload.get("metadata"), dict)
            else None,
            "observation_id": (payload.get("metadata") or {}).get("observation_id")
            if isinstance(payload.get("metadata"), dict)
            else None,
        },
    }


def _compact_action_validation_report(value: Any) -> Any:
    payload = _to_dict(value)
    if not isinstance(payload, dict):
        return payload
    return {
        "allowed": payload.get("allowed"),
        "blocking_errors": _compact_error_list(payload.get("blocking_errors")),
        "observation_id": payload.get("observation_id"),
        "subgoal_id": payload.get("subgoal_id"),
        "checks": _compact_named_check(payload.get("checks")),
    }


def _compact_verification_report(value: Any) -> Any:
    payload = _to_dict(value)
    if not isinstance(payload, dict):
        return payload
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return {
        "success": payload.get("success"),
        "partial_progress": payload.get("partial_progress"),
        "failure_type": payload.get("failure_type"),
        "progress_score": payload.get("progress_score"),
        "should_reobserve": payload.get("should_reobserve"),
        "notes": [_short_text(item) for item in (payload.get("notes") or [])[:MAX_CONTEXT_LIST_ITEMS]],
        "metadata": {
            "source": metadata.get("source"),
            "subgoal_success": metadata.get("subgoal_success"),
            "task_success": metadata.get("task_success"),
            "next_action": metadata.get("next_action"),
            "current_subgoal_id": metadata.get("current_subgoal_id"),
            "execution_report": _compact_execution_report(metadata.get("execution_report")),
        },
    }


def _compact_execution_report(value: Any) -> Any:
    payload = _to_dict(value)
    if not isinstance(payload, dict):
        return payload
    observation = payload.get("observation") if isinstance(payload.get("observation"), dict) else {}
    action_chunk = payload.get("action_chunk") if isinstance(payload.get("action_chunk"), dict) else {}
    action_metadata = action_chunk.get("metadata") if isinstance(action_chunk.get("metadata"), dict) else {}
    commands = action_chunk.get("commands") if isinstance(action_chunk.get("commands"), list) else []
    return {
        "backend": payload.get("backend"),
        "status": payload.get("status"),
        "success": payload.get("success"),
        "executed_steps": payload.get("executed_steps"),
        "task_env_bound": payload.get("task_env_bound"),
        "observation_id": observation.get("observation_id"),
        "action_chunk": {
            "action_type": action_chunk.get("action_type"),
            "command_count": len(commands),
            "control_horizon": action_chunk.get("control_horizon"),
            "metadata": {
                "subgoal_id": action_metadata.get("subgoal_id"),
                "observation_id": action_metadata.get("observation_id"),
                "consumed": action_metadata.get("consumed"),
                "stale": action_metadata.get("stale"),
            },
        },
    }


def _compact_named_check(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    compact = {"status": value.get("status")}
    for key in ("missing", "error", "reason", "vector_length", "observation_id", "task_plan_current_subgoal_id", "current_subgoal_id"):
        if key in value:
            compact[key] = _short_text(value.get(key)) if isinstance(value.get(key), str) else value.get(key)
    return compact


def _compact_object_binding_check(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    return {
        "status": value.get("status"),
        "subgoal_type": value.get("subgoal_type"),
        "source_candidate_id": value.get("source_candidate_id"),
        "target_candidate_id": value.get("target_candidate_id"),
        "target_required": value.get("target_required"),
        "source": value.get("source"),
        "target": value.get("target"),
    }


def _compact_camera_check(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    cameras = value.get("cameras") if isinstance(value.get("cameras"), dict) else {}
    return {
        "status": value.get("status"),
        "required": value.get("required"),
        "openpi_required": value.get("openpi_required"),
        "expected_resolution": value.get("expected_resolution"),
        "cameras": {
            name: {
                "ok": item.get("ok"),
                "reason": item.get("reason"),
                "resolution": item.get("resolution"),
            }
            for name, item in cameras.items()
            if isinstance(item, dict)
        },
    }


def _compact_action_backend_check(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    worker = value.get("worker") if isinstance(value.get("worker"), dict) else {}
    return {
        "status": value.get("status"),
        "backend_present": value.get("backend_present"),
        "enabled": value.get("enabled"),
        "pretrained_path_exists": value.get("pretrained_path_exists"),
        "worker": {
            "ok": worker.get("ok"),
            "mode": worker.get("mode"),
            "host": worker.get("host"),
            "port": worker.get("port"),
            "reason": worker.get("reason"),
        },
    }


def _compact_error_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return [_short_text(value)]
    return [_short_text(item) for item in value[:MAX_CONTEXT_LIST_ITEMS]]


def _short_text(value: Any, limit: int = MAX_CONTEXT_TEXT_CHARS) -> Any:
    if value is None or not isinstance(value, str):
        return value
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...[truncated {len(value) - limit} chars]"
