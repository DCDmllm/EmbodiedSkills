from __future__ import annotations

import math
from typing import Any

from ..blackboard import Blackboard
from ..blackboard_utils import (
    current_observation_id,
    mark_action_chunk_consumed,
    mark_grounding_overlay_stale,
    mark_motion_artifacts_stale,
    metadata_value,
)
from ..notices import emit_human_trace
from ..schema import ActionChunk, MotionGoal, SkillRequest, SkillResult
from ..skills.base import SkillContext, SkillRegistry
from .skill_helpers import get_attr, ok, register_skill, to_dict, unavailable


def register_motion_skills(registry: SkillRegistry) -> None:
    register_skill(registry, "motion", "build_motion_goal", "Build a motion goal from scheduler/subgoal payload.", build_motion_goal)
    register_skill(registry, "motion", "plan_motion", "Plan or select a bounded motion from the motion goal.", plan_motion)
    register_skill(registry, "motion", "emit_action_chunk", "Convert a motion plan to an action chunk.", emit_action_chunk)
    register_skill(registry, "motion", "validate_action_chunk", "Validate an action chunk before execution.", validate_action_chunk)
    register_skill(registry, "motion", "execute_action", "Execute or hand off an action chunk.", execute_action)


def build_motion_goal(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    world_state = blackboard.read("world_state")
    payload = request.payload
    current_subgoal = blackboard.read("current_subgoal")
    source_candidate_id = (
        payload.get("source_candidate_id")
        or get_attr(current_subgoal, "source_candidate_id")
        or get_attr(world_state, "source_candidate_id")
    )
    target_candidate_id = (
        payload.get("target_candidate_id")
        or get_attr(current_subgoal, "target_candidate_id")
        or (None if current_subgoal is not None else get_attr(world_state, "target_candidate_id"))
    )
    source = world_state.candidate_by_id(source_candidate_id) if world_state is not None else None
    target = world_state.candidate_by_id(target_candidate_id) if world_state is not None else None
    target_handle = _target_handle(source, target)
    if target_handle.get("target_type") == "missing_visual_target":
        blackboard.write(
            "last_motion_error",
            {
                "reason": "missing_source_and_target_candidates",
                "source_candidate_id": source_candidate_id,
                "target_candidate_id": target_candidate_id,
                "current_subgoal": to_dict(current_subgoal),
            },
            event_type="motion.build_motion_goal_unavailable",
        )
        return unavailable(
            "motion_goal_unavailable",
            "missing_source_and_target_candidates",
            {
                "target_handle": target_handle,
                "source_candidate_id": source_candidate_id,
                "target_candidate_id": target_candidate_id,
                "current_subgoal": to_dict(current_subgoal),
            },
        )
    goal = MotionGoal(
        skill=str(payload.get("skill") or get_attr(current_subgoal, "type") or "approach"),
        source_candidate_id=source_candidate_id,
        target_candidate_id=target_candidate_id,
        acting_arm=payload.get("acting_arm"),
        motion_hint=payload.get("motion_hint") or get_attr(current_subgoal, "type") or target_handle["target_type"],
        target_pose=target_handle.get("target_pose"),
        constraints=dict(payload.get("constraints", {})) if isinstance(payload.get("constraints"), dict) else {},
        metadata={
            "source": "motion.build_motion_goal",
            "target_handle": target_handle,
            "subgoal": to_dict(current_subgoal),
            "subgoal_id": get_attr(current_subgoal, "subgoal_id"),
            "observation_id": current_observation_id(blackboard),
            "stale": False,
        },
    )
    mark_motion_artifacts_stale(blackboard, "new_motion_goal_built")
    blackboard.write("motion_goal", goal, event_type="motion.build_motion_goal")
    return ok("motion_goal_built", {"motion_goal": goal.to_dict()})


def plan_motion(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    motion_goal = blackboard.read("motion_goal")
    current_subgoal = blackboard.read("current_subgoal")
    target_handle = get_attr(motion_goal, "metadata", {}).get("target_handle", {}) if motion_goal is not None else {}
    target_type = target_handle.get("target_type") if isinstance(target_handle, dict) else None
    if target_type == "image_grounded":
        plan = _image_grounded_plan(blackboard, motion_goal, target_handle, request.payload)
        mark_motion_artifacts_stale(blackboard, "new_motion_plan_built")
        blackboard.write("motion_plan", plan, event_type="motion.plan_motion")
        if plan.get("status") == "motion_plan_unavailable":
            reason = str(plan.get("reason", "motion_plan_unavailable"))
            return unavailable("motion_plan_unavailable", reason, {"motion_plan": plan})
        return ok("image_grounded_motion_plan_built", {"motion_plan": plan})
    plan = {
        "status": "motion_plan_unavailable",
        "reason": _motion_unavailable_reason(target_type),
        "retryable": False,
        "motion_goal": to_dict(motion_goal),
        "required_backend": "controller_or_vla_action_model",
        "available_alternatives": ["wire_visual_servo", "wire_vla_action_model", "wire_metric_motion_controller"],
        "metadata": {
            "source": "motion.plan_motion",
            "subgoal_id": get_attr(current_subgoal, "subgoal_id"),
            "observation_id": current_observation_id(blackboard),
            "stale": False,
        },
    }
    mark_motion_artifacts_stale(blackboard, "new_motion_plan_unavailable")
    blackboard.write("motion_plan", plan, event_type="motion.plan_motion")
    return ok("motion_plan_unavailable", {"motion_plan": plan})


def emit_action_chunk(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    backend = blackboard.read("action_backend")
    motion_plan = blackboard.read("motion_plan")
    motion_goal = blackboard.read("motion_goal")
    current_subgoal = blackboard.read("current_subgoal")
    artifact_metadata = {
        "subgoal_id": get_attr(current_subgoal, "subgoal_id"),
        "observation_id": current_observation_id(blackboard),
        "artifact_prefix": _execution_artifact_prefix(blackboard, request.payload),
        "consumed": False,
        "stale": False,
    }
    motion_plan_status = get_attr(motion_plan, "status")
    if motion_plan_status == "motion_plan_unavailable":
        chunk = _unavailable_chunk("motion_plan_unavailable", {"motion_plan": to_dict(motion_plan)})
        _stamp_chunk_metadata(chunk, artifact_metadata)
        blackboard.write("action_chunk", chunk, event_type="motion.emit_action_chunk")
        output = {"action_chunk": chunk.to_dict(), "motion_plan": to_dict(motion_plan)}
        return unavailable("action_chunk_unavailable", "motion_plan_unavailable", output)
    if backend is not None and hasattr(backend, "build_action_chunk"):
        backend_name = str(getattr(backend, "name", "action_backend"))
        emit_human_trace(
            backend_name,
            "request action chunk",
            detail=f"subgoal={artifact_metadata.get('subgoal_id')} observation={artifact_metadata.get('observation_id')}",
        )
        backend_result = backend.build_action_chunk(
            motion_goal,
            blackboard.read("world_state"),
            blackboard.read("observation"),
            _backend_request(motion_plan, request.payload),
        )
        chunk = backend_result.action_chunk or _unavailable_chunk(backend_result.status, backend_result.to_dict())
        _stamp_chunk_metadata(chunk, artifact_metadata)
        emit_human_trace(
            backend_name if backend_result.success else "failure",
            f"action backend -> {backend_result.status}",
            detail=f"commands={len(getattr(chunk, 'commands', []) or [])}",
        )
        blackboard.write("action_backend_result", backend_result, event_type="motion.action_backend")
        blackboard.write("action_chunk", chunk, event_type="motion.emit_action_chunk")
        output = {"action_chunk": chunk.to_dict(), "backend_result": backend_result.to_dict()}
        if not backend_result.success:
            reason = backend_result.errors[0] if backend_result.errors else backend_result.status
            return unavailable(backend_result.status, reason, output)
        if chunk.action_type == "unavailable" or not chunk.commands:
            reason = str(chunk.metadata.get("reason") if isinstance(chunk.metadata, dict) else "empty_action_chunk")
            return unavailable("action_chunk_unavailable", reason, output)
        return ok(backend_result.status, output)
    chunk = _unavailable_chunk("action_backend_unavailable", {"reason": "missing_action_backend", "retryable": False})
    _stamp_chunk_metadata(chunk, artifact_metadata)
    blackboard.write("action_chunk", chunk, event_type="motion.emit_action_chunk")
    return unavailable("action_chunk_unavailable", "missing_action_backend", {"action_chunk": chunk.to_dict()})


def validate_action_chunk(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    report = _validate_action_chunk_report(blackboard)
    blackboard.write("last_action_validation_report", report, event_type="motion.validate_action_chunk")
    output = {"action_validation_report": report}
    if report["allowed"]:
        return ok("action_chunk_validated", output)
    return unavailable("action_chunk_validation_failed", str(report["blocking_errors"][0]), output)


def execute_action(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    env = blackboard.read("env_adapter")
    action_chunk = blackboard.read("action_chunk")
    validation = _validate_action_chunk_report(blackboard)
    blackboard.write("last_action_validation_report", validation, event_type="motion.execute_action_preflight")
    if not validation["allowed"]:
        return unavailable("action_chunk_validation_failed", str(validation["blocking_errors"][0]), {"action_validation_report": validation})
    if env is not None and hasattr(env, "execute_action"):
        report = env.execute_action(action_chunk)
    else:
        report = {
            "status": "execution_unavailable",
            "reason": "no_env_execute_action_adapter",
            "retryable": False,
        }
    blackboard.write("execution_report", report, event_type="motion.execute_action")
    if env is not None and getattr(env, "last_observation", None) is not None:
        mark_grounding_overlay_stale(blackboard, "post_action_observation")
        blackboard.write("observation", env.last_observation, event_type="motion.execute_action_observation")
    status = report.get("status") if isinstance(report, dict) else None
    output = {"execution_report": to_dict(report)}
    if status != "action_executed":
        reason = str(report.get("reason") if isinstance(report, dict) and report.get("reason") else status or "execution_unavailable")
        return unavailable(str(status or "execution_unavailable"), reason, output)
    emit_human_trace(
        "execute",
        "action chunk executed",
        detail=f"commands={len(getattr(action_chunk, 'commands', []) or [])}",
    )
    mark_action_chunk_consumed(blackboard, "action_executed")
    mark_motion_artifacts_stale(blackboard, "action_executed", include_goal=True)
    return ok(str(status), output)


def _validate_action_chunk_report(blackboard: Blackboard) -> dict[str, Any]:
    chunk = blackboard.read("action_chunk")
    current_subgoal = blackboard.read("current_subgoal")
    obs_id = current_observation_id(blackboard)
    errors: list[str] = []
    checks: dict[str, Any] = {}

    if chunk is None:
        errors.append("missing_action_chunk")
        return _action_validation_report(False, errors, checks, obs_id, current_subgoal)
    if getattr(chunk, "action_type", None) in {None, "unavailable", "noop"}:
        errors.append(f"invalid_action_type:{getattr(chunk, 'action_type', None)}")
    if metadata_value(chunk, "stale", False):
        errors.append("stale_action_chunk")
    if metadata_value(chunk, "consumed", False):
        errors.append("consumed_action_chunk")

    expected_subgoal_id = get_attr(current_subgoal, "subgoal_id")
    chunk_subgoal_id = metadata_value(chunk, "subgoal_id")
    if expected_subgoal_id != chunk_subgoal_id:
        errors.append(f"action_chunk_subgoal_mismatch:{chunk_subgoal_id}->{expected_subgoal_id}")
    chunk_obs_id = metadata_value(chunk, "observation_id")
    if obs_id != chunk_obs_id:
        errors.append(f"action_chunk_observation_mismatch:{chunk_obs_id}->{obs_id}")

    commands = getattr(chunk, "commands", None)
    if not isinstance(commands, list) or not commands:
        errors.append("empty_action_commands")
        commands = []
    expected_dim = _expected_action_dim(getattr(chunk, "action_type", None), blackboard.read("action_backend"))
    if expected_dim is None:
        errors.append(f"unsupported_action_type:{getattr(chunk, 'action_type', None)}")

    bad_command_indexes: list[int] = []
    for index, command in enumerate(commands):
        if not isinstance(command, list):
            bad_command_indexes.append(index)
            continue
        if expected_dim is not None and len(command) != expected_dim:
            bad_command_indexes.append(index)
            continue
        if not all(_finite(item) for item in command):
            bad_command_indexes.append(index)
    if bad_command_indexes:
        errors.append(f"invalid_action_command_indexes:{bad_command_indexes[:5]}")

    checks["action_chunk"] = {
        "action_type": getattr(chunk, "action_type", None),
        "command_count": len(commands),
        "expected_command_dim": expected_dim,
        "subgoal_id": chunk_subgoal_id,
        "expected_subgoal_id": expected_subgoal_id,
        "observation_id": chunk_obs_id,
        "expected_observation_id": obs_id,
        "bad_command_indexes": bad_command_indexes[:20],
    }
    return _action_validation_report(not errors, errors, checks, obs_id, current_subgoal)


def _action_validation_report(
    allowed: bool,
    errors: list[str],
    checks: dict[str, Any],
    obs_id: str | None,
    current_subgoal: object | None,
) -> dict[str, Any]:
    return {
        "allowed": allowed,
        "status": "action_chunk_validated" if allowed else "action_chunk_validation_failed",
        "blocking_errors": list(errors),
        "checks": checks,
        "metadata": {
            "source": "motion.validate_action_chunk",
            "observation_id": obs_id,
            "current_subgoal_id": get_attr(current_subgoal, "subgoal_id"),
        },
    }


def _execution_artifact_prefix(blackboard: Blackboard, payload: dict[str, Any]) -> str:
    requested = payload.get("artifact_prefix") if isinstance(payload, dict) else None
    if requested:
        return str(requested).strip("/")
    run_prefix = blackboard.read("artifact_prefix")
    if run_prefix:
        return f"{str(run_prefix).strip('/')}/execute"
    return "execute"


def _expected_action_dim(action_type: object | None, backend: object | None = None) -> int | None:
    if backend is not None and hasattr(backend, "action_spec"):
        spec = backend.action_spec()
        types = spec.get("types") if isinstance(spec, dict) else None
        if isinstance(types, dict) and action_type in types:
            return int(types[action_type])
    if action_type == "qpos":
        return 14
    if action_type == "ee":
        return 16
    if action_type == "libero_ee_delta":
        return 7
    if action_type in {"calvin_ee_pose_10d", "calvin_ee_delta"}:
        return 10
    return None


def _target_handle(source: object | None, target: object | None) -> dict[str, object]:
    candidates = [candidate for candidate in (source, target) if candidate is not None]
    metric_candidates = [candidate for candidate in candidates if _has_metric_position(candidate)]
    if metric_candidates:
        primary = target if target is not None and _has_metric_position(target) else metric_candidates[0]
        metric = primary.metric_geometry
        return {
            "target_type": "metric_pose",
            "target_candidate_id": primary.candidate_id,
            "target_pose": metric.position_3d,
            "geometry_source": list(metric.source),
            "requires_controller": True,
        }
    primary = source or target
    if primary is None:
        return {
            "target_type": "missing_visual_target",
            "status": "target_handle_unavailable_missing_target",
            "reason": "missing_source_and_target_candidates",
            "requires_perception_grounding": True,
        }
    return {
        "target_type": "image_grounded",
        "target_candidate_id": getattr(primary, "candidate_id", None),
        "image_evidence": _image_evidence(primary),
        "requires_controller": False,
        "requires_visual_servo_or_vla": True,
    }


def _image_evidence(candidate: object | None) -> dict[str, object]:
    if candidate is None:
        return {}
    return {
        "bbox_by_view": dict(getattr(candidate, "bbox_by_view", {})),
        "mask_ref_by_view": dict(getattr(candidate, "mask_ref_by_view", {})),
        "visibility": getattr(candidate, "visibility", "uncertain"),
        "confidence": getattr(candidate, "confidence", 0.0),
    }


def _has_metric_position(candidate: object | None) -> bool:
    metric = getattr(candidate, "metric_geometry", None)
    return bool(getattr(metric, "has_position", False))


def _image_grounded_plan(
    blackboard: Blackboard,
    motion_goal: object | None,
    target_handle: dict[str, object],
    payload: dict[str, object],
) -> dict[str, object]:
    world_state = blackboard.read("world_state")
    current_subgoal = blackboard.read("current_subgoal")
    observation_id = current_observation_id(blackboard)
    vla_prompt = _vla_prompt(blackboard)
    backend = blackboard.read("action_backend")
    backend_name = str(payload.get("backend") or getattr(backend, "name", "vla_action_backend"))
    if not vla_prompt:
        return {
            "status": "motion_plan_unavailable",
            "reason": "missing_current_subgoal_instruction",
            "retryable": False,
            "motion_goal": to_dict(motion_goal),
            "current_subgoal": to_dict(current_subgoal),
            "metadata": {
                "source": "motion.plan_motion",
                "subgoal_id": get_attr(current_subgoal, "subgoal_id"),
                "observation_id": observation_id,
                "stale": False,
            },
        }
    return {
        "status": "image_grounded_motion_plan_built",
        "backend": backend_name,
        "motion_goal": to_dict(motion_goal),
        "vla_prompt": vla_prompt,
        "current_subgoal": to_dict(current_subgoal),
        "source_candidate_id": get_attr(motion_goal, "source_candidate_id"),
        "target_candidate_id": get_attr(motion_goal, "target_candidate_id"),
        "acting_arm": get_attr(motion_goal, "acting_arm"),
        "target_handle": dict(target_handle),
        "bbox_metadata": _candidate_bbox_metadata(world_state, motion_goal),
        "image_paths": _current_image_paths(blackboard),
        "world_state_id": get_attr(world_state, "world_state_id"),
        "retryable": True,
        "metadata": {
            "source": "motion.plan_motion",
            "subgoal_id": get_attr(current_subgoal, "subgoal_id"),
            "observation_id": observation_id,
            "stale": False,
        },
    }


def _vla_prompt(blackboard: Blackboard) -> str | None:
    current_subgoal = blackboard.read("current_subgoal")
    instruction = str(get_attr(current_subgoal, "instruction", "") or "").strip()
    return _sentence(instruction) if instruction else None


def _sentence(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "."
    if text[-1] in ".!?":
        return text
    return f"{text}."


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _backend_request(motion_plan: object | None, payload: dict[str, object]) -> dict[str, object]:
    request = {"motion_plan": to_dict(motion_plan)}
    request.update(dict(payload))
    return request


def _current_image_paths(blackboard: Blackboard) -> list[str]:
    observation = blackboard.read("observation")
    camera_views = getattr(observation, "camera_views", {})
    if not isinstance(camera_views, dict):
        return []
    return [view.rgb_path for view in camera_views.values() if getattr(view, "rgb_path", None)]


def _candidate_bbox_metadata(world_state: object | None, motion_goal: object | None) -> dict[str, object]:
    if world_state is None:
        return {}
    result: dict[str, object] = {}
    for role, candidate_id in {
        "source": get_attr(motion_goal, "source_candidate_id"),
        "target": get_attr(motion_goal, "target_candidate_id"),
    }.items():
        candidate = world_state.candidate_by_id(candidate_id) if candidate_id else None
        if candidate is None:
            continue
        result[role] = {
            "candidate_id": candidate_id,
            "label": getattr(candidate, "label", None),
            "bbox_by_view": dict(getattr(candidate, "bbox_by_view", {})),
            "visibility": getattr(candidate, "visibility", None),
        }
    return result


def _stamp_chunk_metadata(chunk: ActionChunk, metadata: dict[str, object]) -> None:
    chunk.metadata.update({key: value for key, value in metadata.items() if value is not None})


def _unavailable_chunk(reason: str, metadata: dict[str, object]) -> ActionChunk:
    payload = {"status": "action_chunk_unavailable", "reason": reason, "retryable": False}
    payload.update(metadata)
    return ActionChunk(action_type="unavailable", commands=[], control_horizon=0, metadata=payload)


def _motion_unavailable_reason(target_type: object | None) -> str:
    if target_type == "image_grounded":
        return "image_grounded_motion_requires_visual_servo_or_vla_action_model"
    if target_type == "metric_pose":
        return "metric_motion_requires_controller_backend"
    return "missing_motion_goal_or_target_handle"
