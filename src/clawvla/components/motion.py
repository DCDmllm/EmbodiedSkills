from __future__ import annotations

from ..blackboard import Blackboard
from ..blackboard_utils import (
    current_observation_id,
    mark_action_chunk_consumed,
    mark_grounding_overlay_stale,
    mark_motion_artifacts_stale,
)
from ..notices import emit_human_trace
from ..schema import ActionChunk, MotionGoal, SkillRequest, SkillResult
from ..skills.base import SkillContext, SkillRegistry
from .skill_helpers import get_attr, ok, register_skill, to_dict, unavailable


def register_motion_skills(registry: SkillRegistry) -> None:
    register_skill(registry, "motion", "build_motion_goal", "Build a motion goal from scheduler/subgoal payload.", build_motion_goal)
    register_skill(registry, "motion", "plan_motion", "Plan or select a bounded motion from the motion goal.", plan_motion)
    register_skill(registry, "motion", "emit_action_chunk", "Convert a motion plan to an action chunk.", emit_action_chunk)
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
        or get_attr(world_state, "target_candidate_id")
    )
    source = world_state.candidate_by_id(source_candidate_id) if world_state is not None else None
    target = world_state.candidate_by_id(target_candidate_id) if world_state is not None else None
    target_handle = _target_handle(source, target)
    status = (
        "motion_goal_placeholder_missing_target"
        if target_handle.get("target_type") == "missing_visual_target"
        else "motion_goal_built"
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
    return ok(status, {"motion_goal": goal.to_dict()})


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
        emit_human_trace(
            "openpi",
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
            "openpi" if backend_result.success else "failure",
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


def execute_action(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    env = blackboard.read("env_adapter")
    action_chunk = blackboard.read("action_chunk")
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
            "status": "target_handle_placeholder_missing_target",
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
    vla_prompt = _vla_prompt(blackboard, motion_goal, world_state)
    return {
        "status": "image_grounded_motion_plan_built",
        "backend": str(payload.get("backend", "pi05")),
        "motion_goal": to_dict(motion_goal),
        "task_instruction": blackboard.task_instruction,
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


def _vla_prompt(blackboard: Blackboard, motion_goal: object | None, world_state: object | None) -> str:
    task = blackboard.task_instruction or get_attr(world_state, "task_instruction") or "perform the task"
    source_id = get_attr(motion_goal, "source_candidate_id")
    target_id = get_attr(motion_goal, "target_candidate_id")
    current_subgoal = blackboard.read("current_subgoal")
    source = world_state.candidate_by_id(source_id) if world_state is not None and source_id else None
    target = world_state.candidate_by_id(target_id) if world_state is not None and target_id else None
    parts = [f"Task: {task}."]
    if current_subgoal is not None:
        parts.append(
            "Current subgoal: "
            f"id={get_attr(current_subgoal, 'subgoal_id')}, "
            f"type={get_attr(current_subgoal, 'type')}, "
            f"criteria={get_attr(current_subgoal, 'completion_criteria', {})}."
        )
    if source is not None:
        parts.append(f"Source object: {_candidate_prompt(source)}.")
    if target is not None:
        parts.append(f"Target object: {_candidate_prompt(target)}.")
    if source is not None and target is not None and get_attr(current_subgoal, "type") in {"transport", "place", "release"}:
        parts.append("Execute the manipulation using the source object and place it on the target object.")
    elif source is not None:
        parts.append("Execute only the current short-horizon subgoal; do not assume the full task is completed by one chunk.")
    return " ".join(parts)


def _candidate_prompt(candidate: object) -> str:
    return (
        f"id={getattr(candidate, 'candidate_id', None)}, "
        f"label={getattr(candidate, 'label', None)}, "
        f"visibility={getattr(candidate, 'visibility', None)}, "
        f"bboxes={dict(getattr(candidate, 'bbox_by_view', {}))}"
    )


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
