from __future__ import annotations

import json
import math
from pathlib import Path
import socket
from typing import Any

from ..blackboard_utils import current_observation_id, mark_motion_artifacts_stale
from ..schema import SafetyReport, SkillRequest, SkillResult
from ..skills.base import SkillContext, SkillRegistry
from .skill_helpers import get_attr, ok, register_skill


def register_safety_skills(registry: SkillRegistry) -> None:
    register_skill(registry, "safety", "validate_skill_request", "Validate required fields for a skill request.", validate_skill_request)
    register_skill(registry, "safety", "validate_arm_binding", "Validate image-side to robot-arm binding.", validate_arm_binding)
    register_skill(registry, "safety", "check_reachability", "Check if the requested motion is reachable.", check_reachability)
    register_skill(registry, "safety", "check_workspace", "Check workspace and generic safety constraints.", check_workspace)
    register_skill(registry, "safety", "preflight_action", "Aggregate safety checks before execution.", preflight_action)


def validate_skill_request(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    errors = []
    if not request.component:
        errors.append("missing_component")
    if not request.skill:
        errors.append("missing_skill")
    report = SafetyReport(allowed=not errors, status="valid" if not errors else "invalid", errors=errors)
    blackboard.write("last_safety_report", report, event_type="safety.validate_skill_request")
    return SkillResult(success=not errors, status=report.status, output={"safety_report": report.to_dict()}, errors=errors)


def validate_arm_binding(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    perception = blackboard.read("perception")
    binding = getattr(perception, "arm_binding", {}) if perception is not None else {}
    status = "arm_binding_not_required" if not binding else "arm_binding_available"
    report = SafetyReport(allowed=True, status=status, checks={"arm_binding": dict(binding)})
    blackboard.write("last_safety_report", report, event_type="safety.validate_arm_binding")
    return ok(report.status, {"safety_report": report.to_dict()})


def check_reachability(request: SkillRequest, context: SkillContext) -> SkillResult:
    report = _build_preflight_report(context.blackboard)
    check = report.checks.get("object_binding", {})
    status = "reachability_checked" if report.allowed else "reachability_blocked"
    return _write_report(context.blackboard, report, "safety.check_reachability", status_override=status, checks_override={"object_binding": check})


def check_workspace(request: SkillRequest, context: SkillContext) -> SkillResult:
    report = _build_preflight_report(context.blackboard)
    checks = {
        "camera_inputs": report.checks.get("camera_inputs", {}),
        "robot_state": report.checks.get("robot_state", {}),
    }
    status = "workspace_checked" if report.allowed else "workspace_blocked"
    return _write_report(context.blackboard, report, "safety.check_workspace", status_override=status, checks_override=checks)


def preflight_action(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    mark_motion_artifacts_stale(blackboard, "preflight_action", include_goal=True)
    report = _build_preflight_report(blackboard)
    return _write_report(blackboard, report, "safety.preflight_action")


def _build_preflight_report(blackboard: Any) -> SafetyReport:
    checks: dict[str, Any] = {}
    errors: list[str] = []

    observation = blackboard.read("observation")
    perception = blackboard.read("perception")
    world_state = blackboard.read("world_state")
    task_plan = blackboard.read("task_plan")
    current_subgoal = blackboard.read("current_subgoal")
    env = blackboard.read("env_adapter")
    preflight_spec = _preflight_spec(env)
    obs_id = current_observation_id(blackboard)

    _check_task_state(checks, errors, observation, perception, world_state, task_plan, current_subgoal)
    _check_observation_freshness(checks, errors, obs_id, perception, world_state)
    _check_object_binding(checks, errors, world_state, current_subgoal)
    _check_camera_inputs(checks, errors, observation, preflight_spec)
    _check_robot_state(checks, errors, observation, preflight_spec)
    _check_environment(checks, errors, env)
    _check_action_backend(checks, errors, blackboard.read("action_backend"))

    allowed = not errors
    return SafetyReport(
        allowed=allowed,
        status="preflight_passed" if allowed else "preflight_failed",
        checks=checks,
        errors=errors,
        metadata={
            "source": "safety.preflight_action",
            "next_stage": "execute" if allowed else "observe",
            "observation_id": obs_id,
            "blocking_errors": list(errors),
        },
    )


def _check_task_state(
    checks: dict[str, Any],
    errors: list[str],
    observation: Any,
    perception: Any,
    world_state: Any,
    task_plan: Any,
    current_subgoal: Any,
) -> None:
    missing = []
    for name, value in {
        "observation": observation,
        "perception": perception,
        "world_state": world_state,
        "task_plan": task_plan,
        "current_subgoal": current_subgoal,
    }.items():
        if value is None:
            missing.append(name)
            errors.append(f"missing_{name}")

    current_subgoal_id = get_attr(current_subgoal, "subgoal_id")
    plan_current_id = get_attr(task_plan, "current_subgoal_id")
    if task_plan is not None and current_subgoal is not None and current_subgoal_id != plan_current_id:
        errors.append("current_subgoal_mismatch_task_plan")

    if world_state is not None and bool(get_attr(world_state, "needs_reobserve", False)):
        errors.append("world_state_requires_reobserve")

    checks["task_state"] = {
        "status": "passed"
        if not missing
        and "current_subgoal_mismatch_task_plan" not in errors
        and "world_state_requires_reobserve" not in errors
        else "failed",
        "missing": missing,
        "current_subgoal_id": current_subgoal_id,
        "task_plan_current_subgoal_id": plan_current_id,
        "world_state_needs_reobserve": bool(get_attr(world_state, "needs_reobserve", False)),
    }


def _check_observation_freshness(checks: dict[str, Any], errors: list[str], obs_id: str | None, perception: Any, world_state: Any) -> None:
    perception_obs_id = get_attr(perception, "observation_id")
    world_obs_id = get_attr(get_attr(world_state, "metadata", {}), "observation_id")
    if obs_id is None:
        errors.append("missing_observation_id")
    if perception is not None and perception_obs_id != obs_id:
        errors.append("stale_perception")
    if world_state is not None and world_obs_id is not None and world_obs_id != obs_id:
        errors.append("stale_world_state")
    checks["observation_freshness"] = {
        "status": "passed"
        if obs_id is not None
        and (perception is None or perception_obs_id == obs_id)
        and (world_state is None or world_obs_id is None or world_obs_id == obs_id)
        else "failed",
        "observation_id": obs_id,
        "perception_observation_id": perception_obs_id,
        "world_state_observation_id": world_obs_id,
    }


def _check_object_binding(checks: dict[str, Any], errors: list[str], world_state: Any, current_subgoal: Any) -> None:
    subgoal_type = str(get_attr(current_subgoal, "type", "") or "").lower()
    source_id = get_attr(current_subgoal, "source_candidate_id") or get_attr(world_state, "source_candidate_id")
    target_id = get_attr(current_subgoal, "target_candidate_id") or get_attr(world_state, "target_candidate_id")
    target_required = subgoal_type in {"transport", "place", "release"}

    if not source_id:
        errors.append("missing_source_candidate")
    if target_required and not target_id:
        errors.append(f"missing_target_candidate_for_{subgoal_type}")
    if source_id and target_id and source_id == target_id:
        errors.append("source_target_same_candidate")

    source = _candidate_by_id(world_state, source_id)
    target = _candidate_by_id(world_state, target_id)
    _check_candidate("source", source, errors, required=bool(source_id))
    _check_candidate("target", target, errors, required=target_required and bool(target_id))

    checks["object_binding"] = {
        "status": "passed"
        if not any(error.startswith(("missing_source", "missing_target", "source_", "target_")) for error in errors)
        else "failed",
        "subgoal_type": subgoal_type,
        "source_candidate_id": source_id,
        "target_candidate_id": target_id,
        "target_required": target_required,
        "source": _candidate_summary(source),
        "target": _candidate_summary(target),
    }


def _check_candidate(role: str, candidate: Any, errors: list[str], *, required: bool) -> None:
    if not required:
        return
    if candidate is None:
        errors.append(f"{role}_candidate_not_found")
        return
    label = str(get_attr(candidate, "label", "") or "").strip()
    if not label:
        errors.append(f"{role}_label_missing")
    visibility = str(get_attr(candidate, "visibility", "uncertain") or "uncertain").lower()
    if visibility == "no":
        errors.append(f"{role}_visibility_no")


def _check_camera_inputs(checks: dict[str, Any], errors: list[str], observation: Any, preflight_spec: dict[str, Any]) -> None:
    all_required = tuple(str(item) for item in preflight_spec.get("required_cameras", []) or [])
    action_required = tuple(str(item) for item in preflight_spec.get("action_cameras", []) or all_required)
    expected_resolution = preflight_spec.get("expected_resolution")
    camera_views = get_attr(observation, "camera_views", {}) if observation is not None else {}
    cameras: dict[str, Any] = {}
    for name in all_required:
        view = camera_views.get(name) if isinstance(camera_views, dict) else None
        status = _camera_status(view, expected_resolution)
        cameras[name] = status
        if not status["ok"]:
            errors.append(f"camera_{name}_{status['reason']}")
    checks["camera_inputs"] = {
        "status": "passed" if all(item["ok"] for item in cameras.values()) else "failed",
        "required": list(all_required),
        "action_required": list(action_required),
        "expected_resolution": expected_resolution,
        "cameras": cameras,
    }


def _check_robot_state(checks: dict[str, Any], errors: list[str], observation: Any, preflight_spec: dict[str, Any]) -> None:
    state_spec = preflight_spec.get("state", {}) if isinstance(preflight_spec.get("state"), dict) else {}
    required = bool(state_spec.get("required", False))
    expected_dim = state_spec.get("dim")
    if not required:
        checks["robot_state"] = {"status": "passed", "required": False}
        return
    vector, source, read_error = _state_vector(observation, state_spec)
    ok = (
        isinstance(expected_dim, int)
        and len(vector) == expected_dim
        and all(_finite(item) for item in vector)
    )
    if not ok:
        errors.append(read_error or f"missing_{expected_dim}d_robot_state")
    checks["robot_state"] = {
        "status": "passed" if ok else "failed",
        "source": source,
        "vector_length": len(vector),
        "expected_dim": expected_dim,
        "error": read_error,
    }


def _check_environment(checks: dict[str, Any], errors: list[str], env: Any) -> None:
    status = env.status() if env is not None and hasattr(env, "status") else {}
    if not isinstance(status, dict) or "ready" not in status:
        session = get_attr(env, "session")
        task_env = get_attr(session, "task_env") if session is not None else get_attr(env, "bound_task_env")
        status = {
            "backend": "legacy",
            "ready": env is not None and task_env is not None and get_attr(env, "last_observation") is not None,
            "live_env_bound": task_env is not None,
            "last_observation_present": get_attr(env, "last_observation") is not None,
        }
    ok = bool(status.get("ready")) if isinstance(status, dict) else False
    if not ok:
        errors.append("env_unavailable")
    checks["environment"] = {
        "status": "passed" if ok else "failed",
        "env_adapter_present": env is not None,
        "env_status": status,
    }


def _check_action_backend(checks: dict[str, Any], errors: list[str], backend: Any) -> None:
    if backend is not None and hasattr(backend, "health"):
        health = backend.health()
        ok = bool(health.get("ok")) if isinstance(health, dict) else False
        if not ok:
            reason = health.get("reason") if isinstance(health, dict) else "health_unavailable"
            errors.append(_action_backend_error_code(reason))
        checks["action_backend"] = {
            "status": "passed" if ok else "failed",
            "backend_present": backend is not None,
            "health": health,
            "action_spec": backend.action_spec() if hasattr(backend, "action_spec") else None,
        }
        return
    config = get_attr(backend, "config", {})
    enabled = bool(config.get("enabled")) if isinstance(config, dict) else False
    pretrained_path = config.get("pretrained_path") if isinstance(config, dict) else None
    pretrained_exists = bool(pretrained_path and Path(str(pretrained_path)).exists())
    runtime_cfg = config.get("openpi_runtime", {}) if isinstance(config, dict) else {}
    worker_status = _openpi_worker_status(runtime_cfg if isinstance(runtime_cfg, dict) else {})
    ok = backend is not None and enabled and pretrained_exists and worker_status["ok"]
    if backend is None:
        errors.append("action_backend_missing")
    if backend is not None and not enabled:
        errors.append("action_backend_disabled")
    if enabled and not pretrained_exists:
        errors.append("action_backend_pretrained_path_missing")
    if enabled and not worker_status["ok"]:
        errors.append(f"openpi_worker_{worker_status['reason']}")
    checks["action_backend"] = {
        "status": "passed" if ok else "failed",
        "backend_present": backend is not None,
        "enabled": enabled,
        "pretrained_path": str(pretrained_path) if pretrained_path else None,
        "pretrained_path_exists": pretrained_exists,
        "worker": worker_status,
    }


def _action_backend_error_code(reason: Any) -> str:
    text = str(reason or "health_unavailable")
    if text.startswith(("action_backend_", "openpi_worker_")):
        return text
    return f"action_backend_{text}"


def _write_report(
    blackboard: Any,
    report: SafetyReport,
    event_type: str,
    *,
    status_override: str | None = None,
    checks_override: dict[str, Any] | None = None,
) -> SkillResult:
    if status_override is not None or checks_override is not None:
        report = SafetyReport(
            allowed=report.allowed,
            status=status_override or report.status,
            checks=checks_override if checks_override is not None else report.checks,
            errors=list(report.errors),
            metadata=dict(report.metadata),
        )
    blackboard.write("last_safety_report", report, event_type=event_type)
    blackboard.write("preflight_report", report, event_type=event_type)
    output = {"safety_report": report.to_dict(), "preflight_report": report.to_dict()}
    return SkillResult(success=report.allowed, status=report.status, output=output, errors=list(report.errors))


def _candidate_by_id(world_state: Any, candidate_id: Any) -> Any:
    if world_state is None or not candidate_id:
        return None
    if hasattr(world_state, "candidate_by_id"):
        return world_state.candidate_by_id(str(candidate_id))
    for candidate in get_attr(world_state, "candidates", []) or []:
        if get_attr(candidate, "candidate_id") == candidate_id:
            return candidate
    return None


def _candidate_summary(candidate: Any) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "candidate_id": get_attr(candidate, "candidate_id"),
        "label": get_attr(candidate, "label"),
        "visibility": get_attr(candidate, "visibility"),
        "has_valid_bbox": _has_valid_bbox(candidate),
        "bbox_required": False,
    }


def _has_valid_bbox(candidate: Any) -> bool:
    bbox_by_view = get_attr(candidate, "bbox_by_view", {})
    if not isinstance(bbox_by_view, dict):
        return False
    for bbox in bbox_by_view.values():
        if _valid_bbox(bbox):
            return True
    return False


def _valid_bbox(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return all(_finite(item) for item in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1


def _camera_status(view: Any, expected_resolution: Any = None) -> dict[str, Any]:
    if view is None:
        return {"ok": False, "reason": "missing"}
    rgb_path = get_attr(view, "rgb_path")
    if not rgb_path:
        return {"ok": False, "reason": "missing_rgb_path"}
    path = Path(str(rgb_path))
    if not path.exists():
        return {"ok": False, "reason": "rgb_file_missing", "path": str(path)}
    try:
        from PIL import Image

        with Image.open(path) as image:
            size = list(image.size)
    except Exception as exc:
        return {"ok": False, "reason": "rgb_file_unreadable", "path": str(path), "error": str(exc)}
    expected = [int(item) for item in expected_resolution] if isinstance(expected_resolution, list) else None
    ok = True if expected is None else size == expected
    return {"ok": ok, "reason": "ok" if ok else "unexpected_resolution", "path": str(path), "resolution": size}


def _state_vector(observation: Any, state_spec: dict[str, Any]) -> tuple[list[float], str | None, str | None]:
    source_hint_value = state_spec.get("source")
    if isinstance(source_hint_value, list):
        vector, source, error = _raw_or_arm_state_vector(observation, [str(item) for item in source_hint_value])
        if vector or error is not None:
            return vector, source, error
    source_hint = str(source_hint_value or "")
    if source_hint == "libero_state8":
        vector, source, error = _libero_state8_vector(observation)
        if vector or error is not None:
            return vector, source, error
    if source_hint:
        vector, source, error = _raw_or_arm_state_vector(observation, [source_hint])
        if vector or error is not None:
            return vector, source, error
    return _joint_action_vector(observation)


def _raw_or_arm_state_vector(observation: Any, keys: list[str]) -> tuple[list[float], str | None, str | None]:
    if observation is None:
        return [], None, None
    raw = get_attr(observation, "raw", {})
    if isinstance(raw, dict):
        for key in keys:
            vector = _float_list(raw.get(key))
            if vector is not None:
                return vector, f"observation.raw.{key}", None
        summary_ref = raw.get("summary_ref")
        if summary_ref:
            try:
                payload = json.loads(Path(str(summary_ref)).read_text(encoding="utf-8"))
            except Exception as exc:
                return [], "observation.raw.summary_ref", f"summary_ref_unreadable:{type(exc).__name__}:{exc}"
            for key in keys:
                vector = _float_list(payload.get(key))
                if vector is not None:
                    return vector, f"observation.raw.summary_ref.{key}", None
    arms = get_attr(observation, "robot_arms", {})
    if isinstance(arms, dict):
        for arm_name, arm in arms.items():
            metadata = get_attr(arm, "metadata", {})
            if not isinstance(metadata, dict):
                continue
            for key in keys:
                vector = _float_list(metadata.get(key))
                if vector is not None:
                    return vector, f"observation.robot_arms.{arm_name}.metadata.{key}", None
    return [], f"observation.raw.{keys[0] if keys else 'state'}", "state_vector_missing"


def _float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return None
    try:
        return [float(item) for item in _flatten(value)]
    except (TypeError, ValueError):
        return None


def _flatten(value: list[Any]) -> list[Any]:
    result: list[Any] = []
    for item in value:
        if isinstance(item, list):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result


def _libero_state8_vector(observation: Any) -> tuple[list[float], str | None, str | None]:
    if observation is None:
        return [], None, None
    raw = get_attr(observation, "raw", {})
    if isinstance(raw, dict) and isinstance(raw.get("libero_state8"), list):
        try:
            return [float(item) for item in raw["libero_state8"]], "observation.raw.libero_state8", None
        except (TypeError, ValueError) as exc:
            return [], "observation.raw.libero_state8", f"libero_state8_invalid:{type(exc).__name__}:{exc}"
    arms = get_attr(observation, "robot_arms", {})
    panda = arms.get("panda") if isinstance(arms, dict) else None
    metadata = get_attr(panda, "metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("state8"), list):
        try:
            return [float(item) for item in metadata["state8"]], "observation.robot_arms.panda.metadata.state8", None
        except (TypeError, ValueError) as exc:
            return [], "observation.robot_arms.panda.metadata.state8", f"libero_state8_invalid:{type(exc).__name__}:{exc}"
    return [], "observation.raw.libero_state8", "libero_state8_missing"


def _joint_action_vector(observation: Any) -> tuple[list[float], str | None, str | None]:
    if observation is None:
        return [], None, None
    raw = get_attr(observation, "raw", {})
    summary_ref = raw.get("summary_ref") if isinstance(raw, dict) else None
    if summary_ref:
        try:
            payload = json.loads(Path(str(summary_ref)).read_text(encoding="utf-8"))
            vector = payload.get("joint_action_vector")
            if isinstance(vector, list):
                return [float(item) for item in vector], "observation.raw.summary_ref", None
        except Exception as exc:
            return [], "observation.raw.summary_ref", f"summary_ref_unreadable:{type(exc).__name__}:{exc}"
    arms = get_attr(observation, "robot_arms", {})
    if not isinstance(arms, dict):
        return [], None, None
    left = arms.get("left")
    right = arms.get("right")
    left_joints = get_attr(left, "joint_positions")
    right_joints = get_attr(right, "joint_positions")
    left_gripper = get_attr(left, "gripper_value")
    right_gripper = get_attr(right, "gripper_value")
    if left_joints is None or right_joints is None or left_gripper is None or right_gripper is None:
        return [], "observation.robot_arms", "robot_arms_incomplete"
    return [*list(left_joints), float(left_gripper), *list(right_joints), float(right_gripper)], "observation.robot_arms", None


def _openpi_worker_status(runtime_cfg: dict[str, Any]) -> dict[str, Any]:
    if runtime_cfg.get("mode") != "worker":
        return {"ok": False, "mode": runtime_cfg.get("mode"), "reason": "runtime_mode_not_worker"}
    host = str(runtime_cfg.get("host") or "127.0.0.1")
    port = int(runtime_cfg.get("port") or 8765)
    try:
        with socket.create_connection((host, port), timeout=2.0) as sock:
            sock.sendall(b'{"op":"health"}\n')
            payload = sock.recv(4096).split(b"\n", 1)[0]
        response = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        return {"ok": False, "mode": "worker", "host": host, "port": port, "reason": "unreachable", "error": str(exc)}
    ok = response.get("status") == "ok"
    return {"ok": ok, "mode": "worker", "host": host, "port": port, "reason": response.get("status"), "response": response}


def _preflight_spec(env: Any) -> dict[str, Any]:
    if env is not None and hasattr(env, "preflight_spec"):
        spec = env.preflight_spec()
        if isinstance(spec, dict):
            return spec
    return {
        "required_cameras": ["head_camera", "front_camera", "left_camera", "right_camera"],
        "action_cameras": ["head_camera", "left_camera", "right_camera"],
        "expected_resolution": [960, 540],
        "state": {"required": True, "dim": 14, "source": "joint_action_vector"},
        "action": {"required": True, "types": {"qpos": 14, "ee": 16}},
    }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
