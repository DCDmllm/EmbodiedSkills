from __future__ import annotations

from ..blackboard import Blackboard
from ..model_calls import call_component_json
from ..schema import SkillRequest, SkillResult, VerificationReport
from ..skills.base import SkillContext, SkillRegistry
from .skill_helpers import get_attr, ok, register_skill, to_dict, unavailable


VERIFIER_INSTRUCTION = (
    "Verify only the current robot manipulation subgoal from the attached fresh post-execution images and execution report. "
    "Do not judge the whole task when deciding subgoal_success. The task_instruction is background only; "
    "current_subgoal.completion_criteria is the sole authoritative success target. "
    "Judge exactly that criterion without adding, removing, strengthening, or replacing any requirement. "
    "Do not require an action or result that belongs to an adjacent or later subgoal. "
    "Use the attached verification images as the source of truth. Scheduler narration, state_summary, "
    "expected_result, and other prior text are not visual evidence and must not override the images. "
    "Judge the current subgoal first; only mark task_success true if the entire task is complete. "
    "execution_report.full_task_success is RoboTwin's whole-task check_success result, not the current "
    "subgoal result; do not mark the current subgoal failed only because full_task_success is false. "
    "Use failure_type=not_done only when the subgoal is simply incomplete and the current target "
    "can still be completed by continuing execution without replanning. In that case use "
    "next_action=continue_execute. Use observation_stale or ambiguous when visual evidence is "
    "insufficient and next_action=reobserve. Use execution_failed or other only when the executed "
    "action made the state worse, changed the wrong object, destabilized the scene, or cannot be "
    "continued directly; in that case use next_action=recover. Do not use recover for normal "
    "not_done progress."
)


def build_verifier_model_request(
    blackboard: Blackboard,
    current_subgoal: object | None,
    execution_report: object | None,
) -> tuple[str, dict[str, object]]:
    """Build the single verifier request contract shared by Runtime and SFT data."""
    return (
        VERIFIER_INSTRUCTION,
        {
            "task_instruction": blackboard.task_instruction,
            "blackboard": _verifier_blackboard_context(blackboard),
            "current_subgoal": to_dict(current_subgoal),
            "subgoal_success_contract": _subgoal_verification_contract(current_subgoal),
            "execution_report": _compact_execution_report(execution_report),
            "required_schema": {
                "subgoal_success": False,
                "task_success": False,
                "partial_progress": False,
                "failure_type": "none|not_done|observation_stale|execution_failed|ambiguous|other",
                "progress_score": 0.0,
                "should_reobserve": False,
                "next_action": "advance_subgoal|continue_execute|reobserve|recover|finish",
                "notes": ["short evidence note"],
            },
        },
    )


def register_verifier_skills(registry: SkillRegistry) -> None:
    register_skill(registry, "verifier", "verify_progress", "Verify progress from before/after observations.", verify_progress, True)


def verify_progress(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    current_subgoal = blackboard.read("current_subgoal")
    execution_report = blackboard.read("execution_report")
    compact_execution = _compact_execution_report(execution_report)
    subgoal_contract = _subgoal_verification_contract(current_subgoal)
    if context.has_model and request.payload.get("use_model", True):
        image_paths = list(request.payload.get("image_paths") or [])
        if not image_paths:
            return SkillResult(
                success=False,
                status="verification_unavailable",
                output={
                    "reason": "missing_verify_images",
                    "current_subgoal": to_dict(current_subgoal),
                    "execution_report": compact_execution,
                },
                errors=["missing_verify_images"],
            )
        instruction, model_payload = build_verifier_model_request(
            blackboard,
            current_subgoal,
            execution_report,
        )
        raw = call_component_json(
            context,
            instruction=instruction,
            payload=model_payload,
            image_paths=image_paths,
            render_format=request.payload.get("render_format", "json"),
        )
        report = _report_from_payload(raw, current_subgoal, execution_report, source="verifier_model")
        blackboard.write("last_verification_report", report, event_type="verifier.verify_progress")
        status = (
            "task_verified_success"
            if report.metadata.get("task_success")
            else "subgoal_verified_success"
            if report.success
            else "subgoal_verification_failed"
        )
        return ok(status, {"verification_report": report.to_dict()})

    return unavailable(
        "verification_unavailable",
        "verifier_model_unavailable",
        {
            "current_subgoal": to_dict(current_subgoal),
            "subgoal_success_contract": subgoal_contract,
            "execution_report": compact_execution,
        },
    )


def _report_from_payload(payload: dict[str, object], current_subgoal: object | None, execution_report: object | None, *, source: str) -> VerificationReport:
    subgoal_success = _as_bool(payload.get("subgoal_success"), False)
    task_success = _as_bool(payload.get("task_success"), False)
    failure_type = payload.get("failure_type")
    if subgoal_success and (failure_type is None or str(failure_type) in {"", "none"}):
        failure_type = None
    elif failure_type is None:
        failure_type = "not_done"
    should_reobserve = _as_bool(payload.get("should_reobserve"), False)
    raw_next_action = str(payload.get("next_action") or "")
    next_action = _canonical_next_action(
        subgoal_success=subgoal_success,
        task_success=task_success,
        failure_type=str(failure_type) if failure_type is not None else None,
        should_reobserve=should_reobserve,
    )
    notes = payload.get("notes")
    if not isinstance(notes, list):
        notes = []
    report = VerificationReport(
        success=bool(subgoal_success),
        partial_progress=_as_bool(payload.get("partial_progress"), False),
        failure_type=str(failure_type) if failure_type is not None else None,
        progress_score=_as_float_or_none(payload.get("progress_score")),
        residuals=dict(payload.get("residuals", {})) if isinstance(payload.get("residuals"), dict) else {},
        should_reobserve=should_reobserve or next_action == "reobserve",
        notes=[str(item) for item in notes],
        metadata={
            "source": source,
            "subgoal_success": bool(subgoal_success),
            "task_success": bool(task_success),
            "next_action": next_action,
            "raw_next_action": raw_next_action or None,
            "current_subgoal_id": get_attr(current_subgoal, "subgoal_id"),
            "subgoal_success_contract": _subgoal_verification_contract(current_subgoal),
            "execution_report": _compact_execution_report(execution_report),
        },
    )
    return report


def _canonical_next_action(
    *,
    subgoal_success: bool,
    task_success: bool,
    failure_type: str | None,
    should_reobserve: bool,
) -> str:
    _ = task_success
    if subgoal_success:
        return "advance_subgoal"
    normalized_failure = str(failure_type or "not_done").strip().lower()
    if should_reobserve or normalized_failure in {"observation_stale", "ambiguous"}:
        return "reobserve"
    if normalized_failure in {"not_done", "none", ""}:
        return "continue_execute"
    return "recover"


def _verifier_blackboard_context(blackboard: Blackboard) -> dict[str, object]:
    """Return verifier context without scheduler-authored narrative history."""
    payload = blackboard.compact_context()
    payload["last_scheduler_decision"] = None
    payload["recent_loop_history"] = []
    return payload


def _subgoal_verification_contract(current_subgoal: object | None) -> dict[str, object]:
    source_id = get_attr(current_subgoal, "source_candidate_id")
    target_id = get_attr(current_subgoal, "target_candidate_id")
    completion_criteria = dict(get_attr(current_subgoal, "completion_criteria", {}) or {})
    natural_language = str(completion_criteria.get("natural_language") or "").strip()
    base = {
        "subgoal_id": get_attr(current_subgoal, "subgoal_id"),
        "subgoal_instruction": get_attr(current_subgoal, "instruction"),
        "source_candidate_id": source_id,
        "target_candidate_id": target_id,
        "completion_criteria": completion_criteria,
        "judge_only_this_subgoal": True,
        "success_condition": natural_language or "the visual state satisfies current_subgoal.completion_criteria",
        "not_required": [
            "any action or result not explicitly required by current_subgoal.completion_criteria",
            "full task completion unless this is the final subgoal",
        ],
    }
    return base


def _compact_execution_report(execution_report: object | None) -> dict[str, object] | None:
    payload = to_dict(execution_report)
    if not isinstance(payload, dict):
        return None
    observation = payload.get("observation") if isinstance(payload.get("observation"), dict) else {}
    action_chunk = payload.get("action_chunk") if isinstance(payload.get("action_chunk"), dict) else {}
    action_metadata = action_chunk.get("metadata") if isinstance(action_chunk.get("metadata"), dict) else {}
    commands = action_chunk.get("commands") if isinstance(action_chunk.get("commands"), list) else []
    return {
        "backend": payload.get("backend"),
        "status": payload.get("status"),
        "full_task_success": payload.get("success"),
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


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"true", "yes", "1"}:
            return True
        if value.lower() in {"false", "no", "0"}:
            return False
    return default


def _as_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
