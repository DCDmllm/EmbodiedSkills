from __future__ import annotations

from ..blackboard import Blackboard
from ..model_calls import call_component_json
from ..schema import SkillRequest, SkillResult, VerificationReport
from ..skills.base import SkillContext, SkillRegistry
from .skill_helpers import get_attr, ok, register_skill, to_dict


def register_verifier_skills(registry: SkillRegistry) -> None:
    register_skill(registry, "verifier", "verify_progress", "Verify progress from before/after observations.", verify_progress, True)
    register_skill(registry, "verifier", "score_residual", "Score observation residuals.", score_residual)
    register_skill(registry, "verifier", "diagnose_failure", "Diagnose likely failure mode.", diagnose_failure, True)


def verify_progress(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    current_subgoal = blackboard.read("current_subgoal")
    execution_report = blackboard.read("execution_report")
    if context.has_model and request.payload.get("use_model", True):
        raw = call_component_json(
            context,
            instruction=(
                "Verify the current robot manipulation subgoal from the latest observation and execution report. "
                "Judge the current subgoal first; only mark task_success true if the entire task is complete."
            ),
            payload={
                "task_instruction": blackboard.task_instruction,
                "blackboard": blackboard.compact_context(),
                "current_subgoal": to_dict(current_subgoal),
                "execution_report": to_dict(execution_report),
                "required_schema": {
                    "subgoal_success": False,
                    "task_success": False,
                    "partial_progress": False,
                    "failure_type": "none|not_done|observation_stale|execution_failed|ambiguous|other",
                    "progress_score": 0.0,
                    "should_reobserve": False,
                    "next_action": "advance_subgoal|continue_execute|reobserve|replan|recover|finish",
                    "notes": ["short evidence note"],
                },
            },
            image_paths=request.payload.get("image_paths"),
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

    report = VerificationReport(
        success=False,
        partial_progress=False,
        failure_type="verification_placeholder_unavailable",
        progress_score=None,
        should_reobserve=True,
        notes=["Verifier model/backend is unavailable; subgoal was not verified."],
        metadata={
            "source": "verifier_placeholder",
            "placeholder": True,
            "subgoal_success": False,
            "task_success": False,
            "next_action": "reobserve",
            "current_subgoal_id": get_attr(current_subgoal, "subgoal_id"),
            "execution_report": to_dict(execution_report),
        },
    )
    blackboard.write("last_verification_report", report, event_type="verifier.verify_progress")
    return ok(
        "verification_placeholder_unavailable",
        {"verification_report": report.to_dict(), "retryable": False, "reason": "verifier_model_unavailable"},
    )


def score_residual(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    verification = blackboard.read("last_verification_report")
    if verification is not None:
        verification.residuals.setdefault("status", "residual_scorer_unavailable")
        verification.residuals.setdefault("retryable", False)
        blackboard.write("last_verification_report", verification, event_type="verifier.score_residual")
    return ok("residual_scorer_unavailable", {"verification_report": to_dict(verification), "retryable": False})


def diagnose_failure(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    diagnosis = {
        "failure_type": get_attr(blackboard.read("last_verification_report"), "failure_type", "unknown"),
        "status": "diagnosis_unavailable",
        "reason": "failure_diagnosis_not_wired",
        "retryable": False,
    }
    blackboard.write("last_failure_diagnosis", diagnosis, event_type="verifier.diagnose_failure")
    return ok("failure_diagnosis_unavailable", {"diagnosis": diagnosis})


def _report_from_payload(payload: dict[str, object], current_subgoal: object | None, execution_report: object | None, *, source: str) -> VerificationReport:
    next_action = str(payload.get("next_action") or "recover")
    if next_action not in {"advance_subgoal", "continue_execute", "reobserve", "replan", "recover", "finish"}:
        next_action = "recover"
    subgoal_success = _as_bool(payload.get("subgoal_success"), False)
    task_success = _as_bool(payload.get("task_success"), False)
    failure_type = payload.get("failure_type")
    if subgoal_success and (failure_type is None or str(failure_type) in {"", "none"}):
        failure_type = None
    elif failure_type is None:
        failure_type = "not_done"
    notes = payload.get("notes")
    if not isinstance(notes, list):
        notes = []
    report = VerificationReport(
        success=bool(subgoal_success),
        partial_progress=_as_bool(payload.get("partial_progress"), False),
        failure_type=str(failure_type) if failure_type is not None else None,
        progress_score=_as_float_or_none(payload.get("progress_score")),
        residuals=dict(payload.get("residuals", {})) if isinstance(payload.get("residuals"), dict) else {},
        should_reobserve=_as_bool(payload.get("should_reobserve"), next_action == "reobserve"),
        notes=[str(item) for item in notes],
        metadata={
            "source": source,
            "subgoal_success": bool(subgoal_success),
            "task_success": bool(task_success),
            "next_action": next_action,
            "current_subgoal_id": get_attr(current_subgoal, "subgoal_id"),
            "execution_report": to_dict(execution_report),
        },
    )
    return report


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
