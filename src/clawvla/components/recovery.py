from __future__ import annotations

from typing import Any

from ..blackboard_utils import mark_motion_artifacts_stale
from ..model_calls import call_component_json
from ..schema import SkillRequest, SkillResult, Subgoal, TaskPlan
from ..skills.base import SkillContext, SkillRegistry
from .skill_helpers import get_attr, ok, register_skill, to_dict, unavailable


PATCH_TYPES = {"retry_current_subgoal", "replace_current_subgoal", "insert_recovery_subgoal", "replan", "reobserve", "abort"}
NEXT_STAGES = {"preflight", "plan", "observe", "finish"}
PATCH_NEXT_STAGE = {
    "retry_current_subgoal": "preflight",
    "replace_current_subgoal": "preflight",
    "insert_recovery_subgoal": "preflight",
    "replan": "plan",
    "reobserve": "observe",
    "abort": "finish",
}


def register_recovery_skills(registry: SkillRegistry) -> None:
    register_skill(
        registry,
        "recovery",
        "decide_recovery",
        "Diagnose a true verification failure and propose a concrete recovery patch.",
        decide_recovery,
        True,
    )
    register_skill(registry, "recovery", "build_retry_request", "Apply a recovery patch and request the next stage.", build_retry_request)


def decide_recovery(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    verification = blackboard.read("last_verification_report")
    execution_report = blackboard.read("execution_report")
    current_subgoal = blackboard.read("current_subgoal")
    task_plan = blackboard.read("task_plan")
    readiness_error = _recovery_readiness_error(verification, execution_report)
    if readiness_error is not None:
        return unavailable(
            "recovery_unavailable",
            readiness_error,
            {
                "verification": to_dict(verification),
                "execution_report": to_dict(execution_report),
                "current_subgoal": to_dict(current_subgoal),
            },
        )
    if not context.has_model or not request.payload.get("use_model", True):
        return unavailable(
            "recovery_unavailable",
            "recovery_model_required_no_fallback",
            {
                "verification": to_dict(verification),
                "execution_report": to_dict(execution_report),
                "current_subgoal": to_dict(current_subgoal),
            },
        )

    raw = call_component_json(
        context,
        instruction=(
            "Diagnose a true robot manipulation failure after verifier.next_action=recover and propose one concrete "
            "recovery patch. Do not route normal not_done progress here; not_done should have used continue_execute. "
            "A recovery patch must change something meaningful: revise the current subgoal instruction, insert a short "
            "recovery subgoal, request replanning when the plan itself is invalid, request reobservation when the scene "
            "cannot be judged, or abort only if the run cannot continue. Do not output a generic stage router."
        ),
        payload={
            "task_instruction": blackboard.task_instruction,
            "task_plan": to_dict(task_plan),
            "current_subgoal": to_dict(current_subgoal),
            "verification_report": to_dict(verification),
            "execution_report": to_dict(execution_report),
            "required_schema": {
                "recoverable": True,
                "failure_diagnosis": "short concrete visual or execution failure diagnosis",
                "patch_type": "retry_current_subgoal|replace_current_subgoal|insert_recovery_subgoal|replan|reobserve|abort",
                "next_stage": "preflight|plan|observe|finish",
                "repaired_subgoal": {
                    "subgoal_id": "required for subgoal patch types",
                    "type": "short subgoal type",
                    "instruction": "natural-language VLA command for the recovery attempt",
                    "source_candidate_id": "existing source id or null",
                    "target_candidate_id": "existing target id or null",
                    "status": "pending",
                    "completion_criteria": {
                        "natural_language": "visible success condition for this recovery subgoal"
                    },
                },
                "notes": ["short evidence note"],
            },
        },
        image_paths=request.payload.get("image_paths"),
        render_format=request.payload.get("render_format", "json"),
    )
    errors = _recovery_directive_errors(raw)
    if errors:
        return unavailable(
            "recovery_invalid_model_output",
            ";".join(errors),
            {
                "raw_keys": sorted(str(key) for key in raw.keys()),
                "raw_recovery_directive": raw,
                "verification": to_dict(verification),
                "current_subgoal": to_dict(current_subgoal),
            },
        )
    directive = _normalize_recovery_directive(raw, verification, execution_report, current_subgoal)
    blackboard.write("last_recovery_directive", directive, event_type="recovery.decide_recovery")
    return ok("recovery_patch_decided", {"recovery_directive": directive})


def build_retry_request(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    directive = blackboard.read("last_recovery_directive")
    if not isinstance(directive, dict):
        return unavailable("retry_request_unavailable", "missing_recovery_directive", {})
    errors = _recovery_directive_errors(directive)
    if errors:
        return unavailable(
            "retry_request_unavailable",
            ";".join(errors),
            {"recovery_directive": directive},
        )

    patch_type = str(directive["patch_type"])
    if patch_type in {"retry_current_subgoal", "replace_current_subgoal"}:
        apply_error = _apply_current_subgoal_patch(blackboard, directive)
    elif patch_type == "insert_recovery_subgoal":
        apply_error = _insert_recovery_subgoal(blackboard, directive)
    else:
        apply_error = None
    if apply_error is not None:
        return unavailable("retry_request_unavailable", apply_error, {"recovery_directive": directive})

    next_stage = str(directive["next_stage"])
    if next_stage == "preflight":
        blackboard.write("preflight_report", None, event_type="recovery.build_retry_request_clear_preflight_report")
        blackboard.write("safety_report", None, event_type="recovery.build_retry_request_clear_safety_report")
        mark_motion_artifacts_stale(blackboard, "recovery_retry_request", include_goal=True)

    retry_request = {
        "stage": next_stage,
        "control": "finish_run" if next_stage == "finish" else "run_skill",
        "patch_type": patch_type,
        "reason": str(directive["failure_diagnosis"]),
        "recovery_directive": directive,
    }
    blackboard.write("last_retry_request", retry_request, event_type="recovery.build_retry_request")
    return ok("retry_request_built", {"retry_request": retry_request})


def _recovery_readiness_error(verification: object | None, execution_report: object | None) -> str | None:
    if verification is None and execution_report is None:
        return "missing_failure_report_before_recovery"
    if verification is None:
        return None
    if get_attr(verification, "success", False):
        return "recovery_requires_failed_verification"
    next_action = _verification_next_action(verification)
    if next_action != "recover":
        return f"recovery_requires_verification_next_action_recover:{next_action}"
    return None


def _verification_next_action(verification: object | None) -> str | None:
    metadata = get_attr(verification, "metadata", {})
    if isinstance(metadata, dict) and metadata.get("next_action"):
        return str(metadata["next_action"])
    if get_attr(verification, "success", False):
        return "advance_subgoal"
    if get_attr(verification, "should_reobserve", False):
        return "reobserve"
    return None


def _recovery_directive_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    patch_type = str(payload.get("patch_type") or "")
    next_stage = str(payload.get("next_stage") or "")
    diagnosis = str(payload.get("failure_diagnosis") or "").strip()
    if patch_type not in PATCH_TYPES:
        errors.append(f"invalid_patch_type:{patch_type}:expected_{sorted(PATCH_TYPES)}")
    if next_stage not in NEXT_STAGES:
        errors.append(f"invalid_next_stage:{next_stage}:expected_{sorted(NEXT_STAGES)}")
    if patch_type in PATCH_NEXT_STAGE and next_stage and next_stage != PATCH_NEXT_STAGE[patch_type]:
        errors.append(f"next_stage_mismatch_for_patch_type:{patch_type}:{next_stage}:expected_{PATCH_NEXT_STAGE[patch_type]}")
    if not diagnosis:
        errors.append("missing_failure_diagnosis")
    if patch_type in {"retry_current_subgoal", "replace_current_subgoal", "insert_recovery_subgoal"}:
        subgoal = payload.get("repaired_subgoal")
        if not isinstance(subgoal, dict):
            errors.append(f"missing_repaired_subgoal_for_patch_type:{patch_type}")
        else:
            errors.extend(_subgoal_patch_errors(subgoal))
    return errors


def _subgoal_patch_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not str(payload.get("subgoal_id") or "").strip():
        errors.append("missing_repaired_subgoal_id")
    if not str(payload.get("type") or "").strip():
        errors.append("missing_repaired_subgoal_type")
    if not str(payload.get("instruction") or "").strip():
        errors.append("missing_repaired_subgoal_instruction")
    criteria = payload.get("completion_criteria")
    if not isinstance(criteria, dict) or not str(criteria.get("natural_language") or "").strip():
        errors.append("missing_repaired_subgoal_natural_language_completion_criteria")
    return errors


def _normalize_recovery_directive(
    raw: dict[str, Any],
    verification: object | None,
    execution_report: object | None,
    current_subgoal: object | None,
) -> dict[str, Any]:
    directive = {
        "recoverable": bool(raw.get("recoverable", True)),
        "failure_diagnosis": str(raw["failure_diagnosis"]).strip(),
        "patch_type": str(raw["patch_type"]),
        "next_stage": str(raw["next_stage"]),
        "repaired_subgoal": raw.get("repaired_subgoal") if isinstance(raw.get("repaired_subgoal"), dict) else None,
        "notes": [str(item) for item in raw.get("notes", [])] if isinstance(raw.get("notes"), list) else [],
        "verification": to_dict(verification),
        "execution_report": to_dict(execution_report),
        "current_subgoal_id": get_attr(current_subgoal, "subgoal_id"),
        "source": "recovery.decide_recovery",
    }
    return directive


def _apply_current_subgoal_patch(blackboard: Any, directive: dict[str, Any]) -> str | None:
    current = blackboard.read("current_subgoal")
    if current is None:
        return "missing_current_subgoal_for_recovery_patch"
    patch = directive.get("repaired_subgoal")
    if not isinstance(patch, dict):
        return "missing_repaired_subgoal_for_current_subgoal_patch"
    patched = _patched_subgoal(current, patch, directive)
    task_plan = blackboard.read("task_plan")
    if isinstance(task_plan, TaskPlan):
        for index, subgoal in enumerate(task_plan.subgoals):
            if subgoal.subgoal_id == current.subgoal_id:
                task_plan.subgoals[index] = patched
                task_plan.current_subgoal_id = patched.subgoal_id
                break
        else:
            return "current_subgoal_not_found_in_task_plan_for_recovery_patch"
        blackboard.write("task_plan", task_plan, event_type="recovery.patch_task_plan_current_subgoal")
    blackboard.write("current_subgoal", patched, event_type="recovery.patch_current_subgoal")
    return None


def _insert_recovery_subgoal(blackboard: Any, directive: dict[str, Any]) -> str | None:
    task_plan = blackboard.read("task_plan")
    current = blackboard.read("current_subgoal")
    if not isinstance(task_plan, TaskPlan):
        return "missing_task_plan_for_recovery_subgoal_insert"
    if current is None:
        return "missing_current_subgoal_for_recovery_subgoal_insert"
    patch = directive.get("repaired_subgoal")
    if not isinstance(patch, dict):
        return "missing_repaired_subgoal_for_recovery_subgoal_insert"
    if any(subgoal.subgoal_id == str(patch["subgoal_id"]) for subgoal in task_plan.subgoals):
        return f"duplicate_recovery_subgoal_id:{patch['subgoal_id']}"
    inserted = Subgoal.from_payload({**patch, "status": "running"})
    for subgoal in task_plan.subgoals:
        if subgoal.subgoal_id == current.subgoal_id:
            subgoal.status = "pending"
    for index, subgoal in enumerate(task_plan.subgoals):
        if subgoal.subgoal_id == current.subgoal_id:
            task_plan.subgoals.insert(index, inserted)
            break
    else:
        return "current_subgoal_not_found_in_task_plan_for_recovery_subgoal_insert"
    task_plan.current_subgoal_id = inserted.subgoal_id
    blackboard.write("task_plan", task_plan, event_type="recovery.insert_recovery_subgoal")
    blackboard.write("current_subgoal", inserted, event_type="recovery.current_subgoal_inserted")
    return None


def _patched_subgoal(current: Subgoal, patch: dict[str, Any], directive: dict[str, Any]) -> Subgoal:
    patched = Subgoal.from_payload(
        {
            "subgoal_id": patch.get("subgoal_id") or current.subgoal_id,
            "type": patch.get("type") or current.type,
            "instruction": patch.get("instruction") or current.instruction,
            "source_candidate_id": patch.get("source_candidate_id", current.source_candidate_id),
            "target_candidate_id": patch.get("target_candidate_id", current.target_candidate_id),
            "status": "running",
            "completion_criteria": patch.get("completion_criteria") or current.completion_criteria,
            "metadata": {
                **dict(getattr(current, "metadata", {}) or {}),
                "recovery_patch": {
                    "patch_type": directive.get("patch_type"),
                    "failure_diagnosis": directive.get("failure_diagnosis"),
                },
            },
        }
    )
    return patched
