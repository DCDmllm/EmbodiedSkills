from __future__ import annotations

from typing import Any

from ..blackboard_utils import current_observation_id, mark_motion_artifacts_stale, metadata_value
from ..blackboard import Blackboard
from ..loop_types import LoopDecision
from ..model_calls import call_component_json
from ..schema import SchedulerDecision, SkillRequest, SkillResult, Subgoal, TaskPlan
from ..skills.base import SkillContext, SkillRegistry
from .skill_helpers import ok, register_skill, to_dict, unavailable


def register_scheduler_skills(registry: SkillRegistry) -> None:
    register_skill(registry, "scheduler", "select_next_component", "Select the next component to run.", select_next_component, True)
    register_skill(registry, "scheduler", "choose_next_skill", "Choose the next skill request.", choose_next_skill, True)
    register_skill(registry, "scheduler", "build_task_plan", "Build a task-level subgoal plan.", build_task_plan, True)
    register_skill(registry, "scheduler", "select_current_subgoal", "Select the active subgoal from a task plan.", select_current_subgoal)
    register_skill(registry, "scheduler", "advance_subgoal", "Advance to the next subgoal after verification succeeds.", advance_subgoal)
    register_skill(registry, "scheduler", "allocate_budget", "Allocate bounded action or model budget.", allocate_budget)
    register_skill(
        registry,
        "scheduler",
        "repair_stage_transition",
        "Explicit error-repair stage transition with target_stage.",
        repair_stage_transition,
    )


def select_next_component(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    world_state = blackboard.read("world_state")
    if world_state is None or getattr(world_state, "needs_reobserve", False):
        decision = SchedulerDecision("vision", "capture_views", "World state is missing or requested reobservation.", stage="observe")
    elif not getattr(world_state, "source_candidate_id", None):
        decision = SchedulerDecision("vision", "localize_task_objects", "Task source is not localized yet.", stage="observe")
    else:
        decision = SchedulerDecision("motion", "build_motion_goal", "World state exists; build a bounded motion goal.", stage="execute")
    blackboard.write("last_scheduler_decision", decision, event_type="scheduler.select_next_component")
    return ok("next_component_selected", {"decision": decision.to_dict()})


def choose_next_skill(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    if context.has_model and request.payload.get("use_model", True):
        loop_mode = bool(request.payload.get("loop_mode", False))
        payload = {
            "blackboard": blackboard.compact_context(),
            "current_observation_images": list(request.payload.get("image_paths") or []),
            "loop_mode": loop_mode,
            "current_stage": request.payload.get("current_stage") or blackboard.read("stage"),
            "stage_order": request.payload.get("stage_order"),
            "allowed_skills": request.payload.get("allowed_skills"),
            "available_components": request.payload.get("available_components"),
            "available_skills": request.payload.get("available_skills"),
            "runtime_state": request.payload.get("runtime_state"),
            "required_schema": _loop_schema() if loop_mode else _skill_schema(),
        }
        raw = call_component_json(
            context,
            instruction=_scheduler_instruction(loop_mode),
            payload=payload,
            image_paths=request.payload.get("image_paths"),
            render_format=request.payload.get("render_format", "json"),
        )
        loop_decision = LoopDecision.from_payload(raw)
        missing_run_skill_fields = []
        if loop_decision.control == "run_skill":
            if not loop_decision.next_component:
                missing_run_skill_fields.append("next_component")
            if not loop_decision.next_skill:
                missing_run_skill_fields.append("next_skill")
        if missing_run_skill_fields:
            reason = (
                "scheduler_model_output_missing_required_fields: control=run_skill requires both "
                "next_component and next_skill. Missing fields: "
                f"{', '.join(missing_run_skill_fields)}. Retry and add the missing field(s). "
                "If you forgot next_component, use the exact component key from allowed_skills; "
                "next_skill must be one listed under that component."
            )
            error_payload = {
                "reason": reason,
                "missing_fields": missing_run_skill_fields,
                "raw_keys": sorted(str(key) for key in raw.keys()),
                "raw_model_output": raw,
                "allowed_skills": payload.get("allowed_skills"),
                "current_stage": payload.get("current_stage"),
            }
            blackboard.write(
                "last_scheduler_error",
                error_payload,
                event_type="scheduler.choose_next_skill_invalid_model_output",
            )
            return unavailable("scheduler_invalid_model_output", reason, error_payload)

        corrected_loop_decision = LoopDecision(
            control=loop_decision.control,
            stage=loop_decision.stage,
            next_component=str(loop_decision.next_component) if loop_decision.control == "run_skill" else None,
            next_skill=str(loop_decision.next_skill) if loop_decision.control == "run_skill" else None,
            payload=loop_decision.payload,
            reason=str(raw.get("reason") or loop_decision.reason or "model_scheduler"),
            narration=loop_decision.narration or (str(raw["narration"]) if raw.get("narration") is not None else None),
            state_summary=loop_decision.state_summary
            or (str(raw["state_summary"]) if raw.get("state_summary") is not None else None),
            expected_result=loop_decision.expected_result
            or (str(raw["expected_result"]) if raw.get("expected_result") is not None else None),
            budget_steps=loop_decision.budget_steps,
            metadata={
                **loop_decision.metadata,
                "source": "model",
            },
        )
        decision = SchedulerDecision(
            next_component=str(corrected_loop_decision.next_component or ""),
            next_skill=str(corrected_loop_decision.next_skill or ""),
            reason=corrected_loop_decision.reason,
            payload=loop_decision.payload,
            stage=loop_decision.stage,
            budget_steps=loop_decision.budget_steps,
            metadata={"source": "model", "control": loop_decision.control},
        )
        blackboard.write("last_scheduler_decision", decision, event_type="scheduler.choose_next_skill")
        return ok("next_skill_chosen_by_model", {"decision": decision.to_dict(), "loop_decision": corrected_loop_decision.to_dict()})

    reason = (
        "scheduler_model_required_no_fallback: choose_next_skill requires an enabled scheduler model. "
        "No default skill, previous skill, or inferred component will be reused. Start or configure the scheduler "
        "model and retry."
    )
    error_payload = {
        "reason": reason,
        "current_stage": request.payload.get("current_stage") or blackboard.read("stage"),
        "allowed_skills": request.payload.get("allowed_skills"),
        "loop_mode": bool(request.payload.get("loop_mode", False)),
    }
    blackboard.write(
        "last_scheduler_error",
        error_payload,
        event_type="scheduler.choose_next_skill_model_unavailable",
    )
    return unavailable("scheduler_model_unavailable", reason, error_payload)


def build_task_plan(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    world_state = blackboard.read("world_state")
    if world_state is None:
        return unavailable("task_plan_unavailable", "missing_world_state", {})
    source_id = getattr(world_state, "source_candidate_id", None)
    target_id = getattr(world_state, "target_candidate_id", None)
    if not source_id or not target_id:
        return unavailable("task_plan_unavailable", "missing_source_or_target_candidate", {"world_state": to_dict(world_state)})

    if context.has_model and request.payload.get("use_model", True):
        raw = call_component_json(
            context,
            instruction=_task_plan_instruction(),
            payload={
                "task_instruction": blackboard.task_instruction,
                "world_state": to_dict(world_state),
                "required_schema": _task_plan_schema(source_id, target_id),
            },
            image_paths=request.payload.get("image_paths"),
            render_format=request.payload.get("render_format", "json"),
        )
        plan = TaskPlan.from_payload(raw)
        validation_errors = _validate_task_plan_completeness(plan, blackboard.task_instruction, source_id, target_id)
        if not validation_errors:
            plan.metadata["source"] = "scheduler_model"
        else:
            return unavailable(
                "task_plan_invalid_model_output",
                ";".join(validation_errors),
                {
                    "raw_keys": sorted(str(key) for key in raw.keys()),
                    "raw_task_plan": raw,
                    "parsed_task_plan": plan.to_dict(),
                    "source_candidate_id": source_id,
                    "target_candidate_id": target_id,
                },
            )
    else:
        reason = (
            "scheduler_model_required_for_task_plan: build_task_plan requires an enabled scheduler model. "
            "Template task plans are disabled; retry after starting or configuring the scheduler model."
        )
        error_payload = {
            "reason": reason,
            "source_candidate_id": source_id,
            "target_candidate_id": target_id,
            "world_state": to_dict(world_state),
        }
        blackboard.write(
            "last_scheduler_error",
            error_payload,
            event_type="scheduler.build_task_plan_model_unavailable",
        )
        return unavailable("task_plan_unavailable", reason, error_payload)

    blackboard.write("task_plan", plan, event_type="scheduler.build_task_plan")
    blackboard.write("current_subgoal", None, event_type="scheduler.build_task_plan_reset_current_subgoal")
    mark_motion_artifacts_stale(blackboard, "task_plan_rebuilt", include_goal=True)
    return ok("task_plan_built", {"task_plan": plan.to_dict()})


def select_current_subgoal(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    task_plan = blackboard.read("task_plan")
    if not isinstance(task_plan, TaskPlan):
        return unavailable("current_subgoal_unavailable", "missing_task_plan", {})
    subgoal = task_plan.current_subgoal()
    if subgoal is None or subgoal.status in {"succeeded", "failed", "skipped"}:
        subgoal = _first_subgoal_with_status(task_plan, {"pending", "running"})
    if subgoal is None:
        task_plan.status = "succeeded"
        blackboard.write("task_plan", task_plan, event_type="scheduler.select_current_subgoal_complete")
        return ok("task_plan_complete", {"task_plan": task_plan.to_dict(), "current_subgoal": None})
    subgoal.status = "running"
    task_plan.current_subgoal_id = subgoal.subgoal_id
    mark_motion_artifacts_stale(blackboard, "current_subgoal_selected", include_goal=True)
    blackboard.write("task_plan", task_plan, event_type="scheduler.select_current_subgoal_plan")
    blackboard.write("current_subgoal", subgoal, event_type="scheduler.select_current_subgoal")
    return ok("current_subgoal_selected", {"task_plan": task_plan.to_dict(), "current_subgoal": subgoal.to_dict()})


def advance_subgoal(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    task_plan = blackboard.read("task_plan")
    current = blackboard.read("current_subgoal")
    verification = blackboard.read("last_verification_report")
    if not isinstance(task_plan, TaskPlan):
        return unavailable("advance_subgoal_unavailable", "missing_task_plan", {})
    if current is None:
        return unavailable("advance_subgoal_unavailable", "missing_current_subgoal", {"task_plan": task_plan.to_dict()})
    if verification is None:
        return unavailable("advance_subgoal_unavailable", "missing_successful_verification", {"task_plan": task_plan.to_dict()})
    if not getattr(verification, "success", False):
        return unavailable("advance_subgoal_unavailable", "last_verification_not_successful", {"verification": to_dict(verification)})
    current_subgoal_id = getattr(current, "subgoal_id", None)
    verified_subgoal_id = _verified_subgoal_id(verification)
    if not verified_subgoal_id:
        return unavailable(
            "advance_subgoal_unavailable",
            "verification_missing_current_subgoal_id",
            {"verification": to_dict(verification), "current_subgoal_id": current_subgoal_id},
        )
    if verified_subgoal_id != current_subgoal_id:
        return unavailable(
            "advance_subgoal_unavailable",
            "verification_current_subgoal_mismatch",
            {
                "verification": to_dict(verification),
                "verified_subgoal_id": verified_subgoal_id,
                "current_subgoal_id": current_subgoal_id,
            },
        )

    for subgoal in task_plan.subgoals:
        if subgoal.subgoal_id == current_subgoal_id:
            subgoal.status = "succeeded"
            break
    next_subgoal = _first_subgoal_with_status(task_plan, {"pending"})
    if next_subgoal is None:
        task_plan.status = "succeeded"
        task_plan.current_subgoal_id = None
        _archive_and_clear_consumed_verification(blackboard, verification, current_subgoal_id, "task_plan_complete")
        blackboard.write("task_plan", task_plan, event_type="scheduler.advance_subgoal_complete")
        blackboard.write("current_subgoal", None, event_type="scheduler.advance_subgoal_complete")
        return ok("task_plan_complete", {"task_plan": task_plan.to_dict(), "current_subgoal": None, "finish_recommended": True})

    next_subgoal.status = "running"
    task_plan.current_subgoal_id = next_subgoal.subgoal_id
    _archive_and_clear_consumed_verification(blackboard, verification, current_subgoal_id, "subgoal_advanced")
    blackboard.write("preflight_report", None, event_type="scheduler.advance_subgoal_clear_preflight_report")
    blackboard.write("safety_report", None, event_type="scheduler.advance_subgoal_clear_safety_report")
    blackboard.write("stage", "preflight", event_type="scheduler.advance_subgoal_stage_preflight")
    mark_motion_artifacts_stale(blackboard, "subgoal_advanced", include_goal=True)
    blackboard.write("task_plan", task_plan, event_type="scheduler.advance_subgoal_plan")
    blackboard.write("current_subgoal", next_subgoal, event_type="scheduler.advance_subgoal")
    return ok(
        "subgoal_advanced",
        {
            "task_plan": task_plan.to_dict(),
            "current_subgoal": next_subgoal.to_dict(),
            "next_stage": "preflight",
            "consumed_verification_subgoal_id": current_subgoal_id,
        },
    )


def _verified_subgoal_id(verification: object) -> str | None:
    metadata = getattr(verification, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("current_subgoal_id") is not None:
        return str(metadata["current_subgoal_id"])
    return None


def _archive_and_clear_consumed_verification(
    blackboard: Blackboard,
    verification: object,
    subgoal_id: str | None,
    reason: str,
) -> None:
    blackboard.write(
        "last_resolved_verification_report",
        {
            "reason": reason,
            "consumed_subgoal_id": subgoal_id,
            "verification": to_dict(verification),
        },
        event_type="scheduler.advance_subgoal_archive_verification",
    )
    blackboard.write("last_verification_report", None, event_type="scheduler.advance_subgoal_clear_verification")


def allocate_budget(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    budget = int(request.payload.get("budget_steps", request.budget_steps or 1))
    budget = max(1, min(budget, 20))
    blackboard.write("last_budget_steps", budget, event_type="scheduler.allocate_budget")
    return ok("budget_allocated", {"budget_steps": budget})


def repair_stage_transition(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    target_stage = request.payload.get("target_stage")
    if target_stage not in {"observe", "plan", "preflight", "recover"}:
        return unavailable(
            "repair_stage_transition_unavailable",
            "invalid_or_missing_target_stage",
            {"target_stage": target_stage, "allowed_target_stages": ["observe", "plan", "preflight", "recover"]},
        )
    reason = request.payload.get("reason") or request.payload.get("error") or "repair_requested"
    payload = {
        "kind": "repair_stage_transition",
        "from_stage": request.stage,
        "target_stage": str(target_stage),
        "reason": reason,
    }
    blackboard.write(
        "last_stage_transition",
        payload,
        event_type="scheduler.repair_stage_transition_request",
    )
    if target_stage == "plan":
        blackboard.write("task_plan", None, event_type="scheduler.repair_stage_transition_clear_task_plan")
        blackboard.write("current_subgoal", None, event_type="scheduler.repair_stage_transition_clear_current_subgoal")
        mark_motion_artifacts_stale(blackboard, "repair_stage_transition_plan", include_goal=True)
    elif target_stage in {"observe", "preflight", "recover"}:
        mark_motion_artifacts_stale(blackboard, f"repair_stage_transition_{target_stage}", include_goal=target_stage == "observe")
        if target_stage == "observe":
            _clear_visual_state_for_reobserve(blackboard, str(reason))
    if target_stage in {"observe", "plan", "preflight"} and blackboard.read("last_verification_report") is not None:
        blackboard.write(
            "last_resolved_verification_report",
            to_dict(blackboard.read("last_verification_report")),
            event_type="scheduler.repair_stage_transition_archive_verification",
        )
        blackboard.write(
            "last_verification_report",
            None,
            event_type="scheduler.repair_stage_transition_clear_active_verification",
        )
    blackboard.write("stage", str(target_stage), event_type="scheduler.repair_stage_transition")
    return ok("repair_stage_transition_requested", payload)


def _clear_visual_state_for_reobserve(blackboard: Blackboard, reason: str) -> None:
    observation = blackboard.read("observation")
    perception = blackboard.read("perception")
    world_state = blackboard.read("world_state")
    blackboard.write(
        "last_reobserve_request",
        {
            "reason": reason,
            "previous_observation_id": current_observation_id(blackboard),
            "previous_perception_observation_id": getattr(perception, "observation_id", None),
            "previous_world_state_observation_id": metadata_value(world_state, "observation_id"),
            "had_observation": observation is not None,
            "had_perception": perception is not None,
            "had_world_state": world_state is not None,
        },
        event_type="scheduler.reobserve_archive_visual_state",
    )
    blackboard.write("observation", None, event_type="scheduler.reobserve_clear_observation")
    blackboard.write("perception", None, event_type="scheduler.reobserve_clear_perception")
    blackboard.write("world_state", None, event_type="scheduler.reobserve_clear_world_state")
    blackboard.write("grounding_overlay", None, event_type="scheduler.reobserve_clear_grounding_overlay")


def _skill_schema() -> dict[str, object]:
    return {
        "next_component": "string",
        "next_skill": "string",
        "reason": "short string",
        "narration": "one short visible sentence explaining the next action",
        "state_summary": "one short visible sentence describing the key current state",
        "expected_result": "one short visible sentence describing what the skill should produce",
        "stage": "string or null",
        "budget_steps": "integer or null",
        "payload": "object",
    }


def _loop_schema() -> dict[str, object]:
    return {
        "control": "run_skill|advance_stage|finish_run",
        "stage": (
            "For run_skill: current_stage. For advance_stage: null or omitted; runtime advances to the default next stage. "
            "For finish_run: null."
        ),
        "next_component": "required exact component key from allowed_skills when control=run_skill; null otherwise",
        "next_skill": "required exact skill under next_component when control=run_skill; null otherwise",
        "payload": "object",
        "reason": "short string",
        "narration": "one short visible sentence explaining the next action",
        "state_summary": "one short visible sentence describing the key current state",
        "expected_result": "one short visible sentence describing what the skill should produce",
        "budget_steps": "integer or null",
    }


def _task_plan_instruction() -> str:
    return (
        "Build a complete ordered manipulation subgoal plan from the current scene state to final task success. "
        "The plan must cover the whole task instruction, not only the first useful action. "
        "Each subgoal must include an instruction: one natural-language robot command that can be sent directly "
        "to the short-horizon VLA action policy. The instruction should say the concrete object names and relation, "
        "for example 'move close to the container', 'grasp the container', or 'place the container on the plate'. "
        "Each subgoal must also include completion_criteria.natural_language: one concrete visual success condition "
        "for that exact subgoal. The verifier will use that success condition directly, so do not write placeholders "
        "such as 'what must be true' or vague labels. "
        "The runtime will send this exact instruction to the action policy; do not rely on type/source/target being "
        "converted into a better command later. The final subgoal must make the original instruction true. "
        "Use existing candidate ids only; do not invent objects. "
        "For normal multi-step manipulation, prefer a slightly more detailed plan with four or more subgoals "
        "when the task naturally contains that many short-horizon stages. Keep subgoals reasonably atomic: "
        "approaching an object, grasping it, lifting or transporting it, aligning with a target, placing it, "
        "releasing it, and waiting for stability are separate stages when they matter. Do not combine transport "
        "with placement, release, insertion, pressing, or other terminal effects in one subgoal. "
        "For placement tasks, final success requires the object to be released or otherwise visibly stable on/in "
        "the target, not merely held above or touching the target while the gripper still supports it. "
        "Represent that as a separate release/wait_for_stability subgoal when the action policy may need a clean "
        "short-horizon command for it, or make the final placement instruction explicitly include release and stability. "
        "Do not force every task into a place sequence: choose subgoal types that match the task, such as "
        "approach, grasp, lift, transport, align, place, release, press, open, close, pull, push, insert, pour, "
        "stack, handover, fold, wipe, or wait_for_stability. For contact or articulation tasks, include the "
        "reaching/contact stage and the terminal press, open, close, pull, or push stage. A one-subgoal plan is "
        "invalid for normal manipulation tasks unless that single subgoal itself fully completes the entire "
        "instruction from the current state."
    )


def _task_plan_schema(source_id: str, target_id: str) -> dict[str, Any]:
    return {
        "task": "copy the task instruction",
        "subgoals": [
            {
                "subgoal_id": "S1",
                "type": "first short-horizon stage, e.g. approach or grasp",
                "instruction": "one natural-language command for S1 to send directly to the VLA",
                "source_candidate_id": source_id,
                "target_candidate_id": None,
                "status": "pending",
                "completion_criteria": {
                    "natural_language": "concrete visible success condition for S1, not a placeholder"
                },
            },
            {
                "subgoal_id": "S2",
                "type": "next task-specific stage, e.g. lift, transport, press, open, pour, place",
                "instruction": "one natural-language command for S2 to send directly to the VLA",
                "source_candidate_id": source_id,
                "target_candidate_id": target_id,
                "status": "pending",
                "completion_criteria": {
                    "natural_language": "concrete visible success condition for S2, not a placeholder"
                },
            },
            {
                "subgoal_id": "S3",
                "type": "next atomic stage, e.g. lift, transport, align, press, open, pour",
                "instruction": "one natural-language command for S3 to send directly to the VLA",
                "source_candidate_id": source_id,
                "target_candidate_id": target_id,
                "status": "pending",
                "completion_criteria": {
                    "natural_language": "concrete visible success condition for S3, not a placeholder"
                },
            },
            {
                "subgoal_id": "S4",
                "type": "optional or terminal atomic stage that makes the whole instruction true",
                "instruction": "one natural-language command for S4 to send directly to the VLA",
                "source_candidate_id": source_id,
                "target_candidate_id": target_id,
                "status": "pending",
                "completion_criteria": {
                    "natural_language": "concrete visible success condition for S4 or the whole task, not a placeholder"
                },
            },
        ],
        "current_subgoal_id": "S1",
        "status": "pending",
    }


def _validate_task_plan_completeness(
    plan: TaskPlan,
    task_instruction: str | None,
    source_id: str,
    target_id: str,
) -> list[str]:
    _ = task_instruction  # Task-text keyword completeness checks are intentionally disabled.
    errors: list[str] = []
    if not plan.subgoals:
        return ["empty_subgoals_in_model_output"]

    subgoal_ids = [subgoal.subgoal_id for subgoal in plan.subgoals]
    if len(set(subgoal_ids)) != len(subgoal_ids):
        errors.append("duplicate_subgoal_id")
    if plan.current_subgoal_id and plan.current_subgoal_id not in set(subgoal_ids):
        errors.append("current_subgoal_id_not_in_subgoals")

    for subgoal in plan.subgoals:
        instruction = str(getattr(subgoal, "instruction", "") or "").strip()
        if not instruction:
            errors.append(f"missing_subgoal_instruction:{subgoal.subgoal_id}")
        criteria_error = _completion_criteria_error(subgoal)
        if criteria_error is not None:
            errors.append(criteria_error)

    if not any(subgoal.source_candidate_id == source_id for subgoal in plan.subgoals):
        errors.append("missing_source_candidate_in_subgoals")
    if (
        target_id
        and target_id != source_id
        and not any(subgoal.target_candidate_id == target_id for subgoal in plan.subgoals)
    ):
        errors.append("missing_target_candidate_in_subgoals")

    return errors


def _completion_criteria_error(subgoal: Subgoal) -> str | None:
    criteria = getattr(subgoal, "completion_criteria", None)
    if not isinstance(criteria, dict):
        return f"missing_completion_criteria:{subgoal.subgoal_id}"
    natural_language = criteria.get("natural_language")
    if not isinstance(natural_language, str) or not natural_language.strip():
        return f"missing_natural_language_completion_criteria:{subgoal.subgoal_id}"
    if _looks_like_completion_placeholder(natural_language):
        return f"placeholder_completion_criteria:{subgoal.subgoal_id}"
    return None


def _looks_like_completion_placeholder(value: str) -> bool:
    text = _norm_text(value)
    placeholders = {
        "what must be true",
        "success condition",
        "final success condition",
        "concrete visible success condition",
        "not a placeholder",
        "after s1",
        "after s2",
        "after s3",
        "after s4",
    }
    return any(placeholder in text for placeholder in placeholders)


def _norm_text(value: object) -> str:
    return str(value or "").lower().replace("-", "_")


def _first_subgoal_with_status(task_plan: TaskPlan, statuses: set[str]) -> Subgoal | None:
    for subgoal in task_plan.subgoals:
        if subgoal.status in statuses:
            return subgoal
    return None


def _scheduler_instruction(loop_mode: bool) -> str:
    if not loop_mode:
        return "You are the scheduler for a multi-component robot manipulation agent."
    return (
        "You are the scheduler for a multi-component robot manipulation agent. "
        "Choose exactly one next control action. Use run_skill to call an allowed component skill, "
        "advance_stage only for the default next stage, or finish_run when the round should stop. "
        "Do not use advance_stage for repair jumps, backward jumps, or skip-stage jumps. "
        "Only choose skills listed under allowed_skills for "
        "the current stage. The stage_order field only lists legal stage names; it is not a skill list. "
        "For control=run_skill, next_component and next_skill are both required: "
        "next_component must be an exact key in allowed_skills, and next_skill must be one of that "
        "component's listed skills. Never output null, None, or an empty string for next_component or "
        "next_skill when control=run_skill. For control=run_skill, stage must be the current_stage. "
        "The stage field in run_skill is the stage you are currently in, not the stage you want to enter. "
        "If you want to enter the default next stage, use control=advance_stage with stage=null; do not "
        "set run_skill.stage to that next stage. "
        "For control=advance_stage, stage must be null or omitted; the runtime chooses only the default next stage. "
        "For control=finish_run, stage, next_component, and "
        "next_skill must be null. Do not invent components or skills. Use recent_loop_history as conversation "
        "history for the run. You may inspect the attached current observation images to decide whether "
        "the scene is visible, whether a fresh observation is needed, and whether the previous visual "
        "evidence still supports advancing. Do not create new object ids from the images; call vision "
        "skills when object detection, grounding, or uncertainty must be updated. If a skill returned "
        "metric_geometry_unavailable, unavailable, failed, or no new evidence, treat that capability as "
        "currently unavailable and do not retry it unless the inputs changed; choose a different skill, "
        "advance the stage, or finish the run. If blackboard contains last_skill_exception, "
        "last_localization_error, or bootstrap_observe_failures, explicitly account "
        "for that failure in state_summary and choose a corrective next skill instead of pretending the "
        "failed skill succeeded. Respect runtime_state and allowed_skills: if a required artifact is missing or stale "
        "for the current stage, choose a listed skill that can produce it instead of skipping ahead. Do not infer or "
        "invent a skill name from an artifact name. runtime_state is the latest state and overrides older "
        "recent_loop_history entries when they disagree. If runtime_state.next_required_decision is not null, "
        "output that control/stage/next_component/next_skill/payload exactly unless it conflicts with allowed_skills. "
        "Do not use stale verification text from recent_loop_history to override next_required_decision. "
        "In observe, runtime_state.observe_complete and runtime_state.world_state_ready are authoritative. "
        "If runtime_state.observation_present is false, choose vision.capture_views first; do not choose "
        "perceive_scene, localize_task_objects, state.update_world_state, or any non-observe skill before images exist. "
        "If either shows the task objects are already bound and fresh, choose control=advance_stage with stage=null. "
        "Do not call localize_task_objects again when runtime_state.world_state_source_candidate_id and "
        "runtime_state.world_state_target_candidate_id are both set. Metric geometry is optional evidence for this "
        "agent and is not required to leave observe because the PI0.5/OpenPI execution backend receives images. "
        "Do not call lift_depth_cluster or lift_geometry merely because metric_geometry is unavailable. "
        "In observe, the visual state is built in strict semantic order: capture_views obtains images; "
        "perceive_scene detects candidate objects only and does not produce source_candidate_id or target_candidate_id; "
        "localize_task_objects is the required skill that binds top-level source_candidate_id and target_candidate_id; "
        "update_world_state should run only after those top-level bindings exist in perception. If perception has "
        "candidates but source_candidate_id or target_candidate_id is null, choose vision.localize_task_objects, "
        "not state.update_world_state and not another perceive_scene unless the existing candidates are empty or stale. "
        "When runtime_state.observe_complete is true, observe is complete: choose control=advance_stage with stage=null. "
        "Do not run safety.preflight_action in observe; preflight checks are only legal after observe advances through plan "
        "to the preflight stage. Do not repeat perceive_scene or localize_task_objects after observe_complete is true. "
        "Use scheduler.repair_stage_transition only after an explicit failed or blocked previous step. "
        "Its payload must include target_stage and reason. Use target_stage=observe only for verification next_action=reobserve "
        "or missing visual evidence outside preflight. Preflight visual errors must stay in preflight and use "
        "vision.refresh_preflight_observation. Use target_stage=plan "
        "for invalid task_plan, invalid current_subgoal, or bad object binding. Use target_stage=preflight for "
        "another motion attempt after verification next_action=continue_execute or after a retry request. "
        "Use target_stage=recover only for execution or verification failures that need recovery.decide_recovery. "
        "In plan, the only required planning artifacts are task_plan and current_subgoal. If runtime_state.task_plan_present "
        "is false, choose scheduler.build_task_plan. Else if runtime_state.current_subgoal_present is false, choose "
        "scheduler.select_current_subgoal. scheduler.allocate_budget is optional and should be used at most once; if "
        "runtime_state.budget_allocated is true, do not call scheduler.allocate_budget again. Once runtime_state.plan_ready "
        "is true, choose control=advance_stage with stage=null. Missing motion_goal, motion_plan, or action_chunk is "
        "expected in plan and must not be fixed in plan. build_motion_goal is not a scheduler skill, and no build_motion_goal "
        "skill is legal in plan. Motion artifacts are created later, after preflight, in execute, using allowed motion skills. "
        "After task_plan and current_subgoal are ready in plan, choose control=advance_stage with stage=null; never jump "
        "directly from plan to execute. "
        "Preflight is mandatory before every motion attempt. In preflight, run safety.preflight_action "
        "when runtime_state.preflight_error is missing_preflight_report_before_execute or "
        "stale_preflight_report. When runtime_state.preflight_ready is true, choose exactly "
        "control=advance_stage with stage=null to enter execute; do not run safety.preflight_action again, "
        "do not choose any motion skill in preflight, and do not set run_skill.stage to execute. "
        "If preflight_report failed with stale_perception, "
        "stale_world_state, world_state_requires_reobserve, missing observation, missing_observation_id, or camera errors, "
        "do not rerun preflight_action and do not jump to observe: choose vision.refresh_preflight_observation while staying "
        "in preflight. After vision.refresh_preflight_observation succeeds, the old preflight_report may still show the old "
        "stale errors; if runtime_state.visual_state_fresh_for_current_observation is true, run safety.preflight_action once "
        "to issue a fresh preflight report instead of refreshing again. "
        "Choose scheduler.repair_stage_transition target_stage=plan for missing/current-subgoal mismatch, missing source/target, or "
        "bad object binding; choose target_stage=recover for execution or backend failures that need recovery; "
        "finish only when the blocking error is not recoverable within this run. Do not choose motion skills "
        "while still in preflight. In execute, if runtime_state.preflight_ready is false, choose "
        "scheduler.repair_stage_transition target_stage=preflight before any motion skill; otherwise build or rebuild the earliest missing/stale motion artifact "
        "in this order: motion_goal, motion_plan, action_chunk, validate_action_chunk, execute_action. "
        "If last_action_validation_report failed in execute, rebuild the stale or invalid motion artifacts "
        "from the earliest invalid artifact instead of retrying execute_action unchanged. "
        "Task planning artifacts are task_plan then current_subgoal. Motion artifacts are motion_goal, "
        "motion_plan, action_chunk, then execute_action. When choosing motion.emit_action_chunk, payload "
        "must include an integer horizon from 10 to 50. Use horizon=10 for precise grasp, place, release, or "
        "stability checks; horizon=20 for approach, align, lift, press, open, close, pull, push, insert, or pour; "
        "horizon=30 for normal transport; use up to horizon=50 only for clearly longer moves. "
        "After execute_action succeeds, verify the current "
        "subgoal before advancing, continuing execution, replanning, reobserving, recovering, or finishing. "
        "In verify, first capture fresh verification images with vision.capture_verify_views when "
        "runtime_state.verify_observation_fresh is false. capture_verify_views is not observe: it only captures "
        "the current post-execution camera views for verification and must not perceive, localize, plan, preflight, "
        "or change the current subgoal. After runtime_state.verify_observation_fresh is true, choose "
        "verifier.verify_progress exactly once for that execution result. verifier.verify_progress must use the "
        "fresh verify images; do not verify from text-only execution_report when verify images are missing. "
        "When current_stage is verify and runtime_state.verification_present is true, do not run more "
        "vision, state, verifier, or recovery skills inside verify. Obey runtime_state.verification_next_action exactly: "
        "advance_subgoal means output control=run_skill, stage=verify, next_component=scheduler, "
        "next_skill=advance_subgoal. continue_execute means scheduler.repair_stage_transition target_stage=preflight. "
        "reobserve means scheduler.repair_stage_transition target_stage=observe; this applies to verify results, not "
        "preflight stale/missing observation errors. "
        "replan means scheduler.repair_stage_transition target_stage=plan. "
        "recover means scheduler.repair_stage_transition target_stage=recover; this is only for true execution "
        "failures, not normal not_done progress. "
        "finish means output control=finish_run with stage=null. "
        "In recover, run recovery.decide_recovery, then recovery.build_retry_request. recovery.decide_recovery must "
        "produce a concrete recovery patch, not a generic route. After a retry request exists, use "
        "scheduler.repair_stage_transition target_stage=preflight for a patched motion retry, "
        "target_stage=observe for fresh evidence, target_stage=plan for a new plan, or finish_run if the "
        "retry_request stage is finish. Never go directly from recover to execute. "
        "Always include narration, state_summary, and expected_result as concise visible trace text."
    )
