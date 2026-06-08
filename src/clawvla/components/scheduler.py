from __future__ import annotations

from ..blackboard_utils import mark_motion_artifacts_stale
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
    register_skill(registry, "scheduler", "request_reobserve", "Request reobservation when state is uncertain.", request_reobserve)


def select_next_component(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    world_state = blackboard.read("world_state")
    if world_state is None or getattr(world_state, "needs_reobserve", False):
        decision = SchedulerDecision("vision", "capture_views", "World state is missing or requested reobservation.", stage="observe")
    elif not getattr(world_state, "source_candidate_id", None):
        decision = SchedulerDecision("vision", "ground_task_objects", "Task source is not grounded yet.", stage="observe")
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
            "phase_policy": request.payload.get("phase_policy"),
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
        next_skill = loop_decision.next_skill or raw.get("next_skill")
        inferred_component = _component_for_skill(str(next_skill), payload.get("allowed_skills")) if next_skill is not None else None
        next_component = loop_decision.next_component or raw.get("next_component") or inferred_component
        corrected_loop_decision = LoopDecision(
            control=loop_decision.control,
            stage=loop_decision.stage,
            next_component=str(next_component) if next_component is not None and loop_decision.control == "run_skill" else None,
            next_skill=str(next_skill) if next_skill is not None and loop_decision.control == "run_skill" else None,
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
                "component_inferred": loop_decision.next_component is None and next_component is not None,
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

    decision = blackboard.read("last_scheduler_decision") or SchedulerDecision(
        "vision",
        "capture_views",
        "No previous scheduler decision.",
        stage="observe",
    )
    blackboard.write("last_scheduler_decision", decision, event_type="scheduler.choose_next_skill")
    return ok("next_skill_chosen", {"decision": to_dict(decision)})


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
            instruction=(
                "Build a concise manipulation subgoal plan. Use existing candidate ids only. "
                "Each subgoal should be executable by a short-horizon robot action policy."
            ),
            payload={
                "task_instruction": blackboard.task_instruction,
                "world_state": to_dict(world_state),
                "required_schema": {
                    "task": "task string",
                    "subgoals": [
                        {
                            "subgoal_id": "S1",
                            "type": "approach|grasp|transport|place|release|other",
                            "source_candidate_id": source_id,
                            "target_candidate_id": target_id,
                            "status": "pending",
                            "completion_criteria": {},
                        }
                    ],
                    "current_subgoal_id": "S1",
                    "status": "pending",
                },
            },
            image_paths=request.payload.get("image_paths"),
            render_format=request.payload.get("render_format", "json"),
        )
        plan = TaskPlan.from_payload(raw)
        if plan.subgoals:
            plan.metadata["source"] = "scheduler_model"
        else:
            return unavailable(
                "task_plan_invalid_model_output",
                "empty_subgoals_in_model_output",
                {
                    "raw_keys": sorted(str(key) for key in raw.keys()),
                    "source_candidate_id": source_id,
                    "target_candidate_id": target_id,
                },
            )
    else:
        plan = _template_task_plan(blackboard.task_instruction, source_id, target_id, "template_no_model")

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
    if verification is not None and not getattr(verification, "success", False):
        return unavailable("advance_subgoal_unavailable", "last_verification_not_successful", {"verification": to_dict(verification)})

    for subgoal in task_plan.subgoals:
        if subgoal.subgoal_id == getattr(current, "subgoal_id", None):
            subgoal.status = "succeeded"
            break
    next_subgoal = _first_subgoal_with_status(task_plan, {"pending"})
    if next_subgoal is None:
        task_plan.status = "succeeded"
        blackboard.write("task_plan", task_plan, event_type="scheduler.advance_subgoal_complete")
        blackboard.write("current_subgoal", None, event_type="scheduler.advance_subgoal_complete")
        return ok("task_plan_complete", {"task_plan": task_plan.to_dict(), "current_subgoal": None, "finish_recommended": True})

    next_subgoal.status = "running"
    task_plan.current_subgoal_id = next_subgoal.subgoal_id
    mark_motion_artifacts_stale(blackboard, "subgoal_advanced", include_goal=True)
    blackboard.write("task_plan", task_plan, event_type="scheduler.advance_subgoal_plan")
    blackboard.write("current_subgoal", next_subgoal, event_type="scheduler.advance_subgoal")
    return ok("subgoal_advanced", {"task_plan": task_plan.to_dict(), "current_subgoal": next_subgoal.to_dict()})


def allocate_budget(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    budget = int(request.payload.get("budget_steps", request.budget_steps or 1))
    budget = max(1, min(budget, 20))
    blackboard.write("last_budget_steps", budget, event_type="scheduler.allocate_budget")
    return ok("budget_allocated", {"budget_steps": budget})


def request_reobserve(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    blackboard.write("stage", "observe", event_type="scheduler.request_reobserve")
    return ok("reobserve_requested", {"next_component": "vision", "next_skill": "capture_views"})


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
        "stage": "one of observe|plan|preflight|execute|verify|recover, or null",
        "next_component": "required exact component key from allowed_skills when control=run_skill; null otherwise",
        "next_skill": "required exact skill under next_component when control=run_skill; null otherwise",
        "payload": "object",
        "reason": "short string",
        "narration": "one short visible sentence explaining the next action",
        "state_summary": "one short visible sentence describing the key current state",
        "expected_result": "one short visible sentence describing what the skill should produce",
        "budget_steps": "integer or null",
    }


def _component_for_skill(next_skill: str, allowed_skills: object) -> str | None:
    if not isinstance(allowed_skills, dict):
        return None
    matches = [
        str(component)
        for component, skills in allowed_skills.items()
        if isinstance(skills, list) and next_skill in [str(skill) for skill in skills]
    ]
    return matches[0] if len(matches) == 1 else None


def _template_task_plan(task: str | None, source_id: str, target_id: str, source: str) -> TaskPlan:
    subgoals = [
        Subgoal("S1", "approach", source_candidate_id=source_id, status="pending", completion_criteria={"near": source_id}),
        Subgoal("S2", "grasp", source_candidate_id=source_id, status="pending", completion_criteria={"grasped": source_id}),
        Subgoal(
            "S3",
            "transport",
            source_candidate_id=source_id,
            target_candidate_id=target_id,
            status="pending",
            completion_criteria={"above_target": target_id},
        ),
        Subgoal(
            "S4",
            "place",
            source_candidate_id=source_id,
            target_candidate_id=target_id,
            status="pending",
            completion_criteria={"source_on_target": [source_id, target_id]},
        ),
        Subgoal("S5", "release", source_candidate_id=source_id, status="pending", completion_criteria={"released": source_id}),
    ]
    return TaskPlan(task=task, subgoals=subgoals, current_subgoal_id="S1", status="pending", metadata={"source": source})


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
        "advance_stage to move to another stage when the current stage has enough evidence, or "
        "finish_run when the round should stop. Only choose skills listed under allowed_skills for "
        "the current stage. For control=run_skill, next_component and next_skill are both required: "
        "next_component must be an exact key in allowed_skills, and next_skill must be one of that "
        "component's listed skills. Never output null, None, or an empty string for next_component or "
        "next_skill when control=run_skill. Do not invent components or skills. Use recent_loop_history as conversation "
        "history for the run. You may inspect the attached current observation images to decide whether "
        "the scene is visible, whether a fresh observation is needed, and whether the previous visual "
        "evidence still supports advancing. Do not create new object ids from the images; call vision "
        "skills when object detection, grounding, or uncertainty must be updated. If a skill returned "
        "metric_geometry_unavailable, unavailable, failed, or no new evidence, treat that capability as "
        "currently unavailable and do not retry it unless the inputs changed; choose a different skill, "
        "advance the stage, or finish the run. If blackboard contains last_skill_exception, "
        "last_localization_error, last_grounding_error, or bootstrap_observe_failures, explicitly account "
        "for that failure in state_summary and choose a corrective next skill instead of pretending the "
        "failed skill succeeded. Respect runtime_state and allowed_skills: if a required "
        "artifact is missing or stale, choose a skill that can produce it instead of skipping ahead. "
        "Task planning artifacts are task_plan then current_subgoal. Motion artifacts are motion_goal, "
        "motion_plan, action_chunk, then execute_action. After execute_action succeeds, verify the current "
        "subgoal before advancing, continuing execution, replanning, reobserving, recovering, or finishing. "
        "Always include narration, state_summary, and expected_result as concise visible trace text."
    )
