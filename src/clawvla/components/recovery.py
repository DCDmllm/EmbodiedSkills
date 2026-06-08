from __future__ import annotations

from ..blackboard import Blackboard
from ..schema import SkillRequest, SkillResult
from ..skills.base import SkillContext, SkillRegistry
from .skill_helpers import get_attr, ok, register_skill


def register_recovery_skills(registry: SkillRegistry) -> None:
    register_skill(registry, "recovery", "decide_recovery", "Decide recovery policy from verification.", decide_recovery, True)
    register_skill(registry, "recovery", "build_retry_request", "Build a retry skill request.", build_retry_request)


def decide_recovery(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    verification = blackboard.read("last_verification_report")
    execution_report = blackboard.read("execution_report")
    next_action = _verification_next_action(verification)
    if next_action is None:
        next_action = "replan" if get_attr(execution_report, "status") != "action_executed" else "recover"
    directive = _directive_for_next_action(next_action)
    directive.update(
        {
            "reason": get_attr(verification, "failure_type", None) or get_attr(execution_report, "reason", "unknown"),
            "verification": _compact_verification(verification),
            "execution_status": get_attr(execution_report, "status"),
            "source": "recovery.decide_recovery",
        }
    )
    blackboard.write("last_recovery_directive", directive, event_type="recovery.decide_recovery")
    return ok("recovery_route_decided", {"recovery_directive": directive})


def build_retry_request(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    directive = blackboard.read("last_recovery_directive", {})
    if not isinstance(directive, dict):
        directive = {}
    retry_request = {
        "stage": directive.get("stage"),
        "component": directive.get("component"),
        "skill": directive.get("skill"),
        "control": directive.get("control", "run_skill"),
        "next_action": directive.get("next_action"),
        "reason": directive.get("reason"),
    }
    blackboard.write("last_retry_request", retry_request, event_type="recovery.build_retry_request")
    return ok("retry_request_built", {"retry_request": retry_request})


def _verification_next_action(verification: object | None) -> str | None:
    metadata = get_attr(verification, "metadata", {})
    if isinstance(metadata, dict) and metadata.get("next_action"):
        return str(metadata["next_action"])
    if get_attr(verification, "success", False):
        return "advance_subgoal"
    if get_attr(verification, "should_reobserve", False):
        return "reobserve"
    return None


def _directive_for_next_action(next_action: str) -> dict[str, object]:
    routes = {
        "advance_subgoal": {"stage": "verify", "component": "scheduler", "skill": "advance_subgoal"},
        "continue_execute": {"stage": "execute", "component": "motion", "skill": "build_motion_goal"},
        "reobserve": {"stage": "observe", "component": "vision", "skill": "capture_views"},
        "replan": {"stage": "plan", "component": "scheduler", "skill": "build_task_plan"},
        "recover": {"stage": "recover", "component": "recovery", "skill": "build_retry_request"},
        "finish": {"stage": "verify", "component": None, "skill": None, "control": "finish_run"},
    }
    directive = dict(routes.get(next_action, routes["recover"]))
    directive.setdefault("control", "run_skill")
    directive["next_action"] = next_action if next_action in routes else "recover"
    return directive


def _compact_verification(verification: object | None) -> dict[str, object] | None:
    if verification is None:
        return None
    metadata = get_attr(verification, "metadata", {})
    return {
        "success": get_attr(verification, "success"),
        "failure_type": get_attr(verification, "failure_type"),
        "should_reobserve": get_attr(verification, "should_reobserve"),
        "next_action": metadata.get("next_action") if isinstance(metadata, dict) else None,
        "task_success": metadata.get("task_success") if isinstance(metadata, dict) else None,
    }
