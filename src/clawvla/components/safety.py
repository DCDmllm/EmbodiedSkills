from __future__ import annotations

from ..blackboard import Blackboard
from ..schema import SafetyReport, SkillRequest, SkillResult
from ..skills.base import SkillContext, SkillRegistry
from .skill_helpers import ok, register_skill


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
    blackboard = context.blackboard
    return _placeholder_report(blackboard, "safety.check_reachability", "reachability_not_checked", "requires_robotwin_probe")


def check_workspace(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    return _placeholder_report(blackboard, "safety.check_workspace", "workspace_not_checked", "requires_robot_calibration")


def preflight_action(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    report = SafetyReport(allowed=True, status="preflight_placeholder_allowed", checks={"note": "No hard safety checker is wired yet."})
    blackboard.write("last_safety_report", report, event_type="safety.preflight_action")
    return ok(report.status, {"safety_report": report.to_dict()})


def _placeholder_report(blackboard: Blackboard, event_type: str, status: str, required: str) -> SkillResult:
    report = SafetyReport(allowed=True, status=status, checks={"mode": "placeholder", "required": required})
    blackboard.write("last_safety_report", report, event_type=event_type)
    return ok(report.status, {"safety_report": report.to_dict()})
