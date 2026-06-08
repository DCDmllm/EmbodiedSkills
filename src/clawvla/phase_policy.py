from __future__ import annotations

from dataclasses import dataclass, field

from .components import ComponentRegistry


STAGE_ORDER = ["observe", "plan", "preflight", "execute", "verify", "recover"]


DEFAULT_ALLOWED_SKILLS: dict[str, dict[str, list[str]]] = {
    "observe": {
        "vision": [
            "capture_views",
            "perceive_scene",
            "localize_task_objects",
            "ground_task_objects",
            "render_grounding_overlay",
            "lift_depth_cluster",
            "lift_geometry",
            "bind_arm",
            "estimate_uncertainty",
        ],
        "state": ["update_world_state", "track_object_identity", "infer_relations", "summarize_state"],
    },
    "plan": {
        "scheduler": ["build_task_plan", "select_current_subgoal", "advance_subgoal", "allocate_budget", "request_reobserve"],
        "state": ["summarize_state", "infer_relations"],
    },
    "preflight": {
        "safety": [
            "validate_skill_request",
            "validate_arm_binding",
            "check_reachability",
            "check_workspace",
            "preflight_action",
        ],
        "state": ["summarize_state"],
    },
    "execute": {
        "motion": ["build_motion_goal", "plan_motion", "emit_action_chunk", "execute_action"],
        "state": ["summarize_state"],
    },
    "verify": {
        "verifier": ["verify_progress", "score_residual", "diagnose_failure"],
        "state": ["update_world_state", "summarize_state"],
        "vision": ["capture_views", "perceive_scene", "localize_task_objects", "ground_task_objects", "render_grounding_overlay", "estimate_uncertainty"],
        "scheduler": ["advance_subgoal"],
    },
    "recover": {
        "recovery": ["decide_recovery", "build_retry_request"],
        "scheduler": ["request_reobserve"],
        "vision": ["capture_views", "perceive_scene", "localize_task_objects", "ground_task_objects", "render_grounding_overlay", "estimate_uncertainty"],
        "state": ["summarize_state"],
    },
}


@dataclass
class PhasePolicy:
    stage_order: list[str] = field(default_factory=lambda: list(STAGE_ORDER))
    allowed_skills: dict[str, dict[str, list[str]]] = field(default_factory=lambda: dict(DEFAULT_ALLOWED_SKILLS))

    def normalize_stage(self, stage: str | None) -> str:
        if stage in self.stage_order:
            return str(stage)
        return self.stage_order[0]

    def next_stage(self, current_stage: str) -> str:
        if current_stage == "recover":
            return "observe"
        try:
            index = self.stage_order.index(current_stage)
        except ValueError:
            return self.stage_order[0]
        return self.stage_order[min(index + 1, len(self.stage_order) - 1)]

    def allowed_for_stage(self, stage: str) -> dict[str, list[str]]:
        return {name: list(skills) for name, skills in self.allowed_skills.get(stage, {}).items()}

    def full_allowed_skills(self) -> dict[str, dict[str, list[str]]]:
        return {stage: self.allowed_for_stage(stage) for stage in self.stage_order}

    def is_allowed(self, stage: str, component: str, skill: str, components: ComponentRegistry) -> bool:
        if component not in components.names():
            return False
        configured = self.allowed_skills.get(stage, {})
        return skill in configured.get(component, [])
