from __future__ import annotations

from ..blackboard import Blackboard
from ..schema import PerceptionResult, SkillRequest, SkillResult, WorldRelation, WorldState
from ..skills.base import SkillContext, SkillRegistry
from .skill_helpers import get_attr, ok, register_skill, unavailable


def register_state_skills(registry: SkillRegistry) -> None:
    register_skill(
        registry,
        "state",
        "update_world_state",
        "Write world_state from perception after source_candidate_id/target_candidate_id are bound.",
        update_world_state,
    )
    register_skill(registry, "state", "summarize_state", "Summarize compact state for scheduling/model prompts.", summarize_state)


def update_world_state(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    perception = blackboard.read("perception")
    if not isinstance(perception, PerceptionResult):
        reason = "missing_perception_before_world_state_update"
        blackboard.write(
            "last_state_error",
            {"reason": reason, "observation_id": get_attr(blackboard.read("observation"), "observation_id")},
            event_type="state.update_world_state_unavailable",
        )
        return unavailable(
            "world_state_unavailable",
            reason,
            {"observation_id": get_attr(blackboard.read("observation"), "observation_id")},
        )
    if not perception.candidates:
        reason = "empty_perception_candidates_before_world_state_update"
        blackboard.write(
            "last_state_error",
            {
                "reason": reason,
                "observation_id": perception.observation_id,
                "perception_metadata": dict(perception.metadata),
            },
            event_type="state.update_world_state_unavailable",
        )
        return unavailable(
            "world_state_unavailable",
            reason,
            {"perception": perception.to_dict()},
        )
    perception_is_placeholder = "placeholder" in str(perception.metadata).lower()
    if perception_is_placeholder:
        reason = "placeholder_perception_before_world_state_update"
        blackboard.write(
            "last_state_error",
            {
                "reason": reason,
                "observation_id": perception.observation_id,
                "perception_metadata": dict(perception.metadata),
            },
            event_type="state.update_world_state_unavailable",
        )
        return unavailable(
            "world_state_unavailable",
            reason,
            {"perception": perception.to_dict()},
        )
    world_state = WorldState(
        task_instruction=blackboard.task_instruction,
        geometry_summary=dict(perception.geometry_summary),
        candidates=perception.candidates,
        robot_arms=getattr(blackboard.read("observation"), "robot_arms", {}),
        relations=_basic_relations(perception),
        source_candidate_id=perception.source_candidate_id,
        target_candidate_id=perception.target_candidate_id,
        stage=str(request.payload.get("stage") or blackboard.read("stage") or "observe"),
        needs_reobserve=bool(perception.uncertainty.get("needs_reobserve", False)),
        uncertainty_reasons=[str(item) for item in perception.uncertainty.get("reasons", [])],
        metadata={
            "perception_metadata": perception.metadata,
            "placeholder": False,
            "placeholder_reason": None,
            "observation_id": perception.observation_id,
        },
    )
    blackboard.write("world_state", world_state, event_type="state.update_world_state")
    blackboard.write("stage", world_state.stage)
    return ok("world_state_updated", {"world_state": world_state.to_dict()})


def summarize_state(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    context = blackboard.compact_context()
    blackboard.write("state_summary", context, event_type="state.summarize_state")
    return ok("state_summarized", {"summary": context})


def _basic_relations(perception: PerceptionResult) -> list[WorldRelation]:
    if not perception.source_candidate_id or not perception.target_candidate_id:
        return []
    return [
        WorldRelation(
            relation="task_source_target_pair",
            source_candidate_id=perception.source_candidate_id,
            target_candidate_id=perception.target_candidate_id,
            confidence=0.5,
            metadata={"source": "grounding"},
        )
    ]
