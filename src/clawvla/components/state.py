from __future__ import annotations

from ..blackboard import Blackboard
from ..schema import PerceptionResult, SkillRequest, SkillResult, WorldRelation, WorldState
from ..skills.base import SkillContext, SkillRegistry
from .skill_helpers import get_attr, ok, register_skill, to_dict


def register_state_skills(registry: SkillRegistry) -> None:
    register_skill(registry, "state", "update_world_state", "Update blackboard world state from perception.", update_world_state)
    register_skill(registry, "state", "track_object_identity", "Track object identity across observations.", track_object_identity)
    register_skill(registry, "state", "infer_relations", "Infer spatial/task relations from world state.", infer_relations)
    register_skill(registry, "state", "summarize_state", "Summarize compact state for scheduling/model prompts.", summarize_state)


def update_world_state(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    perception = blackboard.read("perception")
    perception_was_missing = not isinstance(perception, PerceptionResult)
    if not isinstance(perception, PerceptionResult):
        perception = PerceptionResult(
            observation_id=get_attr(blackboard.read("observation"), "observation_id"),
            metadata={"mode": "placeholder_perception", "reason": "perception_missing_before_world_state_update"},
        )
    perception_is_placeholder = perception_was_missing or "placeholder" in str(perception.metadata).lower()
    if not perception.candidates:
        perception.metadata.setdefault("world_state_update_note", "empty_perception_candidates")
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
            "placeholder": bool(perception_is_placeholder or not perception.candidates),
            "placeholder_reason": "empty_or_placeholder_perception" if perception_is_placeholder or not perception.candidates else None,
        },
    )
    blackboard.write("world_state", world_state, event_type="state.update_world_state")
    blackboard.write("stage", world_state.stage)
    status = "world_state_updated_from_placeholder_perception" if world_state.metadata["placeholder"] else "world_state_updated"
    return ok(status, {"world_state": world_state.to_dict()})


def track_object_identity(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    world_state = blackboard.read("world_state")
    if world_state is not None:
        world_state.metadata.setdefault("identity_tracking", {"status": "placeholder"})
        blackboard.write("world_state", world_state, event_type="state.track_object_identity")
    return ok("identity_tracking_placeholder", {"world_state": to_dict(world_state)})


def infer_relations(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    world_state = blackboard.read("world_state")
    if world_state is not None:
        world_state.relations = _merge_relations(world_state.relations, _basic_relations_from_world(world_state))
        blackboard.write("world_state", world_state, event_type="state.infer_relations")
    return ok("relations_placeholder", {"world_state": to_dict(world_state)})


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


def _basic_relations_from_world(world_state: WorldState) -> list[WorldRelation]:
    source = world_state.candidate_by_id(world_state.source_candidate_id)
    target = world_state.candidate_by_id(world_state.target_candidate_id)
    if source is None or target is None:
        return []
    relations = _basic_relations(
        PerceptionResult(
            source_candidate_id=source.candidate_id,
            target_candidate_id=target.candidate_id,
        )
    )
    if source.metric_geometry.has_position and target.metric_geometry.has_position:
        relations.append(
            WorldRelation(
                relation="source_and_target_have_metric_geometry",
                source_candidate_id=source.candidate_id,
                target_candidate_id=target.candidate_id,
                confidence=min(source.confidence, target.confidence),
                metadata={"source": "metric_geometry_presence"},
            )
        )
    return relations


def _merge_relations(existing: list[WorldRelation], new_relations: list[WorldRelation]) -> list[WorldRelation]:
    merged = {(item.relation, item.source_candidate_id, item.target_candidate_id): item for item in existing}
    for relation in new_relations:
        merged[(relation.relation, relation.source_candidate_id, relation.target_candidate_id)] = relation
    return list(merged.values())
