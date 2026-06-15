from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from .schema_utils import dict_of_float_lists, safe_float
from .schema_geometry import MetricGeometry


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class DictSerializable:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StageDefinition(DictSerializable):
    name: str
    description: str = ""
    allowed_components: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillSpec(DictSerializable):
    component: str
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    requires_model: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptView(DictSerializable):
    format: str = "json"
    root_tag: str = "clawvla_context"
    include_images: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillRequest(DictSerializable):
    component: str
    skill: str
    payload: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: _new_id("req"))
    stage: str | None = None
    budget_steps: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResult(DictSerializable):
    success: bool
    status: str
    output: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    component: str | None = None
    skill: str | None = None
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CameraView(DictSerializable):
    name: str
    rgb_path: str | None = None
    depth_path: str | None = None
    mask_path: str | None = None
    intrinsics: list[float] | None = None
    extrinsics: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotArmState(DictSerializable):
    arm_name: str
    eef_pose: list[float] | None = None
    gripper_state: str | None = None
    gripper_value: float | None = None
    joint_positions: list[float] | None = None
    image_side: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservationBundle(DictSerializable):
    observation_id: str = field(default_factory=lambda: _new_id("obs"))
    task_instruction: str | None = None
    camera_views: dict[str, CameraView] = field(default_factory=dict)
    robot_arms: dict[str, RobotArmState] = field(default_factory=dict)
    pointcloud_ref: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneCandidate(DictSerializable):
    candidate_id: str
    label: str | None = None
    role_hypotheses: dict[str, float] = field(default_factory=dict)
    bbox_by_view: dict[str, list[float]] = field(default_factory=dict)
    mask_ref_by_view: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    metric_geometry: MetricGeometry = field(default_factory=MetricGeometry)
    support: dict[str, Any] = field(default_factory=dict)
    visibility: str = "uncertain"
    confidence: float = 0.0
    status: str = "candidate"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any], index: int = 0) -> "SceneCandidate":
        role_hypotheses = payload.get("role_hypotheses")
        if not isinstance(role_hypotheses, dict):
            role_hypotheses = {}
        return cls(
            candidate_id=str(payload.get("candidate_id") or payload.get("id") or f"C{index + 1}"),
            label=str(payload["label"]) if payload.get("label") is not None else None,
            role_hypotheses={str(key): safe_float(value, 0.0) for key, value in role_hypotheses.items()},
            bbox_by_view=dict_of_float_lists(payload.get("bbox_by_view")),
            mask_ref_by_view={str(key): str(value) for key, value in (payload.get("mask_ref_by_view") or {}).items()}
            if isinstance(payload.get("mask_ref_by_view"), dict)
            else {},
            evidence=dict(payload.get("evidence", {})) if isinstance(payload.get("evidence"), dict) else {},
            metric_geometry=MetricGeometry.from_candidate_payload(payload),
            support=dict(payload.get("support", {})) if isinstance(payload.get("support"), dict) else {},
            visibility=str(payload.get("visibility", "uncertain")),
            confidence=safe_float(payload.get("confidence"), 0.0),
            status=str(payload.get("status", "candidate")),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {},
        )


@dataclass
class PerceptionResult(DictSerializable):
    observation_id: str | None = None
    candidates: list[SceneCandidate] = field(default_factory=list)
    source_candidate_id: str | None = None
    target_candidate_id: str | None = None
    arm_binding: dict[str, str] = field(default_factory=dict)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    geometry_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PerceptionResult":
        candidates_payload = payload.get("candidates", [])
        candidates = [
            SceneCandidate.from_payload(item, index=index)
            for index, item in enumerate(candidates_payload)
            if isinstance(item, dict)
        ]
        uncertainty = payload.get("uncertainty")
        return cls(
            observation_id=str(payload["observation_id"]) if payload.get("observation_id") is not None else None,
            candidates=candidates,
            source_candidate_id=(
                str(payload["source_candidate_id"]) if payload.get("source_candidate_id") is not None else None
            ),
            target_candidate_id=(
                str(payload["target_candidate_id"]) if payload.get("target_candidate_id") is not None else None
            ),
            arm_binding={str(key): str(value) for key, value in (payload.get("arm_binding") or {}).items()}
            if isinstance(payload.get("arm_binding"), dict)
            else {},
            uncertainty=uncertainty if isinstance(uncertainty, dict) else {},
            geometry_summary=dict(payload.get("geometry_summary", {})) if isinstance(payload.get("geometry_summary"), dict) else {},
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {},
        )


@dataclass
class Subgoal(DictSerializable):
    subgoal_id: str
    type: str
    instruction: str | None = None
    source_candidate_id: str | None = None
    target_candidate_id: str | None = None
    status: str = "pending"
    completion_criteria: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any], index: int = 0) -> "Subgoal":
        return cls(
            subgoal_id=str(payload.get("subgoal_id") or payload.get("id") or f"S{index + 1}"),
            type=str(payload.get("type") or payload.get("skill") or payload.get("name") or "act"),
            instruction=str(payload["instruction"]).strip() if payload.get("instruction") is not None else None,
            source_candidate_id=(
                str(payload["source_candidate_id"])
                if payload.get("source_candidate_id") is not None
                else str(payload["source"])
                if payload.get("source") is not None
                else None
            ),
            target_candidate_id=(
                str(payload["target_candidate_id"])
                if payload.get("target_candidate_id") is not None
                else str(payload["target"])
                if payload.get("target") is not None
                else None
            ),
            status=str(payload.get("status", "pending")),
            completion_criteria=dict(payload.get("completion_criteria", {}))
            if isinstance(payload.get("completion_criteria"), dict)
            else {},
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {},
        )


@dataclass
class TaskPlan(DictSerializable):
    task: str | None = None
    subgoals: list[Subgoal] = field(default_factory=list)
    current_subgoal_id: str | None = None
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TaskPlan":
        subgoals_payload = payload.get("subgoals", [])
        subgoals = [
            Subgoal.from_payload(item, index=index)
            for index, item in enumerate(subgoals_payload)
            if isinstance(item, dict)
        ]
        current_subgoal_id = (
            str(payload["current_subgoal_id"])
            if payload.get("current_subgoal_id") is not None
            else subgoals[0].subgoal_id
            if subgoals
            else None
        )
        return cls(
            task=str(payload["task"]) if payload.get("task") is not None else None,
            subgoals=subgoals,
            current_subgoal_id=current_subgoal_id,
            status=str(payload.get("status", "pending")),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {},
        )

    def current_subgoal(self) -> Subgoal | None:
        if not self.current_subgoal_id:
            return None
        for subgoal in self.subgoals:
            if subgoal.subgoal_id == self.current_subgoal_id:
                return subgoal
        return None


@dataclass
class GroundingOverlay(DictSerializable):
    observation_id: str
    image_paths: dict[str, str] = field(default_factory=dict)
    object_refs: dict[str, Any] = field(default_factory=dict)
    stale: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldRelation(DictSerializable):
    relation: str
    source_candidate_id: str
    target_candidate_id: str
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldState(DictSerializable):
    world_state_id: str = field(default_factory=lambda: _new_id("world"))
    task_instruction: str | None = None
    scene_summary: str = ""
    geometry_summary: dict[str, Any] = field(default_factory=dict)
    candidates: list[SceneCandidate] = field(default_factory=list)
    robot_arms: dict[str, RobotArmState] = field(default_factory=dict)
    relations: list[WorldRelation] = field(default_factory=list)
    source_candidate_id: str | None = None
    target_candidate_id: str | None = None
    stage: str = "observe"
    needs_reobserve: bool = False
    uncertainty_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def candidate_by_id(self, candidate_id: str | None) -> SceneCandidate | None:
        if not candidate_id:
            return None
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        return None


@dataclass
class SchedulerDecision(DictSerializable):
    next_component: str
    next_skill: str
    reason: str
    payload: dict[str, Any] = field(default_factory=dict)
    stage: str | None = None
    budget_steps: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionGoal(DictSerializable):
    skill: str
    source_candidate_id: str | None = None
    target_candidate_id: str | None = None
    acting_arm: str | None = None
    motion_hint: str | None = None
    target_pose: list[float] | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionChunk(DictSerializable):
    action_type: str = "noop"
    commands: list[list[float]] = field(default_factory=list)
    control_horizon: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyReport(DictSerializable):
    allowed: bool
    status: str
    checks: dict[str, Any] = field(default_factory=dict)
    clipped_motion_goal: MotionGoal | None = None
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationReport(DictSerializable):
    success: bool
    partial_progress: bool = False
    failure_type: str | None = None
    progress_score: float | None = None
    residuals: dict[str, Any] = field(default_factory=dict)
    should_reobserve: bool = False
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
