from __future__ import annotations

from pathlib import Path
from typing import Any

from ..blackboard_utils import mark_grounding_overlay_stale, mark_motion_artifacts_stale
from ..blackboard import Blackboard
from ..model_calls import call_component_json
from ..notices import emit_runtime_event
from ..schema import GroundingOverlay, PerceptionResult, SkillRequest, SkillResult
from ..skills.base import SkillContext, SkillRegistry
from .skill_helpers import get_attr, ok, register_skill, to_dict, unavailable
from .vision_geometry import LiftGeometryConfig, lift_perception_geometry


def register_vision_skills(registry: SkillRegistry) -> None:
    register_skill(registry, "vision", "capture_views", "Collect public camera/depth/proprioception observations.", capture_views)
    register_skill(registry, "vision", "perceive_scene", "Produce visual candidates from the current observation.", perceive_scene)
    register_skill(registry, "vision", "localize_task_objects", "Localize task-relevant objects with bbox evidence.", localize_task_objects)
    register_skill(registry, "vision", "ground_task_objects", "Bind task language to visual candidates.", ground_task_objects)
    register_skill(registry, "vision", "render_grounding_overlay", "Render bbox/role overlays for VLM-side prompts.", render_grounding_overlay)
    register_skill(
        registry,
        "vision",
        "lift_depth_cluster",
        "Optionally attach metric depth/pointcloud evidence to candidates.",
        lift_depth_cluster,
        metadata={
            "optional": True,
            "requires_any": ["depth+intrinsics+extrinsics+bbox", "pointcloud_ref+rough_position"],
            "writes": ["candidate.metric_geometry"],
        },
    )
    register_skill(
        registry,
        "vision",
        "lift_geometry",
        "Compatibility alias for lift_depth_cluster.",
        lift_geometry,
        metadata={"optional": True, "alias_for": "lift_depth_cluster"},
    )
    register_skill(registry, "vision", "bind_arm", "Resolve image-side arm references to robot arms.", bind_arm)
    register_skill(registry, "vision", "estimate_uncertainty", "Estimate visual uncertainty and reobserve needs.", estimate_uncertainty)


def capture_views(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    env = blackboard.read("env_adapter")
    try:
        observation = (
            env.capture_views(**request.payload)
            if env is not None and hasattr(env, "capture_views")
            else request.payload.get("observation")
        )
    except Exception as exc:
        report = {
            "backend": "robotwin" if env is not None else "none",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "task_env_bound": bool(getattr(env, "session", None) and getattr(env.session, "task_env", None)),
        }
        blackboard.write("last_observation_error", report, event_type="vision.capture_views_unavailable")
        return unavailable("observation_unavailable", str(exc), {"capture_error": report})
    mark_grounding_overlay_stale(blackboard, "new_observation_captured")
    mark_motion_artifacts_stale(blackboard, "new_observation_captured", include_goal=True)
    blackboard.write("observation", observation, event_type="vision.capture_views")
    return ok("observation_captured", {"observation": to_dict(observation)})


def perceive_scene(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    existing = blackboard.read("perception")
    existing_perception = existing if isinstance(existing, PerceptionResult) else None
    observation_id = get_attr(blackboard.read("observation"), "observation_id")
    if existing_perception is not None and existing_perception.observation_id != observation_id:
        existing_perception = None
    if context.has_model and request.payload.get("use_model", False):
        payload = {
            "task_instruction": blackboard.task_instruction,
            "observation_context": _compact_observation_context(blackboard.read("observation")),
            "current_candidates": _compact_candidates(existing_perception) if existing_perception else [],
            "current_source_candidate_id": existing_perception.source_candidate_id if existing_perception else None,
            "current_target_candidate_id": existing_perception.target_candidate_id if existing_perception else None,
            "required_schema": {
                "candidates": [
                    {
                        "candidate_id": "C1",
                        "label": "short visual label or null",
                        "bbox_by_view": {"camera_name": [0, 0, 0, 0]},
                        "visibility": "yes|partial|no|uncertain",
                        "confidence": 0.0,
                        "status": "short status",
                    }
                ],
                "uncertainty": {"needs_reobserve": False, "reasons": []},
            },
        }
        raw = call_component_json(
            context,
            instruction=(
                "Detect task-relevant visual candidates from public robot observations. STRICT OUTPUT: the "
                "top-level JSON object must contain exactly the keys candidates and uncertainty. The first "
                "top-level key must be candidates. Do not wrap the answer in perception/result/output/data. "
                "Do not echo inputs. Keep stable candidate ids when current_candidates are provided. Do not "
                "use [0, 0, 0, 0] as a real bbox; if an object is not visible, leave that view absent. If "
                "current_source_candidate_id/current_target_candidate_id are already set and the new "
                "observation does not clearly contradict them, preserve those ids. Do not replace an already "
                "grounded container/plate with visibility=no, bbox=[0,0,0,0], and null source/target."
            ),
            payload=payload,
            image_paths=request.payload.get("image_paths"),
            render_format=request.payload.get("render_format", "json"),
        )
        if not isinstance(raw.get("candidates"), list):
            _emit_invalid_model_output("vision.perceive_scene", "missing_candidates_in_model_output", raw)
            if existing_perception is not None:
                blackboard.write(
                    "last_perception_error",
                    {"reason": "missing_candidates_in_model_output", "raw_keys": sorted(str(key) for key in raw.keys())},
                    event_type="vision.perceive_scene_invalid_output",
                )
                return ok(
                    "scene_perception_preserved_existing_after_invalid_model_output",
                    {"perception": existing_perception.to_dict(), "raw_keys": sorted(str(key) for key in raw.keys())},
                )
            return unavailable(
                "scene_perception_invalid_model_output",
                "missing_candidates_in_model_output",
                {"raw_keys": sorted(str(key) for key in raw.keys())},
            )
        perception = PerceptionResult.from_payload(raw)
        if perception.observation_id is None:
            perception.observation_id = observation_id
        perception, preserved_count = _merge_perception_update(existing_perception, perception)
        perception.metadata["source"] = "vision_model"
        if preserved_count:
            perception.metadata["preserved_existing_candidates"] = preserved_count
        blackboard.write("perception", perception, event_type="vision.perceive_scene")
        status = "scene_perceived_by_model" if not preserved_count else "scene_perceived_by_model_merged_existing"
        return ok(status, {"perception": perception.to_dict(), "preserved_existing_candidates": preserved_count})

    perception = blackboard.read("perception") or PerceptionResult(
        observation_id=get_attr(blackboard.read("observation"), "observation_id"),
        metadata={"mode": "placeholder_perception"},
    )
    blackboard.write("perception", perception, event_type="vision.perceive_scene")
    return ok("scene_perceived_placeholder", {"perception": to_dict(perception)})


def localize_task_objects(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    perception = _ensure_perception(blackboard)
    if context.has_model and request.payload.get("use_model", True):
        raw = call_component_json(
            context,
            instruction=(
                "Refine task-relevant object localization for the task. Output only the required schema object. "
                "Do not echo task_instruction, observation, current_perception, required_schema, or any input object. "
                "Top-level keys must be candidates, source_candidate_id, target_candidate_id, and uncertainty. "
                "Keep stable candidate ids when possible. Return bbox_by_view for visible candidates. "
                "Do not use [0, 0, 0, 0] as a real bbox; if an object is not visible, leave that view absent."
            ),
            payload={
                "task_instruction": blackboard.task_instruction,
                "observation_context": _compact_observation_context(blackboard.read("observation")),
                "current_candidates": _compact_candidates(perception),
                "required_schema": {
                    "candidates": [
                        {
                            "candidate_id": "C1",
                            "label": "object label",
                            "bbox_by_view": {"camera_name": [0, 0, 0, 0]},
                            "visibility": "yes|partial|no|uncertain",
                            "confidence": 0.0,
                            "status": "localized object status",
                        }
                    ],
                    "source_candidate_id": "candidate id or null",
                    "target_candidate_id": "candidate id or null",
                    "uncertainty": {"needs_reobserve": False, "reasons": []},
                },
            },
            image_paths=request.payload.get("image_paths"),
            render_format=request.payload.get("render_format", "json"),
        )
        if not isinstance(raw.get("candidates"), list):
            _emit_invalid_model_output("vision.localize_task_objects", "missing_candidates_in_model_output", raw)
            blackboard.write(
                "last_localization_error",
                {"reason": "missing_candidates_in_model_output", "raw_keys": sorted(str(key) for key in raw.keys())},
                event_type="vision.localize_task_objects_invalid_output",
            )
            return unavailable(
                "localization_invalid_model_output",
                "missing_candidates_in_model_output",
                {"raw_keys": sorted(str(key) for key in raw.keys()), "existing_perception": perception.to_dict()},
            )
        localized = PerceptionResult.from_payload(raw)
        if not localized.candidates:
            _emit_invalid_model_output("vision.localize_task_objects", "empty_candidates_in_model_output", raw)
            blackboard.write(
                "last_localization_error",
                {"reason": "empty_candidates_in_model_output", "raw_keys": sorted(str(key) for key in raw.keys())},
                event_type="vision.localize_task_objects_invalid_output",
            )
            return unavailable(
                "localization_invalid_model_output",
                "empty_candidates_in_model_output",
                {"raw_keys": sorted(str(key) for key in raw.keys()), "existing_perception": perception.to_dict()},
            )
        if localized.observation_id is None:
            localized.observation_id = get_attr(blackboard.read("observation"), "observation_id")
        localized.metadata.update(perception.metadata)
        localized.metadata["localization_source"] = "vision_model"
        blackboard.write("perception", localized, event_type="vision.localize_task_objects")
        return ok("task_objects_localized_by_model", {"perception": localized.to_dict()})

    perception.metadata.setdefault("localization_source", "existing_or_placeholder")
    blackboard.write("perception", perception, event_type="vision.localize_task_objects")
    return ok("task_objects_localized_placeholder", {"perception": perception.to_dict()})


def ground_task_objects(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    perception = _ensure_perception(blackboard)
    if context.has_model and request.payload.get("use_model", False):
        raw = call_component_json(
            context,
            instruction=(
                "Ground the task source object and target object to existing visual candidate ids. "
                "Output only the required schema object. Do not echo task_instruction, perception, "
                "required_schema, or any input object. Top-level keys must be source_candidate_id, "
                "target_candidate_id, and uncertainty. If the task is 'place the container on the plate', "
                "the source is the movable container and the target is the plate."
            ),
            payload={
                "task_instruction": blackboard.task_instruction,
                "candidates": _compact_candidates(perception),
                "required_schema": {
                    "source_candidate_id": "candidate id or null",
                    "target_candidate_id": "candidate id or null",
                    "uncertainty": {"needs_reobserve": False, "reasons": []},
                },
            },
            image_paths=request.payload.get("image_paths"),
            render_format=request.payload.get("render_format", "json"),
        )
        source_id = str(raw.get("source_candidate_id")) if raw.get("source_candidate_id") is not None else None
        target_id = str(raw.get("target_candidate_id")) if raw.get("target_candidate_id") is not None else None
        candidate_ids = {candidate.candidate_id for candidate in perception.candidates}
        if source_id not in candidate_ids:
            source_id = None
        if target_id not in candidate_ids:
            target_id = None
        if source_id is None or target_id is None:
            _emit_invalid_model_output("vision.ground_task_objects", "missing_source_or_target_in_model_output", raw)
            blackboard.write(
                "last_grounding_error",
                {
                    "reason": "missing_source_or_target_in_model_output",
                    "raw_keys": sorted(str(key) for key in raw.keys()),
                    "candidate_ids": sorted(candidate_ids),
                    "source_candidate_id": source_id,
                    "target_candidate_id": target_id,
                },
                event_type="vision.ground_task_objects_invalid_output",
            )
            return unavailable(
                "task_grounding_invalid_model_output",
                "missing_source_or_target_in_model_output",
                {
                    "raw_keys": sorted(str(key) for key in raw.keys()),
                    "candidate_ids": sorted(candidate_ids),
                    "perception": perception.to_dict(),
                },
            )
        perception.source_candidate_id = source_id
        perception.target_candidate_id = target_id
        if isinstance(raw.get("uncertainty"), dict):
            perception.uncertainty.update(raw["uncertainty"])
        perception.metadata["grounding_source"] = "vision_model"
        blackboard.write("perception", perception, event_type="vision.ground_task_objects")
        return ok("task_objects_grounded_by_model", {"perception": perception.to_dict()})

    perception.source_candidate_id = request.payload.get("source_candidate_id") or perception.source_candidate_id
    perception.target_candidate_id = request.payload.get("target_candidate_id") or perception.target_candidate_id
    blackboard.write("perception", perception, event_type="vision.ground_task_objects")
    return ok("task_objects_grounded_placeholder", {"perception": perception.to_dict()})


def render_grounding_overlay(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    observation = blackboard.read("observation")
    perception = _ensure_perception(blackboard)
    observation_id = get_attr(observation, "observation_id")
    if observation is None or not observation_id:
        return unavailable("grounding_overlay_unavailable", "missing_observation", {})
    if not perception.candidates:
        return unavailable("grounding_overlay_unavailable", "missing_perception_candidates", {"perception": perception.to_dict()})
    env = blackboard.read("env_adapter")
    artifacts = getattr(env, "artifacts", None)
    if artifacts is None or not hasattr(artifacts, "write_image"):
        return unavailable("grounding_overlay_unavailable", "missing_artifact_store", {})

    image_paths: dict[str, str] = {}
    object_refs: dict[str, object] = {}
    prefix = str(request.payload.get("artifact_prefix") or _artifact_prefix(observation) or "grounding")
    for view_name, view in getattr(observation, "camera_views", {}).items():
        if not getattr(view, "rgb_path", None):
            continue
        rendered = _render_view_overlay(str(view.rgb_path), str(view_name), perception)
        if rendered is None:
            continue
        image_paths[str(view_name)] = artifacts.write_image(f"{prefix}/overlays/{view_name}_grounding.png", rendered)

    for candidate in perception.candidates:
        object_refs[candidate.candidate_id] = {
            "label": candidate.label,
            "bbox_by_view": dict(candidate.bbox_by_view),
            "role": _candidate_role(candidate.candidate_id, perception),
        }
    if not image_paths:
        return unavailable("grounding_overlay_unavailable", "no_renderable_bbox_views", {"perception": perception.to_dict()})

    overlay = GroundingOverlay(
        observation_id=str(observation_id),
        image_paths=image_paths,
        object_refs=object_refs,
        stale=False,
        metadata={"source": "vision.render_grounding_overlay", "artifact_prefix": prefix},
    )
    perception.metadata["grounding_overlay"] = overlay.to_dict()
    blackboard.write("perception", perception, event_type="vision.render_grounding_overlay_perception")
    blackboard.write("grounding_overlay", overlay, event_type="vision.render_grounding_overlay")
    return ok("grounding_overlay_rendered", {"grounding_overlay": overlay.to_dict()})


def lift_depth_cluster(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    perception = _ensure_perception(blackboard)
    observation = blackboard.read("observation")
    env = blackboard.read("env_adapter")
    artifacts = getattr(env, "artifacts", None)
    summary = lift_perception_geometry(
        observation,
        perception,
        artifacts=artifacts,
        config=LiftGeometryConfig(
            artifact_prefix=str(request.payload.get("artifact_prefix", "geometry")),
            bbox_padding_px=int(request.payload.get("bbox_padding_px", 2)),
            min_points=int(request.payload.get("min_points", 12)),
        ),
    )
    blackboard.write("perception", perception, event_type="vision.lift_depth_cluster")
    status = "metric_geometry_lifted" if summary.get("lifted_candidates", 0) else "metric_geometry_unavailable"
    return ok(status, {"geometry_summary": summary, "perception": perception.to_dict()})


def lift_geometry(request: SkillRequest, context: SkillContext) -> SkillResult:
    result = lift_depth_cluster(request, context)
    context.blackboard.append_event("vision.lift_geometry_alias", {"alias_for": "lift_depth_cluster"})
    return result


def bind_arm(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    perception = _ensure_perception(blackboard)
    requested = request.payload.get("binding")
    if isinstance(requested, dict):
        perception.arm_binding.update({str(key): str(value) for key, value in requested.items()})
    blackboard.write("perception", perception, event_type="vision.bind_arm")
    return ok("arm_binding_updated", {"arm_binding": dict(perception.arm_binding)})


def estimate_uncertainty(request: SkillRequest, context: SkillContext) -> SkillResult:
    blackboard = context.blackboard
    perception = _ensure_perception(blackboard)
    if context.has_model and request.payload.get("use_model", False):
        raw = call_component_json(
            context,
            instruction="Estimate whether current visual state is reliable enough for scheduling.",
            payload={
                "task_instruction": blackboard.task_instruction,
                "perception": perception.to_dict(),
                "required_schema": {"needs_reobserve": False, "reasons": [], "notes": []},
            },
            image_paths=request.payload.get("image_paths"),
            render_format=request.payload.get("render_format", "json"),
        )
        perception.uncertainty.update(raw)
        blackboard.write("perception", perception, event_type="vision.estimate_uncertainty")
        return ok("uncertainty_estimated_by_model", {"uncertainty": dict(perception.uncertainty)})

    if not perception.candidates:
        perception.uncertainty.setdefault("reasons", []).append("no_visual_candidates")
        perception.uncertainty["needs_reobserve"] = True
    blackboard.write("perception", perception, event_type="vision.estimate_uncertainty")
    return ok("uncertainty_estimated", {"uncertainty": dict(perception.uncertainty)})


def _ensure_perception(blackboard: Blackboard) -> PerceptionResult:
    perception = blackboard.read("perception")
    if isinstance(perception, PerceptionResult):
        return perception
    perception = PerceptionResult(observation_id=get_attr(blackboard.read("observation"), "observation_id"))
    blackboard.write("perception", perception)
    return perception


def _compact_observation_context(observation: object | None) -> dict[str, Any]:
    camera_views = getattr(observation, "camera_views", {})
    if not isinstance(camera_views, dict):
        camera_views = {}
    return {
        "observation_id": get_attr(observation, "observation_id"),
        "camera_views": {
            str(name): {
                "rgb_path": getattr(view, "rgb_path", None),
                "has_depth": bool(getattr(view, "depth_path", None)),
                "has_mask": bool(getattr(view, "mask_path", None)),
            }
            for name, view in camera_views.items()
        },
    }


def _compact_candidates(perception: PerceptionResult) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": candidate.candidate_id,
            "label": candidate.label,
            "bbox_by_view": dict(candidate.bbox_by_view),
            "visibility": candidate.visibility,
            "confidence": candidate.confidence,
            "status": candidate.status,
        }
        for candidate in perception.candidates
    ]


def _emit_invalid_model_output(source: str, reason: str, raw: dict[str, Any]) -> None:
    import json

    raw_preview = json.dumps(raw, ensure_ascii=True)[:1600]
    emit_runtime_event(
        "clawvla_invalid_model_output",
        {
            "source": source,
            "reason": reason,
            "raw_keys": sorted(str(key) for key in raw.keys()),
            "raw_preview": raw_preview,
        },
    )


def _merge_perception_update(
    existing: PerceptionResult | None,
    incoming: PerceptionResult,
) -> tuple[PerceptionResult, int]:
    if existing is None:
        return incoming, 0
    existing_by_id = {candidate.candidate_id: candidate for candidate in existing.candidates}
    incoming_ids = {candidate.candidate_id for candidate in incoming.candidates}
    merged_candidates = []
    preserved_count = 0
    for candidate in incoming.candidates:
        old = existing_by_id.get(candidate.candidate_id)
        if old is not None and _should_preserve_existing_candidate(old, candidate):
            merged_candidates.append(old)
            preserved_count += 1
        else:
            merged_candidates.append(candidate)
    for role_id in [existing.source_candidate_id, existing.target_candidate_id]:
        if role_id and role_id not in incoming_ids and role_id in existing_by_id:
            merged_candidates.append(existing_by_id[role_id])
            incoming_ids.add(role_id)
            preserved_count += 1
    incoming.candidates = merged_candidates
    if incoming.source_candidate_id is None:
        incoming.source_candidate_id = existing.source_candidate_id
    if incoming.target_candidate_id is None:
        incoming.target_candidate_id = existing.target_candidate_id
    if preserved_count:
        incoming.uncertainty = dict(existing.uncertainty)
    incoming.metadata = {**existing.metadata, **incoming.metadata}
    return incoming, preserved_count


def _should_preserve_existing_candidate(old: object, new: object) -> bool:
    if _candidate_has_real_bbox(new):
        return False
    if not _candidate_has_useful_state(old):
        return False
    visibility = str(getattr(new, "visibility", "")).lower()
    status = str(getattr(new, "status", "")).lower()
    confidence = float(getattr(new, "confidence", 0.0) or 0.0)
    return visibility == "no" or "not visible" in status or confidence <= 0.0


def _candidate_has_useful_state(candidate: object) -> bool:
    return (
        _candidate_has_real_bbox(candidate)
        or str(getattr(candidate, "visibility", "")).lower() not in {"", "no"}
        or float(getattr(candidate, "confidence", 0.0) or 0.0) > 0.0
        or str(getattr(candidate, "status", "")).lower() not in {"", "not visible", "candidate"}
    )


def _candidate_has_real_bbox(candidate: object) -> bool:
    bbox_by_view = getattr(candidate, "bbox_by_view", None)
    if not isinstance(bbox_by_view, dict):
        return False
    for bbox in bbox_by_view.values():
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(item) for item in bbox]
        if (x1, y1, x2, y2) != (0.0, 0.0, 0.0, 0.0) and x2 > x1 and y2 > y1:
            return True
    return False


def _artifact_prefix(observation: object | None) -> str | None:
    metadata = getattr(observation, "metadata", {})
    if isinstance(metadata, dict) and metadata.get("artifact_prefix"):
        return str(metadata["artifact_prefix"])
    return None


def _render_view_overlay(image_path: str, view_name: str, perception: PerceptionResult):
    from PIL import Image, ImageDraw

    image = Image.open(Path(image_path)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    rendered_any = False
    for candidate in perception.candidates:
        bbox = candidate.bbox_by_view.get(view_name)
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(item) for item in bbox]
        role = _candidate_role(candidate.candidate_id, perception)
        color = _role_color(role)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{candidate.candidate_id}:{candidate.label or 'object'}"
        if role:
            label = f"{role}:{label}"
        draw.rectangle([x1, max(0, y1 - 16), x1 + max(80, len(label) * 7), y1], fill=(0, 0, 0, 160))
        draw.text((x1 + 2, max(0, y1 - 15)), label, fill=color)
        rendered_any = True
    return image if rendered_any else None


def _candidate_role(candidate_id: str, perception: PerceptionResult) -> str | None:
    if candidate_id == perception.source_candidate_id:
        return "source"
    if candidate_id == perception.target_candidate_id:
        return "target"
    return None


def _role_color(role: str | None) -> tuple[int, int, int, int]:
    if role == "source":
        return (255, 72, 72, 255)
    if role == "target":
        return (72, 220, 120, 255)
    return (255, 210, 64, 255)
