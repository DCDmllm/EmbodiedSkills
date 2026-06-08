from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..schema import CameraView, ObservationBundle, PerceptionResult, SceneCandidate
from ..schema_geometry import MetricGeometry
from .geometry_utils import (
    MIN_LOCAL_POINTS,
    backproject_view_mask,
    clamp_bbox,
    extent_from_summary,
    float_list,
    float_or_none,
    load_depth,
    load_pointcloud_ref,
    local_points_from_pointcloud_ref,
    merge_points,
    summarize_local_pointcloud,
    summarize_view_geometry,
    unique_strings,
)


@dataclass
class LiftGeometryConfig:
    artifact_prefix: str = "geometry"
    bbox_padding_px: int = 2
    min_points: int = MIN_LOCAL_POINTS


@dataclass
class CandidateGeometry:
    candidate_id: str
    metric_geometry: MetricGeometry

    @property
    def lifted(self) -> bool:
        return self.metric_geometry.available


def lift_perception_geometry(
    observation: ObservationBundle | None,
    perception: PerceptionResult,
    artifacts: Any | None = None,
    config: LiftGeometryConfig | None = None,
) -> dict[str, Any]:
    config = config or LiftGeometryConfig()
    if observation is None:
        return {
            "status": "metric_geometry_unavailable",
            "reason": "missing_observation",
            "retryable": False,
            "lifted_candidates": 0,
            "candidate_count": len(perception.candidates),
        }

    global_support_z = estimate_global_support_z(observation)
    lifted_count = 0
    reports: list[dict[str, Any]] = []
    for candidate in perception.candidates:
        result = lift_candidate_geometry(observation, candidate, global_support_z, artifacts, config)
        if result.lifted:
            apply_candidate_geometry(candidate, result)
            lifted_count += 1
        reports.append(
            {
                "candidate_id": candidate.candidate_id,
                "lifted": result.lifted,
                "point_count": int(result.metric_geometry.pointcloud_local.get("point_count", 0)),
                "geometry_source": list(result.metric_geometry.source),
                "pointcloud_ref": result.metric_geometry.pointcloud_ref,
            }
        )

    perception.geometry_summary = {
        "status": "ok" if lifted_count else "metric_geometry_unavailable",
        "candidate_count": len(perception.candidates),
        "lifted_candidates": lifted_count,
        "global_support_z": global_support_z,
        "candidate_reports": reports,
    }
    if not lifted_count:
        perception.geometry_summary.update(
            {
                "reason": "requires_depth_bbox_or_pointcloud_with_rough_position",
                "retryable": False,
                "available_alternatives": ["image_grounded_motion", "visual_servo", "vla_action_model"],
            }
        )
    perception.metadata["geometry_lift"] = perception.geometry_summary
    return dict(perception.geometry_summary)


def lift_candidate_geometry(
    observation: ObservationBundle,
    candidate: SceneCandidate,
    global_support_z: float | None,
    artifacts: Any | None,
    config: LiftGeometryConfig,
) -> CandidateGeometry:
    points_by_view: dict[str, np.ndarray] = {}
    geometry_views: dict[str, dict[str, Any]] = {}
    sources: list[str] = []

    for view_name, view in observation.camera_views.items():
        mask = candidate_mask_for_view(candidate, view, config.bbox_padding_px)
        if mask is None:
            continue
        points = backproject_view_mask(view, mask)
        if points.shape[0] < config.min_points:
            continue
        points_by_view[view_name] = points
        geometry_views[view_name] = summarize_view_geometry(points)
        sources.append(f"depth:{view_name}")

    points = merge_points(points_by_view.values())
    pointcloud_ref = None
    if points.shape[0] < config.min_points:
        ref_points = local_points_from_pointcloud_ref(observation.pointcloud_ref, candidate.metric_geometry.position_3d)
        if ref_points.shape[0] >= config.min_points:
            points = ref_points
            pointcloud_ref = observation.pointcloud_ref
            sources.append("pointcloud_ref:local_radius")
    if points.shape[0] < config.min_points:
        return CandidateGeometry(candidate.candidate_id, MetricGeometry(source=sources, quality={"reason": "missing_metric_input"}))

    summary = summarize_local_pointcloud(points, global_support_z)
    position_3d = float_list(summary.get("center_xyz"))
    extent_3d = extent_from_summary(summary)
    support_gap = float_or_none(summary.get("support_gap"))
    pointcloud_ref = write_candidate_points(artifacts, config.artifact_prefix, candidate.candidate_id, points) or pointcloud_ref
    sources.append("candidate_bbox_lift")
    metric_geometry = MetricGeometry(
        available=True,
        source=sources,
        pointcloud_local=summary,
        geometry_views=geometry_views,
        position_3d=position_3d,
        extent_3d=extent_3d,
        support_gap=support_gap,
        pointcloud_ref=pointcloud_ref,
        quality={"method": "depth_bbox_lift", "min_points": config.min_points},
    )
    metric_geometry.refresh_available()
    return CandidateGeometry(candidate_id=candidate.candidate_id, metric_geometry=metric_geometry)


def apply_candidate_geometry(candidate: SceneCandidate, geometry: CandidateGeometry) -> None:
    metric = geometry.metric_geometry
    candidate.metric_geometry = metric
    candidate.support.update(
        {
            "support_gap": metric.support_gap,
            "support_z": metric.pointcloud_local.get("support_z"),
            "top_z": metric.pointcloud_local.get("top_z"),
            "source": "depth_bbox_lift",
        }
    )
    candidate.metric_geometry.source = unique_strings(candidate.metric_geometry.source)
    existing_evidence = candidate.evidence.get("metric_geometry")
    if isinstance(existing_evidence, list):
        metric_evidence = [str(item) for item in existing_evidence]
    elif existing_evidence is None:
        metric_evidence = []
    else:
        metric_evidence = [str(existing_evidence)]
    candidate.evidence["metric_geometry"] = unique_strings(metric_evidence + candidate.metric_geometry.source)
    candidate.metadata["metric_geometry_available"] = candidate.metric_geometry.available
    if candidate.metric_geometry.pointcloud_ref:
        candidate.metadata["pointcloud_ref"] = candidate.metric_geometry.pointcloud_ref


def candidate_mask_for_view(candidate: SceneCandidate, view: CameraView, padding: int) -> np.ndarray | None:
    if view.depth_path is None:
        return None
    depth = load_depth(view.depth_path)
    if depth is None or depth.ndim != 2:
        return None
    bbox = candidate.bbox_by_view.get(view.name)
    if bbox is None:
        return None
    x0, y0, x1, y1 = clamp_bbox(bbox, depth.shape[1], depth.shape[0], padding)
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros(depth.shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def estimate_global_support_z(observation: ObservationBundle) -> float | None:
    points = []
    ref_points = load_pointcloud_ref(observation.pointcloud_ref)
    if ref_points.shape[0]:
        points.append(ref_points[:, :3])
    for view in observation.camera_views.values():
        if view.depth_path is None:
            continue
        depth = load_depth(view.depth_path)
        if depth is None or depth.ndim != 2:
            continue
        mask = np.isfinite(depth) & (depth > 0)
        view_points = backproject_view_mask(view, mask)
        if view_points.size:
            points.append(view_points[:, :3])
    merged = merge_points(points)
    if merged.shape[0] == 0:
        return None
    return float(np.quantile(merged[:, 2], 0.05))


def write_candidate_points(artifacts: Any | None, prefix: str, candidate_id: str, points: np.ndarray) -> str | None:
    if artifacts is None or not hasattr(artifacts, "write_pointcloud"):
        return None
    safe_id = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in candidate_id)
    return artifacts.write_pointcloud(f"{prefix}/pointcloud/{safe_id}_local.npy", points)
