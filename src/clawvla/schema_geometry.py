from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .schema_utils import float_list_or_none, float_or_none, string_list


@dataclass
class MetricGeometry:
    """Optional metric evidence for a visual candidate.

    This block is allowed to be empty on real hardware paths that only expose RGB
    or image-space grounding.
    """

    available: bool = False
    source: list[str] = field(default_factory=list)
    position_3d: list[float] | None = None
    extent_3d: list[float] | None = None
    pointcloud_ref: str | None = None
    pointcloud_local: dict[str, Any] = field(default_factory=dict)
    geometry_views: dict[str, dict[str, Any]] = field(default_factory=dict)
    support_gap: float | None = None
    quality: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_candidate_payload(cls, payload: dict[str, Any]) -> "MetricGeometry":
        metric_payload = payload.get("metric_geometry")
        merged = dict(metric_payload) if isinstance(metric_payload, dict) else {}
        legacy_keys = {
            "geometry_source": "source",
            "position_3d": "position_3d",
            "extent_3d": "extent_3d",
            "pointcloud_ref": "pointcloud_ref",
            "pointcloud_local": "pointcloud_local",
            "geometry_views": "geometry_views",
            "support_gap": "support_gap",
        }
        for legacy_key, metric_key in legacy_keys.items():
            if legacy_key in payload and metric_key not in merged:
                merged[metric_key] = payload[legacy_key]
        return cls.from_payload(merged)

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "MetricGeometry":
        if not isinstance(payload, dict):
            return cls()
        pointcloud_local = dict(payload.get("pointcloud_local", {})) if isinstance(payload.get("pointcloud_local"), dict) else {}
        geometry_views = _dict_of_dicts(payload.get("geometry_views"))
        geometry = cls(
            available=bool(payload.get("available", False)),
            source=string_list(payload.get("source") or payload.get("geometry_source")),
            position_3d=float_list_or_none(payload.get("position_3d")),
            extent_3d=float_list_or_none(payload.get("extent_3d")),
            pointcloud_ref=str(payload.get("pointcloud_ref")) if payload.get("pointcloud_ref") is not None else None,
            pointcloud_local=pointcloud_local,
            geometry_views=geometry_views,
            support_gap=float_or_none(payload.get("support_gap")),
            quality=dict(payload.get("quality", {})) if isinstance(payload.get("quality"), dict) else {},
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {},
        )
        geometry.refresh_available()
        return geometry

    @property
    def has_position(self) -> bool:
        return self.position_3d is not None and len(self.position_3d) >= 3

    def refresh_available(self) -> None:
        self.available = bool(
            self.available
            or self.has_position
            or self.extent_3d
            or self.pointcloud_ref
            or self.pointcloud_local
            or self.geometry_views
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dict_of_dicts(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    return {str(key): dict(item) for key, item in value.items() if isinstance(item, dict)}
