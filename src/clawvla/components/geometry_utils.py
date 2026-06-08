from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..schema import CameraView


MAX_POINTS_PER_VIEW = 8192
MIN_LOCAL_POINTS = 12
DEPTH_MM_THRESHOLD = 10.0


def backproject_view_mask(view: CameraView, mask: np.ndarray) -> np.ndarray:
    if view.depth_path is None:
        return np.empty((0, 3), dtype=np.float32)
    depth = load_depth(view.depth_path)
    intrinsics = reshape_matrix(view.intrinsics, (3, 3))
    extrinsics = reshape_matrix(view.extrinsics, (3, 4))
    if depth is None or intrinsics is None or extrinsics is None:
        return np.empty((0, 3), dtype=np.float32)
    if depth.ndim != 2 or mask.shape != depth.shape:
        return np.empty((0, 3), dtype=np.float32)

    valid = mask & np.isfinite(depth) & (depth > 0)
    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if ys.size > MAX_POINTS_PER_VIEW:
        sample_ids = np.linspace(0, ys.size - 1, num=MAX_POINTS_PER_VIEW, dtype=int)
        ys = ys[sample_ids]
        xs = xs[sample_ids]

    z = depth[ys, xs].astype(np.float32) * depth_scale(depth)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if abs(fx) < 1e-6 or abs(fy) < 1e-6:
        return np.empty((0, 3), dtype=np.float32)
    camera_points = np.stack(
        [
            (xs.astype(np.float32) - cx) * z / fx,
            (ys.astype(np.float32) - cy) * z / fy,
            z,
        ],
        axis=1,
    )
    rotation = extrinsics[:, :3].astype(np.float32)
    translation = extrinsics[:, 3].astype(np.float32)
    try:
        return (np.linalg.inv(rotation) @ (camera_points - translation).T).T.astype(np.float32)
    except np.linalg.LinAlgError:
        return np.empty((0, 3), dtype=np.float32)


def local_points_from_pointcloud_ref(path: str | None, center: list[float] | None) -> np.ndarray:
    if center is None or len(center) < 3:
        return np.empty((0, 3), dtype=np.float32)
    points = load_pointcloud_ref(path)
    if points.shape[0] == 0:
        return points
    center_xyz = np.asarray(center[:3], dtype=np.float32)
    distances = np.linalg.norm(points[:, :3] - center_xyz[None, :], axis=1)
    for radius in (0.03, 0.05, 0.08, 0.12):
        mask = distances <= radius
        if int(np.count_nonzero(mask)) >= MIN_LOCAL_POINTS:
            return points[mask, :3].astype(np.float32)
    nearest_count = min(max(MIN_LOCAL_POINTS, 1), points.shape[0])
    ids = np.argsort(distances)[:nearest_count]
    return points[ids, :3].astype(np.float32)


def summarize_local_pointcloud(points: np.ndarray, global_support_z: float | None) -> dict[str, Any]:
    xyz = np.asarray(points[:, :3], dtype=np.float32)
    center = np.median(xyz, axis=0)
    xyz_min = np.min(xyz, axis=0)
    xyz_max = np.max(xyz, axis=0)
    distances = np.linalg.norm(xyz - center[None, :], axis=1)
    support_z = float(np.quantile(xyz[:, 2], 0.1))
    top_z = float(np.quantile(xyz[:, 2], 0.9))
    baseline_support_z = global_support_z if global_support_z is not None else support_z
    return {
        "point_count": int(xyz.shape[0]),
        "center_xyz": as_float_list(center),
        "xyz_min": as_float_list(xyz_min),
        "xyz_max": as_float_list(xyz_max),
        "radius": float(np.quantile(distances, 0.9)) if distances.size else 0.0,
        "support_z": support_z,
        "top_z": top_z,
        "support_gap": float(xyz_min[2] - baseline_support_z) if baseline_support_z is not None else None,
        "support_z_source": "global_scene_depth" if global_support_z is not None else "candidate_depth_quantile",
        "local_relief": float(top_z - support_z),
    }


def summarize_view_geometry(points: np.ndarray) -> dict[str, Any]:
    xyz = np.asarray(points[:, :3], dtype=np.float32)
    return {
        "world_center": as_float_list(np.median(xyz, axis=0)),
        "world_xyz_min": as_float_list(np.min(xyz, axis=0)),
        "world_xyz_max": as_float_list(np.max(xyz, axis=0)),
        "world_point_count": int(xyz.shape[0]),
    }


def load_depth(path: str) -> np.ndarray | None:
    try:
        return np.asarray(np.load(Path(path)), dtype=np.float32)
    except Exception:
        return None


def load_pointcloud_ref(path: str | None) -> np.ndarray:
    if not path:
        return np.empty((0, 3), dtype=np.float32)
    try:
        payload = np.load(Path(path))
    except Exception:
        return np.empty((0, 3), dtype=np.float32)
    if isinstance(payload, np.lib.npyio.NpzFile):
        arrays = [np.asarray(payload[key]) for key in payload.files]
        return merge_points(array.reshape(-1, array.shape[-1]) for array in arrays if array.size and array.ndim >= 2)
    array = np.asarray(payload)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array.reshape(-1, array.shape[-1]).astype(np.float32) if array.shape[-1] >= 3 else np.empty((0, 3))


def merge_points(points: Any) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for item in points:
        array = np.asarray(item)
        if array.size == 0 or array.ndim != 2 or array.shape[1] < 3:
            continue
        arrays.append(array[:, :3].astype(np.float32))
    if not arrays:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(arrays, axis=0).astype(np.float32)


def clamp_bbox(bbox: list[float], width: int, height: int, padding: int) -> tuple[int, int, int, int]:
    values = [float(item) for item in bbox[:4]]
    x0, y0, x1, y1 = values
    if 0.0 <= max(values) <= 1.0:
        x0, x1 = x0 * width, x1 * width
        y0, y1 = y0 * height, y1 * height
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    left = max(int(np.floor(x0)) - padding, 0)
    top = max(int(np.floor(y0)) - padding, 0)
    right = min(int(np.ceil(x1)) + padding, width)
    bottom = min(int(np.ceil(y1)) + padding, height)
    return left, top, right, bottom


def reshape_matrix(value: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size != shape[0] * shape[1]:
        return None
    return array.reshape(shape)


def depth_scale(depth: np.ndarray) -> float:
    finite = np.asarray(depth)[np.isfinite(depth)]
    if finite.size == 0:
        return 1.0
    return 0.001 if float(np.median(finite)) > DEPTH_MM_THRESHOLD else 1.0


def extent_from_summary(summary: dict[str, Any]) -> list[float] | None:
    xyz_min = float_list(summary.get("xyz_min"))
    xyz_max = float_list(summary.get("xyz_max"))
    if xyz_min is None or xyz_max is None:
        return None
    return [float(max_v - min_v) for min_v, max_v in zip(xyz_min[:3], xyz_max[:3], strict=False)]


def as_float_list(value: Any) -> list[float]:
    return [float(item) for item in np.asarray(value).reshape(-1).tolist()]


def float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        return [float(item) for item in np.asarray(value).reshape(-1)[:3].tolist()]
    except (TypeError, ValueError):
        return None


def float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value)
        if text not in result:
            result.append(text)
    return result
