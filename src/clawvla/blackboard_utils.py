from __future__ import annotations

from typing import Any

from .blackboard import Blackboard


def current_observation_id(blackboard: Blackboard) -> str | None:
    observation = blackboard.read("observation")
    value = getattr(observation, "observation_id", None)
    return str(value) if value is not None else None


def mark_grounding_overlay_stale(blackboard: Blackboard, reason: str) -> None:
    overlay = blackboard.read("grounding_overlay")
    if overlay is None:
        return
    _set_metadata_flag(overlay, "stale", True)
    _set_metadata_value(overlay, "stale_reason", reason)
    blackboard.write("grounding_overlay", overlay, event_type="grounding_overlay.stale")


def mark_motion_artifacts_stale(blackboard: Blackboard, reason: str, *, include_goal: bool = False) -> None:
    keys = ["motion_plan", "action_chunk", "action_backend_result"]
    if include_goal:
        keys.insert(0, "motion_goal")
    for key in keys:
        value = blackboard.read(key)
        if value is None:
            continue
        _set_metadata_flag(value, "stale", True)
        _set_metadata_value(value, "stale_reason", reason)
        blackboard.write(key, value, event_type=f"{key}.stale")


def mark_action_chunk_consumed(blackboard: Blackboard, reason: str) -> None:
    chunk = blackboard.read("action_chunk")
    if chunk is None:
        return
    _set_metadata_flag(chunk, "consumed", True)
    _set_metadata_value(chunk, "consumed_reason", reason)
    blackboard.write("action_chunk", chunk, event_type="action_chunk.consumed")


def metadata_value(value: Any, key: str, default: Any = None) -> Any:
    metadata = _metadata(value)
    return metadata.get(key, default) if isinstance(metadata, dict) else default


def _metadata(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        metadata = value.get("metadata")
        if isinstance(metadata, dict):
            return metadata
        value["metadata"] = {}
        return value["metadata"]
    metadata = getattr(value, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    return None


def _set_metadata_flag(value: Any, key: str, flag: bool) -> None:
    if isinstance(value, dict) and key in {"stale"}:
        value[key] = flag
    elif hasattr(value, key):
        setattr(value, key, flag)
    metadata = _metadata(value)
    if metadata is not None:
        metadata[key] = flag


def _set_metadata_value(value: Any, key: str, item: Any) -> None:
    metadata = _metadata(value)
    if metadata is not None:
        metadata[key] = item
