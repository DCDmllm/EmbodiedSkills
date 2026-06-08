from __future__ import annotations

from typing import Any


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def float_list_or_none(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def dict_of_float_lists(value: Any) -> dict[str, list[float]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, list[float]] = {}
    for key, item in value.items():
        parsed = float_list_or_none(item)
        if parsed is not None:
            result[str(key)] = parsed
    return result
