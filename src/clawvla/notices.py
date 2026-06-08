from __future__ import annotations

import json
import os
import sys
from typing import Any


NOTICE_MARKERS = (
    "placeholder",
    "unavailable",
    "not_wired",
    "not_checked",
    "diagnostic_only",
    "skipped",
    "fallback",
    "no_model",
    "requires_",
    "not_implemented",
)

TRACE_COLORS = {
    "scheduler": "\033[36m",
    "skill": "\033[35m",
    "success": "\033[32m",
    "failure": "\033[31m",
    "openpi": "\033[33m",
    "execute": "\033[34m",
    "reset": "\033[0m",
}

TRACE_PREFIX = {
    "scheduler": "==>",
    "skill": "  $",
    "success": "  OK",
    "failure": "  !!",
    "openpi": "  pi",
    "execute": "  >>",
}


def emit_status_notice(
    status: str,
    *,
    success: bool,
    source: str,
    reason: str | None = None,
    payload: Any | None = None,
    always: bool = False,
) -> None:
    markers = sorted(_collect_markers([status, reason, payload]))
    if not always and not markers:
        return
    notice = {
        "event": "clawvla_status_notice",
        "source": source,
        "success": success,
        "status": status,
        "reason": reason,
        "markers": markers,
    }
    print(json.dumps(notice, ensure_ascii=True), file=sys.stderr, flush=True)


def emit_runtime_event(event: str, payload: dict[str, Any]) -> None:
    notice = {"event": event}
    notice.update(_jsonable(payload))
    print(json.dumps(notice, ensure_ascii=True), file=sys.stderr, flush=True)


def emit_human_trace(kind: str, message: str, *, detail: str | None = None) -> None:
    color = "" if os.environ.get("NO_COLOR") else TRACE_COLORS.get(kind, "")
    reset = "" if not color else TRACE_COLORS["reset"]
    prefix = TRACE_PREFIX.get(kind, f"[{kind}]")
    if kind == "scheduler" and detail:
        detail_lines = "\n".join(f"    {part}" for part in detail.split(" | ") if part)
        text = f"{prefix} {message}\n{detail_lines}"
    else:
        suffix = f"  {detail}" if detail else ""
        text = f"{prefix} {message}{suffix}"
    print(f"{color}{text}{reset}", file=sys.stderr, flush=True)


def _collect_markers(value: Any, *, depth: int = 0, budget: list[int] | None = None) -> set[str]:
    if budget is None:
        budget = [256]
    if budget[0] <= 0 or depth > 4:
        return set()
    budget[0] -= 1
    if isinstance(value, dict):
        found: set[str] = set()
        for key, item in value.items():
            found.update(_collect_markers(str(key), depth=depth + 1, budget=budget))
            found.update(_collect_markers(item, depth=depth + 1, budget=budget))
        return found
    if isinstance(value, (list, tuple, set)):
        found = set()
        for item in value:
            found.update(_collect_markers(item, depth=depth + 1, budget=budget))
        return found
    text = str(value).lower()
    return {marker for marker in NOTICE_MARKERS if marker in text}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
