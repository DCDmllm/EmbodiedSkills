from __future__ import annotations

import re
from typing import Iterable


_TARGET_REQUIRED_SUBGOAL_TYPES = {
    "align",
    "dump",
    "handover",
    "hang",
    "insert",
    "place",
    "pour",
    "put",
    "release",
    "stack",
    "transport",
}

_TARGET_REQUIRED_PATTERNS = (
    (
        r"\b(place|put|set|position|insert|drop|move|bring|transfer)\b.*"
        r"\b(on|onto|in|into|inside|near|left of|right of|beside|next to|under|above|over)\b"
    ),
    r"\b(stack|hang|pour|dump)\b",
    r"\b(beat|scan|stamp)\b",
    (
        r"\b(to|into|onto)\s+the\s+"
        r"(pad|mat|plate|basket|box|stand|rack|cabinet|drawer|tray|scale|coaster|pot|pan)\b"
    ),
    r"\b(to|at)\s+(the\s+)?(left\s+|right\s+)?(target|destination|goal|marker|target pose)\b",
)


def task_requires_target(task_instruction: object | None) -> bool:
    text = _normalize_text(task_instruction)
    if not text:
        return False
    return any(re.search(pattern, text) for pattern in _TARGET_REQUIRED_PATTERNS)


def subgoal_requires_target(subgoal_type: object | None, instruction: object | None = None) -> bool:
    type_name = _normalize_identifier(subgoal_type)
    if type_name in _TARGET_REQUIRED_SUBGOAL_TYPES:
        return True
    return task_requires_target(instruction)


def task_plan_requires_target(task_instruction: object | None, subgoals: Iterable[object] | None) -> bool:
    # Target binding is decided from the full task before a plan exists. When
    # that task text is available, do not let incidental subgoal wording such
    # as "move above the bell" or a handover "release" invent a second object.
    if _normalize_text(task_instruction):
        return task_requires_target(task_instruction)
    return any(
        subgoal_requires_target(getattr(subgoal, "type", None), getattr(subgoal, "instruction", None))
        for subgoal in (subgoals or [])
    )


def action_backend_requires_candidate_bindings(action_backend: object | None) -> bool:
    """Whether grounding ids are a hard execution contract for this backend.

    Classical geometric controllers may need candidate ids and poses.  A
    language-conditioned VLA such as PI0.5 consumes the current images, robot
    state, and subgoal text directly, so candidate bindings are optional hints.
    Unknown backends keep the conservative historical behavior.
    """
    return bool(getattr(action_backend, "requires_candidate_bindings", True))


def _normalize_identifier(value: object | None) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_text(value: object | None) -> str:
    return " ".join(str(value or "").strip().lower().replace("-", " ").split())
