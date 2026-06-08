from __future__ import annotations

from typing import Any, Callable

from ..notices import emit_status_notice
from ..schema import SkillRequest, SkillResult, SkillSpec
from ..skills.base import Skill, SkillContext, SkillRegistry


def register_skill(
    registry: SkillRegistry,
    component: str,
    name: str,
    description: str,
    handler: Callable[[SkillRequest, SkillContext], SkillResult],
    requires_model: bool = False,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    registry.register(
        Skill(
            spec=SkillSpec(
                component=component,
                name=name,
                description=description,
                input_schema=dict(input_schema or {}),
                output_schema=dict(output_schema or {}),
                requires_model=requires_model,
                metadata=dict(metadata or {}),
            ),
            handler=handler,
        )
    )


def ok(status: str, output: dict[str, Any] | None = None) -> SkillResult:
    payload = dict(output or {})
    emit_status_notice(status, success=True, source="skill_result", payload=payload)
    return SkillResult(success=True, status=status, output=payload)


def unavailable(status: str, reason: str, output: dict[str, Any] | None = None) -> SkillResult:
    payload = {"status": status, "reason": reason, "retryable": False}
    payload.update(dict(output or {}))
    emit_status_notice(status, success=False, source="skill_result", reason=reason, payload=payload)
    return SkillResult(success=False, status=status, output=payload, errors=[reason])


def get_attr(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def to_dict(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    return value
