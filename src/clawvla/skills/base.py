from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..blackboard import Blackboard
from ..models import ModelRuntime
from ..schema import SkillRequest, SkillResult, SkillSpec


@dataclass
class SkillContext:
    component_name: str
    blackboard: Blackboard
    model_runtime: ModelRuntime | None = None
    model_route: str | None = None

    @property
    def has_model(self) -> bool:
        return bool(self.model_runtime and self.model_runtime.enabled)


SkillFn = Callable[[SkillRequest, SkillContext], SkillResult]


@dataclass
class Skill:
    spec: SkillSpec
    handler: SkillFn

    def run(self, request: SkillRequest, context: SkillContext | Blackboard) -> SkillResult:
        if isinstance(context, Blackboard):
            context = SkillContext(component_name=request.component, blackboard=context)
        result = self.handler(request, context)
        result.request_id = request.request_id
        result.component = request.component
        result.skill = request.skill
        return result


class SkillRegistry:
    def __init__(self):
        self._skills: dict[tuple[str, str], Skill] = {}

    def register(self, skill: Skill) -> None:
        key = (skill.spec.component, skill.spec.name)
        if key in self._skills:
            raise ValueError(f"Skill already registered: {key}")
        self._skills[key] = skill

    def get(self, component: str, skill_name: str) -> Skill:
        key = (component, skill_name)
        if key not in self._skills:
            raise KeyError(f"Unknown skill: {component}.{skill_name}")
        return self._skills[key]

    def has(self, component: str, skill_name: str) -> bool:
        return (component, skill_name) in self._skills

    def specs_for_component(self, component: str) -> list[SkillSpec]:
        return [skill.spec for key, skill in sorted(self._skills.items()) if key[0] == component]

    def all_specs(self) -> list[SkillSpec]:
        return [skill.spec for _, skill in sorted(self._skills.items())]
