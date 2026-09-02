from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..blackboard import Blackboard
from ..config import ComponentConfig
from ..models import ModelRuntime
from ..schema import SkillRequest, SkillResult, SkillSpec
from ..skills import SkillContext, SkillRegistry


@dataclass
class Component:
    name: str
    config: ComponentConfig
    skills: SkillRegistry
    model_runtime: ModelRuntime | None = None
    model_route: str | None = None
    skill_model_runtimes: dict[str, ModelRuntime] = field(default_factory=dict)
    skill_model_routes: dict[str, str] = field(default_factory=dict)

    def run_skill(self, request: SkillRequest, blackboard: Blackboard) -> SkillResult:
        if request.component != self.name:
            raise ValueError(f"Request component {request.component!r} does not match {self.name!r}.")
        skill = self.skills.get(request.component, request.skill)
        model_runtime = self.skill_model_runtimes.get(request.skill, self.model_runtime)
        model_route = self.skill_model_routes.get(request.skill, self.model_route)
        context = SkillContext(
            component_name=self.name,
            blackboard=blackboard,
            model_runtime=model_runtime,
            model_route=model_route,
        )
        return skill.run(request, context)

    def skill_specs(self) -> list[SkillSpec]:
        return self.skills.specs_for_component(self.name)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.config.enabled,
            "model": self.config.model,
            "model_route": self.model_route,
            "skill_model_routes": dict(sorted(self.skill_model_routes.items())),
            "prompt_format": self.config.prompt_format,
            "configured_skills": list(self.config.skills),
            "registered_skills": [spec.name for spec in self.skill_specs()],
            "model_enabled": bool(self.model_runtime and self.model_runtime.enabled),
            "skill_models_enabled": {
                skill: runtime.enabled
                for skill, runtime in sorted(self.skill_model_runtimes.items())
            },
        }


class ComponentRegistry:
    def __init__(self):
        self._components: dict[str, Component] = {}

    def register(self, component: Component) -> None:
        if component.name in self._components:
            raise ValueError(f"Component already registered: {component.name}")
        self._components[component.name] = component

    def get(self, name: str) -> Component:
        if name not in self._components:
            raise KeyError(f"Unknown component: {name}")
        return self._components[name]

    def run(self, request: SkillRequest, blackboard: Blackboard) -> SkillResult:
        return self.get(request.component).run_skill(request, blackboard)

    def names(self) -> list[str]:
        return sorted(self._components)

    def summaries(self) -> dict[str, dict[str, Any]]:
        return {name: self._components[name].summary() for name in self.names()}
