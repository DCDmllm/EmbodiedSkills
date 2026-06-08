from __future__ import annotations

from ..config import AgentConfig, ModelConfig
from ..models import ModelRuntime
from ..skills import SkillRegistry, register_builtin_skills
from .base import Component, ComponentRegistry


def build_skill_registry() -> SkillRegistry:
    registry = SkillRegistry()
    register_builtin_skills(registry)
    return registry


def build_model_runtimes(config: AgentConfig) -> dict[str, ModelRuntime]:
    return {name: ModelRuntime(model_cfg) for name, model_cfg in config.models.items()}


def build_component_registry(
    config: AgentConfig,
    skill_registry: SkillRegistry | None = None,
    model_runtimes: dict[str, ModelRuntime] | None = None,
) -> ComponentRegistry:
    skill_registry = skill_registry or build_skill_registry()
    model_runtimes = model_runtimes or build_model_runtimes(config)
    registry = ComponentRegistry()
    for name, component_cfg in config.components.items():
        if not component_cfg.enabled:
            continue
        model_runtime = model_runtimes.get(component_cfg.model or "")
        if model_runtime is None and component_cfg.model in config.models:
            model_runtime = ModelRuntime(config.models[component_cfg.model])
        registry.register(
            Component(
                name=name,
                config=component_cfg,
                skills=skill_registry,
                model_runtime=model_runtime,
            )
        )
    return registry
