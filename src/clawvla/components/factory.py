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
        model_runtime = _resolve_model_runtime(
            component_cfg.model,
            config=config,
            model_runtimes=model_runtimes,
            route=f"component:{name}",
        )
        skill_model_runtimes = {
            skill: _required_model_runtime(
                model_key,
                config=config,
                model_runtimes=model_runtimes,
                route=f"skill:{name}.{skill}",
            )
            for skill, model_key in component_cfg.skill_models.items()
        }
        registry.register(
            Component(
                name=name,
                config=component_cfg,
                skills=skill_registry,
                model_runtime=model_runtime,
                model_route=component_cfg.model,
                skill_model_runtimes=skill_model_runtimes,
                skill_model_routes=dict(component_cfg.skill_models),
            )
        )
    return registry


def _resolve_model_runtime(
    model_key: str | None,
    *,
    config: AgentConfig,
    model_runtimes: dict[str, ModelRuntime],
    route: str,
) -> ModelRuntime | None:
    if not model_key:
        return None
    runtime = model_runtimes.get(model_key)
    if runtime is not None:
        return runtime
    if model_key in config.models:
        runtime = ModelRuntime(config.models[model_key])
        model_runtimes[model_key] = runtime
        return runtime
    raise ValueError(f"unknown_model_route:{route}:{model_key}")


def _required_model_runtime(
    model_key: str,
    *,
    config: AgentConfig,
    model_runtimes: dict[str, ModelRuntime],
    route: str,
) -> ModelRuntime:
    runtime = _resolve_model_runtime(
        model_key,
        config=config,
        model_runtimes=model_runtimes,
        route=route,
    )
    if runtime is None:
        raise ValueError(f"missing_model_route:{route}")
    return runtime
