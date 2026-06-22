from __future__ import annotations

from ..config import AgentConfig
from .base import RobotEnvAdapter
from .libero import LiberoAdapter
from .robocasa import RoboCasaAdapter
from .robotwin import RoboTwinAdapter


def build_env_adapter(config: AgentConfig) -> RobotEnvAdapter:
    env_type = str(getattr(config.environment, "type", "robotwin") or "robotwin").lower()
    if env_type in {"robotwin", "robo_twin"}:
        return RoboTwinAdapter(config.robotwin)
    if env_type in {"libero", "libero_env"}:
        return LiberoAdapter(config.environment)
    if env_type in {"robocasa", "robo_casa"}:
        return RoboCasaAdapter(config.environment)
    raise ValueError(f"unsupported_environment_type:{env_type}")


def environment_artifact_dir(config: AgentConfig) -> str:
    if getattr(config.environment, "artifact_dir", None):
        return str(config.environment.artifact_dir)
    return str(config.robotwin.artifact_dir)
