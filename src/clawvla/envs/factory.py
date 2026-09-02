from __future__ import annotations

from ..config import AgentConfig
from .base import RobotEnvAdapter
from .libero import LiberoAdapter
from .rmbench import RmbenchAdapter
from .robotwin import RoboTwinAdapter


def build_env_adapter(config: AgentConfig) -> RobotEnvAdapter:
    env_type = str(getattr(config.environment, "type", "robotwin") or "robotwin").lower()
    if env_type in {"robotwin", "robo_twin"}:
        return RoboTwinAdapter(config.robotwin)
    if env_type in {"rmbench", "rm_bench"}:
        return RmbenchAdapter(config.robotwin)
    if env_type in {"libero", "libero_env"}:
        return LiberoAdapter(config.environment)
    raise ValueError(f"unsupported_environment_type:{env_type}")


def environment_artifact_dir(config: AgentConfig) -> str:
    if getattr(config.environment, "artifact_dir", None):
        return str(config.environment.artifact_dir)
    return str(config.robotwin.artifact_dir)
