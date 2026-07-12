from __future__ import annotations

import json
from dataclasses import dataclass, field
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass
from pathlib import Path
from typing import Any

from .schema import StageDefinition


class ModelBackend(StrEnum):
    NONE = "none"
    LOCAL_HF = "local_hf"
    OPENAI_COMPATIBLE = "openai_compatible"
    AZURE_OPENAI = "azure_openai"


@dataclass
class ModelConfig:
    backend: str = ModelBackend.NONE
    model: str | None = None
    api_base_url: str | None = None
    api_base_url_env: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    api_version: str | None = None
    max_new_tokens: int = 2048
    temperature: float = 0.0
    request_timeout: float = 1800.0
    reasoning_effort: str | None = None
    enable_thinking: bool | None = None
    stream: bool = False
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentConfig:
    enabled: bool = True
    model: str | None = None
    skills: list[str] = field(default_factory=list)
    prompt_format: str = "json"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotwinConfig:
    repo_root: str = "/mnt/wangwai/RoboTwin"
    task_name: str = "place_container_plate"
    task_config: str = "demo_clean"
    seed: int = 0
    now_ep_num: int = 0
    is_test: bool = False
    eval_mode: bool = False
    render_freq: int | None = None
    need_plan: bool | None = None
    enable_depth: bool = True
    enable_pointcloud: bool = True
    enable_actor_segmentation: bool = False
    enable_mesh_segmentation: bool = False
    camera_profile: str | None = None
    planner_image_mode: str = "current_rgb_4"
    static_camera_preset: str = "selected_global_4"
    artifact_dir: str = "/mnt/wangwai/vla/clawvla/tmp_artifacts"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentConfig:
    type: str = "robotwin"
    artifact_dir: str | None = None
    task_name: str | None = None
    seed: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeEnvironment:
    conda_env: str = "robotwin-py312"
    conda_bin: str = "/mnt/wangwai/miniconda3/bin/conda"
    pythonpath_prefix: list[str] = field(default_factory=lambda: [
        "/mnt/wangwai/tmp_pytorch3d_target",
        "/mnt/wangwai/vla/clawvla/src",
    ])
    env: dict[str, str] = field(default_factory=lambda: {
        "__EGL_VENDOR_LIBRARY_DIRS": "/usr/share/glvnd/egl_vendor.d",
        "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
    })


@dataclass
class AgentConfig:
    name: str
    task: dict[str, Any] = field(default_factory=dict)
    models: dict[str, ModelConfig] = field(default_factory=dict)
    components: dict[str, ComponentConfig] = field(default_factory=dict)
    stages: list[StageDefinition] = field(default_factory=list)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    robotwin: RobotwinConfig = field(default_factory=RobotwinConfig)
    runtime_environment: RuntimeEnvironment = field(default_factory=RuntimeEnvironment)
    metadata: dict[str, Any] = field(default_factory=dict)

    def enabled_components(self) -> dict[str, ComponentConfig]:
        return {name: cfg for name, cfg in self.components.items() if cfg.enabled}


def _model_config(payload: dict[str, Any]) -> ModelConfig:
    return ModelConfig(**payload)


def _component_config(payload: dict[str, Any]) -> ComponentConfig:
    return ComponentConfig(**payload)


def _stage_definition(payload: dict[str, Any]) -> StageDefinition:
    return StageDefinition(**payload)


def load_config(path: str | Path) -> AgentConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    robotwin = RobotwinConfig(**payload.get("robotwin", {}))
    environment = _environment_config(payload.get("environment"), robotwin)
    return AgentConfig(
        name=str(payload.get("name", "clawvla")),
        task=dict(payload.get("task", {})),
        models={name: _model_config(cfg) for name, cfg in payload.get("models", {}).items()},
        components={name: _component_config(cfg) for name, cfg in payload.get("components", {}).items()},
        stages=[_stage_definition(item) for item in payload.get("stages", []) if isinstance(item, dict)],
        environment=environment,
        robotwin=robotwin,
        runtime_environment=RuntimeEnvironment(**payload.get("runtime_environment", {})),
        metadata=dict(payload.get("metadata", {})),
    )


def _environment_config(payload: Any, robotwin: RobotwinConfig) -> EnvironmentConfig:
    if not isinstance(payload, dict):
        return EnvironmentConfig(
            type="robotwin",
            artifact_dir=robotwin.artifact_dir,
            task_name=robotwin.task_name,
            seed=robotwin.seed,
            params={},
            runtime={},
            metadata={"source": "legacy_robotwin_config"},
        )
    merged = dict(payload)
    env_type = str(merged.get("type") or "robotwin")
    if not merged.get("artifact_dir") and env_type == "robotwin":
        merged["artifact_dir"] = robotwin.artifact_dir
    if not merged.get("task_name") and env_type == "robotwin":
        merged["task_name"] = robotwin.task_name
    if merged.get("seed") is None and env_type == "robotwin":
        merged["seed"] = robotwin.seed
    return EnvironmentConfig(
        type=env_type,
        artifact_dir=merged.get("artifact_dir"),
        task_name=merged.get("task_name"),
        seed=merged.get("seed"),
        params=dict(merged.get("params", {})) if isinstance(merged.get("params"), dict) else {},
        runtime=dict(merged.get("runtime", {})) if isinstance(merged.get("runtime"), dict) else {},
        metadata=dict(merged.get("metadata", {})) if isinstance(merged.get("metadata"), dict) else {},
    )
