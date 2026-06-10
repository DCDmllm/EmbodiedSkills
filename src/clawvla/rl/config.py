from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml


@dataclass
class EnvCommandConfig:
    python: str
    env: dict[str, str] = field(default_factory=dict)
    unset_env: list[str] = field(default_factory=list)
    cwd: str = "/mnt/wangwai/vla/clawvla"


@dataclass
class PolicyConfig:
    model_path: str
    served_model_name: str = "clawvla-policy"
    roles: list[str] = field(default_factory=lambda: ["vision", "scheduler", "verifier", "recovery"])
    max_new_tokens: int = 2048
    temperature: float = 1.0
    request_timeout: float = 1800.0
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 18080
    external_base_url: str | None = None
    api_key: str = "clawvla-rl-policy"


@dataclass
class RolloutConfig:
    base_config: str
    instruction: str
    initial_stage: str = "observe"
    max_steps: int = 25
    run_robotwin: bool = True
    episodes: int = 1
    group_size: int = 4
    artifact_prefix: str = "clawvla_rl"
    task_name: str = "place_container_plate"
    seeds: list[int] = field(default_factory=lambda: [0])
    episode_timeout_s: float = 1800.0
    openpi_port_base: int = 8765
    start_openpi_worker: bool = True


@dataclass
class RewardConfig:
    registry: list[str] = field(default_factory=lambda: ["clawvla.rl.reward_registry:register_builtin_robotwin"])
    task_map: dict[str, str] = field(default_factory=dict)
    step_cost: float = 0.05
    incomplete_episode_penalty: float = -1.0
    invalid_decision_penalty: float = -2.0
    skill_failure_penalty: float = -1.0
    infra_failure_reward: float | None = None


@dataclass
class VerlConfig:
    env: str = "verl-0.8-py310"
    algorithm: str = "grpo"
    train_mode: str = "full"
    lora_merge_for_rollout: bool = False
    lora_target_modules: str | list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_exclude_modules: str | None = None
    force_full_gpu_workers: bool = False
    dtype: str = "bfloat16"
    flash_attention: bool = True
    tensor_parallel_size: int = 2
    rollout_n: int = 4
    total_epochs: int = 1
    total_training_steps: int | None = None
    max_prompt_length: int = 32768
    max_response_length: int = 32768
    max_model_len: int = 65536
    max_num_batched_tokens: int = 65536
    max_num_seqs: int = 16
    fsdp_model_dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.7
    gradient_checkpointing: bool = True
    use_remove_padding: bool = True
    learning_rate: float = 1e-6
    actor_ppo_mini_batch_size: int = 4
    actor_ppo_micro_batch_size_per_gpu: int = 1
    actor_ppo_max_token_len_per_gpu: int = 65536
    rollout_log_prob_micro_batch_size_per_gpu: int = 1
    ref_log_prob_micro_batch_size_per_gpu: int = 1
    fsdp_param_offload: bool = True
    fsdp_optimizer_offload: bool = True
    val_before_train: bool = False
    test_freq: int = -1


@dataclass
class ClusterConfig:
    gpus: list[int] = field(default_factory=lambda: list(range(8)))
    policy_gpus: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    openpi_gpus: list[int] = field(default_factory=lambda: [6])
    robotwin_gpus: list[int] = field(default_factory=lambda: [7])
    ray_num_workers: int = 4


@dataclass
class LoggingConfig:
    run_root: str = "/mnt/wangwai/vla/clawvla/runs/rl"
    tmp_root: str = "/mnt/wangwai/vla/clawvla/tmp_runs/rl"
    rich: bool = True
    wandb_mode: str = "disabled"
    wandb_project: str = "clawvla-agent-rl"
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(default_factory=list)
    upload_artifacts: bool = False


@dataclass
class CheckpointConfig:
    output_dir: str = "/mnt/wangwai/vla/clawvla/checkpoints/rl"
    save_freq: int = 20
    keep_last: int = 3
    resume: str | None = None


@dataclass
class RLConfig:
    name: str = "qwen3vl_pi05_grpo"
    run_id: str | None = None
    trainer: EnvCommandConfig = field(
        default_factory=lambda: EnvCommandConfig(python="/mnt/wangwai/miniconda3/envs/verl-0.8-py310/bin/python")
    )
    robotwin: EnvCommandConfig = field(
        default_factory=lambda: EnvCommandConfig(
            python="/mnt/wangwai/miniconda3/envs/robotwin-py312/bin/python",
            env={"PYTHONPATH": "/mnt/wangwai/tmp_pytorch3d_target:/mnt/wangwai/vla/clawvla/src"},
            unset_env=["PYTHONPATH"],
        )
    )
    openpi: EnvCommandConfig = field(
        default_factory=lambda: EnvCommandConfig(
            python="/mnt/wangwai/miniconda3/envs/openpi-torch-py312/bin/python",
            env={"PYTHONPATH": "/mnt/wangwai/vla/clawvla/src:/mnt/wangwai/RoboTwin/policy/pi05/src"},
        )
    )
    policy: PolicyConfig = field(
        default_factory=lambda: PolicyConfig("/mnt/wangwai/weights/Qwen/Qwen3-VL-8B-Instruct")
    )
    rollout: RolloutConfig = field(
        default_factory=lambda: RolloutConfig(
            base_config="/mnt/wangwai/vla/clawvla/configs/robotwin_pi05_worker_probe.json",
            instruction="place the container on the plate",
        )
    )
    reward: RewardConfig = field(default_factory=RewardConfig)
    verl: VerlConfig = field(default_factory=VerlConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolved_run_id(self) -> str:
        if self.run_id:
            return self.run_id
        return f"{self.name}_{uuid4().hex[:8]}"


def load_rl_config(path: str | Path) -> RLConfig:
    config_path = Path(path)
    payload = _load_yaml_with_extends(config_path)
    return _rl_config(payload)


def dump_resolved_config(config: RLConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(_to_plain(config), sort_keys=False, allow_unicode=False), encoding="utf-8")


def _load_yaml_with_extends(path: Path) -> dict[str, Any]:
    payload = _load_yaml(path)
    extends = payload.pop("extends", [])
    if isinstance(extends, str):
        extends = [extends]
    merged: dict[str, Any] = {}
    for item in extends or []:
        parent_path = Path(item)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        merged = _deep_update(merged, _load_yaml_with_extends(parent_path))
    return _deep_update(merged, payload)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"RL config must be a YAML object: {path}")
    return payload


def _rl_config(payload: dict[str, Any]) -> RLConfig:
    return RLConfig(
        name=str(payload.get("name", "qwen3vl_pi05_grpo")),
        run_id=payload.get("run_id"),
        trainer=_env_command(payload.get("trainer"), RLConfig().trainer),
        robotwin=_env_command(payload.get("robotwin"), RLConfig().robotwin),
        openpi=_env_command(payload.get("openpi"), RLConfig().openpi),
        policy=_dataclass_from_dict(PolicyConfig, payload.get("policy"), RLConfig().policy),
        rollout=_dataclass_from_dict(RolloutConfig, payload.get("rollout"), RLConfig().rollout),
        reward=_dataclass_from_dict(RewardConfig, payload.get("reward"), RLConfig().reward),
        verl=_dataclass_from_dict(VerlConfig, payload.get("verl"), RLConfig().verl),
        cluster=_dataclass_from_dict(ClusterConfig, payload.get("cluster"), RLConfig().cluster),
        logging=_dataclass_from_dict(LoggingConfig, payload.get("logging"), RLConfig().logging),
        checkpoint=_dataclass_from_dict(CheckpointConfig, payload.get("checkpoint"), RLConfig().checkpoint),
        metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {},
    )


def _env_command(payload: object, default: EnvCommandConfig) -> EnvCommandConfig:
    if not isinstance(payload, dict):
        return default
    merged = _deep_update(_to_plain(default), payload)
    return EnvCommandConfig(**merged)


def _dataclass_from_dict(cls: type, payload: object, default: Any) -> Any:
    if not isinstance(payload, dict):
        return default
    merged = _deep_update(_to_plain(default), payload)
    return cls(**merged)


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _to_plain(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    return value
