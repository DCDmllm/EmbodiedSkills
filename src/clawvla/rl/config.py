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
class RolloutTaskConfig:
    task_name: str
    instruction: str
    seeds: list[int] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutEpisodeSpec:
    index: int
    task_index: int
    task_name: str
    instruction: str
    seed: int
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutConfig:
    base_config: str
    instruction: str
    initial_stage: str = "observe"
    max_steps: int = 25
    run_robotwin: bool = True
    run_environment: bool | None = None
    episodes: int = 1
    group_size: int = 4
    artifact_prefix: str = "clawvla_rl"
    task_name: str = "place_container_plate"
    seeds: list[int] = field(default_factory=lambda: [0])
    tasks: list[RolloutTaskConfig] = field(default_factory=list)
    episode_timeout_s: float = 1800.0
    openpi_port_base: int = 8765
    start_openpi_worker: bool = True


@dataclass
class RewardConfig:
    registry: list[str] = field(default_factory=lambda: ["clawvla.rl.reward_registry:register_builtin_robotwin"])
    task_map: dict[str, str] = field(default_factory=dict)
    step_cost: float = 0.05
    incomplete_episode_penalty: float = -1.0
    premature_finish_penalty: float = -3.0
    invalid_decision_penalty: float = -2.0
    skill_failure_penalty: float = -1.0
    recoverable_preflight_penalty: float = -0.1
    infra_failure_reward: float | None = None


@dataclass
class OpenRLHFConfig:
    env: str = "openrlhf-py310-cu128"
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
        default_factory=lambda: EnvCommandConfig(
            python="/mnt/wangwai/vla/clawvla/.venv-openrlhf-py310-cu128/bin/python"
        )
    )
    robotwin: EnvCommandConfig = field(
        default_factory=lambda: EnvCommandConfig(
            python="/mnt/wangwai/miniconda3/envs/robotwin-py312/bin/python",
            env={"PYTHONPATH": "/mnt/wangwai/tmp_pytorch3d_target:/mnt/wangwai/vla/clawvla/src"},
            unset_env=["PYTHONPATH"],
        )
    )
    environment: EnvCommandConfig = field(
        default_factory=lambda: EnvCommandConfig(
            python="/mnt/wangwai/miniconda3/envs/robotwin-py312/bin/python",
            env={"PYTHONPATH": "/mnt/wangwai/tmp_pytorch3d_target:/mnt/wangwai/vla/clawvla/src"},
            unset_env=["PYTHONPATH"],
        )
    )
    openpi: EnvCommandConfig = field(
        default_factory=lambda: EnvCommandConfig(
            python="/mnt/wangwai/miniconda3/envs/openpi-torch-py312/bin/python",
            env={
                "PYTHONPATH": (
                    "/mnt/linyutong/wangwai_mirror/vla/clawvla/src:"
                    "/mnt/linyutong/wangwai_mirror/pi0.5/src"
                )
            },
        )
    )
    policy: PolicyConfig = field(
        default_factory=lambda: PolicyConfig("/mnt/wangwai/weights/Qwen/Qwen3-VL-8B-Instruct")
    )
    rollout: RolloutConfig = field(
        default_factory=lambda: RolloutConfig(
            base_config=(
                "/mnt/linyutong/wangwai_mirror/vla/clawvla/configs/robotwin_pi05_subtasks_25k.json"
            ),
            instruction="place the container on the plate",
        )
    )
    reward: RewardConfig = field(default_factory=RewardConfig)
    openrlhf: OpenRLHFConfig = field(default_factory=OpenRLHFConfig)
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
    rollout = _dataclass_from_dict(RolloutConfig, payload.get("rollout"), RLConfig().rollout)
    rollout.tasks = _normalize_rollout_tasks(rollout.tasks)
    return RLConfig(
        name=str(payload.get("name", "qwen3vl_pi05_grpo")),
        run_id=payload.get("run_id"),
        trainer=_env_command(payload.get("trainer"), RLConfig().trainer),
        robotwin=_env_command(payload.get("robotwin"), RLConfig().robotwin),
        environment=_env_command(payload.get("environment"), _env_command(payload.get("robotwin"), RLConfig().environment)),
        openpi=_env_command(payload.get("openpi"), RLConfig().openpi),
        policy=_dataclass_from_dict(PolicyConfig, payload.get("policy"), RLConfig().policy),
        rollout=rollout,
        reward=_dataclass_from_dict(RewardConfig, payload.get("reward"), RLConfig().reward),
        openrlhf=_dataclass_from_dict(OpenRLHFConfig, payload.get("openrlhf"), RLConfig().openrlhf),
        cluster=_dataclass_from_dict(ClusterConfig, payload.get("cluster"), RLConfig().cluster),
        logging=_dataclass_from_dict(LoggingConfig, payload.get("logging"), RLConfig().logging),
        checkpoint=_dataclass_from_dict(CheckpointConfig, payload.get("checkpoint"), RLConfig().checkpoint),
        metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {},
    )


def build_rollout_episode_specs(config: RLConfig) -> list[RolloutEpisodeSpec]:
    specs: list[RolloutEpisodeSpec] = []
    tasks = rollout_tasks(config)
    for task_index, task in enumerate(tasks):
        seeds = list(task.seeds or config.rollout.seeds or [0])
        for seed in seeds:
            specs.append(
                RolloutEpisodeSpec(
                    index=len(specs),
                    task_index=task_index,
                    task_name=task.task_name,
                    instruction=task.instruction,
                    seed=int(seed),
                    params=dict(task.params),
                )
            )
    if config.rollout.tasks:
        return specs

    count = max(1, config.rollout.episodes)
    seeds = list(config.rollout.seeds or [0])
    return [
        RolloutEpisodeSpec(
            index=index,
            task_index=0,
            task_name=config.rollout.task_name,
            instruction=config.rollout.instruction,
            seed=int(seeds[index % len(seeds)]),
            params={},
        )
        for index in range(count)
    ]


def rollout_tasks(config: RLConfig) -> list[RolloutTaskConfig]:
    if config.rollout.tasks:
        return list(config.rollout.tasks)
    return [
        RolloutTaskConfig(
            task_name=config.rollout.task_name,
            instruction=config.rollout.instruction,
            seeds=list(config.rollout.seeds),
            params={},
        )
    ]


def _normalize_rollout_tasks(value: object) -> list[RolloutTaskConfig]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"rollout.tasks must be a list, got {type(value).__name__}")
    tasks = []
    for index, item in enumerate(value):
        if isinstance(item, RolloutTaskConfig):
            tasks.append(item)
            continue
        if not isinstance(item, dict):
            raise TypeError(f"rollout.tasks[{index}] must be an object, got {type(item).__name__}")
        task_name = str(item.get("task_name") or "").strip()
        instruction = str(item.get("instruction") or "").strip()
        if not task_name:
            raise ValueError(f"rollout.tasks[{index}] is missing task_name")
        if not instruction:
            raise ValueError(f"rollout.tasks[{index}] is missing instruction")
        seeds_value = item.get("seeds", [])
        if seeds_value is None:
            seeds = []
        elif isinstance(seeds_value, list):
            seeds = [int(seed) for seed in seeds_value]
        else:
            raise TypeError(f"rollout.tasks[{index}].seeds must be a list, got {type(seeds_value).__name__}")
        params_value = item.get("params", {})
        if params_value is None:
            params = {}
        elif isinstance(params_value, dict):
            params = dict(params_value)
        else:
            raise TypeError(f"rollout.tasks[{index}].params must be an object, got {type(params_value).__name__}")
        tasks.append(RolloutTaskConfig(task_name=task_name, instruction=instruction, seeds=seeds, params=params))
    return tasks


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
