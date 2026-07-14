from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import random
from typing import Any
from uuid import uuid4

import yaml


_PROJECT_ROOT_PATH = Path(
    os.environ.get(
        "CLAWVLA_PROJECT_ROOT",
        Path(__file__).resolve().parents[3],
    )
).expanduser().resolve()
_DEFAULT_WORKSPACE_ROOT = (
    _PROJECT_ROOT_PATH.parent.parent
    if _PROJECT_ROOT_PATH.parent.name == "vla"
    else _PROJECT_ROOT_PATH.parent
)
PROJECT_ROOT = str(_PROJECT_ROOT_PATH)
WORKSPACE_ROOT = str(
    Path(os.environ.get("CLAWVLA_WORKSPACE_ROOT", _DEFAULT_WORKSPACE_ROOT))
    .expanduser()
    .resolve()
)


@dataclass
class EnvCommandConfig:
    python: str
    env: dict[str, str] = field(default_factory=dict)
    unset_env: list[str] = field(default_factory=list)
    cwd: str = PROJECT_ROOT


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
    source: str = "configured"
    planner_reference_available: bool = False


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
    persistent_openpi_workers: bool = False
    openpi_worker_count: int | None = None
    persistent_robotwin_workers: bool = False
    robotwin_worker_count: int | None = None
    robotwin_worker_port_base: int = 18765


@dataclass
class RolloutSeedMixConfig:
    enabled: bool = False
    valid_seed_cache_dir: str = (
        f"{PROJECT_ROOT}/runs/eval/robotwin_train_grounding_valid_seeds_seed2_30/valid_seeds"
    )
    expert_plan_ratio: float = 0.6
    shuffle_seed: int = 42
    max_grounding_seeds: int | None = None
    max_prompts: int | None = None


@dataclass
class RewardConfig:
    registry: list[str] = field(default_factory=lambda: ["clawvla.rl.reward_registry:register_builtin_robotwin"])
    task_map: dict[str, str] = field(default_factory=dict)
    step_cost: float = 0.05
    incomplete_episode_penalty: float = -1.0
    premature_finish_penalty: float = -4.0
    stalled_loop_penalty: float = -8.0
    invalid_decision_penalty: float = -2.0
    skill_failure_penalty: float = -1.0
    recoverable_preflight_penalty: float = -0.1
    infra_failure_reward: float | None = None


@dataclass
class PlannerAuxConfig:
    enabled: bool = False
    dataset_root: str = f"{PROJECT_ROOT}/runs/data/robotwin_expert_subtasks_train_50x50_merged"
    repair_ledger: str = (
        f"{PROJECT_ROOT}/runs/data/robotwin_expert_subtasks_train_50x50_merged/annotation_repairs/"
        "subtask_repairs_gpt-5.6-sol_all_unpolished.jsonl"
    )
    split_manifest: str = (
        f"{PROJECT_ROOT}/runs/data/robotwin_expert_subtasks_train_50x50_merged/splits/"
        "task_stratified_seed42_val5.json"
    )
    split_name: str = "train"
    advantage_weight: float = 0.2
    max_reference_plans_per_task: int = 64


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
    enforce_eager: bool = True
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
    kl_init_coef: float = 1e-3
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
    run_root: str = f"{PROJECT_ROOT}/runs/rl"
    tmp_root: str = f"{PROJECT_ROOT}/tmp_runs/rl"
    rich: bool = True
    wandb_mode: str = "disabled"
    wandb_project: str = "clawvla-agent-rl"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_tags: list[str] = field(default_factory=list)
    wandb_sample_log_freq: int = 20
    upload_artifacts: bool = False


@dataclass
class CheckpointConfig:
    output_dir: str = f"{PROJECT_ROOT}/checkpoints/rl"
    save_freq: int = 20
    # None preserves every periodic checkpoint. A positive integer enables
    # OpenRLHF's normal checkpoint rotation.
    keep_last: int | None = 3
    resume: str | None = None


@dataclass
class RLConfig:
    name: str = "qwen3vl_pi05_grpo"
    run_id: str | None = None
    trainer: EnvCommandConfig = field(
        default_factory=lambda: EnvCommandConfig(
            python=f"{PROJECT_ROOT}/.venv-openrlhf-py310-cu128/bin/python"
        )
    )
    robotwin: EnvCommandConfig = field(
        default_factory=lambda: EnvCommandConfig(
            python="/mnt/wangwai/miniconda3/envs/robotwin-py312/bin/python",
            env={"PYTHONPATH": f"/mnt/wangwai/tmp_pytorch3d_target:{PROJECT_ROOT}/src"},
            unset_env=["PYTHONPATH"],
        )
    )
    environment: EnvCommandConfig = field(
        default_factory=lambda: EnvCommandConfig(
            python="/mnt/wangwai/miniconda3/envs/robotwin-py312/bin/python",
            env={"PYTHONPATH": f"/mnt/wangwai/tmp_pytorch3d_target:{PROJECT_ROOT}/src"},
            unset_env=["PYTHONPATH"],
        )
    )
    openpi: EnvCommandConfig = field(
        default_factory=lambda: EnvCommandConfig(
            python="/mnt/wangwai/miniconda3/envs/openpi-torch-py312/bin/python",
            env={
                "PYTHONPATH": f"{PROJECT_ROOT}/src:{WORKSPACE_ROOT}/pi0.5/src"
            },
        )
    )
    policy: PolicyConfig = field(
        default_factory=lambda: PolicyConfig("/mnt/wangwai/weights/Qwen/Qwen3-VL-8B-Instruct")
    )
    rollout: RolloutConfig = field(
        default_factory=lambda: RolloutConfig(
            base_config=f"{PROJECT_ROOT}/configs/robotwin_pi05_subtasks_25k.json",
            instruction="place the container on the plate",
        )
    )
    reward: RewardConfig = field(default_factory=RewardConfig)
    planner_aux: PlannerAuxConfig = field(default_factory=PlannerAuxConfig)
    rollout_seed_mix: RolloutSeedMixConfig = field(default_factory=RolloutSeedMixConfig)
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
    return _expand_config_value(payload)


def _expand_config_value(value: Any) -> Any:
    """Resolve portable repository tokens and ordinary environment variables."""
    if isinstance(value, dict):
        return {key: _expand_config_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_config_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_config_value(item) for item in value)
    if not isinstance(value, str):
        return value
    expanded = value.replace("${PROJECT_ROOT}", PROJECT_ROOT).replace(
        "${WORKSPACE_ROOT}", WORKSPACE_ROOT
    )
    return os.path.expandvars(expanded)


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
        planner_aux=_dataclass_from_dict(PlannerAuxConfig, payload.get("planner_aux"), RLConfig().planner_aux),
        rollout_seed_mix=_dataclass_from_dict(
            RolloutSeedMixConfig, payload.get("rollout_seed_mix"), RLConfig().rollout_seed_mix
        ),
        openrlhf=_dataclass_from_dict(OpenRLHFConfig, payload.get("openrlhf"), RLConfig().openrlhf),
        cluster=_dataclass_from_dict(ClusterConfig, payload.get("cluster"), RLConfig().cluster),
        logging=_dataclass_from_dict(LoggingConfig, payload.get("logging"), RLConfig().logging),
        checkpoint=_dataclass_from_dict(CheckpointConfig, payload.get("checkpoint"), RLConfig().checkpoint),
        metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {},
    )


def build_rollout_episode_specs(config: RLConfig) -> list[RolloutEpisodeSpec]:
    if config.rollout_seed_mix.enabled:
        return _build_online_seed_mix_specs(config)
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


def _build_online_seed_mix_specs(config: RLConfig) -> list[RolloutEpisodeSpec]:
    from .planner_similarity import load_planner_reference_index

    ratio = float(config.rollout_seed_mix.expert_plan_ratio)
    if not 0 < ratio <= 1:
        raise ValueError(f"rollout_seed_mix.expert_plan_ratio must be in (0, 1], got {ratio}")
    if not config.planner_aux.enabled:
        raise ValueError("rollout_seed_mix requires planner_aux.enabled=true")

    tasks = rollout_tasks(config)
    task_indices = {task.task_name: index for index, task in enumerate(tasks)}
    references = load_planner_reference_index(
        config.planner_aux.dataset_root,
        config.planner_aux.repair_ledger,
        config.planner_aux.split_manifest,
        config.planner_aux.split_name,
        config.planner_aux.max_reference_plans_per_task,
    )
    expert_specs = [
        RolloutEpisodeSpec(
            index=0,
            task_index=task_indices[reference.task_name],
            task_name=reference.task_name,
            instruction=reference.task_instruction,
            seed=reference.seed,
            params={},
            source="expert_subgoals",
            planner_reference_available=True,
        )
        for task_name in task_indices
        for reference in references.get(task_name, ())
    ]
    if not expert_specs:
        raise ValueError("rollout_seed_mix found no expert plan episodes in the configured split")

    grounding_specs = _load_valid_seed_specs(
        Path(config.rollout_seed_mix.valid_seed_cache_dir),
        tasks,
        shuffle_seed=int(config.rollout_seed_mix.shuffle_seed),
    )
    grounding_count = round(len(expert_specs) * (1.0 - ratio) / ratio)
    if config.rollout_seed_mix.max_grounding_seeds is not None:
        grounding_count = min(grounding_count, max(0, int(config.rollout_seed_mix.max_grounding_seeds)))
    if grounding_count > len(grounding_specs):
        raise ValueError(
            "rollout_seed_mix does not have enough distinct grounding seeds: "
            f"need={grounding_count} available={len(grounding_specs)}"
        )

    mixed = [*expert_specs, *grounding_specs[:grounding_count]]
    random.Random(int(config.rollout_seed_mix.shuffle_seed)).shuffle(mixed)
    if config.rollout_seed_mix.max_prompts is not None:
        max_prompts = int(config.rollout_seed_mix.max_prompts)
        if max_prompts <= 0:
            raise ValueError(f"rollout_seed_mix.max_prompts must be positive, got {max_prompts}")
        mixed = mixed[:max_prompts]
    return [
        RolloutEpisodeSpec(
            index=index,
            task_index=spec.task_index,
            task_name=spec.task_name,
            instruction=spec.instruction,
            seed=spec.seed,
            params=dict(spec.params),
            source=spec.source,
            planner_reference_available=spec.planner_reference_available,
        )
        for index, spec in enumerate(mixed)
    ]


def _load_valid_seed_specs(
    cache_dir: Path,
    tasks: list[RolloutTaskConfig],
    *,
    shuffle_seed: int,
) -> list[RolloutEpisodeSpec]:
    cache_dir = cache_dir.expanduser().resolve()
    if (cache_dir / "valid_seeds").is_dir():
        cache_dir = cache_dir / "valid_seeds"
    per_task: dict[str, list[RolloutEpisodeSpec]] = {}
    for task_index, task in enumerate(tasks):
        path = cache_dir / f"{task.task_name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing valid seed cache for {task.task_name}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("valid") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise ValueError(f"valid seed cache has no valid[] list: {path}")
        task_specs = []
        seen: set[int] = set()
        for entry in entries:
            if not isinstance(entry, dict) or "seed" not in entry:
                continue
            seed = int(entry["seed"])
            if seed in seen:
                continue
            seen.add(seed)
            task_specs.append(
                RolloutEpisodeSpec(
                    index=0,
                    task_index=task_index,
                    task_name=task.task_name,
                    instruction=str(entry.get("instruction") or task.instruction),
                    seed=seed,
                    params={},
                    source="official_valid_grounding",
                    planner_reference_available=False,
                )
            )
        random.Random(int(shuffle_seed) + task_index).shuffle(task_specs)
        per_task[task.task_name] = task_specs

    # Round-robin keeps the selected grounding subset balanced over all tasks.
    ordered: list[RolloutEpisodeSpec] = []
    max_count = max((len(items) for items in per_task.values()), default=0)
    for item_index in range(max_count):
        for task in tasks:
            items = per_task[task.task_name]
            if item_index < len(items):
                ordered.append(items[item_index])
    return ordered


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
