from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import netrc
import os
from pathlib import Path
import shutil
from typing import Any

from .config import PROJECT_ROOT, RLConfig, build_rollout_episode_specs, dump_resolved_config, load_rl_config
from .persistent_services import persistent_rollout_services, rollout_service_specs
from .service_pool import run_logged_subprocess


OPENRLHF_TRAINER_MODULE = "openrlhf.cli.train_ppo_ray"
DEFAULT_OPENRLHF_PYTHON = Path(PROJECT_ROOT) / ".venv-openrlhf-py310-cu128/bin/python"
RL_CONFIG_PRESETS = {
    "robotwin-multitask": "configs/rl/qwen3vl_pi05_multitask_1update.yaml",
    "robotwin-real5": "configs/rl/qwen3vl_pi05_real_5step_1update.yaml",
    "robotwin-real1": "configs/rl/qwen3vl_pi05_real_1update.yaml",
    "libero-multitask": "configs/rl/qwen3vl_pi05_libero_multitask_1update.yaml",
    "libero-single": "configs/rl/qwen3vl_pi05_libero_multitask_1update_single_gpu.yaml",
    "robocasa-rollout": "configs/rl/qwen3vl_groot_robocasa_rollout_smoke.yaml",
    "robocasa-1update": "configs/rl/qwen3vl_groot_robocasa_1update.yaml",
    "calvin-xvla": "configs/rl/qwen3vl_calvin_xvla_1update.yaml",
    "calvin-long-smoke": "configs/rl/qwen3vl_calvin_xvla_1update_long_smoke.yaml",
    "rynnbrain-real1": "configs/rl/rynnbrain2b_pi05_real_1update.yaml",
    "rynnbrain-train-smoke": "configs/rl/rynnbrain2b_pi05_train_smoke.yaml",
    "train-smoke": "configs/rl/qwen3vl_pi05_train_smoke.yaml",
    "rollout-smoke": "configs/rl/qwen3vl_pi05_rollout_smoke.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ClawVLA agent RL through OpenRLHF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=os.environ.get("CLAWVLA_RL_CONFIG"), help="RL config YAML path.")
    parser.add_argument(
        "--preset",
        choices=sorted(RL_CONFIG_PRESETS),
        default=os.environ.get("CLAWVLA_RL_PRESET"),
        help="Named RL config preset. --config takes precedence when both are set.",
    )
    parser.add_argument(
        "--mode",
        choices=["dry-run", "train"],
        default="dry-run",
        help="Print command or run training.",
    )
    parser.add_argument("--run-id", default=None, help="Run directory name under logging.run_root.")
    parser.add_argument(
        "--python",
        default=os.environ.get("CLAWVLA_OPENRLHF_PYTHON", str(DEFAULT_OPENRLHF_PYTHON)),
        help="Python executable used to launch OpenRLHF.",
    )
    args = parser.parse_args()

    config = load_rl_config(_resolve_config_path(args.config, args.preset))
    run_id = args.run_id or f"{config.resolved_run_id()}_openrlhf"
    run_dir = Path(config.logging.run_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("logs", "artifacts", "checkpoints"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)

    resolved_config_path = run_dir / "resolved_config.yaml"
    dump_resolved_config(config, resolved_config_path)
    train_file = _write_prompt_dataset(config, run_dir)
    command = _openrlhf_train_command(
        config,
        python=Path(args.python),
        run_dir=run_dir,
        train_file=train_file,
    )
    env = _openrlhf_env(config, run_dir, resolved_config_path)

    print(" ".join(command))
    if args.mode == "dry-run":
        for spec in rollout_service_specs(config, run_dir):
            print(
                f"persistent_service={spec.kind}[{spec.index}] "
                f"gpu={spec.gpu} port={spec.port} log={spec.log_path}"
            )
        print(f"run_dir={run_dir}")
        print(f"dataset={train_file}")
        return

    with persistent_rollout_services(config, run_dir, env):
        completed = run_logged_subprocess(
            command,
            cwd=config.trainer.cwd,
            log_path=run_dir / "logs" / "openrlhf_train.log",
            env=env,
            timeout=None,
            writer=None,
            event_prefix="clawvla_openrlhf_train",
        )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _resolve_config_path(config: str | None, preset: str | None) -> Path:
    if config:
        return _repo_relative_path(config)
    if preset:
        return _repo_relative_path(RL_CONFIG_PRESETS[preset])
    return _repo_relative_path(RL_CONFIG_PRESETS["robotwin-multitask"])


def _repo_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_prompt_dataset(config: RLConfig, run_dir: Path) -> Path:
    path = run_dir / "artifacts" / "openrlhf_prompts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    specs = build_rollout_episode_specs(config)
    with path.open("w", encoding="utf-8") as handle:
        for spec in specs:
            label = {
                "index": spec.index,
                "task_index": spec.task_index,
                "seed": spec.seed,
                "task_name": spec.task_name,
                "instruction": spec.instruction,
                "params": dict(spec.params),
                "source": spec.source,
                "planner_reference_available": spec.planner_reference_available,
            }
            row = {
                "input": (
                    "Run one ClawVLA embodied-agent episode. The policy must choose model outputs for "
                    f"the task: {spec.instruction}"
                ),
                "label": json.dumps(label, ensure_ascii=True),
                "datasource": "clawvla_environment",
            }
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return path


def _openrlhf_train_command(config: RLConfig, *, python: Path, run_dir: Path, train_file: Path) -> list[str]:
    policy_gpus = list(config.cluster.policy_gpus)
    tensor_parallel_size = max(1, int(config.openrlhf.tensor_parallel_size))
    if len(policy_gpus) % tensor_parallel_size != 0:
        raise ValueError(
            "OpenRLHF policy GPU count must divide tensor_parallel_size: "
            f"policy_gpus={policy_gpus} tensor_parallel_size={tensor_parallel_size}"
        )
    ds_tensor_parallel_size = _openrlhf_ds_tensor_parallel_size(config)
    if len(policy_gpus) % ds_tensor_parallel_size != 0:
        raise ValueError(
            "OpenRLHF policy GPU count must divide DeepSpeed tensor parallel size: "
            f"policy_gpus={policy_gpus} ds_tensor_parallel_size={ds_tensor_parallel_size}"
        )
    num_engines = max(1, len(policy_gpus) // tensor_parallel_size)
    group_size = max(1, int(config.rollout.group_size), int(config.openrlhf.rollout_n))
    train_batch_size = group_size
    rollout_batch_size = max(1, config.rollout.episodes)
    prompt_count = len(build_rollout_episode_specs(config))
    max_len = int(
        config.openrlhf.max_model_len or config.openrlhf.max_prompt_length + config.openrlhf.max_response_length
    )
    max_new_tokens = int(config.policy.max_new_tokens)
    zero_stage = _openrlhf_zero_stage(config)
    gpu_memory_utilization = _vllm_gpu_memory_utilization(config)
    attn_implementation = _openrlhf_attn_implementation(config)
    command = [
        str(python),
        "-m",
        OPENRLHF_TRAINER_MODULE,
        "--actor.model_name_or_path",
        config.policy.model_path,
        "--train.agent_func_path",
        str(Path(config.trainer.cwd) / "src" / "clawvla" / "rl" / "openrlhf_agent.py"),
        "--data.prompt_dataset",
        str(train_file),
        "--data.input_key",
        "input",
        "--data.label_key",
        "label",
        "--data.prompt_split",
        "train",
        "--data.max_len",
        str(max_len),
        "--data.max_samples",
        str(prompt_count),
        "--data.max_images_per_prompt",
        "16",
        "--rollout.max_new_tokens",
        str(max_new_tokens),
        "--rollout.batch_size",
        str(rollout_batch_size),
        "--rollout.vllm_generate_batch_size",
        str(rollout_batch_size),
        "--rollout.n_samples_per_prompt",
        str(group_size),
        "--rollout.micro_batch_size",
        "1",
        "--rollout.temperature",
        str(config.policy.temperature),
        "--train.batch_size",
        str(train_batch_size),
        "--train.micro_batch_size",
        "1",
        "--train.max_tokens_per_gpu",
        str(int(config.openrlhf.actor_ppo_max_token_len_per_gpu)),
        "--train.max_epochs",
        "1",
        "--train.num_episodes",
        str(max(1, config.openrlhf.total_training_steps or config.openrlhf.total_epochs)),
        "--algo.advantage.estimator",
        "group_norm" if group_size > 1 else "reinforce",
        "--algo.kl.init_coef",
        str(float(config.openrlhf.kl_init_coef)),
        "--actor.num_nodes",
        "1",
        "--actor.num_gpus_per_node",
        str(max(1, len(policy_gpus))),
        "--ref.num_nodes",
        "1",
        "--ref.num_gpus_per_node",
        str(max(1, len(policy_gpus))),
        "--vllm.num_engines",
        str(num_engines),
        "--vllm.tensor_parallel_size",
        str(tensor_parallel_size),
        "--vllm.gpu_memory_utilization",
        str(gpu_memory_utilization),
        "--vllm.enable_prefix_caching",
        "--train.colocate_all",
        "--ds.zero_stage",
        "2",
        "--ds.param_dtype",
        str(config.openrlhf.dtype).replace("bfloat16", "bf16"),
        "--ds.attn_implementation",
        attn_implementation,
        "--ds.tensor_parallel_size",
        str(ds_tensor_parallel_size),
        "--actor.adam.lr",
        str(config.openrlhf.learning_rate),
        "--ckpt.output_dir",
        str(run_dir / "checkpoints"),
        "--ckpt.path",
        str(run_dir / "checkpoints"),
        "--ckpt.save_steps",
        str(config.checkpoint.save_freq),
        "--ckpt.max_num",
        str(_checkpoint_max_num(config)),
        "--logger.logging_steps",
        "1",
        "--logger.wandb.project",
        config.logging.wandb_project,
        "--logger.wandb.run_name",
        run_dir.name,
    ]
    if _env_bool("CLAWVLA_OPENRLHF_VLLM_ENFORCE_EAGER", default=bool(config.openrlhf.enforce_eager)):
        command.append("--vllm.enforce_eager")
    command.extend(_wandb_command_args(config))
    if _env_bool("CLAWVLA_OPENRLHF_VLLM_ENABLE_SLEEP", default=True):
        command.append("--vllm.enable_sleep")
    if _env_bool("CLAWVLA_OPENRLHF_DS_ENABLE_SLEEP", default=bool(config.openrlhf.fsdp_param_offload)):
        command.append("--ds.enable_sleep")
    if _env_bool("CLAWVLA_OPENRLHF_ADAM_OFFLOAD", default=bool(config.openrlhf.fsdp_optimizer_offload)):
        command.append("--ds.adam_offload")
    if _env_bool("CLAWVLA_OPENRLHF_GRADIENT_CHECKPOINTING", default=bool(config.openrlhf.gradient_checkpointing)):
        command.append("--actor.gradient_checkpointing_enable")
    grad_accum_dtype = os.environ.get("CLAWVLA_OPENRLHF_GRAD_ACCUM_DTYPE")
    if grad_accum_dtype:
        command.extend(["--ds.grad_accum_dtype", grad_accum_dtype])
    if _env_bool("CLAWVLA_OPENRLHF_OVERLAP_COMM", default=False):
        command.append("--ds.overlap_comm")
    if _env_bool("CLAWVLA_OPENRLHF_FREEZE_VISUAL_ENCODER", default=False):
        command.append("--actor.freeze_visual_encoder")

    zero_stage_index = command.index("--ds.zero_stage") + 1
    command[zero_stage_index] = str(zero_stage)

    if config.openrlhf.train_mode == "lora":
        target_modules = _lora_target_modules(config.openrlhf.lora_target_modules)
        command.extend(
            [
                "--ds.lora.rank",
                "32",
                "--ds.lora.target_modules",
                *target_modules,
            ]
        )
    return command


def _lora_target_modules(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _openrlhf_zero_stage(config: RLConfig) -> int:
    override = os.environ.get("CLAWVLA_OPENRLHF_ZERO_STAGE")
    if override is not None:
        return int(override)
    return 3 if config.openrlhf.train_mode == "full" else 2


def _checkpoint_max_num(config: RLConfig) -> int:
    keep_last = config.checkpoint.keep_last
    if keep_last is None:
        # OpenRLHF requires an integer and has no explicit unlimited sentinel.
        # Its eviction check is count-based, so this effectively disables it.
        return 2_147_483_647
    value = int(keep_last)
    if value <= 0:
        raise ValueError(f"checkpoint.keep_last must be positive or null, got {keep_last}")
    return value


def _wandb_command_args(config: RLConfig) -> list[str]:
    mode = str(config.logging.wandb_mode or "disabled").strip().lower()
    if mode == "disabled":
        return []
    if mode not in {"online", "offline"}:
        raise ValueError(f"logging.wandb_mode must be disabled, online, or offline, got {mode!r}")
    if mode == "online" and not _wandb_auth_available():
        raise RuntimeError(
            "WandB online logging is enabled but no credentials were found. Run "
            "`.venv-openrlhf-py310-cu128/bin/python -m wandb login` or export WANDB_API_KEY."
        )
    # OpenRLHF uses this argument as the logger enable switch. The actual
    # credential remains in WANDB_API_KEY/~/.netrc and is never exposed in the
    # printed subprocess command or process list.
    args = ["--logger.wandb.key", "clawvla-auth-is-preconfigured"]
    if config.logging.wandb_entity:
        args.extend(["--logger.wandb.org", str(config.logging.wandb_entity)])
    if config.logging.wandb_group:
        args.extend(["--logger.wandb.group", str(config.logging.wandb_group)])
    return args


def _wandb_auth_available() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        credentials = netrc.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False
    return bool(credentials and credentials[2])


def _openrlhf_ds_tensor_parallel_size(config: RLConfig) -> int:
    del config
    override = os.environ.get("CLAWVLA_OPENRLHF_DS_TENSOR_PARALLEL_SIZE")
    if override is not None:
        return max(1, int(override))
    return 1


def _openrlhf_attn_implementation(config: RLConfig) -> str:
    override = os.environ.get("CLAWVLA_OPENRLHF_ATTN_IMPLEMENTATION")
    if override:
        return override
    return "flash_attention_2" if config.openrlhf.flash_attention else "eager"


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _vllm_gpu_memory_utilization(config: RLConfig) -> float:
    override = os.environ.get("CLAWVLA_OPENRLHF_VLLM_GPU_MEMORY_UTILIZATION")
    if override is not None:
        value = float(override)
    else:
        value = float(config.openrlhf.gpu_memory_utilization)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"vLLM gpu_memory_utilization must be in (0, 1], got {value}")
    return value


def _openrlhf_env(config: RLConfig, run_dir: Path, resolved_config_path: Path) -> dict[str, str]:
    tmp_override = os.environ.get("CLAWVLA_OPENRLHF_TMPDIR")
    tmp_dir = (
        Path(tmp_override).expanduser()
        if tmp_override
        else Path("/dev/shm")
        / "cvla_openrlhf"
        / hashlib.sha1(str(run_dir).encode("utf-8")).hexdigest()[:10]
    )
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(config.trainer.env)
    _ensure_ninja_on_path(env)
    wandb_dir = run_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "PYTHONPATH": str(Path(config.trainer.cwd) / "src"),
            "CUDA_VISIBLE_DEVICES": ",".join(str(item) for item in config.cluster.policy_gpus),
            "CLAWVLA_ENABLE_OPENRLHF_RUNTIME_PATCHES": "1",
            "CLAWVLA_OPENRLHF_RUNTIME_PATCHES_STRICT": "1",
            "CLAWVLA_OPENRLHF_TOKENIZER_COMPAT": "1",
            "CLAWVLA_OPENRLHF_RL_CONFIG": str(resolved_config_path),
            "CLAWVLA_OPENRLHF_RUN_DIR": str(run_dir),
            "WANDB_MODE": config.logging.wandb_mode,
            "WANDB_DIR": str(wandb_dir),
            "WANDB_RUN_ID": hashlib.sha1(run_dir.name.encode("utf-8")).hexdigest()[:16],
            "WANDB_RESUME": "allow",
            "CLAWVLA_WANDB_SAMPLE_LOG_FREQ": str(max(0, int(config.logging.wandb_sample_log_freq))),
            "TMPDIR": str(tmp_dir),
            "TMP": str(tmp_dir),
            "TEMP": str(tmp_dir),
            "RAY_TMPDIR": str(tmp_dir),
            "TRITON_CACHE_DIR": str(tmp_dir / "triton"),
        }
    )
    if str(config.logging.wandb_mode).strip().lower() == "offline":
        env.setdefault("WANDB_API_KEY", "clawvla-offline")
    if config.logging.wandb_tags:
        env["WANDB_TAGS"] = ",".join(str(tag) for tag in config.logging.wandb_tags)
    env.setdefault("RAY_CGRAPH_submit_timeout", "300")
    env.setdefault("RAY_CGRAPH_get_timeout", "300")
    if _env_bool("CLAWVLA_OPENRLHF_VLLM_ENABLE_SLEEP", default=True):
        allocator_conf = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments:True" in allocator_conf:
            env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    else:
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def _ensure_ninja_on_path(env: dict[str, str]) -> None:
    """Expose the ninja executable bundled with the active Python environment.

    The RL launcher intentionally invokes its venv's Python directly instead of
    activating the venv.  On the shared training machines, Python packages can
    also be supplied by a base conda environment.  In both cases the matching
    ``bin`` directory may be absent from PATH, which makes FlashInfer JIT fail
    even though the ``ninja`` Python package is installed.
    """
    path = env.get("PATH", os.defpath)
    if shutil.which("ninja", path=path):
        return

    candidates = [Path(os.sys.executable).resolve().parent / "ninja"]
    spec = importlib.util.find_spec("ninja")
    if spec is not None and spec.origin:
        origin = Path(spec.origin).resolve()
        candidates.extend(parent / "bin" / "ninja" for parent in origin.parents)

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            env["PATH"] = f"{candidate.parent}{os.pathsep}{path}"
            return


if __name__ == "__main__":
    main()
