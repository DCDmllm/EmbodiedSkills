from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .config import RLConfig, build_rollout_episode_specs, dump_resolved_config, load_rl_config
from .service_pool import run_logged_subprocess


OPENRLHF_TRAINER_MODULE = "openrlhf.cli.train_ppo_ray"
DEFAULT_OPENRLHF_PYTHON = Path("/mnt/wangwai/vla/clawvla/.venv-openrlhf-py310-cu128/bin/python")
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
        print(f"run_dir={run_dir}")
        print(f"dataset={train_file}")
        return

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
        str(max(1, config.openrlhf.total_epochs)),
        "--algo.advantage.estimator",
        "group_norm" if group_size > 1 else "reinforce",
        "--algo.kl.init_coef",
        "0",
        "--actor.num_nodes",
        "1",
        "--actor.num_gpus_per_node",
        str(max(1, len(policy_gpus))),
        "--vllm.num_engines",
        str(num_engines),
        "--vllm.tensor_parallel_size",
        str(tensor_parallel_size),
        "--vllm.gpu_memory_utilization",
        str(gpu_memory_utilization),
        "--vllm.enable_prefix_caching",
        "--vllm.enforce_eager",
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
        "--logger.logging_steps",
        "1",
        "--logger.wandb.project",
        config.logging.wandb_project,
        "--logger.wandb.run_name",
        run_dir.name,
    ]
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
        return float(override)
    cap = 0.5 if int(config.openrlhf.tensor_parallel_size) == 1 else 0.25
    return min(float(config.openrlhf.gpu_memory_utilization), cap)


def _openrlhf_env(config: RLConfig, run_dir: Path, resolved_config_path: Path) -> dict[str, str]:
    tmp_dir = Path("/tmp") / "cvla_openrlhf" / hashlib.sha1(str(run_dir).encode("utf-8")).hexdigest()[:10]
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(config.trainer.env)
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
            "TMPDIR": str(tmp_dir),
            "TMP": str(tmp_dir),
            "TEMP": str(tmp_dir),
            "TRITON_CACHE_DIR": str(tmp_dir / "triton"),
        }
    )
    env.setdefault("RAY_CGRAPH_submit_timeout", "300")
    env.setdefault("RAY_CGRAPH_get_timeout", "300")
    if _env_bool("CLAWVLA_OPENRLHF_VLLM_ENABLE_SLEEP", default=True):
        allocator_conf = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments:True" in allocator_conf:
            env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    else:
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


if __name__ == "__main__":
    main()
