from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

from ..archive import create_run_archive, write_preflight_report
from ..config import RLConfig, build_rollout_episode_specs, dump_resolved_config, load_rl_config, rollout_tasks
from ..policy_proxy import OpenAIForwardBackend, PolicyBackend, PolicyProxy, StaticPolicyBackend
from ..reward_registry import build_reward_registry
from ..rollout_worker import run_rollout_episode
from ..service_pool import command_env, run_logged_subprocess
from ..terminal import RLTerminal
from ..trajectory import EpisodeRecord, TrajectoryWriter, build_policy_call_adapter, load_jsonl


VALID_MODES = {"dry-run", "rollout-only", "train", "replay-reward", "replay-adapter"}
VERL_TRAINER_MODULE = "verl.trainer.main_ppo_sync"


def main() -> None:
    args = _parse_args()
    config = load_rl_config(args.config)
    if args.run_id:
        config.run_id = args.run_id
    terminal = RLTerminal(enabled=config.logging.rich and not args.no_rich)
    run_dir = create_run_archive(config, mode=args.mode)
    writer = TrajectoryWriter(run_dir / "events.jsonl")
    terminal.event("archive_created", run_dir=run_dir)
    report = _preflight(config, mode=args.mode, static_policy=bool(args.policy_response))
    write_preflight_report(run_dir, report)
    terminal.event("preflight", success=report["success"], errors=report["errors"])
    if not report["success"]:
        raise SystemExit(2)

    if args.mode == "dry-run":
        return
    if args.mode == "rollout-only":
        episodes = _run_rollout_only(
            config,
            run_dir=run_dir,
            writer=writer,
            terminal=terminal,
            policy_response=args.policy_response,
        )
        _maybe_log_wandb(config, run_dir, episodes)
        return
    if args.mode == "train":
        _run_train(config, run_dir=run_dir, writer=writer, terminal=terminal)
        return
    if args.mode == "replay-reward":
        _replay_reward(args.replay_path, terminal)
        return
    if args.mode == "replay-adapter":
        _replay_adapter(args.replay_path, terminal)
        return
    raise ValueError(f"Unsupported mode: {args.mode}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ClawVLA Agent RL runner.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="dry-run")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--policy-response",
        default=None,
        help="Explicit static model response for rollout pipeline smoke tests.",
    )
    parser.add_argument("--replay-path", default=None)
    parser.add_argument("--no-rich", action="store_true")
    return parser.parse_args()


def _preflight(config: RLConfig, *, mode: str, static_policy: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    paths = {
        "trainer_python": config.trainer.python,
        "robotwin_python": config.robotwin.python,
        "base_config": config.rollout.base_config,
    }
    if mode == "train":
        paths["model_path"] = config.policy.model_path
    for name, value in paths.items():
        if value and not Path(str(value)).exists():
            errors.append(f"missing_path:{name}:{value}")
    try:
        registry = build_reward_registry(config.reward.registry, config.reward.task_map)
        for task in rollout_tasks(config):
            registry.handler_for_task(task.task_name)
    except Exception as exc:
        errors.append(f"reward_registry_error:{type(exc).__name__}:{exc}")
    if mode == "rollout-only" and not static_policy and not config.policy.external_base_url:
        errors.append("rollout_only_requires_policy_response_or_policy_external_base_url")
    if mode == "train":
        try:
            subprocess.run(
                [
                    config.trainer.python,
                    "-c",
                    "import transfer_queue, verl, torch, vllm; import clawvla.rl.verl_agent_loop",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=command_env(
                    config.trainer,
                    {
                        "PYTHONPATH": "/mnt/wangwai/vla/clawvla/src",
                        "CLAWVLA_ENABLE_VERL_RUNTIME_PATCHES": "1",
                        "CLAWVLA_VERL_RUNTIME_PATCHES_STRICT": "1",
                    },
                ),
                timeout=60,
            )
        except Exception as exc:
            errors.append(f"verl_import_error:{type(exc).__name__}:{exc}")
    return {"success": not errors, "errors": errors, "warnings": warnings, "mode": mode, "config_name": config.name}


def _run_rollout_only(
    config: RLConfig,
    *,
    run_dir: Path,
    writer: TrajectoryWriter,
    terminal: RLTerminal,
    policy_response: str | None,
) -> list[EpisodeRecord]:
    backend = _policy_backend(config, policy_response=policy_response)
    proxy = PolicyProxy(
        host=config.policy.proxy_host,
        port=config.policy.proxy_port,
        backend=backend,
        trajectory_writer=writer,
    )
    proxy.start()
    terminal.event("service_policy_proxy_ready", base_url=proxy.base_url)
    episodes: list[EpisodeRecord] = []
    try:
        for spec in build_rollout_episode_specs(config):
            before_call_count = len(proxy.calls)
            terminal.event("episode_start", index=spec.index, seed=spec.seed, task=spec.task_name)
            episode = run_rollout_episode(
                config,
                run_dir=run_dir,
                episode_index=spec.index,
                seed=spec.seed,
                policy_base_url=proxy.base_url,
                task_name=spec.task_name,
                instruction=spec.instruction,
            )
            episode.policy_calls = list(proxy.calls[before_call_count:])
            if episode.status == "infra_failure":
                terminal.event(
                    "episode_failed",
                    episode_id=episode.episode_id,
                    status=episode.status,
                    errors=episode.errors,
                )
            else:
                terminal.event(
                    "episode_finish",
                    episode_id=episode.episode_id,
                    status=episode.status,
                    reward=episode.reward_score,
                    policy_calls=len(episode.policy_calls),
                )
            writer.write_episode(episode)
            episodes.append(episode)
    finally:
        proxy.stop()
        terminal.event("service_policy_proxy_stopped", base_url=proxy.base_url)
    return episodes


def _policy_backend(config: RLConfig, *, policy_response: str | None) -> PolicyBackend:
    if policy_response is not None:
        return StaticPolicyBackend(policy_response)
    if config.policy.external_base_url:
        return OpenAIForwardBackend(
            base_url=config.policy.external_base_url,
            api_key=config.policy.api_key,
            model=config.policy.served_model_name,
        )
    raise RuntimeError("policy_backend_unavailable")


def _run_train(config: RLConfig, *, run_dir: Path, writer: TrajectoryWriter, terminal: RLTerminal) -> None:
    train_file = _write_verl_dataset(config, run_dir, split="train")
    val_file = _write_verl_dataset(config, run_dir, split="val")
    agent_loop_config = _write_verl_agent_loop_config(config, run_dir)
    resolved_config_path = run_dir / "resolved_config.yaml"
    dump_resolved_config(config, resolved_config_path)
    command = _verl_train_command(config, run_dir, train_file, val_file, agent_loop_config, resolved_config_path)
    trainer_tmp_dir = Path("/tmp") / "cvla" / hashlib.sha1(str(run_dir).encode("utf-8")).hexdigest()[:10]
    trainer_tmp_dir.mkdir(parents=True, exist_ok=True)
    env = command_env(
        config.trainer,
        {
            "PYTHONPATH": "/mnt/wangwai/vla/clawvla/src",
            "CUDA_VISIBLE_DEVICES": ",".join(str(item) for item in config.cluster.policy_gpus),
            "CLAWVLA_VERL_FULL_GPU_WORKERS": "1" if config.verl.force_full_gpu_workers else "0",
            "CLAWVLA_ENABLE_VERL_RUNTIME_PATCHES": "1",
            "CLAWVLA_VERL_RUNTIME_PATCHES_STRICT": "1",
            "TMPDIR": str(trainer_tmp_dir),
            "TMP": str(trainer_tmp_dir),
            "TEMP": str(trainer_tmp_dir),
            "RAY_TMPDIR": str(trainer_tmp_dir),
            "WANDB_MODE": config.logging.wandb_mode,
        },
    )
    terminal.event("train_start", command=" ".join(command), log=run_dir / "logs" / "verl_train.log")
    completed = run_logged_subprocess(
        command,
        cwd=config.trainer.cwd,
        log_path=run_dir / "logs" / "verl_train.log",
        env=env,
        timeout=None,
        writer=writer,
        event_prefix="clawvla_rl_verl_train",
    )
    if completed.returncode != 0:
        terminal.event("train_failed", return_code=completed.returncode, log=run_dir / "logs" / "verl_train.log")
        raise SystemExit(completed.returncode)
    terminal.event("train_finish", return_code=completed.returncode, log=run_dir / "logs" / "verl_train.log")


def _write_verl_dataset(config: RLConfig, run_dir: Path, *, split: str) -> Path:
    path = run_dir / "artifacts" / f"{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    specs = build_rollout_episode_specs(config)
    if split != "train":
        specs = specs[: min(2, len(specs))]
    with path.open("w", encoding="utf-8") as handle:
        for spec in specs:
            row = {
                "prompt": [
                    {
                        "role": "user",
                        "content": (
                            "Run one ClawVLA RoboTwin episode. The policy must choose model outputs for "
                            f"the task: {spec.instruction}"
                        ),
                    }
                ],
                "data_source": "clawvla_robotwin",
                "agent_name": "clawvla_agent",
                "reward_model": {"ground_truth": spec.task_name},
                "extra_info": {
                    "index": spec.index,
                    "task_index": spec.task_index,
                    "seed": spec.seed,
                    "task_name": spec.task_name,
                    "instruction": spec.instruction,
                },
            }
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return path


def _write_verl_agent_loop_config(config: RLConfig, run_dir: Path) -> Path:
    path = run_dir / "artifacts" / "verl_agent_loop.yaml"
    payload = [
        {
            "name": "clawvla_agent",
            "_target_": "clawvla.rl.verl_agent_loop.ClawVLAAgentLoop",
            "rl_config_path": str(run_dir / "resolved_config.yaml"),
            "run_dir": str(run_dir),
        }
    ]
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return path


def _verl_train_command(
    config: RLConfig,
    run_dir: Path,
    train_file: Path,
    val_file: Path,
    agent_loop_config: Path,
    resolved_config_path: Path,
) -> list[str]:
    logger = "['console','wandb']" if config.logging.wandb_mode != "disabled" else "['console']"
    lora_rank = "32" if config.verl.train_mode == "lora" else "0"
    command = [
        config.trainer.python,
        "-m",
        VERL_TRAINER_MODULE,
        "algorithm.adv_estimator=grpo",
        f"trainer.project_name={config.logging.wandb_project}",
        f"trainer.experiment_name={config.resolved_run_id()}",
        f"trainer.logger={logger}",
        f"trainer.n_gpus_per_node={len(config.cluster.policy_gpus)}",
        "trainer.nnodes=1",
        f"trainer.default_local_dir={run_dir / 'checkpoints'}",
        f"trainer.save_freq={config.checkpoint.save_freq}",
        f"trainer.max_actor_ckpt_to_keep={config.checkpoint.keep_last}",
        f"trainer.total_epochs={config.verl.total_epochs}",
        f"trainer.val_before_train={str(config.verl.val_before_train)}",
        f"trainer.test_freq={config.verl.test_freq}",
        "trainer.critic_warmup=0",
        "+ray_kwargs.ray_init.include_dashboard=False",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        "data.prompt_key=prompt",
        f"data.max_prompt_length={config.verl.max_prompt_length}",
        f"data.max_response_length={config.verl.max_response_length}",
        f"data.train_batch_size={max(1, config.rollout.group_size)}",
        f"data.val_batch_size={max(1, min(2, config.rollout.group_size))}",
        "data.return_raw_chat=True",
        "data.truncation=error",
        "data.trust_remote_code=True",
        f"actor_rollout_ref.model.path={config.policy.model_path}",
        "actor_rollout_ref.model.trust_remote_code=True",
        f"actor_rollout_ref.model.use_remove_padding={str(config.verl.use_remove_padding)}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={str(config.verl.gradient_checkpointing)}",
        f"actor_rollout_ref.model.lora_rank={lora_rank}",
        f"actor_rollout_ref.model.lora.rank={lora_rank}",
        f"actor_rollout_ref.model.lora.merge={str(config.verl.lora_merge_for_rollout)}",
        f"actor_rollout_ref.actor.optim.lr={config.verl.learning_rate}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={config.verl.actor_ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={config.verl.actor_ppo_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={config.verl.actor_ppo_max_token_len_per_gpu}",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.entropy_coeff=0",
        f"actor_rollout_ref.actor.fsdp_config.model_dtype={config.verl.fsdp_model_dtype}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={str(config.verl.fsdp_param_offload)}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={str(config.verl.fsdp_optimizer_offload)}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        f"actor_rollout_ref.rollout.dtype={config.verl.dtype}",
        f"actor_rollout_ref.rollout.n={config.verl.rollout_n}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={config.verl.tensor_parallel_size}",
        f"actor_rollout_ref.rollout.max_model_len={config.verl.max_model_len}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={config.verl.max_num_batched_tokens}",
        f"actor_rollout_ref.rollout.max_num_seqs={config.verl.max_num_seqs}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={config.verl.gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={config.verl.rollout_log_prob_micro_batch_size_per_gpu}",
        "actor_rollout_ref.rollout.enable_prefix_caching=True",
        "actor_rollout_ref.rollout.enable_chunked_prefill=True",
        "actor_rollout_ref.rollout.agent.default_agent_loop=clawvla_agent",
        f"actor_rollout_ref.rollout.agent.num_workers={config.cluster.ray_num_workers}",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={agent_loop_config}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={config.verl.ref_log_prob_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.ref.fsdp_config.model_dtype={config.verl.fsdp_model_dtype}",
        f"actor_rollout_ref.ref.fsdp_config.param_offload={str(config.verl.fsdp_param_offload)}",
        "reward.reward_model.enable=False",
        "critic.enable=False",
        f"+clawvla_rl_config={resolved_config_path}",
    ]
    if config.verl.train_mode == "lora":
        command.append(f"actor_rollout_ref.model.target_modules={_hydra_value(config.verl.lora_target_modules)}")
        if config.verl.lora_exclude_modules:
            command.append(f"actor_rollout_ref.model.exclude_modules={config.verl.lora_exclude_modules}")
    if config.verl.total_training_steps is not None:
        command.append(f"trainer.total_training_steps={config.verl.total_training_steps}")
    return command


def _hydra_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ",".join(str(item) for item in value) + "]"
    return str(value)


def _replay_reward(path: str | None, terminal: RLTerminal) -> None:
    if not path:
        raise ValueError("--replay-path is required for replay-reward")
    if not Path(path).exists():
        terminal.event("replay_failed", success=False, reason=f"missing_replay_path:{path}")
        raise SystemExit(2)
    records = load_jsonl(path)
    rewards = [item for item in records if str(item.get("event", "")).endswith("reward_record")]
    total = sum(
        float((item.get("reward") or {}).get("reward", 0.0))
        for item in rewards
        if isinstance(item.get("reward"), dict)
    )
    terminal.event("reward_replay", records=len(records), rewards=len(rewards), total=total)


def _replay_adapter(path: str | None, terminal: RLTerminal) -> None:
    if not path:
        raise ValueError("--replay-path is required for replay-adapter")
    records = load_jsonl(path)
    episodes = [
        item.get("episode")
        for item in records
        if item.get("event") == "clawvla_rl_episode" and isinstance(item.get("episode"), dict)
    ]
    checked = 0
    image_payload_unavailable = 0
    for episode in episodes:
        calls = episode.get("policy_calls") or []
        from ..trajectory import PolicyCallTrace

        traces = [PolicyCallTrace(**call) for call in calls if isinstance(call, dict)]
        for trace in traces:
            build_policy_call_adapter(trace, require_multimodal_payload=False)
            checked += 1
            if trace.image_refs:
                image_payload_unavailable += 1
    terminal.event(
        "adapter_replay",
        episodes=len(episodes),
        checked=checked,
        image_payload_unavailable_in_replay=image_payload_unavailable,
    )


def _maybe_log_wandb(config: RLConfig, run_dir: Path, episodes: list[EpisodeRecord]) -> None:
    if config.logging.wandb_mode == "disabled":
        return
    try:
        import wandb
    except Exception as exc:
        TrajectoryWriter(run_dir / "events.jsonl").write_event(
            "clawvla_rl_wandb_unavailable",
            {"error": f"{type(exc).__name__}: {exc}"},
        )
        return
    run = wandb.init(
        project=config.logging.wandb_project,
        entity=config.logging.wandb_entity,
        name=config.resolved_run_id(),
        tags=config.logging.wandb_tags,
        dir=str(run_dir),
        mode=config.logging.wandb_mode,
        config=asdict(config),
    )
    for index, episode in enumerate(episodes):
        wandb.log(
            {
                "rollout/episode_index": index,
                "rollout/reward": episode.reward_score,
                "rollout/policy_calls": len(episode.policy_calls),
                "rollout/skill_failures": sum(1 for item in episode.skill_calls if not item.success),
                "rollout/status": episode.status,
            }
        )
    if config.logging.upload_artifacts:
        artifact = wandb.Artifact(f"{config.resolved_run_id()}-logs", type="clawvla-rl-run")
        artifact.add_dir(str(run_dir))
        run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
