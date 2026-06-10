from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
from typing import Any

from .config import RLConfig, load_rl_config
from .service_pool import command_env, run_logged_subprocess
from .trajectory import EpisodeRecord, RewardRecord, SkillCallTrace, TrajectoryWriter


def run_rollout_episode(
    config: RLConfig,
    *,
    run_dir: Path,
    episode_index: int,
    seed: int,
    policy_base_url: str,
) -> EpisodeRecord:
    episode = EpisodeRecord.new(task_name=config.rollout.task_name, instruction=config.rollout.instruction, seed=seed)
    episode.status = "running"
    writer = TrajectoryWriter(run_dir / "events.jsonl")
    writer.write_event(
        "clawvla_rl_episode_start",
        {
            "episode_id": episode.episode_id,
            "episode_index": episode_index,
            "seed": seed,
            "policy_base_url": policy_base_url,
        },
    )
    openpi_port = _allocate_openpi_port(config, run_dir)
    config_path = _write_agent_config(
        config,
        run_dir=run_dir,
        episode=episode,
        episode_index=episode_index,
        seed=seed,
        policy_base_url=policy_base_url,
        openpi_port=openpi_port,
    )
    result_path = run_dir / "trajectories" / f"{episode.episode_id}_result.json"
    log_path = run_dir / "logs" / f"{episode.episode_id}_agent.log"
    reward_path = run_dir / "rewards" / f"{episode.episode_id}_reward.jsonl"
    artifact_prefix = f"{config.rollout.artifact_prefix}_{episode_index}_{episode.episode_id}"
    command = [
        config.robotwin.python,
        "-m",
        "clawvla.scripts.run_loop",
        "--config",
        str(config_path),
        "--instruction",
        config.rollout.instruction,
        "--artifact-prefix",
        artifact_prefix,
        "--initial-stage",
        config.rollout.initial_stage,
        "--max-steps",
        str(config.rollout.max_steps),
        "--result-output",
        str(result_path),
    ]
    if config.rollout.run_robotwin:
        command.append("--run")
    env_extra = {
        "OPENAI_COMPATIBLE_API_KEY": config.policy.api_key,
        "OPENAI_COMPATIBLE_API_BASE_URL": policy_base_url,
        "CLAWVLA_RL_REWARD_JSONL": str(reward_path),
        "CLAWVLA_RL_TASK_NAME": config.rollout.task_name,
        "CLAWVLA_RL_STEP_COST": str(config.reward.step_cost),
    }
    if config.cluster.robotwin_gpus:
        env_extra["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in config.cluster.robotwin_gpus)
    env = command_env(config.robotwin, env_extra)
    completed = run_logged_subprocess(
        command,
        cwd=config.robotwin.cwd,
        log_path=log_path,
        env=env,
        timeout=config.rollout.episode_timeout_s,
        writer=writer,
        event_prefix="clawvla_rl_agent",
    )
    episode.artifacts.update({
        "agent_log": str(log_path),
        "result_json": str(result_path),
        "agent_config": str(config_path),
        "reward_jsonl": str(reward_path),
    })
    if completed.returncode != 0:
        episode.status = "infra_failure"
        episode.errors.append(f"agent_return_code:{completed.returncode}")
        writer.write_event(
            "clawvla_rl_episode_infra_failure",
            {"episode_id": episode.episode_id, "errors": episode.errors},
        )
        writer.write_episode(episode)
        return episode
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        _populate_episode_from_result(episode, payload)
        _populate_episode_rewards(episode, reward_path)
        _append_episode_terminal_reward(episode, reward_path, config)
    else:
        episode.status = "infra_failure"
        episode.errors.append(f"missing_result_json:{result_path}")
    writer.write_event(
        "clawvla_rl_episode_finish",
        {"episode_id": episode.episode_id, "status": episode.status, "errors": episode.errors},
    )
    writer.write_episode(episode)
    return episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one ClawVLA RL rollout episode in the RoboTwin environment.")
    parser.add_argument("--rl-config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy-base-url", required=True)
    args = parser.parse_args()
    config = load_rl_config(args.rl_config)
    episode = run_rollout_episode(
        config,
        run_dir=Path(args.run_dir),
        episode_index=args.episode_index,
        seed=args.seed,
        policy_base_url=args.policy_base_url,
    )
    print(
        json.dumps(
            {"episode_id": episode.episode_id, "status": episode.status, "reward": episode.reward_score},
            ensure_ascii=True,
        ),
        flush=True,
    )


def _write_agent_config(
    config: RLConfig,
    *,
    run_dir: Path,
    episode: EpisodeRecord,
    episode_index: int,
    seed: int,
    policy_base_url: str,
    openpi_port: int,
) -> Path:
    base = json.loads(Path(config.rollout.base_config).read_text(encoding="utf-8"))
    base.setdefault("task", {})["instruction"] = config.rollout.instruction
    base.setdefault("robotwin", {})["task_name"] = config.rollout.task_name
    base["robotwin"]["seed"] = seed
    base["robotwin"]["artifact_dir"] = str(run_dir / "artifacts" / episode.episode_id)
    for role in config.policy.roles:
        base.setdefault("models", {})[role] = {
            **dict(base.get("models", {}).get(role, {})),
            "backend": "openai_compatible",
            "model": f"{config.policy.served_model_name}:{role}",
            "api_base_url": policy_base_url,
            "api_base_url_env": None,
            "api_key": config.policy.api_key,
            "api_key_env": None,
            "max_new_tokens": config.policy.max_new_tokens,
            "temperature": config.policy.temperature,
            "request_timeout": config.policy.request_timeout,
            "reasoning_effort": None,
        }
    _override_openpi_runtime(base, config, openpi_port)
    path = run_dir / "artifacts" / episode.episode_id / "agent_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _allocate_openpi_port(config: RLConfig, run_dir: Path) -> int:
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    lock_path = artifact_dir / "openpi_ports.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        text = handle.read().strip()
        next_offset = int(text) if text else 0
        port = int(config.rollout.openpi_port_base) + next_offset
        if port > 65535:
            raise RuntimeError(f"openpi_port_range_exhausted: base={config.rollout.openpi_port_base}")
        handle.seek(0)
        handle.truncate()
        handle.write(str(next_offset + 1))
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)
    return port


def _override_openpi_runtime(base: dict[str, Any], config: RLConfig, openpi_port: int) -> None:
    action_backend = base.setdefault("metadata", {}).setdefault("action_backend", {})
    if not isinstance(action_backend, dict):
        return
    runtime = action_backend.setdefault("openpi_runtime", {})
    if not isinstance(runtime, dict):
        return
    runtime["conda_env"] = "openpi-torch-py312"
    runtime["auto_start"] = bool(config.rollout.start_openpi_worker)
    runtime["pythonpath"] = config.openpi.env.get("PYTHONPATH", runtime.get("pythonpath"))
    runtime["port"] = int(openpi_port)
    if config.cluster.openpi_gpus:
        runtime["cuda_visible_devices"] = ",".join(str(item) for item in config.cluster.openpi_gpus)


def _populate_episode_from_result(episode: EpisodeRecord, payload: dict[str, Any]) -> None:
    loop = payload.get("loop") if isinstance(payload.get("loop"), dict) else {}
    episode.status = str(loop.get("status") or "unknown")
    if loop.get("reason"):
        episode.metadata["loop_reason"] = loop.get("reason")
    for item in loop.get("steps") or []:
        if not isinstance(item, dict):
            continue
        decision = item.get("decision") if isinstance(item.get("decision"), dict) else {}
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        episode.skill_calls.append(
            SkillCallTrace(
                step_index=item.get("step_index"),
                stage=item.get("stage_before"),
                component=str(decision.get("next_component") or ""),
                skill=str(decision.get("next_skill") or ""),
                status=str(item.get("status") or result.get("status") or ""),
                success=bool(result.get("success")),
                errors=[str(error) for error in result.get("errors") or []],
                output_keys=sorted(str(key) for key in (result.get("output") or {}).keys())
                if isinstance(result.get("output"), dict)
                else [],
            )
        )
    if episode.status in {"finished"}:
        episode.reward_score = 0.0
    else:
        episode.reward_score = None


def _populate_episode_rewards(episode: EpisodeRecord, reward_path: Path) -> None:
    if not reward_path.exists():
        return
    for line in reward_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("event") != "clawvla_rl_reward_record":
            continue
        reward_payload = payload.get("reward") if isinstance(payload.get("reward"), dict) else {}
        episode.rewards.append(
            RewardRecord(
                step_index=payload.get("step_index"),
                task_name=str(reward_payload.get("task_name") or episode.task_name),
                reward=float(reward_payload.get("reward", 0.0)),
                family=reward_payload.get("family"),
                reason=str(reward_payload.get("reason") or ""),
                events=dict(reward_payload.get("events") or {}),
                metrics=dict(reward_payload.get("metrics") or {}),
                milestones=dict(reward_payload.get("milestones") or {}),
            )
        )
    if episode.rewards:
        episode.reward_score = float(sum(item.reward for item in episode.rewards))


def _append_episode_terminal_reward(episode: EpisodeRecord, reward_path: Path, config: RLConfig) -> None:
    invalid_decisions = sum(1 for skill in episode.skill_calls if skill.status == "invalid_decision")
    failed_skills = sum(1 for skill in episode.skill_calls if skill.status != "invalid_decision" and not skill.success)
    incomplete = episode.status != "finished"
    penalty = (
        (float(config.reward.incomplete_episode_penalty) if incomplete else 0.0)
        + invalid_decisions * float(config.reward.invalid_decision_penalty)
        + failed_skills * float(config.reward.skill_failure_penalty)
    )
    terminal_success = episode.status == "finished"
    reward = RewardRecord(
        step_index=None,
        task_name=episode.task_name,
        reward=float(penalty),
        family="episode_terminal",
        reason=_terminal_reward_reason(episode.status, invalid_decisions, failed_skills),
        events={
            "episode_finished": terminal_success,
            "episode_incomplete": incomplete,
            "invalid_decision_seen": invalid_decisions > 0,
            "skill_failure_seen": failed_skills > 0,
        },
        metrics={
            "incomplete_episode": 1.0 if incomplete else 0.0,
            "invalid_decisions": float(invalid_decisions),
            "failed_skills": float(failed_skills),
            "skill_calls": float(len(episode.skill_calls)),
        },
        metadata={"episode_status": episode.status, "errors": list(episode.errors)},
    )
    episode.rewards.append(reward)
    episode.reward_score = float(sum(item.reward for item in episode.rewards))
    payload = {
        "event": "clawvla_rl_reward_record",
        "step_index": reward.step_index,
        "reward": {
            "task_name": reward.task_name,
            "reward": reward.reward,
            "family": reward.family,
            "reason": reward.reason,
            "events": reward.events,
            "metrics": reward.metrics,
            "milestones": reward.milestones,
            "metadata": reward.metadata,
        },
    }
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    with reward_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _terminal_reward_reason(status: str, invalid_decisions: int, failed_skills: int) -> str:
    parts = [f"episode_status={status}"]
    if status != "finished":
        parts.append("incomplete_episode=1")
    if invalid_decisions:
        parts.append(f"invalid_decisions={invalid_decisions}")
    if failed_skills:
        parts.append(f"failed_skills={failed_skills}")
    return ";".join(parts)


if __name__ == "__main__":
    main()
