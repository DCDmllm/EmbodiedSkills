from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
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
    task_name: str | None = None,
    instruction: str | None = None,
    task_params: dict[str, Any] | None = None,
) -> EpisodeRecord:
    task_name = task_name or config.rollout.task_name
    instruction = instruction or config.rollout.instruction
    task_params = dict(task_params or {})
    episode = EpisodeRecord.new(task_name=task_name, instruction=instruction, seed=seed)
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
    rollout_lane = _allocate_rollout_lane(config, run_dir)
    openpi_port = _openpi_port_for_lane(config, rollout_lane)
    config_path = _write_agent_config(
        config,
        run_dir=run_dir,
        episode=episode,
        episode_index=episode_index,
        seed=seed,
        policy_base_url=policy_base_url,
        openpi_port=openpi_port,
        rollout_lane=rollout_lane,
        task_name=task_name,
        instruction=instruction,
        task_params=task_params,
    )
    result_path = run_dir / "trajectories" / f"{episode.episode_id}_result.json"
    log_path = run_dir / "logs" / f"{episode.episode_id}_agent.log"
    reward_path = run_dir / "rewards" / f"{episode.episode_id}_reward.jsonl"
    artifact_prefix = f"{config.rollout.artifact_prefix}_{episode_index}_{episode.episode_id}"
    environment_cmd = config.environment
    command = [
        environment_cmd.python,
        "-m",
        "clawvla.scripts.run_loop",
        "--config",
        str(config_path),
        "--instruction",
        instruction,
        "--artifact-prefix",
        artifact_prefix,
        "--initial-stage",
        config.rollout.initial_stage,
        "--max-steps",
        str(config.rollout.max_steps),
        "--result-output",
        str(result_path),
    ]
    if _should_run_environment(config):
        command.append("--run")
    env_extra = {
        "OPENAI_COMPATIBLE_API_KEY": config.policy.api_key,
        "OPENAI_COMPATIBLE_API_BASE_URL": policy_base_url,
        "CLAWVLA_RL_REWARD_JSONL": str(reward_path),
        "CLAWVLA_RL_TASK_NAME": task_name,
        "CLAWVLA_RL_STEP_COST": str(config.reward.step_cost),
        "CLAWVLA_RL_REWARD_REGISTRY": json.dumps(config.reward.registry, ensure_ascii=True),
        "CLAWVLA_RL_REWARD_TASK_MAP": json.dumps(config.reward.task_map, ensure_ascii=True),
    }
    if config.cluster.robotwin_gpus:
        env_extra["CUDA_VISIBLE_DEVICES"] = _gpu_for_rollout_lane(config.cluster.robotwin_gpus, rollout_lane)
    env = command_env(environment_cmd, env_extra)
    robotwin_ports = _service_pool_ports("CLAWVLA_ROBOTWIN_POOL_PORTS")
    if robotwin_ports:
        robotwin_port = robotwin_ports[rollout_lane % len(robotwin_ports)]
        completed = _run_persistent_robotwin_episode(
            port=robotwin_port,
            config_path=config_path,
            instruction=instruction,
            artifact_prefix=artifact_prefix,
            initial_stage=config.rollout.initial_stage,
            max_steps=config.rollout.max_steps,
            result_path=result_path,
            log_path=log_path,
            env_extra=env_extra,
            run_environment=_should_run_environment(config),
            timeout_s=config.rollout.episode_timeout_s,
        )
    else:
        completed = run_logged_subprocess(
            command,
            cwd=environment_cmd.cwd,
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
        if episode.status != "infra_failure":
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
    parser = argparse.ArgumentParser(description="Run one ClawVLA RL rollout episode in the configured environment.")
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
    rollout_lane: int = 0,
    task_name: str,
    instruction: str,
    task_params: dict[str, Any] | None = None,
) -> Path:
    base = json.loads(Path(config.rollout.base_config).read_text(encoding="utf-8"))
    base.setdefault("task", {})["instruction"] = instruction
    environment = base.setdefault("environment", {})
    if not isinstance(environment, dict):
        raise TypeError("agent config environment must be an object when present")
    environment["task_name"] = task_name
    environment["seed"] = seed
    environment["artifact_dir"] = str(run_dir / "artifacts" / episode.episode_id)
    if task_params:
        params = environment.setdefault("params", {})
        if not isinstance(params, dict):
            raise TypeError("agent config environment.params must be an object when present")
        params.update(dict(task_params))
    if str(environment.get("type") or "robotwin").lower() == "robotwin" or "robotwin" in base:
        base.setdefault("robotwin", {})["task_name"] = task_name
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
    _override_action_backend_runtime(base, config, openpi_port, rollout_lane)
    path = run_dir / "artifacts" / episode.episode_id / "agent_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _should_run_environment(config: RLConfig) -> bool:
    if config.rollout.run_environment is not None:
        return bool(config.rollout.run_environment)
    return bool(config.rollout.run_robotwin)


def _allocate_rollout_lane(config: RLConfig, run_dir: Path) -> int:
    del config
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    lock_path = artifact_dir / "openpi_ports.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        text = handle.read().strip()
        next_offset = int(text) if text else 0
        handle.seek(0)
        handle.truncate()
        handle.write(str(next_offset + 1))
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)
    return next_offset


def _allocate_openpi_port(config: RLConfig, run_dir: Path) -> int:
    """Compatibility wrapper for callers that only need a unique/pool port."""
    return _openpi_port_for_lane(config, _allocate_rollout_lane(config, run_dir))


def _openpi_port_for_lane(config: RLConfig, rollout_lane: int) -> int:
    ports = _service_pool_ports("CLAWVLA_OPENPI_POOL_PORTS")
    if ports:
        return ports[int(rollout_lane) % len(ports)]
    port = int(config.rollout.openpi_port_base) + int(rollout_lane)
    if port > 65535:
        raise RuntimeError(f"openpi_port_range_exhausted: base={config.rollout.openpi_port_base}")
    return port


def _service_pool_ports(name: str) -> list[int]:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return []
    ports = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not ports or any(port <= 0 or port > 65535 for port in ports):
        raise ValueError(f"Invalid {name}: {raw!r}")
    return ports


def _run_persistent_robotwin_episode(
    *,
    port: int,
    config_path: Path,
    instruction: str,
    artifact_prefix: str,
    initial_stage: str,
    max_steps: int,
    result_path: Path,
    log_path: Path,
    env_extra: dict[str, str],
    run_environment: bool,
    timeout_s: float,
) -> subprocess.CompletedProcess[str]:
    request = {
        "op": "run",
        "config_path": str(config_path),
        "instruction": instruction,
        "artifact_prefix": artifact_prefix,
        "initial_stage": initial_stage,
        "max_steps": int(max_steps),
        "result_output": str(result_path),
        "log_path": str(log_path),
        "run_environment": bool(run_environment),
        "initial_observe": False,
        "env": dict(env_extra),
    }
    command = ["persistent_robotwin_lane", f"127.0.0.1:{int(port)}"]
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=min(30.0, timeout_s)) as connection:
            connection.settimeout(float(timeout_s))
            stream = connection.makefile("rwb")
            stream.write((json.dumps(request, ensure_ascii=True) + "\n").encode("utf-8"))
            stream.flush()
            line = stream.readline()
        if not line:
            raise ConnectionError(f"RoboTwin lane {port} closed without a response")
        response = json.loads(line.decode("utf-8"))
        if not response.get("ok"):
            _append_lane_error(log_path, response)
            return subprocess.CompletedProcess(command, 1, json.dumps(response), "")
        return subprocess.CompletedProcess(command, 0, json.dumps(response), "")
    except Exception as exc:
        payload = {
            "ok": False,
            "status": "persistent_robotwin_request_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "port": int(port),
        }
        _append_lane_error(log_path, payload)
        return subprocess.CompletedProcess(command, 1, json.dumps(payload), "")


def _append_lane_error(log_path: Path, payload: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _override_action_backend_runtime(
    base: dict[str, Any],
    config: RLConfig,
    openpi_port: int,
    rollout_lane: int,
) -> None:
    action_backend = base.setdefault("metadata", {}).setdefault("action_backend", {})
    if not isinstance(action_backend, dict):
        return
    backend_type = str(action_backend.get("type", "pi05")).lower()
    if backend_type in {"pi05", "pi0.5", "pi_05"}:
        _override_openpi_runtime(action_backend, config, openpi_port, rollout_lane)
        return
    if backend_type in {"groot", "gr00t", "gr00t_n1_5", "gr00t-n1.5"}:
        _override_groot_runtime(action_backend, config, openpi_port, rollout_lane)
        return


def _override_openpi_runtime(
    action_backend: dict[str, Any],
    config: RLConfig,
    openpi_port: int,
    rollout_lane: int,
) -> None:
    runtime = action_backend.setdefault("openpi_runtime", {})
    if not isinstance(runtime, dict):
        return
    runtime["conda_env"] = "openpi-torch-py312"
    runtime["auto_start"] = bool(config.rollout.start_openpi_worker) and not bool(
        _service_pool_ports("CLAWVLA_OPENPI_POOL_PORTS")
    )
    runtime["pythonpath"] = config.openpi.env.get("PYTHONPATH", runtime.get("pythonpath"))
    runtime["port"] = int(openpi_port)
    if config.cluster.openpi_gpus:
        runtime["cuda_visible_devices"] = _gpu_for_rollout_lane(config.cluster.openpi_gpus, rollout_lane)


def _override_groot_runtime(
    action_backend: dict[str, Any],
    config: RLConfig,
    openpi_port: int,
    rollout_lane: int,
) -> None:
    runtime = action_backend.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        return
    runtime["mode"] = "worker"
    runtime["auto_start"] = bool(config.rollout.start_openpi_worker)
    runtime["port"] = int(openpi_port)
    if config.environment.env.get("PYTHONPATH"):
        runtime.setdefault("pythonpath", config.environment.env["PYTHONPATH"])
    if config.cluster.openpi_gpus:
        runtime["cuda_visible_devices"] = _gpu_for_rollout_lane(config.cluster.openpi_gpus, rollout_lane)


def _gpu_for_rollout_lane(gpus: list[int], rollout_lane: int) -> str:
    if not gpus:
        raise ValueError("cannot select a rollout GPU from an empty list")
    return str(gpus[int(rollout_lane) % len(gpus)])


def _populate_episode_from_result(episode: EpisodeRecord, payload: dict[str, Any]) -> None:
    loop = payload.get("loop") if isinstance(payload.get("loop"), dict) else {}
    episode.status = str(loop.get("status") or "unknown")
    task_status = payload.get("task_status") if isinstance(payload.get("task_status"), dict) else {}
    episode.metadata["task_status"] = dict(task_status)
    episode.metadata["official_task_success"] = bool(task_status.get("success", False))
    episode.metadata["task_status_available"] = bool(task_status.get("available", bool(task_status)))
    task_status_reason = str(task_status.get("reason") or "")
    if not episode.metadata["task_status_available"] and task_status_reason.startswith("task_status_failed:"):
        episode.status = "infra_failure"
        episode.errors.append(task_status_reason)
    if loop.get("reason"):
        episode.metadata["loop_reason"] = loop.get("reason")
    for item in loop.get("steps") or []:
        if not isinstance(item, dict):
            continue
        decision = item.get("decision") if isinstance(item.get("decision"), dict) else {}
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        record_status = str(item.get("status") or result.get("status") or "")
        episode.skill_calls.append(
            SkillCallTrace(
                step_index=item.get("step_index"),
                stage=item.get("stage_before"),
                component=str(decision.get("next_component") or ""),
                skill=str(decision.get("next_skill") or ""),
                status=record_status,
                success=bool(result.get("success")) and record_status not in {
                    "invalid_decision",
                    "scheduler_failed",
                    "skill_exception",
                    "skill_failed",
                },
                errors=[str(error) for error in result.get("errors") or []],
                output_keys=sorted(str(key) for key in (result.get("output") or {}).keys())
                if isinstance(result.get("output"), dict)
                else [],
            )
        )
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


RECOVERABLE_PREFLIGHT_ERRORS = {
    "stale_perception",
    "stale_world_state",
    "world_state_requires_reobserve",
    "missing_observation",
    "missing_observation_id",
}


def _append_episode_terminal_reward(episode: EpisodeRecord, reward_path: Path, config: RLConfig) -> None:
    stalled_loop = episode.status == "stalled_loop"
    budget_exhausted = episode.status in {"max_steps_reached", "max_steps_reached_with_failures"}
    stalled_or_exhausted = stalled_loop or budget_exhausted
    penalty_skill_calls = _skill_calls_excluding_stall_trigger(episode)
    invalid_decisions = sum(1 for skill in penalty_skill_calls if skill.status == "invalid_decision")
    recoverable_preflight_failures = sum(1 for skill in penalty_skill_calls if _is_recoverable_preflight_failure(skill))
    failed_skills = sum(
        1
        for skill in penalty_skill_calls
        if skill.status != "invalid_decision" and not skill.success and not _is_recoverable_preflight_failure(skill)
    )
    official_success, success_source = _episode_task_success(episode, config)
    loop_finished = episode.status == "finished"
    incomplete = not official_success
    premature_finish = bool(loop_finished and not official_success)
    penalty = (
        (float(config.reward.incomplete_episode_penalty) if incomplete and not stalled_or_exhausted else 0.0)
        + (float(config.reward.premature_finish_penalty) if premature_finish else 0.0)
        + (
            float(config.reward.stalled_loop_penalty)
            if stalled_or_exhausted and not official_success
            else 0.0
        )
        + invalid_decisions * float(config.reward.invalid_decision_penalty)
        + failed_skills * float(config.reward.skill_failure_penalty)
        + recoverable_preflight_failures * float(config.reward.recoverable_preflight_penalty)
    )
    terminal_success = official_success
    reward = RewardRecord(
        step_index=None,
        task_name=episode.task_name,
        reward=float(penalty),
        family="episode_terminal",
        reason=_terminal_reward_reason(
            episode.status,
            incomplete,
            premature_finish,
            stalled_loop,
            budget_exhausted,
            invalid_decisions,
            failed_skills,
            recoverable_preflight_failures,
        ),
        events={
            "episode_finished": terminal_success,
            "loop_finished": loop_finished,
            "official_task_success": official_success,
            "episode_incomplete": incomplete,
            "premature_finish": premature_finish,
            "stalled_loop": stalled_loop,
            "budget_exhausted": budget_exhausted,
            "invalid_decision_seen": invalid_decisions > 0,
            "skill_failure_seen": failed_skills > 0,
            "recoverable_preflight_failure_seen": recoverable_preflight_failures > 0,
        },
        metrics={
            "incomplete_episode": 1.0 if incomplete else 0.0,
            "loop_finished": 1.0 if loop_finished else 0.0,
            "official_task_success": 1.0 if official_success else 0.0,
            "premature_finish": 1.0 if premature_finish else 0.0,
            "stalled_loop": 1.0 if stalled_loop else 0.0,
            "budget_exhausted": 1.0 if budget_exhausted else 0.0,
            "invalid_decisions": float(invalid_decisions),
            "failed_skills": float(failed_skills),
            "recoverable_preflight_failures": float(recoverable_preflight_failures),
            "skill_calls": float(len(episode.skill_calls)),
            "penalized_skill_calls": float(len(penalty_skill_calls)),
        },
        metadata={
            "episode_status": episode.status,
            "success_source": success_source,
            "task_status": dict(episode.metadata.get("task_status") or {}),
            "errors": list(episode.errors),
        },
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


def _is_recoverable_preflight_failure(skill: SkillCallTrace) -> bool:
    if skill.success:
        return False
    if skill.component != "safety" or skill.skill != "preflight_action" or skill.status != "preflight_failed":
        return False
    errors = {str(error) for error in skill.errors}
    return bool(errors) and errors.issubset(RECOVERABLE_PREFLIGHT_ERRORS)


def _terminal_reward_reason(
    status: str,
    incomplete: bool,
    premature_finish: bool,
    stalled_loop: bool,
    budget_exhausted: bool,
    invalid_decisions: int,
    failed_skills: int,
    recoverable_preflight_failures: int = 0,
) -> str:
    parts = [f"episode_status={status}"]
    if incomplete:
        parts.append("incomplete_episode=1")
    if premature_finish:
        parts.append("premature_finish=1")
    if stalled_loop:
        parts.append("stalled_loop=1")
    if budget_exhausted:
        parts.append("budget_exhausted=1")
    if invalid_decisions:
        parts.append(f"invalid_decisions={invalid_decisions}")
    if failed_skills:
        parts.append(f"failed_skills={failed_skills}")
    if recoverable_preflight_failures:
        parts.append(f"recoverable_preflight_failures={recoverable_preflight_failures}")
    return ";".join(parts)


def _skill_calls_excluding_stall_trigger(episode: EpisodeRecord) -> list[SkillCallTrace]:
    calls = list(episode.skill_calls)
    reason = str(episode.metadata.get("loop_reason") or "")
    if episode.status != "stalled_loop" or not reason.startswith("repeated_failed_skill:") or len(calls) < 5:
        return calls
    tail = calls[-5:]
    signature = {(call.component, call.skill, call.status, call.success) for call in tail}
    if len(signature) == 1 and (not tail[0].success or tail[0].status == "invalid_decision"):
        return calls[:-5]
    return calls


def _episode_task_success(episode: EpisodeRecord, config: RLConfig) -> tuple[bool, str]:
    task_status = episode.metadata.get("task_status")
    if isinstance(task_status, dict) and "success" in task_status:
        return bool(task_status.get("success")), "environment_task_status"
    if "official_task_success" in episode.metadata:
        return bool(episode.metadata.get("official_task_success")), "episode_metadata"
    if not _should_run_environment(config):
        return episode.status == "finished", "non_environment_loop_status"
    return False, "missing_environment_task_status"


if __name__ == "__main__":
    main()
