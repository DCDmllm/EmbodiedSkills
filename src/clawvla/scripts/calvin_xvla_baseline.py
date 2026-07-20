from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from clawvla.action_backends.factory import build_action_backend
from clawvla.artifacts import _jsonable
from clawvla.config import load_config
from clawvla.envs import build_env_adapter
from clawvla.schema import MotionGoal, WorldState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a planner-free X-VLA baseline episode in the configured CALVIN task."
    )
    parser.add_argument("--config", default="configs/calvin_xvla_enabled_probe.json")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--artifact-prefix", default="calvin_xvla_baseline")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated fixed environment seeds; overrides --episodes when provided.",
    )
    parser.add_argument("--max-env-steps", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--inference-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError(f"calvin_xvla_episodes_must_be_positive:{args.episodes}")
    for name in ("max_env_steps", "horizon", "inference_steps"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"calvin_xvla_{name}_must_be_positive:{value}")

    config = load_config(args.config)
    seeds = _fixed_seeds(args.seeds, args.episodes, config.environment.seed)
    instruction = args.instruction or str(config.task.get("instruction") or "")
    if not instruction:
        raise ValueError("instruction is required for CALVIN X-VLA baseline")
    adapter = build_env_adapter(config)
    backend = build_action_backend(config)
    infrastructure_ok = True
    try:
        episodes: list[dict[str, object]] = []
        for episode_index, seed in enumerate(seeds):
            config.environment.seed = seed
            observation = adapter.capture_views(
                setup=True,
                instruction=instruction,
                artifact_prefix=(
                    f"{args.artifact_prefix}/episode_{episode_index:03d}/reset"
                ),
            )
            max_env_steps = args.max_env_steps or adapter.max_episode_steps
            chunks: list[dict[str, object]] = []
            started = time.perf_counter()
            failure_reason = None
            while adapter.step_count < max_env_steps:
                remaining = max_env_steps - adapter.step_count
                request: dict[str, object] = {
                    "horizon": min(args.horizon or backend.action_spec()["horizon"], remaining),
                    "motion_plan": {
                        "status": "image_grounded_motion_plan_built",
                        "vla_prompt": instruction,
                    },
                }
                if args.inference_steps is not None:
                    request["inference_steps"] = args.inference_steps
                inference_started = time.perf_counter()
                result = backend.build_action_chunk(
                    MotionGoal(skill="act", motion_hint=instruction),
                    WorldState(task_instruction=instruction),
                    observation,
                    request,
                )
                inference_latency = time.perf_counter() - inference_started
                if not result.success or result.action_chunk is None:
                    infrastructure_ok = False
                    failure_reason = "xvla_action_request_failed"
                    chunks.append(
                        {
                            "index": len(chunks),
                            "inference_latency_seconds": inference_latency,
                            "backend_result": result.to_dict(),
                            "execution": None,
                        }
                    )
                    break
                result.action_chunk.metadata["artifact_prefix"] = (
                    f"{args.artifact_prefix}/episode_{episode_index:03d}/chunk_{len(chunks):03d}"
                )
                execution = adapter.execute_action(result.action_chunk)
                chunks.append(
                    {
                        "index": len(chunks),
                        "inference_latency_seconds": inference_latency,
                        "backend_result": result.to_dict(),
                        "execution": execution,
                    }
                )
                if execution.get("status") != "action_executed":
                    infrastructure_ok = False
                    failure_reason = str(execution.get("reason") or execution.get("status"))
                    break
                observation = adapter.last_observation
                if execution.get("success") or execution.get("done"):
                    break

            task_status = adapter.task_status()
            success = bool(task_status.get("success"))
            if not success and failure_reason is None:
                failure_reason = (
                    "environment_done"
                    if task_status.get("done") and adapter.step_count < max_env_steps
                    else "step_budget_exhausted"
                )
            episodes.append(
                {
                    "episode_index": episode_index,
                    "seed": config.environment.seed,
                    "initial_state": config.environment.params.get("initial_state"),
                    "success": success,
                    "failure_reason": failure_reason,
                    "step_count": adapter.step_count,
                    "chunk_count": len(chunks),
                    "elapsed_seconds": time.perf_counter() - started,
                    "task_status": task_status,
                    "chunks": chunks,
                }
            )

        success_count = sum(bool(item["success"]) for item in episodes)
        payload = _jsonable(
            {
                "status": (
                    "calvin_xvla_baseline_completed"
                    if infrastructure_ok
                    else "calvin_xvla_baseline_infrastructure_failed"
                ),
                "config": str(Path(args.config).resolve()),
                "instruction": instruction,
                "task_name": config.environment.task_name,
                "episodes": episodes,
                "summary": {
                    "episode_count": len(seeds),
                    "success_count": success_count,
                    "success_rate": success_count / len(seeds),
                    "environment_or_http_error_count": sum(
                        item["failure_reason"] not in {None, "step_budget_exhausted", "environment_done"}
                        for item in episodes
                    ),
                },
                "backend_health": backend.health(),
                "action_spec": backend.action_spec(),
            }
        )
        evidence_path = adapter.artifacts.write_json(
            f"{args.artifact_prefix}/baseline_report.json", payload
        )
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    **payload["summary"],
                    "episode_steps": [item["step_count"] for item in episodes],
                    "failure_reasons": [item["failure_reason"] for item in episodes],
                    "evidence_path": evidence_path,
                },
                ensure_ascii=True,
                indent=2,
            )
        )
    finally:
        adapter.close()
    if not infrastructure_ok:
        raise SystemExit(1)


def _fixed_seeds(value: str | None, episodes: int, default_seed: int | None) -> list[int]:
    if value is None:
        return [int(default_seed or 0)] * int(episodes)
    seeds = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not seeds:
        raise ValueError("calvin_xvla_seeds_must_not_be_empty")
    if len(seeds) != len(set(seeds)):
        raise ValueError("calvin_xvla_seeds_must_be_unique")
    return seeds


if __name__ == "__main__":
    main()
