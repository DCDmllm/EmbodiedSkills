from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from clawvla.action_backends.factory import build_action_backend
from clawvla.artifacts import _jsonable
from clawvla.config import load_config
from clawvla.envs import build_env_adapter
from clawvla.schema import MotionGoal, WorldState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Request contract-checked X-VLA actions and optionally execute one bounded chunk in CALVIN."
    )
    parser.add_argument("--config", default="configs/calvin_xvla_enabled_probe.json")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--artifact-prefix", default="calvin_xvla_action_smoke")
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the first successful action chunk after all requests pass.",
    )
    return parser.parse_args()


def _request_payload(args: argparse.Namespace, instruction: str) -> dict[str, object]:
    request: dict[str, object] = {
        "motion_plan": {
            "status": "image_grounded_motion_plan_built",
            "vla_prompt": instruction,
        }
    }
    if args.horizon is not None:
        request["horizon"] = args.horizon
    if args.inference_steps is not None:
        request["inference_steps"] = args.inference_steps
    return request


def main() -> None:
    args = parse_args()
    if args.requests <= 0:
        raise ValueError(f"calvin_xvla_requests_must_be_positive:{args.requests}")
    if args.horizon is not None and args.horizon <= 0:
        raise ValueError(f"calvin_xvla_horizon_must_be_positive:{args.horizon}")
    if args.inference_steps is not None and args.inference_steps <= 0:
        raise ValueError(
            f"calvin_xvla_inference_steps_must_be_positive:{args.inference_steps}"
        )

    config = load_config(args.config)
    instruction = args.instruction or str(config.task.get("instruction") or "")
    if not instruction:
        raise ValueError("instruction is required for CALVIN X-VLA action smoke")
    adapter = build_env_adapter(config)
    exit_code = 0
    try:
        observation = adapter.capture_views(
            setup=True,
            instruction=instruction,
            artifact_prefix=f"{args.artifact_prefix}/observation",
        )
        backend = build_action_backend(config)
        request = _request_payload(args, instruction)
        request_results: list[dict[str, object]] = []
        first_chunk = None
        for index in range(args.requests):
            started = time.perf_counter()
            result = backend.build_action_chunk(
                MotionGoal(skill="act", motion_hint=instruction),
                WorldState(task_instruction=instruction),
                observation,
                request,
            )
            latency = time.perf_counter() - started
            chunk = result.action_chunk
            commands = np.asarray(chunk.commands if chunk is not None else [], dtype=np.float32)
            contract_ok = bool(
                result.success
                and chunk is not None
                and commands.ndim == 2
                and commands.shape[0] > 0
                and commands.shape[1] == 10
                and np.isfinite(commands).all()
            )
            if contract_ok and first_chunk is None:
                first_chunk = chunk
            if not contract_ok:
                exit_code = 1
            request_results.append(
                {
                    "index": index,
                    "latency_seconds": latency,
                    "contract_ok": contract_ok,
                    "result": result.to_dict(),
                }
            )

        execution = None
        if args.execute:
            if first_chunk is None:
                exit_code = 1
            else:
                first_chunk.metadata["artifact_prefix"] = args.artifact_prefix
                execution = adapter.execute_action(first_chunk)
                if execution.get("status") != "action_executed":
                    exit_code = 1

        latencies = [float(item["latency_seconds"]) for item in request_results]
        payload = _jsonable(
            {
                "status": (
                    "calvin_xvla_action_smoke_passed"
                    if exit_code == 0
                    else "calvin_xvla_action_smoke_failed"
                ),
                "config": str(Path(args.config).resolve()),
                "instruction": instruction,
                "request_count": args.requests,
                "contract_pass_count": sum(
                    bool(item["contract_ok"]) for item in request_results
                ),
                "latency_seconds": {
                    "min": min(latencies),
                    "max": max(latencies),
                    "mean": sum(latencies) / len(latencies),
                },
                "backend_health": backend.health(),
                "action_spec": backend.action_spec(),
                "observation_id": observation.observation_id,
                "requests": request_results,
                "execute_requested": args.execute,
                "execution": execution,
                "task_status": adapter.task_status(),
            }
        )
        evidence_path = adapter.artifacts.write_json(
            f"{args.artifact_prefix}/action_smoke_report.json", payload
        )
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "request_count": args.requests,
                    "contract_pass_count": payload["contract_pass_count"],
                    "latency_seconds": payload["latency_seconds"],
                    "execute_requested": args.execute,
                    "execution_status": (
                        execution.get("status") if isinstance(execution, dict) else None
                    ),
                    "oracle_success": (
                        execution.get("success") if isinstance(execution, dict) else None
                    ),
                    "evidence_path": evidence_path,
                },
                ensure_ascii=True,
                indent=2,
            )
        )
    finally:
        adapter.close()
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
