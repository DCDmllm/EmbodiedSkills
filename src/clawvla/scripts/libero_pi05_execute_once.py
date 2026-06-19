from __future__ import annotations

import argparse
import json

from clawvla.action_backends.factory import build_action_backend
from clawvla.artifacts import _jsonable
from clawvla.config import load_config
from clawvla.envs import build_env_adapter
from clawvla.schema import MotionGoal, WorldState


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and execute one short pi05 LIBERO action chunk.")
    parser.add_argument("--config", default="configs/libero_pi05_enabled_probe.json")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--artifact-prefix", default="libero_pi05_execute_once")
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config)
    instruction = args.instruction or str(config.task.get("instruction") or "")
    if not instruction:
        raise ValueError("instruction is required for LIBERO pi05 execute smoke")
    adapter = build_env_adapter(config)
    try:
        observation = adapter.capture_views(setup=True, instruction=instruction, artifact_prefix=args.artifact_prefix)
        backend = build_action_backend(config)
        backend_result = backend.build_action_chunk(
            MotionGoal(skill="act", motion_hint=instruction),
            WorldState(task_instruction=instruction),
            observation,
            {
                "horizon": args.horizon,
                "motion_plan": {
                    "status": "image_grounded_motion_plan_built",
                    "vla_prompt": instruction,
                },
            },
        )
        if not backend_result.success or backend_result.action_chunk is None:
            print(json.dumps(backend_result.to_dict(), ensure_ascii=True, indent=2))
            raise SystemExit(1)
        execution = adapter.execute_action(backend_result.action_chunk)
        payload = _jsonable(
            {
                "status": execution.get("status"),
                "backend_result": backend_result.to_dict(),
                "execution": execution,
                "task_status": adapter.task_status() if hasattr(adapter, "task_status") else None,
            }
        )
        print(
            json.dumps(
                payload,
                ensure_ascii=True,
                indent=2,
            )
        )
        if execution.get("status") != "action_executed":
            raise SystemExit(1)
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
