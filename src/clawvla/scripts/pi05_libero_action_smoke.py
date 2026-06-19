from __future__ import annotations

import argparse
import json

from clawvla.action_backends.factory import build_action_backend
from clawvla.config import load_config
from clawvla.envs import build_env_adapter
from clawvla.schema import MotionGoal, WorldState


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a pi05 LIBERO action chunk without executing it.")
    parser.add_argument("--config", default="configs/libero_pi05_enabled_probe.json")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--artifact-prefix", default="pi05_libero_action_smoke")
    parser.add_argument("--horizon", type=int, default=3)
    args = parser.parse_args()
    config = load_config(args.config)
    instruction = args.instruction or str(config.task.get("instruction") or "")
    if not instruction:
        raise ValueError("instruction is required for pi05 LIBERO action smoke")
    adapter = build_env_adapter(config)
    try:
        observation = adapter.capture_views(setup=True, instruction=instruction, artifact_prefix=args.artifact_prefix)
        backend = build_action_backend(config)
        result = backend.build_action_chunk(
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
        print(
            json.dumps(
                {
                    "status": result.status,
                    "success": result.success,
                    "errors": result.errors,
                    "backend_health": backend.health() if hasattr(backend, "health") else None,
                    "action_spec": backend.action_spec() if hasattr(backend, "action_spec") else None,
                    "action_chunk": result.action_chunk.to_dict() if result.action_chunk is not None else None,
                },
                ensure_ascii=True,
                indent=2,
            )
        )
        if not result.success:
            raise SystemExit(1)
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
