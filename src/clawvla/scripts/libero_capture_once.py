from __future__ import annotations

import argparse
import json

from clawvla.config import load_config
from clawvla.envs import build_env_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one LIBERO observation through the ClawVLA env adapter.")
    parser.add_argument("--config", default="configs/libero_pi05_enabled_probe.json")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--artifact-prefix", default="libero_capture_once")
    args = parser.parse_args()
    config = load_config(args.config)
    adapter = build_env_adapter(config)
    try:
        observation = adapter.capture_views(
            setup=True,
            instruction=args.instruction or config.task.get("instruction"),
            artifact_prefix=args.artifact_prefix,
        )
        print(
            json.dumps(
                {
                    "status": "libero_observation_captured",
                    "adapter": adapter.metadata(),
                    "observation_id": observation.observation_id,
                    "camera_views": {name: view.to_dict() for name, view in observation.camera_views.items()},
                    "robot_arms": {name: arm.to_dict() for name, arm in observation.robot_arms.items()},
                    "raw": observation.raw,
                },
                ensure_ascii=True,
                indent=2,
            )
        )
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
