from __future__ import annotations

import argparse
import json

from clawvla.config import load_config
from clawvla.envs import RoboTwinAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test ClawVLA artifact capture without starting RoboTwin.")
    parser.add_argument("--config", default="configs/robotwin_default.json")
    parser.add_argument("--artifact-prefix", default="smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    adapter = RoboTwinAdapter(config.robotwin)
    raw_observation = {
        "observation": {
            "front_camera": {
                "rgb": [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
                "depth": [[0.1, 0.2], [0.3, 0.4]],
                "intrinsics": [1.0, 0.0, 0.0, 1.0],
                "extrinsics": [1.0, 0.0, 0.0, 0.0],
            }
        },
        "endpose": {
            "left_endpose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "left_gripper": 1.0,
            "right_endpose": [0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "right_gripper": 0.0,
        },
        "joint_action": {
            "left_arm": [0.0, 0.1],
            "right_arm": [0.2, 0.3],
        },
    }
    observation = adapter.capture_views(
        raw_observation=raw_observation,
        instruction="pick up the container and place it on the plate",
        artifact_prefix=args.artifact_prefix,
    )
    print(json.dumps(observation.to_dict(), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
