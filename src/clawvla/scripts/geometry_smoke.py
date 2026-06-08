from __future__ import annotations

import argparse
import json

from clawvla.config import load_config
from clawvla.envs import RoboTwinAdapter
from clawvla.runtime import AgentRuntime
from clawvla.schema import PerceptionResult, SceneCandidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test ClawVLA depth bbox geometry lifting.")
    parser.add_argument("--config", default="configs/robotwin_default.json")
    parser.add_argument("--artifact-prefix", default="geometry_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = AgentRuntime(config)
    adapter = RoboTwinAdapter(config.robotwin)
    runtime.blackboard.write("env_adapter", adapter)

    observation = adapter.capture_views(
        raw_observation=_raw_observation(),
        instruction="pick up the block",
        artifact_prefix=args.artifact_prefix,
    )
    perception = PerceptionResult(
        observation_id=observation.observation_id,
        candidates=[
            SceneCandidate(
                candidate_id="C1",
                label="block",
                bbox_by_view={"front_camera": [1, 1, 4, 4]},
                confidence=0.8,
                visibility="yes",
            )
        ],
        source_candidate_id="C1",
    )
    runtime.blackboard.write("observation", observation)
    runtime.blackboard.write("perception", perception)
    lift_result = runtime.run_skill(
        "vision",
        "lift_geometry",
        {"artifact_prefix": f"{args.artifact_prefix}/geometry", "min_points": 4},
    )
    state_result = runtime.run_skill("state", "update_world_state")
    print(
        json.dumps(
            {
                "lift": lift_result.output.get("geometry_summary"),
                "world_state": state_result.output.get("world_state"),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


def _raw_observation() -> dict[str, object]:
    return {
        "observation": {
            "front_camera": {
                "rgb": [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ],
                "depth": [
                    [0.50, 0.50, 0.50, 0.50, 0.50],
                    [0.50, 0.42, 0.42, 0.42, 0.50],
                    [0.50, 0.42, 0.40, 0.42, 0.50],
                    [0.50, 0.42, 0.42, 0.42, 0.50],
                    [0.50, 0.50, 0.50, 0.50, 0.50],
                ],
                "intrinsics": [100.0, 0.0, 2.0, 0.0, 100.0, 2.0, 0.0, 0.0, 1.0],
                "extrinsics": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            }
        },
        "endpose": {
            "left_endpose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "right_endpose": [0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        },
    }


if __name__ == "__main__":
    main()
