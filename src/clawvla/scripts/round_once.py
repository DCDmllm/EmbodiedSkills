from __future__ import annotations

import argparse
import json

from clawvla.config import load_config
from clawvla.envs import RoboTwinAdapter
from clawvla.runtime import AgentRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one ClawVLA observe/plan round.")
    parser.add_argument("--config", default="configs/robotwin_default.json")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--artifact-prefix", default="round_once")
    parser.add_argument("--run", action="store_true", help="Actually instantiate RoboTwin and call setup_demo/get_obs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = AgentRuntime(config)
    adapter = RoboTwinAdapter(config.robotwin)
    runtime.blackboard.write("env_adapter", adapter)
    runtime.blackboard.task_instruction = args.instruction

    capture_payload = {
        "instruction": args.instruction,
        "artifact_prefix": args.artifact_prefix,
    }
    if args.run:
        capture_payload["setup"] = True
    capture_result = runtime.run_skill("vision", "capture_views", capture_payload)
    observation = runtime.blackboard.read("observation")
    image_paths = [view.rgb_path for view in observation.camera_views.values() if view.rgb_path]

    perceive_result = runtime.run_skill(
        "vision",
        "perceive_scene",
        {"use_model": True, "image_paths": image_paths},
    )
    ground_result = runtime.run_skill(
        "vision",
        "ground_task_objects",
        {"use_model": True, "image_paths": image_paths},
    )
    uncertainty_result = runtime.run_skill(
        "vision",
        "estimate_uncertainty",
        {"use_model": True, "image_paths": image_paths},
    )
    world_state_result = runtime.run_skill("state", "update_world_state")
    decision_result = runtime.run_skill(
        "scheduler",
        "choose_next_skill",
        {
            "use_model": True,
            "available_components": list(config.components.keys()),
            "available_skills": {name: list(component.skills) for name, component in config.components.items()},
        },
    )

    print(
        json.dumps(
            {
                "capture": capture_result.to_dict(),
                "perceive": perceive_result.to_dict(),
                "ground": ground_result.to_dict(),
                "uncertainty": uncertainty_result.to_dict(),
                "world_state": world_state_result.output.get("world_state"),
                "decision": decision_result.output.get("decision"),
                "blackboard": runtime.blackboard.compact_context(),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
