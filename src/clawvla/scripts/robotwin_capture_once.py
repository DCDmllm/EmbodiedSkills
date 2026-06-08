from __future__ import annotations

import argparse
import json

from clawvla.config import load_config
from clawvla.envs import RoboTwinAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one public RoboTwin observation into ClawVLA artifacts.")
    parser.add_argument("--config", default="configs/robotwin_default.json")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--artifact-prefix", default="robotwin_capture_once")
    parser.add_argument("--run", action="store_true", help="Actually instantiate RoboTwin and call setup_demo/get_obs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    adapter = RoboTwinAdapter(config.robotwin)
    if not args.run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "message": "Pass --run to instantiate RoboTwin and capture get_obs().",
                    "robotwin": config.robotwin.__dict__,
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return

    observation = adapter.capture_views(
        setup=True,
        instruction=args.instruction,
        artifact_prefix=args.artifact_prefix,
    )
    print(json.dumps(observation.to_dict(), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
