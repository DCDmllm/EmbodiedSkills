from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clawvla.action_backends.groot import GrootActionBackend, observation_from_artifact
from clawvla.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GR00T backend inference path without executing RoboCasa.")
    parser.add_argument("--config", default="configs/robocasa_groot_enabled_probe.json")
    parser.add_argument("--artifact-dir", default="tmp_artifacts/robocasa")
    parser.add_argument("--prompt", default="pick up the object and place it in the target receptacle")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--pythonpath",
        action="append",
        default=[],
        help="Prepend an extra import path before probing, e.g. /mnt/wangwai/lerobot/src.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in reversed(args.pythonpath):
        if path and path not in sys.path:
            sys.path.insert(0, path)

    config = load_config(args.config)
    backend = GrootActionBackend(dict(config.metadata.get("action_backend", {})))
    observation = observation_from_artifact(Path(args.artifact_dir), args.prompt)
    result = backend.build_action_chunk(
        motion_goal=None,
        world_state=None,
        observation=observation,
        request={"motion_plan": {"vla_prompt": args.prompt}, "horizon": args.horizon},
    )
    payload = result.to_dict()
    rendered = json.dumps(payload, indent=2, ensure_ascii=True)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
