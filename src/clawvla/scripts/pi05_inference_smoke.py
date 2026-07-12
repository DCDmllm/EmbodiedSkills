from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clawvla.config import load_config
from clawvla.action_backends.pi05 import Pi05ActionBackend
from clawvla.schema import CameraView, ObservationBundle, RobotArmState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pi0.5 backend inference path without executing RoboTwin.")
    parser.add_argument("--config", default="configs/robotwin_pi05_enabled_probe.json")
    parser.add_argument("--artifact-dir", default="tmp_artifacts/gpu_loop_execute_smoke")
    parser.add_argument("--prompt", default="place the container on the plate")
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--pythonpath",
        action="append",
        default=[],
        help="Prepend an extra import path before probing, e.g. RoboTwin/policy/pi05/src.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in reversed(args.pythonpath):
        if path and path not in sys.path:
            sys.path.insert(0, path)

    config = load_config(args.config)
    backend = Pi05ActionBackend(dict(config.metadata.get("action_backend", {})))
    observation = _observation_from_artifact(Path(args.artifact_dir), args.prompt)
    result = backend.build_action_chunk(
        motion_goal=None,
        world_state=None,
        observation=observation,
        request={
            "motion_plan": {"vla_prompt": args.prompt},
            "num_steps": args.num_steps,
            "horizon": args.horizon,
        },
    )
    payload = result.to_dict()
    rendered = json.dumps(payload, indent=2, ensure_ascii=True)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


def _observation_from_artifact(artifact_dir: Path, prompt: str) -> ObservationBundle:
    image_dir = artifact_dir / "images"
    summary_path = artifact_dir / "raw_observation_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    vector = payload.get("joint_action_vector")
    robot_arms = {}
    if isinstance(vector, list) and len(vector) == 14:
        robot_arms = {
            "left": RobotArmState(
                arm_name="left",
                joint_positions=[float(item) for item in vector[:6]],
                gripper_value=float(vector[6]),
            ),
            "right": RobotArmState(
                arm_name="right",
                joint_positions=[float(item) for item in vector[7:13]],
                gripper_value=float(vector[13]),
            ),
        }
    return ObservationBundle(
        task_instruction=prompt,
        camera_views={
            "head_camera": CameraView(name="head_camera", rgb_path=str(image_dir / "head_camera_rgb.png")),
            "left_camera": CameraView(name="left_camera", rgb_path=str(image_dir / "left_camera_rgb.png")),
            "right_camera": CameraView(name="right_camera", rgb_path=str(image_dir / "right_camera_rgb.png")),
        },
        robot_arms=robot_arms,
        raw={"summary_ref": str(summary_path)} if summary_path.exists() else {},
    )


if __name__ == "__main__":
    main()
