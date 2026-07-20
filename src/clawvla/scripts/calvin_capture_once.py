from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from clawvla.artifacts import _jsonable
from clawvla.config import load_config
from clawvla.envs import build_env_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reset CALVIN and capture a contract-checked static/gripper observation."
    )
    parser.add_argument("--config", default="configs/calvin_xvla_enabled_probe.json")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--artifact-prefix", default="calvin_capture_once")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat reset/capture against the same configured initial state.",
    )
    return parser.parse_args()


def _capture_contract(observation) -> dict[str, object]:
    camera_names = sorted(observation.camera_views)
    if camera_names != ["gripper", "static"]:
        raise ValueError(f"calvin_capture_camera_contract_invalid:{camera_names}")
    for name, view in observation.camera_views.items():
        if not view.rgb_path or not Path(view.rgb_path).is_file():
            raise ValueError(f"calvin_capture_rgb_artifact_missing:{name}:{view.rgb_path}")
    proprio = np.asarray(observation.raw.get("calvin_proprio"), dtype=np.float32)
    if proprio.shape != (20,) or not np.isfinite(proprio).all():
        raise ValueError(f"calvin_capture_proprio_contract_invalid:{list(proprio.shape)}")
    return {
        "observation_id": observation.observation_id,
        "camera_views": {
            name: view.to_dict() for name, view in observation.camera_views.items()
        },
        "proprio_dim": int(proprio.shape[0]),
        "proprio": proprio.tolist(),
        "robot_obs": observation.raw.get("robot_obs"),
        "scene_obs": observation.raw.get("scene_obs"),
        "summary_ref": observation.raw.get("summary_ref"),
    }


def main() -> None:
    args = parse_args()
    if args.repeat <= 0:
        raise ValueError(f"calvin_capture_repeat_must_be_positive:{args.repeat}")
    config = load_config(args.config)
    instruction = args.instruction or str(config.task.get("instruction") or "")
    adapter = build_env_adapter(config)
    try:
        captures: list[dict[str, object]] = []
        for index in range(args.repeat):
            observation = adapter.capture_views(
                setup=True,
                instruction=instruction or None,
                artifact_prefix=f"{args.artifact_prefix}/reset_{index:03d}",
            )
            initial_info_ref = adapter.artifacts.write_json(
                f"{args.artifact_prefix}/reset_{index:03d}/initial_info.json",
                adapter.start_info,
            )
            task_status = adapter.task_status()
            captures.append(
                {
                    "index": index,
                    **_capture_contract(observation),
                    "initial_info_ref": initial_info_ref,
                    "task_status": {
                        key: task_status.get(key)
                        for key in (
                            "backend",
                            "task_name",
                            "task_language",
                            "subtask",
                            "success",
                            "done",
                            "step_count",
                            "reward",
                        )
                    },
                }
            )
        reference = np.asarray(captures[0]["proprio"], dtype=np.float32)
        max_reset_delta = max(
            float(np.max(np.abs(np.asarray(item["proprio"]) - reference)))
            for item in captures
        )
        payload = _jsonable(
            {
                "status": "calvin_capture_contract_passed",
                "config": str(Path(args.config).resolve()),
                "repeat": args.repeat,
                "instruction": instruction,
                "max_reset_proprio_abs_delta": max_reset_delta,
                "adapter": adapter.metadata(),
                "captures": captures,
            }
        )
        evidence_path = adapter.artifacts.write_json(
            f"{args.artifact_prefix}/capture_report.json", payload
        )
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "repeat": args.repeat,
                    "max_reset_proprio_abs_delta": max_reset_delta,
                    "camera_names": sorted(captures[0]["camera_views"]),
                    "proprio_dim": captures[0]["proprio_dim"],
                    "evidence_path": evidence_path,
                },
                ensure_ascii=True,
                indent=2,
            )
        )
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
