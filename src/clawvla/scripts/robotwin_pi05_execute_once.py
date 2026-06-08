from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from clawvla.artifacts import _jsonable
from clawvla.config import load_config
from clawvla.envs import RoboTwinAdapter
from clawvla.schema import ActionChunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture RoboTwin, infer pi0.5 actions, and execute them once.")
    parser.add_argument("--config", default="configs/robotwin_pi05_enabled_probe.json")
    parser.add_argument("--instruction", default="place the container on the plate")
    parser.add_argument("--artifact-prefix", default="pi05_execute_once")
    parser.add_argument("--openpi-env", default="openpi-torch-py312")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    adapter = RoboTwinAdapter(config.robotwin)
    round_reports = []
    for round_index in range(args.rounds):
        before_prefix = f"{args.artifact_prefix}/step_{round_index:02d}/before"
        observation = adapter.capture_views(
            setup=round_index == 0,
            instruction=args.instruction,
            artifact_prefix=before_prefix,
        )
        artifact_dir = Path(config.robotwin.artifact_dir) / before_prefix
        infer_output = artifact_dir / "pi05_action_chunk.json"
        backend_result = _run_openpi_inference(
            config_path=Path(args.config),
            artifact_dir=artifact_dir,
            instruction=args.instruction,
            conda_bin=Path(config.runtime_environment.conda_bin),
            conda_env=args.openpi_env,
            cuda_visible_devices=args.cuda_visible_devices,
            num_steps=args.num_steps,
            horizon=args.horizon,
            output_path=infer_output,
        )
        chunk_payload = backend_result["action_chunk"]
        chunk = ActionChunk(
            action_type=str(chunk_payload["action_type"]),
            commands=[[float(item) for item in command] for command in chunk_payload["commands"]],
            control_horizon=int(chunk_payload.get("control_horizon") or len(chunk_payload["commands"])),
            metadata={
                **dict(chunk_payload.get("metadata") or {}),
                "artifact_prefix": f"{args.artifact_prefix}/step_{round_index:02d}/execute",
            },
        )
        execution_report = adapter.execute_action(chunk)
        round_reports.append(
            {
                "round_index": round_index,
                "capture": observation.to_dict(),
                "backend_result": backend_result,
                "execution_report": execution_report,
            }
        )
    print(
        json.dumps(
            _jsonable(
                {
                    "rounds": round_reports,
                }
            ),
            indent=2,
            ensure_ascii=True,
        )
    )


def _run_openpi_inference(
    *,
    config_path: Path,
    artifact_dir: Path,
    instruction: str,
    conda_bin: Path,
    conda_env: str,
    cuda_visible_devices: str | None,
    num_steps: int,
    horizon: int,
    output_path: Path,
) -> dict:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = "/mnt/wangwai/vla/clawvla/src:/mnt/wangwai/RoboTwin/policy/pi05/src"
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    command = [
        str(conda_bin),
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "python",
        "-m",
        "clawvla.scripts.pi05_inference_smoke",
        "--config",
        str(config_path),
        "--artifact-dir",
        str(artifact_dir),
        "--prompt",
        instruction,
        "--num-steps",
        str(num_steps),
        "--horizon",
        str(horizon),
        "--output",
        str(output_path),
    ]
    subprocess.run(command, check=True, env=env)
    return json.loads(output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
