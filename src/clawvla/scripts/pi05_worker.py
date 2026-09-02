from __future__ import annotations

import argparse
import json
import os
import socketserver
import threading
from pathlib import Path

from clawvla.action_backends.pi05 import Pi05ActionBackend
from clawvla.config import load_config
from clawvla.schema import CameraView, ObservationBundle, RobotArmState


class Pi05WorkerServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, handler_class, backend: Pi05ActionBackend):
        super().__init__(server_address, handler_class)
        self.backend = backend
        # Multiple rollout lanes may share one resident model.  Accept their
        # sockets concurrently so health checks remain responsive, while the
        # lock keeps GPU inference serialized and avoids racing shared backend
        # state or doubling peak model memory.
        self.inference_lock = threading.Lock()


class Pi05WorkerHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            line = self.rfile.readline()
            if not line:
                return
            request = json.loads(line.decode("utf-8"))
            if request.get("op") == "health":
                self._write({"status": "ok", "backend": "pi05_worker"})
                return
            artifact_dir = Path(str(request["artifact_dir"]))
            motion_plan = request.get("motion_plan")
            prompt = _worker_prompt(motion_plan)
            observation = _observation_from_artifact(artifact_dir, prompt)
            with self.server.inference_lock:  # type: ignore[attr-defined]
                result = self.server.backend.build_action_chunk(  # type: ignore[attr-defined]
                    motion_goal=None,
                    world_state=None,
                    observation=observation,
                    request={
                        "motion_plan": motion_plan,
                        "num_steps": request.get("num_steps"),
                        "horizon": request.get("horizon"),
                    },
                )
            self._write(result.to_dict())
        except Exception as exc:
            self._write(
                {
                    "success": False,
                    "status": "pi05_worker_exception",
                    "action_chunk": None,
                    "metadata": {"exception_type": type(exc).__name__},
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            )

    def _write(self, payload: dict) -> None:
        self.wfile.write((json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8"))


def _worker_prompt(motion_plan: object) -> str:
    if isinstance(motion_plan, dict) and motion_plan.get("vla_prompt"):
        return str(motion_plan["vla_prompt"])
    raise ValueError("motion_plan.vla_prompt is required by pi05_worker; request must come from motion.plan_motion.")


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
            name: CameraView(name=name, rgb_path=str(image_dir / f"{name}_rgb.png"))
            for name in ("head_camera", "left_camera", "right_camera")
        },
        robot_arms=robot_arms,
        raw={"summary_ref": str(summary_path)} if summary_path.exists() else {},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent pi0.5 OpenPI worker.")
    parser.add_argument("--config", default="configs/runtime/robotwin.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CLAWVLA_PI05_DIRECT"] = "1"
    config = load_config(args.config)
    backend_cfg = dict(config.metadata.get("action_backend", {}))
    runtime_cfg = dict(backend_cfg.get("openpi_runtime", {})) if isinstance(backend_cfg.get("openpi_runtime"), dict) else {}
    runtime_cfg["mode"] = "direct"
    backend_cfg["openpi_runtime"] = runtime_cfg
    backend = Pi05ActionBackend(backend_cfg)
    diagnosis = backend.diagnose(load_policy=False)
    backend._load_openpi_torch_runtime(diagnosis["policy_summary"])
    with Pi05WorkerServer((args.host, args.port), Pi05WorkerHandler, backend) as server:
        print(
            json.dumps(
                {
                    "status": "pi05_worker_ready",
                    "host": args.host,
                    "port": args.port,
                    "pid": os.getpid(),
                    "runtime": "openpi_torch_cached",
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
