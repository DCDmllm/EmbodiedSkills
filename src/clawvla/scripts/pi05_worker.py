from __future__ import annotations

import argparse
import json
import os
import socketserver
from pathlib import Path

from clawvla.action_backends.pi05 import Pi05ActionBackend
from clawvla.config import load_config
from clawvla.scripts.pi05_inference_smoke import _observation_from_artifact


class Pi05WorkerServer(socketserver.TCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, handler_class, backend: Pi05ActionBackend):
        super().__init__(server_address, handler_class)
        self.backend = backend


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent pi0.5 OpenPI worker.")
    parser.add_argument("--config", default="configs/robotwin_pi05_enabled_probe.json")
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
                    "runtime": "openpi_torch_cached",
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
