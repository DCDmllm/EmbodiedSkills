from __future__ import annotations

import argparse
import json
import os
import socketserver
from pathlib import Path

from clawvla.action_backends.groot import GrootActionBackend, observation_from_artifact
from clawvla.config import load_config


class GrootWorkerServer(socketserver.TCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, handler_class, backend: GrootActionBackend):
        super().__init__(server_address, handler_class)
        self.backend = backend


class GrootWorkerHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            line = self.rfile.readline()
            if not line:
                return
            request = json.loads(line.decode("utf-8"))
            if request.get("op") == "health":
                self._write({"status": "ok", "backend": "groot_worker"})
                return
            artifact_dir = Path(str(request["artifact_dir"]))
            motion_plan = request.get("motion_plan")
            prompt = _worker_prompt(motion_plan, request)
            observation = observation_from_artifact(artifact_dir, prompt)
            result = self.server.backend.build_action_chunk(  # type: ignore[attr-defined]
                motion_goal=None,
                world_state=None,
                observation=observation,
                request={"motion_plan": {"vla_prompt": prompt}, "horizon": request.get("horizon")},
            )
            self._write(result.to_dict())
        except Exception as exc:
            self._write(
                {
                    "success": False,
                    "status": "groot_worker_exception",
                    "action_chunk": None,
                    "metadata": {"exception_type": type(exc).__name__},
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            )

    def _write(self, payload: dict) -> None:
        self.wfile.write((json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8"))


def _worker_prompt(motion_plan: object, request: dict) -> str:
    if isinstance(motion_plan, dict) and motion_plan.get("vla_prompt"):
        return str(motion_plan["vla_prompt"])
    if request.get("prompt"):
        return str(request["prompt"])
    raise ValueError("motion_plan.vla_prompt is required by groot_worker.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent GR00T worker.")
    parser.add_argument("--config", default="configs/robocasa_groot_enabled_probe.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--load-policy", action="store_true", help="Load GR00T before reporting ready.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CLAWVLA_GROOT_DIRECT"] = "1"
    config = load_config(args.config)
    backend_cfg = dict(config.metadata.get("action_backend", {}))
    runtime_cfg = dict(backend_cfg.get("runtime", {})) if isinstance(backend_cfg.get("runtime"), dict) else {}
    runtime_cfg["mode"] = "direct"
    backend_cfg["runtime"] = runtime_cfg
    backend = GrootActionBackend(backend_cfg)
    if args.load_policy:
        backend.diagnose(load_policy=True)
    with GrootWorkerServer((args.host, args.port), GrootWorkerHandler, backend) as server:
        print(
            json.dumps(
                {
                    "status": "groot_worker_ready",
                    "host": args.host,
                    "port": args.port,
                    "runtime": "groot_cached" if args.load_policy else "groot_lazy",
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
