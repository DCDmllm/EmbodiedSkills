from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import socketserver
import sys
import traceback
from typing import Iterator

from clawvla.scripts.run_loop import run_agent_loop


class RoboTwinLaneServer(socketserver.TCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, handler_class, *, lane_index: int):
        super().__init__(server_address, handler_class)
        self.lane_index = int(lane_index)


class RoboTwinLaneHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        line = self.rfile.readline()
        if not line:
            return
        try:
            request = json.loads(line.decode("utf-8"))
            if request.get("op") == "health":
                self._write(
                    {
                        "ok": True,
                        "status": "robotwin_lane_worker_ready",
                        "lane_index": self.server.lane_index,  # type: ignore[attr-defined]
                    }
                )
                return
            if request.get("op") != "run":
                raise ValueError(f"unsupported RoboTwin lane operation: {request.get('op')!r}")
            log_path = Path(str(request["log_path"]))
            with _temporary_environment(dict(request.get("env") or {})):
                with _redirect_process_output(log_path):
                    output_path = run_agent_loop(
                        config_path=str(request["config_path"]),
                        instruction=str(request["instruction"]),
                        artifact_prefix=str(request["artifact_prefix"]),
                        initial_stage=str(request.get("initial_stage") or "observe"),
                        max_steps=int(request["max_steps"]),
                        result_output=str(request["result_output"]),
                        run_environment=bool(request.get("run_environment", True)),
                        initial_observe=bool(request.get("initial_observe", False)),
                    )
            self._write(
                {
                    "ok": True,
                    "status": "episode_finished",
                    "lane_index": self.server.lane_index,  # type: ignore[attr-defined]
                    "result_output": str(output_path),
                }
            )
        except Exception as exc:
            self._write(
                {
                    "ok": False,
                    "status": "episode_exception",
                    "lane_index": self.server.lane_index,  # type: ignore[attr-defined]
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=12),
                }
            )
        finally:
            gc.collect()

    def _write(self, payload: dict) -> None:
        self.wfile.write((json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8"))


@contextmanager
def _temporary_environment(updates: dict[str, object]) -> Iterator[None]:
    previous = {str(key): os.environ.get(str(key)) for key in updates}
    for key, value in updates.items():
        os.environ[str(key)] = str(value)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _redirect_process_output(path: Path) -> Iterator[None]:
    """Capture Python and native simulator output for one persistent job."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    log_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(log_fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent RoboTwin agent-loop lane worker.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--lane-index", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with RoboTwinLaneServer(
        (args.host, args.port),
        RoboTwinLaneHandler,
        lane_index=args.lane_index,
    ) as server:
        print(
            json.dumps(
                {
                    "status": "robotwin_lane_worker_ready",
                    "host": args.host,
                    "port": args.port,
                    "lane_index": args.lane_index,
                    "pid": os.getpid(),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
