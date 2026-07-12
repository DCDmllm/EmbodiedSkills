from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import Request, urlopen

from clawvla.notices import emit_status_notice
from clawvla.terminal_ui import TerminalRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ClawVLA with a temporary local vLLM OpenAI-compatible server.")
    parser.add_argument(
        "--base-config",
        default="/mnt/linyutong/wangwai_mirror/vla/clawvla/configs/robotwin_pi05_subtasks_25k.json",
    )
    parser.add_argument("--model", required=True, help="Local HF path or model id served by vLLM.")
    parser.add_argument("--served-model-name", default="local-scheduler")
    parser.add_argument(
        "--model-keys",
        default="vision,scheduler,verifier,recovery",
        help="Comma-separated config model keys to route to vLLM.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default="local-vllm")
    parser.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES for vLLM, e.g. 0,1.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--conda-bin", default="/mnt/wangwai/miniconda3/bin/conda")
    parser.add_argument("--conda-env", default="vllm")
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--vllm-arg", action="append", default=[], help="Extra raw argument passed to vLLM api_server.")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--artifact-prefix", default="agent_loop_local_vllm")
    parser.add_argument("--initial-stage", default="observe")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--initial-observe", action="store_true", help="Run the fixed bootstrap observation before scheduler loop.")
    parser.add_argument("--no-initial-observe", action="store_true", help="Deprecated compatibility flag; bootstrap is off by default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    renderer = TerminalRenderer()
    run_dir = Path("/mnt/wangwai/vla/clawvla/tmp_runs")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{args.artifact_prefix}_vllm.log"
    agent_log_path = run_dir / f"{args.artifact_prefix}_agent.log"
    result_path = run_dir / f"{args.artifact_prefix}_result.json"
    temp_config = run_dir / f"{args.artifact_prefix}_vllm_config.json"
    _write_vllm_config(args, temp_config)
    process = _start_vllm(args, log_path)
    previous_handlers = _install_cleanup_handlers(process)
    try:
        _wait_for_vllm(args, process, log_path)
        renderer.render_event(
            {
                "status": "vllm_ready",
                "log": str(log_path),
                "agent_log": str(agent_log_path),
                "result": str(result_path),
                "config": str(temp_config),
            }
        )
        _run_loop(args, temp_config, agent_log_path, result_path)
    finally:
        _restore_cleanup_handlers(previous_handlers, process)
        _stop_process_group(process)
        renderer.render_event({"status": "vllm_stopped", "log": str(log_path)})


def _write_vllm_config(args: argparse.Namespace, output_path: Path) -> None:
    payload = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    base_url = f"http://{args.host}:{args.port}/v1"
    for key in _split_csv(args.model_keys):
        payload.setdefault("models", {})[key] = {
            **dict(payload.get("models", {}).get(key, {})),
            "backend": "openai_compatible",
            "model": args.served_model_name,
            "api_base_url": base_url,
            "api_key": args.api_key,
            "api_key_env": "OPENAI_COMPATIBLE_API_KEY",
            "reasoning_effort": None,
        }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _start_vllm(args: argparse.Namespace, log_path: Path) -> subprocess.Popen[bytes]:
    command = [
        *_python_command_prefix(args.conda_bin, args.conda_env, source="run_loop_with_vllm.vllm"),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--api-key",
        args.api_key,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        *args.vllm_arg,
    ]
    env = dict(os.environ)
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=env, preexec_fn=_parent_death_preexec)
    log_file.close()
    return process


def _wait_for_vllm(args: argparse.Namespace, process: subprocess.Popen[bytes], log_path: Path) -> None:
    deadline = time.time() + args.startup_timeout
    url = f"http://{args.host}:{args.port}/v1/models"
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited before ready. log={log_path}")
        request = Request(url, headers={"Authorization": f"Bearer {args.api_key}"})
        try:
            with urlopen(request, timeout=5.0) as response:
                if 200 <= response.status < 300:
                    return
        except Exception:
            time.sleep(2.0)
    raise TimeoutError(f"vLLM did not become ready within {args.startup_timeout:.1f}s. log={log_path}")


def _run_loop(args: argparse.Namespace, config_path: Path, agent_log_path: Path, result_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "clawvla.scripts.run_loop",
        "--config",
        str(config_path),
        "--instruction",
        args.instruction,
        "--artifact-prefix",
        args.artifact_prefix,
        "--initial-stage",
        args.initial_stage,
        "--max-steps",
        str(args.max_steps),
        "--result-output",
        str(result_path),
    ]
    if args.run:
        command.append("--run")
    if args.initial_observe and not args.no_initial_observe:
        command.append("--initial-observe")
    env = dict(os.environ)
    env["OPENAI_COMPATIBLE_API_KEY"] = args.api_key
    agent_log_path.parent.mkdir(parents=True, exist_ok=True)
    with agent_log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        assert process.stdout is not None
        renderer = TerminalRenderer()
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            renderer.render_agent_line(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=20.0)


def _python_command_prefix(conda_bin: str, conda_env: str, *, source: str) -> list[str]:
    conda_path = Path(conda_bin)
    python_path = conda_path.parent.parent / "envs" / conda_env / "bin" / "python"
    if python_path.exists():
        return [str(python_path)]
    emit_status_notice(
        "conda_env_python_unavailable",
        success=True,
        source=source,
        reason=f"env python not found, falling back to conda run: {python_path}",
    )
    return [str(conda_path), "run", "--no-capture-output", "-n", conda_env, "python"]


def _parent_death_preexec() -> None:
    import ctypes

    os.setsid()
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)


def _install_cleanup_handlers(process: subprocess.Popen[bytes]) -> dict[int, signal.Handlers]:
    previous_handlers: dict[int, signal.Handlers] = {}

    def cleanup() -> None:
        _stop_process_group(process)

    def handle_signal(signum: int, frame: object) -> None:
        cleanup()
        previous = previous_handlers.get(signum)
        if callable(previous):
            previous(signum, frame)
            return
        raise SystemExit(128 + signum)

    process._clawvla_atexit_cleanup = cleanup  # type: ignore[attr-defined]
    atexit.register(cleanup)
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)
    return previous_handlers


def _restore_cleanup_handlers(previous_handlers: dict[int, signal.Handlers], process: subprocess.Popen[bytes]) -> None:
    for signum, previous in previous_handlers.items():
        signal.signal(signum, previous)
    cleanup = getattr(process, "_clawvla_atexit_cleanup", None)
    if cleanup is not None:
        atexit.unregister(cleanup)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
