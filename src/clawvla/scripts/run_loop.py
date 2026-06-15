from __future__ import annotations

import argparse
import atexit
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Iterator

from clawvla.artifacts import _jsonable
from clawvla.config import AgentConfig
from clawvla.config import load_config
from clawvla.envs import RoboTwinAdapter
from clawvla.notices import emit_status_notice
from clawvla.runtime import AgentRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the model-driven ClawVLA agent loop.")
    parser.add_argument("--config", default="configs/robotwin_default.json")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--artifact-prefix", default="agent_loop")
    parser.add_argument("--initial-stage", default="observe")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--run", action="store_true", help="Instantiate RoboTwin and capture a real initial observation.")
    parser.add_argument("--initial-observe", action="store_true", help="Run the fixed bootstrap observation before scheduler loop.")
    parser.add_argument("--no-initial-observe", action="store_true", help="Deprecated compatibility flag; bootstrap is off by default.")
    parser.add_argument("--result-output", default=None, help="Path for the final loop JSON. Defaults to tmp_runs/<prefix>_result.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _apply_runtime_environment(config)
    result_output = Path(args.result_output) if args.result_output else _default_run_path(config, args.artifact_prefix, "result.json")
    with _openpi_worker_lifecycle(config, args.config, args.artifact_prefix):
        runtime = AgentRuntime(config)
        adapter = RoboTwinAdapter(config.robotwin)
        runtime.blackboard.write("env_adapter", adapter)
        runtime.blackboard.write("run_robotwin", bool(args.run))
        runtime.blackboard.write("artifact_prefix", args.artifact_prefix)
        runtime.blackboard.task_instruction = args.instruction
        _install_rl_reward_tracker(runtime, config)

        if args.initial_observe and not args.no_initial_observe:
            bootstrap = _bootstrap_observe(runtime, args.instruction, args.artifact_prefix, args.run)
            if not bootstrap.success:
                _print_result(runtime, {"status": bootstrap.status, "reason": bootstrap.errors[0] if bootstrap.errors else None}, result_output)
                return

        result = runtime.run_loop(max_steps=args.max_steps, initial_stage=args.initial_stage)
        _print_result(runtime, result.to_dict(), result_output)


@contextmanager
def _openpi_worker_lifecycle(config: AgentConfig, config_path: str, artifact_prefix: str) -> Iterator[None]:
    runtime_cfg = _openpi_runtime_cfg(config)
    if runtime_cfg.get("mode") != "worker" or not runtime_cfg.get("auto_start", True):
        yield
        return

    process, log_path = _start_openpi_worker(config_path, artifact_prefix, runtime_cfg, config)
    previous_handlers: dict[int, signal.Handlers] = {}

    def stop_worker() -> None:
        _stop_process_group(process)

    def handle_signal(signum: int, frame: object) -> None:
        stop_worker()
        previous = previous_handlers.get(signum)
        if callable(previous):
            previous(signum, frame)
            return
        raise SystemExit(128 + signum)

    atexit.register(stop_worker)
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)
    try:
        yield
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        atexit.unregister(stop_worker)
        stop_worker()
        print(json.dumps({"status": "pi05_worker_stopped", "log": str(log_path)}, ensure_ascii=True), file=sys.stderr, flush=True)


def _openpi_runtime_cfg(config: AgentConfig) -> dict[str, object]:
    backend_cfg = config.metadata.get("action_backend", {})
    if not isinstance(backend_cfg, dict):
        return {}
    runtime_cfg = backend_cfg.get("openpi_runtime", {})
    return dict(runtime_cfg) if isinstance(runtime_cfg, dict) else {}


def _apply_runtime_environment(config: AgentConfig) -> None:
    env = getattr(config.runtime_environment, "env", {})
    for key, value in env.items():
        os.environ[str(key)] = str(value)


def _start_openpi_worker(
    config_path: str,
    artifact_prefix: str,
    runtime_cfg: dict[str, object],
    config: AgentConfig,
) -> tuple[subprocess.Popen[bytes], Path]:
    log_path = _default_run_path(config, artifact_prefix, "pi05_worker.log")
    log_file = log_path.open("w", encoding="utf-8")
    command = [
        *_python_command_prefix(
            runtime_cfg.get("conda_bin"),
            runtime_cfg.get("conda_env") or "openpi-torch-py312",
            source="run_loop.openpi_worker",
        ),
        "-m",
        "clawvla.scripts.pi05_worker",
        "--config",
        str(config_path),
        "--host",
        str(runtime_cfg.get("host") or "127.0.0.1"),
        "--port",
        str(runtime_cfg.get("port") or 8765),
    ]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = str(
        runtime_cfg.get("pythonpath") or "/mnt/wangwai/vla/clawvla/src:/mnt/wangwai/RoboTwin/policy/pi05/src"
    )
    env["CLAWVLA_PI05_DIRECT"] = "1"
    if runtime_cfg.get("cuda_visible_devices"):
        env["CUDA_VISIBLE_DEVICES"] = str(runtime_cfg["cuda_visible_devices"])
    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=env, preexec_fn=_parent_death_preexec)
    log_file.close()
    try:
        _wait_for_openpi_worker_ready(process, log_path, float(runtime_cfg.get("startup_timeout", 600.0)))
    except BaseException:
        _stop_process_group(process)
        raise
    print(json.dumps({"status": "pi05_worker_started", "log": str(log_path)}, ensure_ascii=True), file=sys.stderr, flush=True)
    return process, log_path


def _wait_for_openpi_worker_ready(process: subprocess.Popen[bytes], log_path: Path, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if "pi05_worker_ready" in text:
                return
        if process.poll() is not None:
            raise RuntimeError(f"pi05 worker exited before ready. log={log_path}")
        time.sleep(1.0)
    raise TimeoutError(f"pi05 worker did not become ready within {timeout:.1f}s. log={log_path}")


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10.0)


def _python_command_prefix(conda_bin: object | None, conda_env: object | None, *, source: str) -> list[str]:
    conda_path = Path(str(conda_bin or "/mnt/wangwai/miniconda3/bin/conda"))
    env_name = str(conda_env or "")
    python_path = conda_path.parent.parent / "envs" / env_name / "bin" / "python"
    if python_path.exists():
        return [str(python_path)]
    emit_status_notice(
        "conda_env_python_unavailable",
        success=True,
        source=source,
        reason=f"env python not found, falling back to conda run: {python_path}",
    )
    return [str(conda_path), "run", "--no-capture-output", "-n", env_name, "python"]


def _parent_death_preexec() -> None:
    import ctypes

    os.setsid()
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)


def _default_run_path(config: AgentConfig, artifact_prefix: str, suffix: str) -> Path:
    run_dir = Path(config.robotwin.artifact_dir).parent / "tmp_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / f"{artifact_prefix}_{suffix}"


def _print_result(runtime: AgentRuntime, loop_payload: dict[str, object], output_path: Path) -> None:
    payload = _jsonable(
        {
            "loop": loop_payload,
            "blackboard": runtime.blackboard.compact_context(),
            "history_length": len(runtime.history),
            "model_calls": _model_call_summary(runtime),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "clawvla_result_written",
                "path": str(output_path),
                "loop_status": loop_payload.get("status"),
                "final_stage": loop_payload.get("final_stage"),
                "reason": loop_payload.get("reason"),
                "history_length": len(runtime.history),
            },
            ensure_ascii=True,
        ),
        file=sys.stderr,
        flush=True,
    )


def _bootstrap_observe(runtime: AgentRuntime, instruction: str, artifact_prefix: str, run_robotwin: bool):
    payload = {"instruction": instruction, "artifact_prefix": artifact_prefix}
    if run_robotwin:
        payload["setup"] = True
    capture = runtime.run_skill("vision", "capture_views", payload, stage="observe")
    if not capture.success:
        return capture
    observation = runtime.blackboard.read("observation")
    image_paths = [view.rgb_path for view in getattr(observation, "camera_views", {}).values() if view.rgb_path]
    failures = []
    for component, skill, skill_payload in [
        ("vision", "perceive_scene", {"use_model": True, "image_paths": image_paths}),
        ("vision", "localize_task_objects", {"use_model": True, "image_paths": image_paths}),
        ("vision", "estimate_uncertainty", {"use_model": True, "image_paths": image_paths}),
    ]:
        result = runtime.run_skill(component, skill, skill_payload, stage="observe")
        if not result.success:
            failures.append(
                {
                    "component": component,
                    "skill": skill,
                    "status": result.status,
                    "errors": result.errors[:3],
                    "reason": result.output.get("reason"),
                }
            )
            runtime.blackboard.write("bootstrap_observe_failures", failures, event_type="bootstrap.observe_skill_failed")
    return runtime.run_skill("state", "update_world_state", {"stage": "observe"}, stage="observe")


def _model_call_summary(runtime: AgentRuntime) -> list[dict[str, object]]:
    calls = []
    for event_index, event in enumerate(runtime.blackboard.events):
        if event.event_type in {"model.call", "model.output"}:
            calls.append({"event_index": event_index, "event_type": event.event_type, **event.payload})
    return calls[-32:]


def _install_rl_reward_tracker(runtime: AgentRuntime, config: AgentConfig) -> None:
    output_path = os.environ.get("CLAWVLA_RL_REWARD_JSONL")
    if not output_path:
        return
    from clawvla.rl.reward_tracker import RuntimeRewardTracker

    task_name = os.environ.get("CLAWVLA_RL_TASK_NAME") or config.robotwin.task_name
    step_cost = float(os.environ.get("CLAWVLA_RL_STEP_COST") or 0.05)
    tracker = RuntimeRewardTracker(task_name=task_name, output_path=output_path, step_cost=step_cost)
    runtime.blackboard.write("rl_reward_tracker", tracker, event_type="rl.reward_tracker_installed")
    emit_status_notice(
        "rl_reward_tracker_installed",
        success=True,
        source="run_loop",
        reason=f"task={task_name} output={output_path}",
        always=True,
    )


if __name__ == "__main__":
    main()
