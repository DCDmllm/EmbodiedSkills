from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
from typing import Any

from .config import EnvCommandConfig
from .trajectory import TrajectoryWriter

_LOCAL_NO_PROXY = ("127.0.0.1", "localhost", "::1")


def command_env(config: EnvCommandConfig, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    for key in config.unset_env:
        env.pop(key, None)
    env.update({str(key): str(value) for key, value in config.env.items()})
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    _ensure_local_no_proxy(env)
    return env


def _ensure_local_no_proxy(env: dict[str, str]) -> None:
    for key in ("NO_PROXY", "no_proxy"):
        existing = [item.strip() for item in env.get(key, "").split(",") if item.strip()]
        values = list(dict.fromkeys([*existing, *_LOCAL_NO_PROXY]))
        env[key] = ",".join(values)


def run_logged_subprocess(
    command: list[str],
    *,
    cwd: str | Path,
    log_path: str | Path,
    env: dict[str, str],
    timeout: float | None,
    writer: TrajectoryWriter | None = None,
    event_prefix: str = "clawvla_rl_subprocess",
) -> subprocess.CompletedProcess[str]:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if writer is not None:
        writer.write_event(f"{event_prefix}_start", {"command": command, "cwd": str(cwd), "log": str(log_path)})
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=_parent_death_preexec,
    )
    lines: list[str] = []
    timed_out = False
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            assert process.stdout is not None
            for line in process.stdout:
                lines.append(line)
                handle.write(line)
                handle.flush()
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _stop_process_group(process)
        return_code = process.returncode if process.returncode is not None else -9
    output = "".join(lines)
    result = subprocess.CompletedProcess(command, return_code, output, "")
    status = "timeout" if timed_out else "finished"
    if writer is not None:
        writer.write_event(
            f"{event_prefix}_{status}",
            {"return_code": return_code, "log": str(log_path), "timed_out": timed_out, "output_tail": output[-4000:]},
        )
    return result


def _parent_death_preexec() -> None:
    import ctypes

    os.setsid()
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)


def _stop_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=20.0)
