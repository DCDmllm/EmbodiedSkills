from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Iterator

from .config import RLConfig
from .service_pool import command_env


@dataclass(frozen=True)
class RolloutServiceSpec:
    kind: str
    index: int
    gpu: int
    port: int
    command: tuple[str, ...]
    cwd: str
    env: dict[str, str]
    log_path: Path
    ready_pattern: str
    startup_timeout_s: float


@dataclass
class RolloutService:
    spec: RolloutServiceSpec
    process: subprocess.Popen[bytes]


def rollout_service_specs(config: RLConfig, run_dir: Path) -> list[RolloutServiceSpec]:
    specs: list[RolloutServiceSpec] = []
    if config.rollout.persistent_openpi_workers:
        count = _worker_count(config.rollout.openpi_worker_count, config.cluster.openpi_gpus, "OpenPI")
        for index in range(count):
            gpu = config.cluster.openpi_gpus[index % len(config.cluster.openpi_gpus)]
            port = int(config.rollout.openpi_port_base) + index
            specs.append(
                RolloutServiceSpec(
                    kind="openpi",
                    index=index,
                    gpu=gpu,
                    port=port,
                    command=(
                        str(config.openpi.python),
                        "-m",
                        "clawvla.scripts.pi05_worker",
                        "--config",
                        str(config.rollout.base_config),
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                    ),
                    cwd=str(config.openpi.cwd),
                    env=command_env(
                        config.openpi,
                        {
                            "CUDA_VISIBLE_DEVICES": str(gpu),
                            "PYTHONUNBUFFERED": "1",
                            "CLAWVLA_PI05_DIRECT": "1",
                        },
                    ),
                    log_path=run_dir / "logs" / f"openpi_pool_{index}.log",
                    ready_pattern="pi05_worker_ready",
                    startup_timeout_s=600.0,
                )
            )

    if config.rollout.persistent_robotwin_workers:
        count = _worker_count(
            config.rollout.robotwin_worker_count,
            config.cluster.robotwin_gpus,
            "RoboTwin",
        )
        for index in range(count):
            gpu = config.cluster.robotwin_gpus[index % len(config.cluster.robotwin_gpus)]
            port = int(config.rollout.robotwin_worker_port_base) + index
            specs.append(
                RolloutServiceSpec(
                    kind="robotwin",
                    index=index,
                    gpu=gpu,
                    port=port,
                    command=(
                        str(config.environment.python),
                        "-m",
                        "clawvla.rl.robotwin_lane_worker",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                        "--lane-index",
                        str(index),
                    ),
                    cwd=str(config.environment.cwd),
                    env=command_env(
                        config.environment,
                        {
                            "CUDA_VISIBLE_DEVICES": str(gpu),
                            "PYTHONUNBUFFERED": "1",
                        },
                    ),
                    log_path=run_dir / "logs" / f"robotwin_pool_{index}.log",
                    ready_pattern="robotwin_lane_worker_ready",
                    startup_timeout_s=180.0,
                )
            )

    ports = [spec.port for spec in specs]
    if len(ports) != len(set(ports)):
        raise ValueError(f"Persistent rollout service ports overlap: {ports}")
    return specs


@contextmanager
def persistent_rollout_services(
    config: RLConfig,
    run_dir: Path,
    trainer_env: dict[str, str],
) -> Iterator[list[RolloutService]]:
    specs = rollout_service_specs(config, run_dir)
    openpi_ports = [spec.port for spec in specs if spec.kind == "openpi"]
    robotwin_ports = [spec.port for spec in specs if spec.kind == "robotwin"]
    if openpi_ports:
        trainer_env["CLAWVLA_OPENPI_POOL_PORTS"] = ",".join(str(port) for port in openpi_ports)
    if robotwin_ports:
        trainer_env["CLAWVLA_ROBOTWIN_POOL_PORTS"] = ",".join(str(port) for port in robotwin_ports)

    services: list[RolloutService] = []
    try:
        for spec in specs:
            spec.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = spec.log_path.open("wb")
            try:
                process = subprocess.Popen(
                    list(spec.command),
                    cwd=spec.cwd,
                    env=spec.env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    preexec_fn=_parent_death_preexec,
                )
            finally:
                log_file.close()
            services.append(RolloutService(spec=spec, process=process))

        for service in services:
            _wait_for_service_ready(service)
        yield services
    finally:
        for service in reversed(services):
            _stop_process_group(service.process)


def _worker_count(configured: int | None, gpus: list[int], name: str) -> int:
    if not gpus:
        raise ValueError(f"Persistent {name} workers require at least one configured GPU")
    count = len(gpus) if configured is None else int(configured)
    if count <= 0:
        raise ValueError(f"Persistent {name} worker count must be positive, got {configured}")
    return count


def _wait_for_service_ready(service: RolloutService) -> None:
    deadline = time.monotonic() + service.spec.startup_timeout_s
    while time.monotonic() < deadline:
        if service.process.poll() is not None:
            raise RuntimeError(
                f"{service.spec.kind} worker {service.spec.index} exited before ready: "
                f"returncode={service.process.returncode} log={service.spec.log_path}"
            )
        if service.spec.log_path.exists():
            text = service.spec.log_path.read_text(encoding="utf-8", errors="replace")
            if service.spec.ready_pattern in text:
                return
        time.sleep(0.5)
    raise TimeoutError(
        f"{service.spec.kind} worker {service.spec.index} was not ready within "
        f"{service.spec.startup_timeout_s:.1f}s: log={service.spec.log_path}"
    )


def _parent_death_preexec() -> None:
    import ctypes

    os.setsid()
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=20.0)
