from __future__ import annotations

import argparse
import atexit
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from urllib.request import Request, urlopen

from clawvla.notices import emit_status_notice
from clawvla.terminal_ui import TerminalRenderer


LORA_RANK_CHOICES = (1, 8, 16, 32, 64, 128, 256, 320, 512)
LORA_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class LoraModuleSpec:
    name: str
    path: Path
    rank: int
    base_model_name_or_path: str | None

    @property
    def cli_value(self) -> str:
        return f"{self.name}={self.path}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EmbodiedSkills with a local vLLM server and resident LoRAs.")
    parser.add_argument(
        "--base-config",
        default="configs/runtime/robotwin.json",
    )
    parser.add_argument("--model", required=True, help="Local HF path or model id served by vLLM.")
    parser.add_argument(
        "--served-model-name",
        default="local-scheduler",
        help="Served name for the base model. LoRA adapters use --lora-module names.",
    )
    parser.add_argument(
        "--model-keys",
        default="vision,scheduler,verifier,recovery",
        help="Comma-separated config model keys to route to vLLM.",
    )
    parser.add_argument(
        "--model-route",
        action="append",
        default=[],
        metavar="MODEL_KEY=SERVED_NAME",
        help=(
            "Explicit config-model route. Repeat to send scheduler/vision/verifier model keys "
            "to different statically loaded LoRA served names. When present, this replaces --model-keys routing."
        ),
    )
    parser.add_argument(
        "--skill-model-route",
        action="append",
        default=[],
        metavar="COMPONENT.SKILL=MODEL_KEY",
        help=(
            "Override a skill's config model key. Example: "
            "scheduler.build_task_plan=planner while scheduler.choose_next_skill keeps the scheduler/selector model."
        ),
    )
    parser.add_argument(
        "--lora-module",
        action="append",
        default=[],
        metavar="SERVED_NAME=ADAPTER_PATH",
        help="Static LoRA loaded once during vLLM startup. Repeat for multiple resident adapters.",
    )
    parser.add_argument(
        "--max-loras",
        type=int,
        default=None,
        help="GPU-resident/concurrent LoRA capacity; defaults to all configured static adapters.",
    )
    parser.add_argument(
        "--max-cpu-loras",
        type=int,
        default=None,
        help="Total LoRA cache capacity; defaults to all configured static adapters.",
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        choices=LORA_RANK_CHOICES,
        default=None,
        help="vLLM LoRA rank capacity; defaults to the smallest supported value covering every adapter.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default="local-vllm")
    parser.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES for vLLM, e.g. 0,1.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--conda-bin", default="conda")
    parser.add_argument("--conda-env", default="vllm")
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--vllm-arg", action="append", default=[], help="Extra raw argument passed to vLLM api_server.")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--artifact-prefix", default="agent_loop_local_vllm")
    parser.add_argument("--initial-stage", default="observe")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate adapters/routes and print the resolved vLLM command without starting a server.",
    )
    parser.add_argument("--initial-observe", action="store_true", help="Run the fixed bootstrap observation before scheduler loop.")
    parser.add_argument("--no-initial-observe", action="store_true", help="Deprecated compatibility flag; bootstrap is off by default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lora_modules = _parse_lora_modules(args)
    model_routes = _resolve_model_routes(args, lora_modules)
    renderer = TerminalRenderer()
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{args.artifact_prefix}_vllm.log"
    agent_log_path = run_dir / f"{args.artifact_prefix}_agent.log"
    result_path = run_dir / f"{args.artifact_prefix}_result.json"
    temp_config = run_dir / f"{args.artifact_prefix}_vllm_config.json"
    _write_vllm_config(args, temp_config, model_routes=model_routes, lora_modules=lora_modules)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "vllm_command": _build_vllm_command(
                        args,
                        lora_modules=lora_modules,
                    ),
                    "resolved_config": str(temp_config),
                    "model_routes": model_routes,
                    "skill_model_routes": {
                        f"{component}.{skill}": model_key
                        for (component, skill), model_key in sorted(
                            _parse_skill_model_routes(args).items()
                        )
                    },
                    "resident_loras": [module.name for module in lora_modules],
                    "request_time_adapter_loading": False,
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return
    process = _start_vllm(args, log_path, lora_modules=lora_modules)
    previous_handlers = _install_cleanup_handlers(process)
    try:
        _wait_for_vllm(
            args,
            process,
            log_path,
            expected_models={args.served_model_name, *(module.name for module in lora_modules)},
        )
        renderer.render_event(
            {
                "status": "vllm_ready",
                "log": str(log_path),
                "agent_log": str(agent_log_path),
                "result": str(result_path),
                "config": str(temp_config),
                "model_routes": model_routes,
                "resident_loras": [module.name for module in lora_modules],
            }
        )
        _run_loop(args, temp_config, agent_log_path, result_path)
    finally:
        _restore_cleanup_handlers(previous_handlers, process)
        _stop_process_group(process)
        renderer.render_event({"status": "vllm_stopped", "log": str(log_path)})


def _write_vllm_config(
    args: argparse.Namespace,
    output_path: Path,
    *,
    model_routes: dict[str, str] | None = None,
    lora_modules: list[LoraModuleSpec] | None = None,
) -> None:
    payload = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    base_url = f"http://{args.host}:{args.port}/v1"
    modules = lora_modules if lora_modules is not None else _parse_lora_modules(args)
    routes = model_routes if model_routes is not None else _resolve_model_routes(args, modules)
    configured_models = payload.setdefault("models", {})
    for key, served_name in routes.items():
        existing_model = dict(configured_models.get(key, {}))
        payload.setdefault("models", {})[key] = {
            **existing_model,
            "backend": "openai_compatible",
            "model": served_name,
            "api_base_url": base_url,
            "api_key": args.api_key,
            "api_key_env": "OPENAI_COMPATIBLE_API_KEY",
            "reasoning_effort": None,
            "metadata": {
                **dict(existing_model.get("metadata", {})),
                "vllm_base_model": args.served_model_name,
                "vllm_served_model": served_name,
                "vllm_adapter": served_name if served_name != args.served_model_name else None,
                "static_adapter_loaded_at_startup": served_name != args.served_model_name,
            },
        }
    skill_routes = _parse_skill_model_routes(args)
    for (component, skill), model_key in skill_routes.items():
        component_payload = payload.get("components", {}).get(component)
        if not isinstance(component_payload, dict):
            raise ValueError(f"skill_model_route_unknown_component:{component}.{skill}")
        configured_skills = component_payload.get("skills")
        if isinstance(configured_skills, list) and configured_skills and skill not in configured_skills:
            raise ValueError(f"skill_model_route_unknown_skill:{component}.{skill}")
        if model_key not in configured_models:
            raise ValueError(
                f"skill_model_route_unknown_model_key:{component}.{skill}:{model_key}"
            )
        component_payload.setdefault("skill_models", {})[skill] = model_key
    payload.setdefault("metadata", {})["multilora_routing"] = {
        "base_model": str(args.model),
        "base_served_model_name": args.served_model_name,
        "model_routes": routes,
        "skill_model_routes": {
            f"{component}.{skill}": model_key
            for (component, skill), model_key in sorted(skill_routes.items())
        },
        "static_lora_modules": {
            module.name: {
                "path": str(module.path),
                "rank": module.rank,
                "base_model_name_or_path": module.base_model_name_or_path,
            }
            for module in modules
        },
        "request_time_adapter_loading": False,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _start_vllm(
    args: argparse.Namespace,
    log_path: Path,
    *,
    lora_modules: list[LoraModuleSpec] | None = None,
) -> subprocess.Popen[bytes]:
    command = _build_vllm_command(
        args,
        lora_modules=lora_modules if lora_modules is not None else _parse_lora_modules(args),
    )
    env = dict(os.environ)
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=env, preexec_fn=_parent_death_preexec)
    log_file.close()
    return process


def _build_vllm_command(
    args: argparse.Namespace,
    *,
    lora_modules: list[LoraModuleSpec],
) -> list[str]:
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
    ]
    if lora_modules:
        max_loras = int(args.max_loras or len(lora_modules))
        max_cpu_loras = int(args.max_cpu_loras or len(lora_modules))
        if max_loras < len(lora_modules):
            raise ValueError(
                f"max_loras_must_cover_all_static_adapters:{max_loras}<{len(lora_modules)}"
            )
        if max_cpu_loras < len(lora_modules):
            raise ValueError(
                f"max_cpu_loras_must_cover_all_static_adapters:{max_cpu_loras}<{len(lora_modules)}"
            )
        required_rank = max(module.rank for module in lora_modules)
        max_lora_rank = int(args.max_lora_rank or _lora_rank_bucket(required_rank))
        if max_lora_rank < required_rank:
            raise ValueError(
                f"max_lora_rank_too_small:{max_lora_rank}<{required_rank}"
            )
        command.extend(
            [
                "--enable-lora",
                "--lora-modules",
                *(module.cli_value for module in lora_modules),
                "--max-loras",
                str(max_loras),
                "--max-cpu-loras",
                str(max_cpu_loras),
                "--max-lora-rank",
                str(max_lora_rank),
            ]
        )
    command.extend(args.vllm_arg)
    return command


def _wait_for_vllm(
    args: argparse.Namespace,
    process: subprocess.Popen[bytes],
    log_path: Path,
    *,
    expected_models: set[str] | None = None,
) -> None:
    deadline = time.time() + args.startup_timeout
    url = f"http://{args.host}:{args.port}/v1/models"
    expected = set(expected_models or {args.served_model_name})
    last_missing = set(expected)
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited before ready. log={log_path}")
        request = Request(url, headers={"Authorization": f"Bearer {args.api_key}"})
        try:
            with urlopen(request, timeout=5.0) as response:
                if 200 <= response.status < 300:
                    payload = json.loads(response.read().decode("utf-8"))
                    available = {
                        str(item.get("id"))
                        for item in payload.get("data", [])
                        if isinstance(item, dict) and item.get("id")
                    }
                    last_missing = expected - available
                    if not last_missing:
                        return
                    time.sleep(2.0)
        except Exception:
            time.sleep(2.0)
    raise TimeoutError(
        f"vLLM did not expose every static model within {args.startup_timeout:.1f}s. "
        f"missing={sorted(last_missing)} log={log_path}"
    )


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

    process._embodiedskills_atexit_cleanup = cleanup  # type: ignore[attr-defined]
    atexit.register(cleanup)
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)
    return previous_handlers


def _restore_cleanup_handlers(previous_handlers: dict[int, signal.Handlers], process: subprocess.Popen[bytes]) -> None:
    for signum, previous in previous_handlers.items():
        signal.signal(signum, previous)
    cleanup = getattr(process, "_embodiedskills_atexit_cleanup", None)
    if cleanup is not None:
        atexit.unregister(cleanup)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_assignments(values: list[str], *, label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"invalid_{label}_assignment:{raw}")
        key, value = (part.strip() for part in raw.split("=", 1))
        if not key or not value:
            raise ValueError(f"invalid_{label}_assignment:{raw}")
        if key in parsed:
            raise ValueError(f"duplicate_{label}_assignment:{key}")
        parsed[key] = value
    return parsed


def _parse_lora_modules(args: argparse.Namespace) -> list[LoraModuleSpec]:
    assignments = _parse_assignments(
        list(getattr(args, "lora_module", []) or []),
        label="lora_module",
    )
    modules: list[LoraModuleSpec] = []
    for name, raw_path in assignments.items():
        if not LORA_NAME_RE.fullmatch(name):
            raise ValueError(f"invalid_lora_served_name:{name}")
        if name == args.served_model_name:
            raise ValueError(f"lora_name_conflicts_with_base_served_name:{name}")
        path = Path(raw_path).expanduser().resolve()
        adapter_config_path = path / "adapter_config.json"
        if not adapter_config_path.is_file():
            raise FileNotFoundError(adapter_config_path)
        if not any((path / filename).is_file() for filename in ("adapter_model.safetensors", "adapter_model.bin")):
            raise FileNotFoundError(f"adapter_weights_missing:{path}")
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        if str(adapter_config.get("peft_type") or "").upper() != "LORA":
            raise ValueError(f"adapter_is_not_lora:{name}:{adapter_config.get('peft_type')}")
        rank = int(adapter_config.get("r") or 0)
        if rank <= 0 or rank > max(LORA_RANK_CHOICES):
            raise ValueError(f"unsupported_lora_rank:{name}:{rank}")
        modules.append(
            LoraModuleSpec(
                name=name,
                path=path,
                rank=rank,
                base_model_name_or_path=(
                    str(adapter_config["base_model_name_or_path"])
                    if adapter_config.get("base_model_name_or_path")
                    else None
                ),
            )
        )
    _validate_lora_base_models(str(args.model), modules)
    return modules


def _resolve_model_routes(
    args: argparse.Namespace,
    lora_modules: list[LoraModuleSpec],
) -> dict[str, str]:
    explicit = _parse_assignments(
        list(getattr(args, "model_route", []) or []),
        label="model_route",
    )
    routes = explicit or {
        key: args.served_model_name for key in _split_csv(args.model_keys)
    }
    available = {args.served_model_name, *(module.name for module in lora_modules)}
    unknown = {served for served in routes.values() if served not in available}
    if unknown:
        raise ValueError(
            f"model_routes_reference_unserved_models:{sorted(unknown)}:available={sorted(available)}"
        )
    return routes


def _lora_rank_bucket(required_rank: int) -> int:
    for rank in LORA_RANK_CHOICES:
        if rank >= required_rank:
            return rank
    raise ValueError(f"unsupported_lora_rank:{required_rank}")


def _parse_skill_model_routes(
    args: argparse.Namespace,
) -> dict[tuple[str, str], str]:
    assignments = _parse_assignments(
        list(getattr(args, "skill_model_route", []) or []),
        label="skill_model_route",
    )
    parsed: dict[tuple[str, str], str] = {}
    for raw_skill, model_key in assignments.items():
        if "." not in raw_skill:
            raise ValueError(f"invalid_skill_model_route_target:{raw_skill}")
        component, skill = (part.strip() for part in raw_skill.split(".", 1))
        if not component or not skill or not model_key:
            raise ValueError(f"invalid_skill_model_route_target:{raw_skill}")
        parsed[(component, skill)] = model_key
    return parsed


def _validate_lora_base_models(
    base_model: str,
    modules: list[LoraModuleSpec],
) -> None:
    for module in modules:
        adapter_base = module.base_model_name_or_path
        if not adapter_base:
            continue
        if _same_model_reference(base_model, adapter_base):
            continue
        raise ValueError(
            f"lora_base_model_mismatch:{module.name}:adapter={adapter_base}:served={base_model}"
        )


def _same_model_reference(left: str, right: str) -> bool:
    left_path = Path(left).expanduser()
    right_path = Path(right).expanduser()
    if left_path.exists() and right_path.exists():
        return left_path.resolve() == right_path.resolve()
    return left.rstrip("/") == right.rstrip("/")


if __name__ == "__main__":
    main()
