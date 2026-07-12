from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
from typing import Any
from urllib.request import Request, urlopen

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
OPENPI_SRC = WORKSPACE_ROOT / "pi0.5" / "src"
ROBOTWIN_ROOT = WORKSPACE_ROOT / "RoboTwin"
PYTORCH3D_TARGET = Path("/mnt/wangwai/tmp_pytorch3d_target")

DEFAULT_TASKS_CONFIG = str(PROJECT_ROOT / "configs/rl/tasks/robotwin_all.yaml")
DEFAULT_BASE_CONFIG = str(PROJECT_ROOT / "configs/robotwin_pi05_subtasks_25k.json")
DEFAULT_QWEN3VL = "/mnt/wangwai/weights/Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_ROBOTWIN_PYTHON = "/mnt/wangwai/miniconda3/envs/robotwin-py312/bin/python"
DEFAULT_CONDA_BIN = "/mnt/wangwai/miniconda3/bin/conda"
DEFAULT_MODEL_KEYS = "vision,state,scheduler,verifier,recovery"
DEFAULT_VALID_SEED_CACHE = "/mnt/wangwai/vla/clawvla/runs/eval/robotwin_valid_seeds_demo_clean_seed0/valid_seeds"


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    instruction: str


@dataclass
class Lane:
    index: int
    robotwin_gpu: str | None
    openpi_gpu: str | None
    openpi_port: int
    openpi_process: subprocess.Popen[bytes] | None = None
    openpi_log: str | None = None


@dataclass
class EpisodeJob:
    task_name: str
    task_instruction: str
    eval_index: int
    seed: int
    seed_arg: int
    instruction: str
    valid_seed_tries: int


@dataclass
class EpisodeResult:
    task_name: str
    task_config: str
    eval_index: int
    seed_arg: int
    seed: int
    instruction: str
    success: bool
    success_source: str | None
    return_code: int
    status: str
    final_stage: str | None
    reason: str | None
    duration_s: float
    lane: int
    robotwin_gpu: str | None
    openpi_port: int
    action_executions: int
    executed_steps: int
    model_calls: int
    result_json: str | None
    artifact_dir: str | None
    agent_log: str | None
    error_tail: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official-style RoboTwin 50-task evaluation with persistent Qwen3-VL vLLM and pi0.5 workers."
    )
    parser.add_argument("--tasks-config", default=DEFAULT_TASKS_CONFIG)
    parser.add_argument("--task-name", action="append", default=[], help="Restrict evaluation to one or more task names.")
    parser.add_argument("--task-limit", type=int, default=None, help="Restrict evaluation to the first N tasks after filtering.")
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="Official eval seed argument; start seed is 100000*(1+seed).")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "runs/eval"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--interleave", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--official-seed-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--valid-seed-cache-dir",
        default=None,
        help="Directory containing precomputed <task>.json valid seed files, or a run dir containing valid_seeds/.",
    )
    parser.add_argument("--seed-check-timeout", type=float, default=900.0)
    parser.add_argument("--episode-timeout", type=float, default=3600.0)
    parser.add_argument("--robotwin-python", default=DEFAULT_ROBOTWIN_PYTHON)
    parser.add_argument("--robotwin-gpus", default="7", help="Comma-separated GPU ids for RoboTwin eval workers.")
    parser.add_argument(
        "--seed-check-gpus",
        default=None,
        help="CUDA_VISIBLE_DEVICES for official expert seed filtering. Defaults to --robotwin-gpus.",
    )
    parser.add_argument("--robotwin-cwd", default=str(PROJECT_ROOT))
    parser.add_argument("--model", default=DEFAULT_QWEN3VL)
    parser.add_argument("--served-model-name", default="local-scheduler")
    parser.add_argument("--model-keys", default=DEFAULT_MODEL_KEYS)
    parser.add_argument("--api-key", default="local-vllm")
    parser.add_argument("--vllm-base-url", default=None, help="Use an already running OpenAI-compatible server.")
    parser.add_argument("--no-start-vllm", action="store_true")
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=18080)
    parser.add_argument("--vllm-gpus", default="0,1")
    parser.add_argument("--vllm-conda-bin", default=DEFAULT_CONDA_BIN)
    parser.add_argument("--vllm-conda-env", default="vllm")
    parser.add_argument("--vllm-startup-timeout", type=float, default=900.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--vllm-arg", action="append", default=[])
    parser.add_argument("--openpi-conda-bin", default=DEFAULT_CONDA_BIN)
    parser.add_argument("--openpi-conda-env", default="openpi-torch-py312")
    parser.add_argument("--openpi-gpus", default="6", help="Comma-separated GPU ids for pi0.5 workers.")
    parser.add_argument("--openpi-port-base", type=int, default=9365)
    parser.add_argument("--no-start-openpi-workers", action="store_true")
    parser.add_argument("--openpi-startup-timeout", type=float, default=900.0)
    parser.add_argument("--keep-agent-logs", choices=["none", "failures", "all"], default="none")
    parser.add_argument("--keep-result-json", choices=["none", "failures", "all"], default="none")
    parser.add_argument("--keep-artifacts", choices=["none", "failures", "all"], default="none")
    parser.add_argument("--agent-log-tail-lines", type=int, default=80)
    parser.add_argument("--progress-task-limit", type=int, default=60)
    parser.add_argument("--progress-event-limit", type=int, default=10)
    parser.add_argument("--initial-observe", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build the run plan and write config files without launching episodes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = _filter_tasks(_load_tasks(Path(args.tasks_config)), args)
    run_dir = _run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "run_config.json", _run_config_payload(args, tasks))

    base_url = args.vllm_base_url or f"http://{args.vllm_host}:{args.vllm_port}/v1"
    vllm_process: subprocess.Popen[bytes] | None = None
    lanes = _make_lanes(args)
    try:
        if args.dry_run:
            asyncio.run(_run_eval(args, tasks, run_dir, base_url, lanes))
            return

        if not args.no_start_vllm and not args.vllm_base_url:
            vllm_process = _start_vllm(args, run_dir)
            _wait_for_openai_server(base_url, args.api_key, vllm_process, run_dir / "logs" / "vllm.log", args.vllm_startup_timeout)
        else:
            _wait_for_openai_server(base_url, args.api_key, None, None, 60.0)

        if not args.no_start_openpi_workers:
            _start_openpi_workers(args, run_dir, lanes)

        asyncio.run(_run_eval(args, tasks, run_dir, base_url, lanes))
    finally:
        for lane in lanes:
            if lane.openpi_process is not None:
                _stop_process_group(lane.openpi_process)
        if vllm_process is not None:
            _stop_process_group(vllm_process)


async def _run_eval(
    args: argparse.Namespace,
    tasks: list[TaskSpec],
    run_dir: Path,
    base_url: str,
    lanes: list[Lane],
) -> None:
    results_path = run_dir / "episodes.jsonl"
    failures_path = run_dir / "failures.jsonl"
    seed_dir = run_dir / "valid_seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)
    if args.valid_seed_cache_dir:
        cache_dir = _resolve_valid_seed_cache_dir(Path(args.valid_seed_cache_dir))
        _materialize_valid_seed_cache(cache_dir, seed_dir, tasks, args)
        print(f"valid seed cache loaded: {cache_dir}")

    if args.dry_run:
        _write_json(
            run_dir / "plan.json",
            {
                "jobs": _planned_job_count(args, tasks),
                "tasks": [asdict(t) for t in tasks],
                "valid_seed_cache_dir": str(args.valid_seed_cache_dir) if args.valid_seed_cache_dir else None,
            },
        )
        print(f"dry_run plan written: {run_dir / 'plan.json'}")
        return

    (run_dir / "configs").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "raw_results").mkdir(exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)

    completed = _load_completed(results_path) if args.resume else {}
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"done": 0, "success": 0})
    for result in completed.values():
        item = stats[result["task_name"]]
        item["done"] += 1
        item["success"] += int(bool(result.get("success")))
    if completed:
        _write_summary(run_dir, args, tasks, completed)

    queue: asyncio.Queue[EpisodeJob | None] = asyncio.Queue(maxsize=max(1, len(lanes) * 2))
    write_lock = asyncio.Lock()
    progress = _Progress(args=args, tasks=tasks, lanes=lanes, run_dir=run_dir, completed=completed)
    progress.start()
    try:
        workers = [
            asyncio.create_task(
                _lane_worker(
                    args=args,
                    tasks=tasks,
                    lane=lane,
                    queue=queue,
                    run_dir=run_dir,
                    base_url=base_url,
                    completed=completed,
                    stats=stats,
                    results_path=results_path,
                    failures_path=failures_path,
                    write_lock=write_lock,
                    progress=progress,
                )
            )
            for lane in lanes
        ]
        producer = asyncio.create_task(_produce_jobs(args, tasks, queue, seed_dir, completed))
        await producer
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)
    finally:
        progress.stop()
    _write_summary(run_dir, args, tasks, completed)
    _write_task_results(run_dir, tasks, completed)
    print(f"results: {results_path}")
    print(f"summary: {run_dir / 'summary.json'}")


async def _produce_jobs(
    args: argparse.Namespace,
    tasks: list[TaskSpec],
    queue: asyncio.Queue[EpisodeJob | None],
    seed_dir: Path,
    completed: dict[str, dict[str, Any]],
) -> None:
    ordered = _job_order(args, tasks)
    seed_states = _load_seed_states(seed_dir, tasks, args.seed)
    for task, eval_index in ordered:
        key = _episode_key(task.task_name, eval_index)
        if key in completed:
            continue
        seed, instruction, tries = await asyncio.to_thread(
            _ensure_valid_seed,
            args,
            task,
            eval_index,
            seed_dir,
            seed_states[task.task_name],
        )
        await queue.put(
            EpisodeJob(
                task_name=task.task_name,
                task_instruction=task.instruction,
                eval_index=eval_index,
                seed=seed,
                seed_arg=args.seed,
                instruction=instruction or task.instruction,
                valid_seed_tries=tries,
            )
        )


async def _lane_worker(
    *,
    args: argparse.Namespace,
    tasks: list[TaskSpec],
    lane: Lane,
    queue: asyncio.Queue[EpisodeJob | None],
    run_dir: Path,
    base_url: str,
    completed: dict[str, dict[str, Any]],
    stats: dict[str, dict[str, Any]],
    results_path: Path,
    failures_path: Path,
    write_lock: asyncio.Lock,
    progress: "_Progress",
) -> None:
    while True:
        job = await queue.get()
        if job is None:
            queue.task_done()
            return
        started = time.time()
        progress.start_job(lane, job)
        try:
            result = await asyncio.to_thread(_run_episode, args, lane, job, run_dir, base_url)
        except Exception as exc:
            result = EpisodeResult(
                task_name=job.task_name,
                task_config=args.task_config,
                eval_index=job.eval_index,
                seed_arg=args.seed,
                seed=job.seed,
                instruction=job.instruction,
                success=False,
                success_source=None,
                return_code=-1,
                status="runner_exception",
                final_stage=None,
                reason=f"{type(exc).__name__}: {exc}",
                duration_s=time.time() - started,
                lane=lane.index,
                robotwin_gpu=lane.robotwin_gpu,
                openpi_port=lane.openpi_port,
                action_executions=0,
                executed_steps=0,
                model_calls=0,
                result_json=None,
                artifact_dir=None,
                agent_log=None,
                error_tail=[],
            )
        payload = asdict(result)
        async with write_lock:
            key = _episode_key(result.task_name, result.eval_index)
            if key not in completed:
                completed[key] = payload
                _append_jsonl(results_path, payload)
                if not result.success:
                    _append_jsonl(failures_path, payload)
                task_stats = stats[result.task_name]
                task_stats["done"] += 1
                task_stats["success"] += int(result.success)
                _write_summary(run_dir, args, tasks, completed)
            progress.advance(payload, completed)
        queue.task_done()


def _run_episode(args: argparse.Namespace, lane: Lane, job: EpisodeJob, run_dir: Path, base_url: str) -> EpisodeResult:
    episode_id = f"{job.eval_index:03d}_{job.task_name}_seed{job.seed}_lane{lane.index}"
    config_path = run_dir / "configs" / f"{episode_id}.json"
    result_path = run_dir / "raw_results" / f"{episode_id}_result.json"
    log_path = run_dir / "logs" / f"{episode_id}_agent.log"
    artifact_dir = run_dir / "artifacts" / episode_id
    _write_episode_config(args, lane, job, config_path, artifact_dir, base_url)
    command = [
        args.robotwin_python,
        "-m",
        "clawvla.scripts.run_loop",
        "--config",
        str(config_path),
        "--instruction",
        job.instruction,
        "--artifact-prefix",
        episode_id,
        "--initial-stage",
        "observe",
        "--max-steps",
        str(args.max_steps),
        "--run",
        "--result-output",
        str(result_path),
    ]
    if args.initial_observe:
        command.append("--initial-observe")
    env = _episode_env(args, lane, base_url)
    start = time.time()
    tail: deque[str] = deque(maxlen=max(0, int(args.agent_log_tail_lines)))
    temp_log_path: Path | None = None
    process: subprocess.Popen[bytes] | subprocess.Popen[str] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f"{episode_id}_",
            suffix=".log",
            dir=str(run_dir / "logs"),
            delete=False,
        ) as temp_log:
            temp_log_path = Path(temp_log.name)
            process = subprocess.Popen(
                command,
                cwd=args.robotwin_cwd,
                stdout=temp_log,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                preexec_fn=_parent_death_preexec,
            )
            process.wait(timeout=max(1.0, float(args.episode_timeout)))
            return_code = int(process.returncode or 0)
    except subprocess.TimeoutExpired:
        if process is not None:
            _stop_process_group(process)  # type: ignore[arg-type]
        return_code = -9
        tail.append(f"episode_timeout:{args.episode_timeout}")
    if temp_log_path is not None and temp_log_path.exists() and args.agent_log_tail_lines > 0:
        tail.extend(_read_tail_lines(temp_log_path, int(args.agent_log_tail_lines)))
    duration = time.time() - start
    payload = _read_result_payload(result_path)
    summary = _summarize_episode_payload(payload, return_code)
    success = bool(summary["success"])
    keep_log = args.keep_agent_logs == "all" or (args.keep_agent_logs == "failures" and not success)
    if temp_log_path is not None and temp_log_path.exists():
        if keep_log:
            temp_log_path.replace(log_path)
        else:
            temp_log_path.unlink()
    if not keep_log and log_path.exists():
        log_path.unlink()
    keep_result = args.keep_result_json == "all" or (args.keep_result_json == "failures" and not success)
    if not keep_result and result_path.exists():
        result_path.unlink()
    keep_artifacts = args.keep_artifacts == "all" or (args.keep_artifacts == "failures" and not success)
    if not keep_artifacts and artifact_dir.exists():
        shutil.rmtree(artifact_dir, ignore_errors=True)
    return EpisodeResult(
        task_name=job.task_name,
        task_config=args.task_config,
        eval_index=job.eval_index,
        seed_arg=job.seed_arg,
        seed=job.seed,
        instruction=job.instruction,
        success=success,
        success_source=summary["success_source"],
        return_code=return_code,
        status=str(summary["status"]),
        final_stage=summary["final_stage"],
        reason=summary["reason"],
        duration_s=duration,
        lane=lane.index,
        robotwin_gpu=lane.robotwin_gpu,
        openpi_port=lane.openpi_port,
        action_executions=int(summary["action_executions"]),
        executed_steps=int(summary["executed_steps"]),
        model_calls=int(summary["model_calls"]),
        result_json=str(result_path) if keep_result and result_path.exists() else None,
        artifact_dir=str(artifact_dir) if keep_artifacts and artifact_dir.exists() else None,
        agent_log=str(log_path) if keep_log and log_path.exists() else None,
        error_tail=list(tail) if return_code != 0 or not success else [],
    )


def _ensure_valid_seed(
    args: argparse.Namespace,
    task: TaskSpec,
    eval_index: int,
    seed_dir: Path,
    state: dict[str, Any],
) -> tuple[int, str | None, int]:
    valid = state.setdefault("valid", [])
    if len(valid) > eval_index:
        item = valid[eval_index]
        return int(item["seed"]), item.get("instruction"), 0
    tries = 0
    while len(valid) <= eval_index:
        candidate = int(state["next_seed"])
        state["next_seed"] = candidate + 1
        tries += 1
        if args.official_seed_filter:
            check = _run_seed_check(args, task.task_name, candidate, eval_index)
            if not check.get("ok"):
                _save_seed_state(seed_dir, task.task_name, state)
                continue
            instruction = str(check.get("instruction") or task.instruction)
        else:
            instruction = task.instruction
        valid.append({"eval_index": len(valid), "seed": candidate, "instruction": instruction})
        _save_seed_state(seed_dir, task.task_name, state)
    item = valid[eval_index]
    return int(item["seed"]), item.get("instruction"), tries


def _run_seed_check(args: argparse.Namespace, task_name: str, seed: int, eval_index: int) -> dict[str, Any]:
    command = [
        args.robotwin_python,
        "-m",
        "clawvla.scripts.robotwin_official_seed_check",
        "--task-name",
        task_name,
        "--task-config",
        args.task_config,
        "--seed",
        str(seed),
        "--episode-index",
        str(eval_index),
        "--instruction-type",
        args.instruction_type,
    ]
    base_cfg = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    camera_profile = ((base_cfg.get("robotwin") or {}).get("camera_profile") if isinstance(base_cfg.get("robotwin"), dict) else None)
    if camera_profile:
        command.extend(["--camera-profile", str(camera_profile)])
    env = _base_robotwin_env(args)
    seed_check_gpus = args.seed_check_gpus if args.seed_check_gpus is not None else args.robotwin_gpus
    if seed_check_gpus:
        env["CUDA_VISIBLE_DEVICES"] = str(seed_check_gpus)
    completed = subprocess.run(
        command,
        cwd=args.robotwin_cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=max(1.0, float(args.seed_check_timeout)),
    )
    if completed.returncode != 0:
        return {"ok": False, "status": "seed_check_process_failed", "stderr": completed.stderr[-2000:]}
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return {"ok": False, "status": "seed_check_empty_output"}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        return {"ok": False, "status": "seed_check_bad_json", "error": str(exc), "stdout": completed.stdout[-2000:]}


def _write_episode_config(
    args: argparse.Namespace,
    lane: Lane,
    job: EpisodeJob,
    output_path: Path,
    artifact_dir: Path,
    base_url: str,
) -> None:
    payload = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    payload.setdefault("task", {})["instruction"] = job.instruction
    robotwin = payload.setdefault("robotwin", {})
    robotwin["task_name"] = job.task_name
    robotwin["task_config"] = args.task_config
    robotwin["seed"] = job.seed
    robotwin["now_ep_num"] = job.eval_index
    robotwin["is_test"] = True
    robotwin["eval_mode"] = True
    robotwin["render_freq"] = 0
    robotwin["need_plan"] = True
    robotwin["artifact_dir"] = str(artifact_dir)
    environment = payload.setdefault("environment", {})
    if isinstance(environment, dict):
        environment["type"] = "robotwin"
        environment["task_name"] = job.task_name
        environment["seed"] = job.seed
        environment["artifact_dir"] = str(artifact_dir)
    for key in _split_csv(args.model_keys):
        model_cfg = payload.setdefault("models", {}).setdefault(key, {})
        model_cfg.update(
            {
                "backend": "openai_compatible",
                "model": args.served_model_name,
                "api_base_url": base_url,
                "api_base_url_env": None,
                "api_key": args.api_key,
                "api_key_env": None,
                "reasoning_effort": None,
                "temperature": 0.0,
            }
        )
    backend = payload.setdefault("metadata", {}).setdefault("action_backend", {})
    runtime = backend.setdefault("openpi_runtime", {})
    runtime["mode"] = "worker"
    runtime["auto_start"] = False
    runtime["host"] = "127.0.0.1"
    runtime["port"] = lane.openpi_port
    runtime["conda_bin"] = args.openpi_conda_bin
    runtime["conda_env"] = args.openpi_conda_env
    if lane.openpi_gpu:
        runtime["cuda_visible_devices"] = lane.openpi_gpu
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _summarize_episode_payload(payload: dict[str, Any] | None, return_code: int) -> dict[str, Any]:
    if payload is None:
        return {
            "success": False,
            "success_source": None,
            "status": "missing_result_json" if return_code == 0 else "process_failed",
            "final_stage": None,
            "reason": None,
            "action_executions": 0,
            "executed_steps": 0,
            "model_calls": 0,
        }
    loop = payload.get("loop") if isinstance(payload.get("loop"), dict) else {}
    reports = list(_find_execution_reports(payload))
    success_source = None
    success = False
    for index, report in enumerate(reports):
        if report.get("status") == "action_executed" and report.get("success") is True:
            success = True
            success_source = f"execution_report[{index}].success"
            break
    return {
        "success": success,
        "success_source": success_source,
        "status": str(loop.get("status") or "unknown"),
        "final_stage": loop.get("final_stage"),
        "reason": loop.get("reason"),
        "action_executions": sum(1 for report in reports if report.get("status") == "action_executed"),
        "executed_steps": sum(int(report.get("executed_steps") or 0) for report in reports if isinstance(report, dict)),
        "model_calls": len(payload.get("model_calls") or []) if isinstance(payload.get("model_calls"), list) else 0,
    }


def _find_execution_reports(value: Any):
    if isinstance(value, dict):
        report = value.get("execution_report")
        if isinstance(report, dict):
            yield report
        for item in value.values():
            yield from _find_execution_reports(item)
    elif isinstance(value, list):
        for item in value:
            yield from _find_execution_reports(item)


def _start_vllm(args: argparse.Namespace, run_dir: Path) -> subprocess.Popen[bytes]:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "vllm.log"
    command = [
        *_python_command_prefix(args.vllm_conda_bin, args.vllm_conda_env),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.vllm_host,
        "--port",
        str(args.vllm_port),
        "--api-key",
        args.api_key,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        *_vllm_args(args),
    ]
    env = dict(os.environ)
    if args.vllm_gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.vllm_gpus
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=env, preexec_fn=_parent_death_preexec)
    print(f"vLLM starting: {log_path}")
    return process


def _start_openpi_workers(args: argparse.Namespace, run_dir: Path, lanes: list[Lane]) -> None:
    for lane in lanes:
        log_path = run_dir / "logs" / f"openpi_worker_lane{lane.index}.log"
        config_path = _openpi_worker_config(args, lane, run_dir)
        command = [
            *_python_command_prefix(args.openpi_conda_bin, args.openpi_conda_env),
            "-m",
            "clawvla.scripts.pi05_worker",
            "--config",
            str(config_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(lane.openpi_port),
        ]
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["PYTHONPATH"] = f"{PROJECT_SRC}:{OPENPI_SRC}"
        env["CLAWVLA_PI05_DIRECT"] = "1"
        if lane.openpi_gpu:
            env["CUDA_VISIBLE_DEVICES"] = lane.openpi_gpu
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=env, preexec_fn=_parent_death_preexec)
        lane.openpi_process = process
        lane.openpi_log = str(log_path)

    for lane in lanes:
        if lane.openpi_process is None or lane.openpi_log is None:
            raise RuntimeError(f"pi0.5 worker lane={lane.index} did not start")
        log_path = Path(lane.openpi_log)
        _wait_for_log(lane.openpi_process, log_path, "pi05_worker_ready", args.openpi_startup_timeout)
        print(f"pi0.5 worker lane={lane.index} port={lane.openpi_port} gpu={lane.openpi_gpu} log={log_path}")


def _openpi_worker_config(args: argparse.Namespace, lane: Lane, run_dir: Path) -> Path:
    payload = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    runtime = payload.setdefault("metadata", {}).setdefault("action_backend", {}).setdefault("openpi_runtime", {})
    runtime["mode"] = "direct"
    runtime["host"] = "127.0.0.1"
    runtime["port"] = lane.openpi_port
    runtime["conda_bin"] = args.openpi_conda_bin
    runtime["conda_env"] = args.openpi_conda_env
    if lane.openpi_gpu:
        runtime["cuda_visible_devices"] = lane.openpi_gpu
    path = run_dir / "configs" / f"openpi_worker_lane{lane.index}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _wait_for_openai_server(
    base_url: str,
    api_key: str,
    process: subprocess.Popen[bytes] | None,
    log_path: Path | None,
    timeout: float,
) -> None:
    deadline = time.time() + timeout
    url = base_url.rstrip("/") + "/models"
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"vLLM exited before ready. log={log_path}")
        request = Request(url, headers={"Authorization": f"Bearer {api_key}"})
        try:
            with urlopen(request, timeout=5.0) as response:
                if 200 <= response.status < 300:
                    print(f"vLLM ready: {base_url}")
                    return
        except Exception:
            time.sleep(2.0)
    raise TimeoutError(f"OpenAI-compatible server not ready within {timeout:.1f}s: {base_url}")


def _wait_for_log(process: subprocess.Popen[bytes], log_path: Path, pattern: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if log_path.exists() and pattern in log_path.read_text(encoding="utf-8", errors="replace"):
            return
        if process.poll() is not None:
            raise RuntimeError(f"process exited before ready. pattern={pattern} log={log_path}")
        time.sleep(1.0)
    raise TimeoutError(f"process did not become ready within {timeout:.1f}s. pattern={pattern} log={log_path}")


class _Progress:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        tasks: list[TaskSpec],
        lanes: list[Lane],
        run_dir: Path,
        completed: dict[str, dict[str, Any]],
    ):
        self.args = args
        self.tasks = tasks
        self.lanes = lanes
        self.run_dir = run_dir
        self.total = len(tasks) * int(args.episodes_per_task)
        self.completed: dict[str, dict[str, Any]] = dict(completed)
        self.active: dict[int, dict[str, Any]] = {}
        self.events: deque[dict[str, Any]] = deque(maxlen=max(1, int(args.progress_event_limit)))
        self.started = time.time()
        self.live = None
        self.last_printed_done = -1

    def start(self) -> None:
        try:
            from rich.live import Live

            self.live = Live(self._render(), refresh_per_second=2, transient=False)
            self.live.start()
        except Exception:
            self.live = None
            print(f"RoboTwin eval: {len(self.completed)}/{self.total}")

    def start_job(self, lane: Lane, job: EpisodeJob) -> None:
        self.active[lane.index] = {
            "lane": lane.index,
            "robotwin_gpu": lane.robotwin_gpu,
            "openpi_port": lane.openpi_port,
            "task_name": job.task_name,
            "eval_index": job.eval_index,
            "seed": job.seed,
            "started": time.time(),
        }
        if self.live is not None:
            self.live.update(self._render())

    def advance(self, payload: dict[str, Any], completed: dict[str, dict[str, Any]]) -> None:
        self.completed = dict(completed)
        lane = payload.get("lane")
        if isinstance(lane, int):
            self.active.pop(lane, None)
        self.events.appendleft(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "task_name": payload.get("task_name"),
                "eval_index": payload.get("eval_index"),
                "seed": payload.get("seed"),
                "success": bool(payload.get("success")),
                "status": payload.get("status"),
                "duration_s": payload.get("duration_s"),
                "lane": payload.get("lane"),
                "reason": payload.get("reason"),
            }
        )
        if self.live is not None:
            self.live.update(self._render())
            return
        done = len(self.completed)
        ok = sum(1 for item in self.completed.values() if item.get("success"))
        acc = ok / done if done else 0.0
        if done != self.last_printed_done or not payload.get("success"):
            self.last_printed_done = done
            print(
                f"[{done}/{self.total}] acc={acc:.2%} ok={ok} "
                f"task={payload.get('task_name')} ep={payload.get('eval_index')} success={payload.get('success')}"
            )

    def stop(self) -> None:
        if self.live is not None:
            self.live.update(self._render())
            self.live.stop()

    def _render(self) -> Any:
        from rich import box
        from rich.console import Group
        from rich.panel import Panel
        from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn
        from rich.table import Table

        done = len(self.completed)
        ok = sum(1 for item in self.completed.values() if item.get("success"))
        failed = done - ok
        acc = ok / done if done else 0.0
        elapsed = max(1e-6, time.time() - self.started)
        episode_rate = done / elapsed
        eta = (self.total - done) / episode_rate if episode_rate > 0 and done < self.total else 0.0

        progress = Progress(
            TextColumn("[bold cyan]Overall"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            expand=True,
        )
        progress.add_task("eval", total=self.total, completed=done)

        overview = Table.grid(expand=True, padding=(0, 2))
        overview.add_column(ratio=2)
        overview.add_column(ratio=1)
        overview.add_column(ratio=1)
        overview.add_column(ratio=1)
        overview.add_row(
            f"[bold]Run[/bold] {self.run_dir.name}",
            f"[bold]Tasks[/bold] {len(self.tasks)}",
            f"[bold]Workers[/bold] {len(self.lanes)}",
            f"[bold]Active[/bold] {len(self.active)}",
        )
        overview.add_row(
            f"[bold]Config[/bold] {self.args.task_config}",
            f"[bold]Done[/bold] {done}/{self.total}",
            f"[bold green]OK[/bold green] {ok}",
            f"[bold red]Fail[/bold red] {failed}",
        )
        overview.add_row(
            f"[bold]Elapsed[/bold] {self._format_duration(elapsed)}",
            f"[bold]ETA[/bold] {self._format_duration(eta)}",
            f"[bold]Rate[/bold] {episode_rate:.3f}/s",
            f"[bold]Acc[/bold] {acc:.2%}",
        )

        active_table = Table(title="Active workers", box=box.SIMPLE_HEAVY, expand=True)
        active_table.add_column("Lane", justify="right")
        active_table.add_column("GPU", justify="right")
        active_table.add_column("pi0.5", justify="right")
        active_table.add_column("Task", style="cyan", overflow="fold")
        active_table.add_column("Ep", justify="right")
        active_table.add_column("Seed", justify="right")
        active_table.add_column("Runtime", justify="right")
        if self.active:
            for item in sorted(self.active.values(), key=lambda value: int(value["lane"])):
                active_table.add_row(
                    str(item["lane"]),
                    str(item.get("robotwin_gpu") or "-"),
                    str(item.get("openpi_port") or "-"),
                    str(item.get("task_name") or "-"),
                    str(item.get("eval_index") or 0),
                    str(item.get("seed") or "-"),
                    self._format_duration(time.time() - float(item.get("started") or time.time())),
                )
        else:
            active_table.add_row("-", "-", "-", "idle", "-", "-", "-")

        task_table = Table(title="Per-task accuracy", box=box.SIMPLE_HEAVY, expand=True)
        task_table.add_column("Task", style="cyan", no_wrap=True)
        task_table.add_column("Done", justify="right")
        task_table.add_column("OK", justify="right")
        task_table.add_column("Acc", justify="right")
        task_table.add_column("Last seed", justify="right")
        task_table.add_column("Last status", overflow="fold")
        task_limit = max(1, int(self.args.progress_task_limit))
        for task_name, items in self._task_rows()[:task_limit]:
            task_done = len(items)
            task_ok = sum(1 for item in items if item.get("success"))
            task_acc = task_ok / task_done if task_done else 0.0
            last = max(items, key=lambda item: int(item.get("eval_index", -1))) if items else {}
            status_style = "green" if task_done >= int(self.args.episodes_per_task) else "yellow"
            task_table.add_row(
                task_name,
                f"[{status_style}]{task_done}/{self.args.episodes_per_task}[/{status_style}]",
                str(task_ok),
                f"{task_acc:.1%}" if task_done else "-",
                str(last.get("seed") or "-"),
                str(last.get("status") or "-"),
            )
        if len(self.tasks) > task_limit:
            task_table.add_row(f"... {len(self.tasks) - task_limit} more", "", "", "", "", "use --progress-task-limit")

        events_table = Table(title="Recent episodes", box=box.SIMPLE, expand=True)
        events_table.add_column("Time", no_wrap=True)
        events_table.add_column("Lane", justify="right")
        events_table.add_column("Task", style="cyan", overflow="fold")
        events_table.add_column("Ep", justify="right")
        events_table.add_column("Seed", justify="right")
        events_table.add_column("Result", no_wrap=True)
        events_table.add_column("Sec", justify="right")
        events_table.add_column("Status", overflow="fold")
        for event in self.events:
            result_style = "green" if event.get("success") else "red"
            duration = event.get("duration_s")
            events_table.add_row(
                str(event.get("time") or ""),
                str(event.get("lane") or "-"),
                str(event.get("task_name") or ""),
                str(event.get("eval_index") or 0),
                str(event.get("seed") or ""),
                f"[{result_style}]{'success' if event.get('success') else 'fail'}[/{result_style}]",
                f"{float(duration):.1f}" if isinstance(duration, (int, float)) else "-",
                str(event.get("status") or ""),
            )

        return Group(
            Panel(Group(overview, progress), title="RoboTwin official eval", border_style="cyan"),
            active_table,
            task_table,
            events_table,
        )

    def _task_rows(self) -> list[tuple[str, list[dict[str, Any]]]]:
        by_task: dict[str, list[dict[str, Any]]] = {task.task_name: [] for task in self.tasks}
        for item in self.completed.values():
            task_name = str(item.get("task_name"))
            by_task.setdefault(task_name, []).append(item)

        def sort_key(row: tuple[str, list[dict[str, Any]]]) -> tuple[int, float, str]:
            task_name, items = row
            done = len(items)
            ok = sum(1 for item in items if item.get("success"))
            acc = ok / done if done else 0.0
            incomplete = done < int(self.args.episodes_per_task)
            return (0 if incomplete else 1, acc, task_name)

        return sorted(by_task.items(), key=sort_key)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h{minutes:02d}m"
        if minutes:
            return f"{minutes}m{secs:02d}s"
        return f"{secs}s"


def _load_tasks(path: Path) -> list[TaskSpec]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = (((payload or {}).get("rollout") or {}).get("tasks") or []) if isinstance(payload, dict) else []
    tasks: list[TaskSpec] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        task_name = str(item.get("task_name") or "").strip()
        instruction = str(item.get("instruction") or task_name).strip()
        if task_name:
            tasks.append(TaskSpec(task_name=task_name, instruction=instruction))
    if not tasks:
        raise ValueError(f"no tasks found in {path}")
    return tasks


def _filter_tasks(tasks: list[TaskSpec], args: argparse.Namespace) -> list[TaskSpec]:
    selected = [str(name).strip() for name in args.task_name if str(name).strip()]
    if selected:
        wanted = set(selected)
        filtered = [task for task in tasks if task.task_name in wanted]
        missing = sorted(wanted - {task.task_name for task in filtered})
        if missing:
            raise ValueError(f"unknown RoboTwin task_name(s): {', '.join(missing)}")
        tasks = filtered
    if args.task_limit is not None:
        tasks = tasks[: max(0, int(args.task_limit))]
    if not tasks:
        raise ValueError("task filtering produced an empty task list")
    return tasks


def _job_order(args: argparse.Namespace, tasks: list[TaskSpec]) -> list[tuple[TaskSpec, int]]:
    if args.interleave:
        return [(task, index) for index in range(args.episodes_per_task) for task in tasks]
    return [(task, index) for task in tasks for index in range(args.episodes_per_task)]


def _planned_job_count(args: argparse.Namespace, tasks: list[TaskSpec]) -> int:
    return len(tasks) * int(args.episodes_per_task)


def _load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        completed[_episode_key(str(item["task_name"]), int(item["eval_index"]))] = item
    return completed


def _load_seed_states(seed_dir: Path, tasks: list[TaskSpec], seed_arg: int) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    start_seed = 100000 * (1 + int(seed_arg))
    for task in tasks:
        path = seed_dir / f"{task.task_name}.json"
        if path.exists():
            states[task.task_name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            states[task.task_name] = {"task_name": task.task_name, "next_seed": start_seed, "valid": []}
    return states


def _resolve_valid_seed_cache_dir(path: Path) -> Path:
    resolved = path.resolve()
    nested = resolved / "valid_seeds"
    if nested.is_dir():
        return nested
    if resolved.is_dir():
        return resolved
    raise FileNotFoundError(f"valid seed cache dir does not exist: {path}")


def _materialize_valid_seed_cache(source_dir: Path, seed_dir: Path, tasks: list[TaskSpec], args: argparse.Namespace) -> None:
    required = int(args.episodes_per_task)
    copied = 0
    for task in tasks:
        source_path = source_dir / f"{task.task_name}.json"
        if not source_path.exists():
            raise FileNotFoundError(f"missing valid seed cache for task {task.task_name}: {source_path}")
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        valid = payload.get("valid")
        if not isinstance(valid, list):
            raise ValueError(f"valid seed cache has no valid[] list: {source_path}")
        if len(valid) < required:
            raise ValueError(f"valid seed cache for {task.task_name} has {len(valid)} seeds, need {required}: {source_path}")
        trimmed_valid = []
        seen_seeds: set[int] = set()
        for index, item in enumerate(valid[:required]):
            if not isinstance(item, dict):
                raise ValueError(f"invalid seed entry for {task.task_name} index={index}: {item!r}")
            entry_index = int(item.get("eval_index", index))
            if entry_index != index:
                raise ValueError(f"seed cache eval_index mismatch for {task.task_name}: got {entry_index}, expected {index}")
            seed = int(item["seed"])
            if seed in seen_seeds:
                raise ValueError(f"duplicate seed for {task.task_name}: {seed}")
            seen_seeds.add(seed)
            trimmed_valid.append(
                {
                    "eval_index": index,
                    "seed": seed,
                    "instruction": str(item.get("instruction") or task.instruction),
                }
            )
        state = dict(payload)
        state.update(
            {
                "task_name": task.task_name,
                "task_config": args.task_config,
                "seed_arg": int(args.seed),
                "target_valid": required,
                "valid": trimmed_valid,
                "source_valid_seed_cache": str(source_path),
            }
        )
        destination = seed_dir / f"{task.task_name}.json"
        if source_path.resolve() != destination.resolve():
            _write_json(destination, state)
            copied += 1
    _write_json(
        seed_dir / "_cache_manifest.json",
        {
            "source_dir": str(source_dir),
            "tasks": len(tasks),
            "episodes_per_task": required,
            "copied_files": copied,
            "loaded_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def _save_seed_state(seed_dir: Path, task_name: str, state: dict[str, Any]) -> None:
    path = seed_dir / f"{task_name}.json"
    path.write_text(json.dumps(state, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _make_lanes(args: argparse.Namespace) -> list[Lane]:
    workers = max(1, int(args.workers))
    robotwin_gpus = _split_csv(args.robotwin_gpus)
    openpi_gpus = _split_csv(args.openpi_gpus)
    lanes = []
    for index in range(workers):
        lanes.append(
            Lane(
                index=index,
                robotwin_gpu=robotwin_gpus[index % len(robotwin_gpus)] if robotwin_gpus else None,
                openpi_gpu=openpi_gpus[index % len(openpi_gpus)] if openpi_gpus else None,
                openpi_port=int(args.openpi_port_base) + index,
            )
        )
    return lanes


def _episode_env(args: argparse.Namespace, lane: Lane, base_url: str) -> dict[str, str]:
    env = _base_robotwin_env(args)
    env["OPENAI_COMPATIBLE_API_BASE_URL"] = base_url
    env["OPENAI_COMPATIBLE_API_KEY"] = args.api_key
    if lane.robotwin_gpu:
        env["CUDA_VISIBLE_DEVICES"] = lane.robotwin_gpu
    return env


def _base_robotwin_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = ":".join(str(path) for path in (PYTORCH3D_TARGET, PROJECT_SRC, OPENPI_SRC))
    library_paths = _robotwin_library_paths(args)
    if library_paths:
        existing = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = ":".join([*library_paths, existing] if existing else library_paths)
    env["__EGL_VENDOR_LIBRARY_DIRS"] = "/usr/share/glvnd/egl_vendor.d"
    env["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
    return env


def _robotwin_library_paths(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []
    python_path = Path(str(args.robotwin_python))
    if python_path.name == "python":
        candidate = python_path.parent.parent / "lib"
        if candidate.exists():
            paths.append(str(candidate))
    return paths


def _write_summary(run_dir: Path, args: argparse.Namespace, tasks: list[TaskSpec], completed: dict[str, dict[str, Any]]) -> None:
    per_task = {}
    for task in tasks:
        items = [item for item in completed.values() if item.get("task_name") == task.task_name]
        done = len(items)
        success = sum(1 for item in items if item.get("success"))
        per_task[task.task_name] = {
            "done": done,
            "success": success,
            "target": args.episodes_per_task,
            "accuracy": success / done if done else None,
        }
    total_done = len(completed)
    total_success = sum(1 for item in completed.values() if item.get("success"))
    summary = {
        "run_id": run_dir.name,
        "task_config": args.task_config,
        "valid_seed_cache_dir": args.valid_seed_cache_dir,
        "episodes_per_task": args.episodes_per_task,
        "task_count": len(tasks),
        "target_episodes": len(tasks) * args.episodes_per_task,
        "completed_episodes": total_done,
        "successes": total_success,
        "overall_accuracy": total_success / total_done if total_done else None,
        "per_task": per_task,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(run_dir / "summary.json", summary)


def _write_task_results(run_dir: Path, tasks: list[TaskSpec], completed: dict[str, dict[str, Any]]) -> None:
    out_dir = run_dir / "task_results"
    out_dir.mkdir(exist_ok=True)
    for task in tasks:
        items = sorted(
            [item for item in completed.values() if item.get("task_name") == task.task_name],
            key=lambda item: int(item.get("eval_index", -1)),
        )
        success = sum(1 for item in items if item.get("success"))
        _write_json(
            out_dir / f"{task.task_name}.json",
            {
                "task_name": task.task_name,
                "episodes": len(items),
                "success": success,
                "accuracy": success / len(items) if items else None,
                "items": items,
            },
        )


def _run_dir(args: argparse.Namespace) -> Path:
    run_id = args.run_id or f"robotwin_official_qwen3vl8b_pi05_{args.task_config}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return Path(args.output_root) / run_id


def _run_config_payload(args: argparse.Namespace, tasks: list[TaskSpec]) -> dict[str, Any]:
    payload = vars(args).copy()
    return {"args": payload, "task_count": len(tasks), "tasks": [asdict(task) for task in tasks]}


def _read_result_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _episode_key(task_name: str, eval_index: int) -> str:
    return f"{task_name}:{eval_index}"


def _python_command_prefix(conda_bin: str, conda_env: str) -> list[str]:
    conda_path = Path(conda_bin)
    python_path = conda_path.parent.parent / "envs" / conda_env / "bin" / "python"
    if python_path.exists():
        return [str(python_path)]
    return [str(conda_path), "run", "--no-capture-output", "-n", conda_env, "python"]


def _vllm_args(args: argparse.Namespace) -> list[str]:
    if args.vllm_arg:
        return list(args.vllm_arg)
    return ["--trust-remote-code", "--dtype", "bfloat16", "--max-model-len", "16384"]


def _read_tail_lines(path: Path, max_lines: int) -> list[str]:
    if max_lines <= 0:
        return []
    lines: deque[str] = deque(maxlen=max_lines)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                lines.append(line.rstrip("\n"))
    except Exception:
        return []
    return list(lines)


def _parent_death_preexec() -> None:
    import ctypes

    os.setsid()
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=20.0)


if __name__ == "__main__":
    main()
