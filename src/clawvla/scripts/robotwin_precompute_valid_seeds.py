from __future__ import annotations

import argparse
import asyncio
from collections import deque
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, TextIO

import yaml

from clawvla.scripts.robotwin_official_seed_check import _check_seed, _ensure_repo


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
ROBOTWIN_ROOT = WORKSPACE_ROOT / "RoboTwin"
PYTORCH3D_TARGET = Path("/mnt/wangwai/tmp_pytorch3d_target")

DEFAULT_TASKS_CONFIG = str(PROJECT_ROOT / "configs/rl/tasks/robotwin_all.yaml")
DEFAULT_BASE_CONFIG = str(PROJECT_ROOT / "configs/robotwin_pi05_worker_probe.json")
DEFAULT_ROBOTWIN_PYTHON = "/mnt/wangwai/miniconda3/envs/robotwin-py312/bin/python"


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    instruction: str


@dataclass
class Lane:
    index: int
    gpu: str | None
    process: asyncio.subprocess.Process | None = None
    stdout_log: TextIO | None = None
    stderr_log: TextIO | None = None


@dataclass
class CandidateJob:
    request_id: str
    task_name: str
    seed: int
    episode_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute official-style RoboTwin valid eval seeds with parallel expert checks."
    )
    parser.add_argument("--tasks-config", default=DEFAULT_TASKS_CONFIG)
    parser.add_argument("--task-name", action="append", default=[], help="Restrict to one or more RoboTwin task names.")
    parser.add_argument("--task-limit", type=int, default=None, help="Restrict to the first N tasks after filtering.")
    parser.add_argument("--repo-root", default=str(ROBOTWIN_ROOT))
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--target-valid", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="Official eval seed arg; start seed is 100000*(1+seed).")
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--language-num", type=int, default=100)
    parser.add_argument("--camera-profile", default=None, help="Override camera profile; defaults to base config robotwin.camera_profile.")
    parser.add_argument("--output-root", default="/mnt/wangwai/vla/clawvla/runs/eval")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default=None, help="Direct output directory. Overrides --output-root/--run-id.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--robotwin-python", default=DEFAULT_ROBOTWIN_PYTHON)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Comma-separated GPU ids for seed-check lanes.")
    parser.add_argument("--lanes-per-gpu", type=int, default=1)
    parser.add_argument("--workers", type=int, default=None, help="Override worker count; GPUs are assigned round-robin.")
    parser.add_argument("--seed-check-timeout", type=float, default=900.0)
    parser.add_argument("--max-candidates-per-task", type=int, default=100000)
    parser.add_argument(
        "--max-infra-failures",
        type=int,
        default=3,
        help="Stop a task after this many consecutive rendering/runtime infrastructure failures.",
    )
    parser.add_argument("--status-interval", type=float, default=5.0)
    parser.add_argument("--progress-task-limit", type=int, default=60)
    parser.add_argument("--progress-event-limit", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        _worker_main(args)
        return
    asyncio.run(_manager_main(args))


async def _manager_main(args: argparse.Namespace) -> None:
    tasks = _filter_tasks(_load_tasks(Path(args.tasks_config)), args)
    run_dir = _run_dir(args)
    seed_dir = run_dir / "valid_seeds"
    log_dir = run_dir / "logs"
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    resolved_camera_profile = args.camera_profile or _camera_profile_from_base(Path(args.base_config))
    args.camera_profile = resolved_camera_profile
    start_seed = 100000 * (1 + int(args.seed))
    states = _load_seed_states(seed_dir, tasks, args, start_seed)
    lanes = _make_lanes(args)

    _write_json(
        run_dir / "valid_seed_run_config.json",
        {
            "args": vars(args),
            "resolved_camera_profile": resolved_camera_profile,
            "start_seed": start_seed,
            "task_count": len(tasks),
            "tasks": [asdict(task) for task in tasks],
            "created_or_updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    if args.dry_run:
        _write_summary(run_dir, tasks, states, args)
        print(f"dry_run: tasks={len(tasks)} lanes={len(lanes)} run_dir={run_dir}")
        return

    scheduler = _Scheduler(args=args, tasks=tasks, states=states, seed_dir=seed_dir, run_dir=run_dir)
    progress = _Progress(args=args, tasks=tasks, states=states, run_dir=run_dir)
    progress.start()
    try:
        await _start_lanes(args, lanes, run_dir)
        workers = [asyncio.create_task(_lane_loop(args, lane, scheduler, progress, run_dir)) for lane in lanes]
        reporter = asyncio.create_task(progress.report_loop())
        await asyncio.gather(*workers)
        reporter.cancel()
        await asyncio.gather(reporter, return_exceptions=True)
    finally:
        progress.stop()
        await _stop_lanes(lanes)
        _write_summary(run_dir, tasks, states, args)
    print(f"valid seeds: {seed_dir}")
    print(f"summary: {run_dir / 'valid_seed_summary.json'}")


async def _start_lanes(args: argparse.Namespace, lanes: list[Lane], run_dir: Path) -> None:
    for lane in lanes:
        await _start_lane(args, lane, run_dir)


async def _start_lane(args: argparse.Namespace, lane: Lane, run_dir: Path) -> None:
    stdout_log = (run_dir / "logs" / f"seed_worker_lane{lane.index}.stdout.log").open("a", encoding="utf-8")
    stderr_log = (run_dir / "logs" / f"seed_worker_lane{lane.index}.stderr.log").open("a", encoding="utf-8")
    command = [
        args.robotwin_python,
        "-m",
        "clawvla.scripts.robotwin_precompute_valid_seeds",
        "--worker",
        "--repo-root",
        args.repo_root,
        "--task-config",
        args.task_config,
        "--instruction-type",
        args.instruction_type,
        "--language-num",
        str(args.language_num),
    ]
    if args.camera_profile:
        command.extend(["--camera-profile", str(args.camera_profile)])
    env = _robotwin_env(args, lane.gpu)
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=stderr_log,
        preexec_fn=_parent_death_preexec,
    )
    lane.process = process
    lane.stdout_log = stdout_log
    lane.stderr_log = stderr_log
    print(f"seed worker lane={lane.index} gpu={lane.gpu} pid={process.pid}")


async def _stop_lanes(lanes: list[Lane]) -> None:
    for lane in lanes:
        process = lane.process
        if process is not None and process.returncode is None:
            try:
                if process.stdin is not None:
                    process.stdin.write(json.dumps({"control": "stop"}).encode("utf-8") + b"\n")
                    await process.stdin.drain()
            except Exception:
                pass
    for lane in lanes:
        process = lane.process
        if process is not None and process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                _terminate_process_group(process)
                await process.wait()
        if lane.stdout_log is not None:
            lane.stdout_log.close()
        if lane.stderr_log is not None:
            lane.stderr_log.close()


async def _lane_loop(
    args: argparse.Namespace,
    lane: Lane,
    scheduler: "_Scheduler",
    progress: "_Progress",
    run_dir: Path,
) -> None:
    while True:
        job = await scheduler.next_job()
        if job is None:
            return
        started = time.time()
        try:
            response = await _request_worker(args, lane, job)
        except Exception as exc:
            response = {
                "id": job.request_id,
                "ok": False,
                "task_name": job.task_name,
                "seed": job.seed,
                "status": "manager_worker_exception",
                "error": f"{type(exc).__name__}: {exc}",
            }
            await _restart_lane(args, lane, run_dir)
        response["duration_s"] = time.time() - started
        await scheduler.finish_job(job, response)
        progress.advance(response)


async def _restart_lane(args: argparse.Namespace, lane: Lane, run_dir: Path) -> None:
    process = lane.process
    if process is not None and process.returncode is None:
        _terminate_process_group(process)
        try:
            await asyncio.wait_for(process.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
    if lane.stdout_log is not None:
        lane.stdout_log.close()
        lane.stdout_log = None
    if lane.stderr_log is not None:
        lane.stderr_log.close()
        lane.stderr_log = None
    await _start_lane(args, lane, run_dir)


async def _request_worker(args: argparse.Namespace, lane: Lane, job: CandidateJob) -> dict[str, Any]:
    process = lane.process
    if process is None or process.stdin is None or process.stdout is None:
        raise RuntimeError(f"lane {lane.index} has no active worker")
    if process.returncode is not None:
        raise RuntimeError(f"lane {lane.index} worker exited with code {process.returncode}")

    payload = {
        "id": job.request_id,
        "task_name": job.task_name,
        "seed": job.seed,
        "episode_index": job.episode_index,
    }
    process.stdin.write(json.dumps(payload, ensure_ascii=True).encode("utf-8") + b"\n")
    await process.stdin.drain()
    deadline = time.time() + max(1.0, float(args.seed_check_timeout))
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            _terminate_process_group(process)
            raise TimeoutError(f"seed check timeout lane={lane.index} task={job.task_name} seed={job.seed}")
        line = await asyncio.wait_for(process.stdout.readline(), timeout=remaining)
        if not line:
            raise RuntimeError(f"lane {lane.index} worker terminated before response")
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            continue
        try:
            item = json.loads(text)
        except json.JSONDecodeError:
            if lane.stdout_log is not None:
                lane.stdout_log.write(text + "\n")
                lane.stdout_log.flush()
            continue
        if item.get("id") == job.request_id:
            return item
        if lane.stdout_log is not None:
            lane.stdout_log.write(json.dumps(item, ensure_ascii=True) + "\n")
            lane.stdout_log.flush()


class _Scheduler:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        tasks: list[TaskSpec],
        states: dict[str, dict[str, Any]],
        seed_dir: Path,
        run_dir: Path,
    ) -> None:
        self.args = args
        self.tasks = tasks
        self.states = states
        self.seed_dir = seed_dir
        self.run_dir = run_dir
        self.lock = asyncio.Lock()
        self.in_flight: set[str] = set()
        self.cursor = 0
        self.request_counter = 0

    async def next_job(self) -> CandidateJob | None:
        async with self.lock:
            if self._done_locked():
                return None
            for offset in range(len(self.tasks)):
                index = (self.cursor + offset) % len(self.tasks)
                task = self.tasks[index]
                state = self.states[task.task_name]
                if task.task_name in self.in_flight:
                    continue
                if len(state.get("valid", [])) >= int(self.args.target_valid):
                    continue
                if state.get("blocked"):
                    continue
                attempts = int(state.get("attempts", 0))
                if attempts >= int(self.args.max_candidates_per_task):
                    continue
                seed = int(state["next_seed"])
                episode_index = len(state.get("valid", []))
                state["next_seed"] = seed + 1
                state["active_seed"] = seed
                state["active_since"] = time.time()
                self.in_flight.add(task.task_name)
                self.cursor = (index + 1) % len(self.tasks)
                self.request_counter += 1
                return CandidateJob(
                    request_id=f"{task.task_name}:{seed}:{self.request_counter}",
                    task_name=task.task_name,
                    seed=seed,
                    episode_index=episode_index,
                )
            if self.in_flight:
                return None
            return None

    async def finish_job(self, job: CandidateJob, response: dict[str, Any]) -> None:
        async with self.lock:
            state = self.states[job.task_name]
            state["attempts"] = int(state.get("attempts", 0)) + 1
            state["next_seed"] = max(int(state.get("next_seed", job.seed + 1)), job.seed + 1)
            state["last_checked_seed"] = job.seed
            state["updated_at"] = datetime.now().isoformat(timespec="seconds")
            valid = state.setdefault("valid", [])
            if bool(response.get("ok")) and len(valid) < int(self.args.target_valid):
                instruction = response.get("instruction") or _task_instruction(self.tasks, job.task_name)
                valid.append(
                    {
                        "eval_index": len(valid),
                        "seed": int(job.seed),
                        "instruction": str(instruction),
                    }
                )
                state["last_valid_seed"] = job.seed
                state["consecutive_infra_failures"] = 0
            else:
                state["failed"] = int(state.get("failed", 0)) + 1
                state["last_failure"] = {
                    "seed": job.seed,
                    "status": response.get("status"),
                    "error": response.get("error"),
                }
                if _is_infra_failure(response):
                    state["consecutive_infra_failures"] = int(state.get("consecutive_infra_failures", 0)) + 1
                    if int(state["consecutive_infra_failures"]) >= int(self.args.max_infra_failures):
                        state["blocked"] = True
                        state["blocked_reason"] = state["last_failure"]
                else:
                    state["consecutive_infra_failures"] = 0
            self.in_flight.discard(job.task_name)
            state.pop("active_seed", None)
            state.pop("active_since", None)
            _save_seed_state(self.seed_dir, job.task_name, state)
            _write_summary(self.run_dir, self.tasks, self.states, self.args)

    def _done_locked(self) -> bool:
        for task in self.tasks:
            state = self.states[task.task_name]
            if len(state.get("valid", [])) >= int(self.args.target_valid):
                continue
            if state.get("blocked"):
                continue
            if int(state.get("attempts", 0)) >= int(self.args.max_candidates_per_task):
                continue
            return False
        return not self.in_flight


class _Progress:
    def __init__(self, *, args: argparse.Namespace, tasks: list[TaskSpec], states: dict[str, dict[str, Any]], run_dir: Path):
        self.args = args
        self.tasks = tasks
        self.states = states
        self.run_dir = run_dir
        self.started = time.time()
        self.live = None
        self.last_printed_valid = -1
        self.events: deque[dict[str, Any]] = deque(maxlen=max(1, int(args.progress_event_limit)))

    def start(self) -> None:
        total = len(self.tasks) * int(self.args.target_valid)
        completed = self._valid_count()
        try:
            from rich.live import Live

            self.live = Live(self._render(), refresh_per_second=2, transient=False)
            self.live.start()
        except Exception:
            self.live = None
            print(f"RoboTwin valid seeds: {completed}/{total}")

    def advance(self, response: dict[str, Any]) -> None:
        self.events.appendleft(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "task": response.get("task_name"),
                "seed": response.get("seed"),
                "ok": bool(response.get("ok")),
                "status": response.get("status"),
                "duration_s": response.get("duration_s"),
            }
        )
        if self.live is not None:
            self.live.update(self._render())
            return
        valid = self._valid_count()
        if valid != self.last_printed_valid or not response.get("ok"):
            self.last_printed_valid = valid
            print(
                f"valid={valid}/{len(self.tasks) * int(self.args.target_valid)} "
                f"attempts={self._attempt_count()} task={response.get('task_name')} "
                f"seed={response.get('seed')} ok={response.get('ok')} status={response.get('status')}"
            )

    async def report_loop(self) -> None:
        while True:
            await asyncio.sleep(max(1.0, float(self.args.status_interval)))
            _write_summary(self.run_dir, self.tasks, self.states, self.args)
            if self.live is not None:
                self.live.update(self._render())

    def stop(self) -> None:
        if self.live is not None:
            self.live.update(self._render())
            self.live.stop()

    def _valid_count(self) -> int:
        return sum(min(len(state.get("valid", [])), int(self.args.target_valid)) for state in self.states.values())

    def _attempt_count(self) -> int:
        return sum(int(state.get("attempts", 0)) for state in self.states.values())

    def _render(self) -> Any:
        from rich import box
        from rich.console import Group
        from rich.panel import Panel
        from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn
        from rich.table import Table

        total = len(self.tasks) * int(self.args.target_valid)
        completed = self._valid_count()
        attempts = self._attempt_count()
        elapsed = max(1e-6, time.time() - self.started)
        valid_rate = completed / elapsed
        seed_yield = completed / attempts if attempts else 0.0
        eta = (total - completed) / valid_rate if valid_rate > 0 and completed < total else 0.0
        blocked = sum(1 for state in self.states.values() if state.get("blocked"))
        active = sum(1 for state in self.states.values() if state.get("active_seed") is not None)

        progress = Progress(
            TextColumn("[bold cyan]Overall"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            expand=True,
        )
        progress.add_task("overall", total=total, completed=completed)

        overview = Table.grid(expand=True, padding=(0, 2))
        overview.add_column(ratio=2)
        overview.add_column(ratio=1)
        overview.add_column(ratio=1)
        overview.add_column(ratio=1)
        overview.add_row(
            f"[bold]Run[/bold] {self.run_dir.name}",
            f"[bold]Tasks[/bold] {len(self.tasks)}",
            f"[bold]Active[/bold] {active}",
            f"[bold]Blocked[/bold] {blocked}",
        )
        overview.add_row(
            f"[bold]Config[/bold] {self.args.task_config}",
            f"[bold]Seed[/bold] {self.args.seed}",
            f"[bold]Target[/bold] {self.args.target_valid}/task",
            f"[bold]Attempts[/bold] {attempts}",
        )
        overview.add_row(
            f"[bold]Elapsed[/bold] {self._format_duration(elapsed)}",
            f"[bold]ETA[/bold] {self._format_duration(eta)}",
            f"[bold]Valid/s[/bold] {valid_rate:.3f}",
            f"[bold]Yield[/bold] {seed_yield:.1%}",
        )

        task_table = Table(
            title="Per-task seed filtering",
            box=box.SIMPLE_HEAVY,
            expand=True,
            show_lines=False,
        )
        task_table.add_column("Task", style="cyan", no_wrap=True)
        task_table.add_column("Valid", justify="right")
        task_table.add_column("Attempts", justify="right")
        task_table.add_column("Yield", justify="right")
        task_table.add_column("Next", justify="right")
        task_table.add_column("Last OK", justify="right")
        task_table.add_column("Status", no_wrap=True)
        task_table.add_column("Last failure", overflow="fold")

        task_limit = max(1, int(self.args.progress_task_limit))
        rows = sorted(self.tasks, key=lambda task: self._task_sort_key(task.task_name))
        shown = rows[:task_limit]
        for task in shown:
            state = self.states[task.task_name]
            valid_count = min(len(state.get("valid", [])), int(self.args.target_valid))
            target = int(self.args.target_valid)
            attempts_for_task = int(state.get("attempts", 0))
            task_yield = valid_count / attempts_for_task if attempts_for_task else 0.0
            last_failure = state.get("last_failure") if isinstance(state.get("last_failure"), dict) else {}
            status, status_style = self._task_status(state, valid_count, target)
            task_table.add_row(
                task.task_name,
                f"{valid_count}/{target}",
                str(attempts_for_task),
                f"{task_yield:.1%}" if attempts_for_task else "-",
                str(state.get("next_seed", "-")),
                str(state.get("last_valid_seed", "-")),
                f"[{status_style}]{status}[/{status_style}]",
                str(last_failure.get("status") or "-"),
            )
        if len(rows) > len(shown):
            task_table.add_row(
                f"... {len(rows) - len(shown)} more",
                "",
                "",
                "",
                "",
                "",
                "use --progress-task-limit",
                "",
            )

        events_table = Table(title="Recent seed checks", box=box.SIMPLE, expand=True)
        events_table.add_column("Time", no_wrap=True)
        events_table.add_column("Task", style="cyan", overflow="fold")
        events_table.add_column("Seed", justify="right")
        events_table.add_column("Result", no_wrap=True)
        events_table.add_column("Status", overflow="fold")
        events_table.add_column("Sec", justify="right")
        for event in list(self.events):
            result_style = "green" if event.get("ok") else "yellow"
            duration = event.get("duration_s")
            events_table.add_row(
                str(event.get("time") or ""),
                str(event.get("task") or ""),
                str(event.get("seed") or ""),
                f"[{result_style}]{'valid' if event.get('ok') else 'skip'}[/{result_style}]",
                str(event.get("status") or ""),
                f"{float(duration):.1f}" if isinstance(duration, (int, float)) else "-",
            )

        return Group(
            Panel(Group(overview, progress), title="RoboTwin valid seed precompute", border_style="cyan"),
            task_table,
            events_table,
        )

    def _task_sort_key(self, task_name: str) -> tuple[int, int, str]:
        state = self.states[task_name]
        target = int(self.args.target_valid)
        valid_count = min(len(state.get("valid", [])), target)
        if state.get("blocked"):
            bucket = 0
        elif valid_count < target and state.get("active_seed") is not None:
            bucket = 1
        elif valid_count < target:
            bucket = 2
        else:
            bucket = 3
        return (bucket, -valid_count, task_name)

    def _task_status(self, state: dict[str, Any], valid_count: int, target: int) -> tuple[str, str]:
        if state.get("blocked"):
            return "blocked", "red"
        active_seed = state.get("active_seed")
        if active_seed is not None:
            active_since = state.get("active_since")
            active_s = time.time() - float(active_since) if isinstance(active_since, (int, float)) else 0.0
            return f"checking {active_seed} ({active_s:.0f}s)", "blue"
        if valid_count >= target:
            return "done", "green"
        return "waiting", "yellow"

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


def _worker_main(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    _ensure_worker_env()
    _ensure_repo(repo_root)
    original_stdout = sys.stdout
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        if request.get("control") == "stop":
            return
        request_id = request.get("id")
        seed_args = argparse.Namespace(
            repo_root=str(repo_root),
            task_name=str(request["task_name"]),
            task_config=args.task_config,
            seed=int(request["seed"]),
            episode_index=int(request["episode_index"]),
            instruction_type=args.instruction_type,
            language_num=int(args.language_num),
            camera_profile=args.camera_profile,
        )
        try:
            with redirect_stdout(sys.stderr):
                payload = _check_seed(seed_args, repo_root)
        except Exception as exc:
            payload = {
                "ok": False,
                "task_name": seed_args.task_name,
                "seed": seed_args.seed,
                "episode_index": seed_args.episode_index,
                "status": "worker_seed_check_exception",
                "error": f"{type(exc).__name__}: {exc}",
            }
        payload["id"] = request_id
        print(json.dumps(payload, ensure_ascii=True), file=original_stdout, flush=True)
        gc.collect()


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


def _load_seed_states(
    seed_dir: Path,
    tasks: list[TaskSpec],
    args: argparse.Namespace,
    start_seed: int,
) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for task in tasks:
        path = seed_dir / f"{task.task_name}.json"
        if args.resume and path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            state.setdefault("task_name", task.task_name)
            state.setdefault("next_seed", start_seed)
            state.setdefault("valid", [])
            state.setdefault("attempts", max(0, int(state.get("next_seed", start_seed)) - start_seed))
        else:
            state = {
                "task_name": task.task_name,
                "task_config": args.task_config,
                "seed_arg": int(args.seed),
                "start_seed": start_seed,
                "next_seed": start_seed,
                "target_valid": int(args.target_valid),
                "valid": [],
                "attempts": 0,
                "failed": 0,
            }
        states[task.task_name] = state
    return states


def _save_seed_state(seed_dir: Path, task_name: str, state: dict[str, Any]) -> None:
    path = seed_dir / f"{task_name}.json"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _write_summary(run_dir: Path, tasks: list[TaskSpec], states: dict[str, dict[str, Any]], args: argparse.Namespace) -> None:
    per_task: dict[str, Any] = {}
    total_valid = 0
    total_attempts = 0
    incomplete = []
    for task in tasks:
        state = states[task.task_name]
        valid_count = min(len(state.get("valid", [])), int(args.target_valid))
        attempts = int(state.get("attempts", 0))
        total_valid += valid_count
        total_attempts += attempts
        if valid_count < int(args.target_valid):
            incomplete.append(task.task_name)
        per_task[task.task_name] = {
            "valid": valid_count,
            "target": int(args.target_valid),
            "attempts": attempts,
            "next_seed": int(state.get("next_seed", 0)),
            "blocked": bool(state.get("blocked")),
            "blocked_reason": state.get("blocked_reason"),
            "last_checked_seed": state.get("last_checked_seed"),
            "last_valid_seed": state.get("last_valid_seed"),
            "yield": valid_count / attempts if attempts else None,
        }
    _write_json(
        run_dir / "valid_seed_summary.json",
        {
            "run_id": run_dir.name,
            "task_config": args.task_config,
            "seed_arg": int(args.seed),
            "target_valid_per_task": int(args.target_valid),
            "task_count": len(tasks),
            "total_target": len(tasks) * int(args.target_valid),
            "total_valid": total_valid,
            "total_attempts": total_attempts,
            "complete": not incomplete,
            "incomplete_tasks": incomplete,
            "per_task": per_task,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _is_infra_failure(response: dict[str, Any]) -> bool:
    status = str(response.get("status") or "").lower()
    error = str(response.get("error") or "").lower()
    text = f"{status}\n{error}"
    markers = (
        "failed to find a rendering device",
        "vulkan",
        "vk_icd",
        "egl",
        "libcuda",
        "cuda driver",
        "manager_worker_exception",
        "worker terminated",
        "seed check timeout",
    )
    return any(marker in text for marker in markers)


def _run_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    run_id = args.run_id or f"robotwin_valid_seeds_{args.task_config}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return Path(args.output_root) / run_id


def _camera_profile_from_base(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    robotwin = payload.get("robotwin")
    if isinstance(robotwin, dict) and robotwin.get("camera_profile"):
        return str(robotwin["camera_profile"])
    return None


def _make_lanes(args: argparse.Namespace) -> list[Lane]:
    gpus = _split_csv(args.gpus)
    if args.workers is None:
        workers = max(1, len(gpus) * max(1, int(args.lanes_per_gpu))) if gpus else max(1, int(args.lanes_per_gpu))
    else:
        workers = max(1, int(args.workers))
    lanes: list[Lane] = []
    for index in range(workers):
        lanes.append(Lane(index=index, gpu=gpus[index % len(gpus)] if gpus else None))
    return lanes


def _robotwin_env(args: argparse.Namespace, gpu: str | None) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = f"{PYTORCH3D_TARGET}:{PROJECT_SRC}"
    library_paths = _robotwin_library_paths(args)
    if library_paths:
        existing = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = ":".join([*library_paths, existing] if existing else library_paths)
    env["__EGL_VENDOR_LIBRARY_DIRS"] = "/usr/share/glvnd/egl_vendor.d"
    env["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _robotwin_library_paths(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []
    python_path = Path(str(args.robotwin_python))
    if python_path.name == "python":
        candidate = python_path.parent.parent / "lib"
        if candidate.exists():
            paths.append(str(candidate))
    return paths


def _ensure_worker_env() -> None:
    env_lib = Path(sys.executable).resolve().parent.parent / "lib"
    if env_lib.exists():
        existing = os.environ.get("LD_LIBRARY_PATH")
        prefix = str(env_lib)
        if not existing:
            os.environ["LD_LIBRARY_PATH"] = prefix
        elif prefix not in existing.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{prefix}:{existing}"
    os.environ["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
    os.environ["__EGL_VENDOR_LIBRARY_DIRS"] = "/usr/share/glvnd/egl_vendor.d"


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _task_instruction(tasks: list[TaskSpec], task_name: str) -> str:
    for task in tasks:
        if task.task_name == task_name:
            return task.instruction
    return task_name


def _parent_death_preexec() -> None:
    import ctypes

    os.setsid()
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)


def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


if __name__ == "__main__":
    main()
