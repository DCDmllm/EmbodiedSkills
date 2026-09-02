from __future__ import annotations

import argparse
import asyncio
from collections import deque
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time
import traceback
from typing import Any

import yaml

from clawvla.scripts.robotwin_official_seed_check import (
    _close_env,
    _cwd,
    _ensure_repo,
    _episode_instruction,
    _instantiate_task,
    _task_args,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent.parent


def _default_robotwin_root() -> Path:
    configured = os.environ.get("ROBOTWIN_ROOT") or os.environ.get("CLAWVLA_ROBOTWIN_ROOT")
    if configured:
        return Path(configured).expanduser()
    candidates = (
        WORKSPACE_ROOT / "RoboTwin",
        PROJECT_ROOT.parent / "RoboTwin",
        PROJECT_ROOT / "RoboTwin",
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])


DEFAULT_TASKS_CONFIG = PROJECT_ROOT / "configs/rl/tasks/robotwin_all.yaml"
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "configs/robotwin_pi05_worker_probe.json"
DEFAULT_ROBOTWIN_PYTHON = os.environ.get("ROBOTWIN_PYTHON", sys.executable)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs/data"
DEFAULT_POLISH_CONFIG = PROJECT_ROOT / "configs/krill_gpt55.local.json"
DEFAULT_ROBOTWIN_ROOT = _default_robotwin_root()
OFFICIAL_SEED_RANGE = "100000:199999"


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    instruction: str


@dataclass(frozen=True)
class Lane:
    index: int
    gpu: str | None


@dataclass(frozen=True)
class CollectionJob:
    task_name: str
    task_instruction: str
    episode_index: int
    start_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect RoboTwin expert trajectories with per-self.move subtask segment traces. "
            "Jobs are scheduled interleaved: task1 episode0, task2 episode0, ..., then episode1."
        )
    )
    parser.add_argument("--tasks-config", default=str(DEFAULT_TASKS_CONFIG))
    parser.add_argument(
        "--task-name",
        action="append",
        default=[],
        help="Restrict to one or more RoboTwin task names.",
    )
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument(
        "--repo-root",
        default=str(DEFAULT_ROBOTWIN_ROOT),
        help="RoboTwin repository root. Defaults to $ROBOTWIN_ROOT or a nearby RoboTwin checkout.",
    )
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument(
        "--task-config",
        default="demo_clean",
        help="RoboTwin task config name, normally demo_clean or demo_randomized.",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "custom"])
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=None, help="Defaults: train=200000, val=300000.")
    parser.add_argument(
        "--forbidden-seed-range",
        action="append",
        default=[OFFICIAL_SEED_RANGE],
        help="Inclusive seed range to skip, e.g. 100000:199999. Can be repeated.",
    )
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--language-num", type=int, default=100)
    parser.add_argument("--camera-profile", default=None)
    parser.add_argument("--save-freq", type=int, default=15)
    parser.add_argument("--max-candidates-per-episode", type=int, default=2000)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Comma-separated GPU ids for collection lanes.")
    parser.add_argument(
        "--robotwin-python",
        default=DEFAULT_ROBOTWIN_PYTHON,
        help="Python executable for the RoboTwin environment. Defaults to $ROBOTWIN_PYTHON or current Python.",
    )
    parser.add_argument(
        "--pythonpath-prefix",
        action="append",
        default=[],
        help=(
            "Extra directory prepended to worker PYTHONPATH. May be repeated, for example for a local "
            "pytorch3d installation. The ClawVLA src directory is always added automatically."
        ),
    )
    parser.add_argument("--episode-timeout", type=float, default=3600.0)
    parser.add_argument("--status-interval", type=float, default=5.0)
    parser.add_argument("--progress-task-limit", type=int, default=60)
    parser.add_argument("--progress-event-limit", type=int, default=10)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument(
        "--episode-cache-root",
        default=os.environ.get("CLAWVLA_EPISODE_CACHE_ROOT", ""),
        help=(
            "Optional local scratch root for RoboTwin per-frame .pkl cache. "
            "Use /dev/shm/... or local NVMe to avoid shared-filesystem write storms."
        ),
    )
    parser.add_argument(
        "--merge-hdf5-mode",
        choices=["memory", "stream"],
        default="memory",
        help="memory preserves RoboTwin's fast in-memory merge; stream uses lower peak RAM.",
    )
    parser.add_argument(
        "--rgb-input-order",
        choices=["rgb", "bgr"],
        default="rgb",
        help=(
            "Channel order of RoboTwin PKL image arrays. Official RoboTwin observations are RGB; "
            "the value is converted explicitly before OpenCV JPEG encoding."
        ),
    )
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also encode RoboTwin mp4 videos. HDF5 image observations are always saved.",
    )
    parser.add_argument(
        "--polish-subgoals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rewrite raw expert comments into VLA-style subgoal commands with the configured model.",
    )
    parser.add_argument("--subgoal-polish-config", default=str(DEFAULT_POLISH_CONFIG))
    parser.add_argument("--subgoal-polish-n", type=int, default=4)
    parser.add_argument("--subgoal-polish-temperature", type=float, default=0.35)
    parser.add_argument("--subgoal-polish-max-tokens", type=int, default=2000)
    parser.add_argument("--subgoal-polish-retries", type=int, default=2)
    parser.add_argument(
        "--subgoal-polish-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attach sampled observation frames to the subgoal polishing request.",
    )
    parser.add_argument("--subgoal-polish-image-camera", default="head_camera")
    parser.add_argument("--subgoal-polish-image-samples", default="mid", help="Comma list from start,mid,end.")
    parser.add_argument("--subgoal-polish-image-size", type=int, default=384)
    parser.add_argument("--subgoal-polish-image-quality", type=int, default=70)
    parser.add_argument("--subgoal-polish-max-images", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-payload", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_one:
        _worker_one(args)
        return
    asyncio.run(_manager_main(args))


async def _manager_main(args: argparse.Namespace) -> None:
    _validate_collection_inputs(args)
    tasks = _filter_tasks(_load_tasks(Path(args.tasks_config)), args)
    args.start_seed = _resolve_start_seed(args)
    seed_ranges = _parse_seed_ranges(args.forbidden_seed_range)
    _assert_non_eval_seed_plan(args.start_seed, seed_ranges)

    run_dir = _run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("logs", "states", "raw", "segments", "summaries"):
        (run_dir / child).mkdir(exist_ok=True)

    camera_profile = args.camera_profile or _camera_profile_from_base(Path(args.base_config))
    args.camera_profile = camera_profile
    states = _load_states(run_dir / "states", tasks, args)
    repaired_tasks = _reconcile_states_from_episode_log(run_dir, tasks, states, args)
    for task_name in repaired_tasks:
        _save_state(run_dir / "states", task_name, states[task_name])
    lanes = _make_lanes(args)

    _write_json(
        run_dir / "run_config.json",
        {
            "args": vars(args),
            "task_count": len(tasks),
            "tasks": [asdict(task) for task in tasks],
            "seed_ranges_forbidden": seed_ranges,
            "created_or_updated_at": datetime.now().isoformat(timespec="seconds"),
            "notes": (
                "Official eval seed range is skipped by default; subtask instructions come from env comments "
                "when available."
            ),
        },
    )

    if args.dry_run:
        _write_summary(run_dir, tasks, states, args)
        print(f"dry_run: tasks={len(tasks)} workers={len(lanes)} run_dir={run_dir}")
        return

    scheduler = _Scheduler(args=args, tasks=tasks, states=states, state_dir=run_dir / "states", run_dir=run_dir)
    progress = _Progress(args=args, tasks=tasks, states=states, run_dir=run_dir)
    progress.start()
    reporter = asyncio.create_task(progress.report_loop())
    try:
        workers = [
            asyncio.create_task(_lane_loop(args, lane, scheduler, progress, run_dir, seed_ranges))
            for lane in lanes
        ]
        await asyncio.gather(*workers)
    finally:
        reporter.cancel()
        await asyncio.gather(reporter, return_exceptions=True)
        progress.stop()
        _write_summary(run_dir, tasks, states, args)
        _write_task_manifests(run_dir, tasks, states)
    print(f"raw data: {run_dir / 'raw'}")
    print(f"segments: {run_dir / 'segments'}")
    print(f"summary: {run_dir / 'summary.json'}")


async def _lane_loop(
    args: argparse.Namespace,
    lane: Lane,
    scheduler: "_Scheduler",
    progress: "_Progress",
    run_dir: Path,
    seed_ranges: list[tuple[int, int]],
) -> None:
    while True:
        job = await scheduler.next_job()
        if job is None:
            return
        started = time.time()
        try:
            response = await _run_worker_job(args, lane, job, run_dir, seed_ranges)
        except Exception as exc:
            response = {
                "ok": False,
                "task_name": job.task_name,
                "episode_index": job.episode_index,
                "start_seed": job.start_seed,
                "status": "manager_worker_exception",
                "error": f"{type(exc).__name__}: {exc}",
            }
        response["duration_s"] = time.time() - started
        await scheduler.finish_job(job, response)
        progress.advance(response)


async def _run_worker_job(
    args: argparse.Namespace,
    lane: Lane,
    job: CollectionJob,
    run_dir: Path,
    seed_ranges: list[tuple[int, int]],
) -> dict[str, Any]:
    payload = {
        "job": asdict(job),
        "seed_ranges": seed_ranges,
    }
    log_path = run_dir / "logs" / f"{job.episode_index:04d}_{job.task_name}_lane{lane.index}.log"
    command = [
        args.robotwin_python,
        "-m",
        "clawvla.scripts.robotwin_collect_expert_subtasks",
        "--worker-one",
        "--repo-root",
        args.repo_root,
        "--task-config",
        args.task_config,
        "--instruction-type",
        args.instruction_type,
        "--language-num",
        str(args.language_num),
        "--save-freq",
        str(args.save_freq),
        "--max-candidates-per-episode",
        str(args.max_candidates_per_episode),
        "--episode-cache-root",
        str(args.episode_cache_root),
        "--merge-hdf5-mode",
        str(args.merge_hdf5_mode),
        "--rgb-input-order",
        str(args.rgb_input_order),
        "--output-dir",
        str(run_dir),
        "--subgoal-polish-config",
        str(args.subgoal_polish_config),
        "--subgoal-polish-n",
        str(args.subgoal_polish_n),
        "--subgoal-polish-temperature",
        str(args.subgoal_polish_temperature),
        "--subgoal-polish-max-tokens",
        str(args.subgoal_polish_max_tokens),
        "--subgoal-polish-retries",
        str(args.subgoal_polish_retries),
        "--subgoal-polish-image-camera",
        str(args.subgoal_polish_image_camera),
        "--subgoal-polish-image-samples",
        str(args.subgoal_polish_image_samples),
        "--subgoal-polish-image-size",
        str(args.subgoal_polish_image_size),
        "--subgoal-polish-image-quality",
        str(args.subgoal_polish_image_quality),
        "--subgoal-polish-max-images",
        str(args.subgoal_polish_max_images),
        "--worker-payload",
        json.dumps(payload, ensure_ascii=True),
    ]
    command.append("--polish-subgoals" if args.polish_subgoals else "--no-polish-subgoals")
    command.append("--save-video" if args.save_video else "--no-save-video")
    command.append("--subgoal-polish-images" if args.subgoal_polish_images else "--no-subgoal-polish-images")
    if args.camera_profile:
        command.extend(["--camera-profile", str(args.camera_profile)])
    if args.keep_cache:
        command.append("--keep-cache")

    env = _robotwin_env(args, lane.gpu)
    with log_path.open("ab") as log:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=log,
            preexec_fn=_parent_death_preexec,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=float(args.episode_timeout))
        except asyncio.TimeoutError:
            _terminate_process_group(process, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                _terminate_process_group(process, signal.SIGKILL)
                await process.wait()
            return {
                "ok": False,
                "task_name": job.task_name,
                "episode_index": job.episode_index,
                "start_seed": job.start_seed,
                "status": "episode_timeout",
                "log_path": str(log_path),
            }
    text = stdout.decode("utf-8", errors="replace")
    parsed = _last_json_line(text)
    if parsed is None:
        return {
            "ok": False,
            "task_name": job.task_name,
            "episode_index": job.episode_index,
            "start_seed": job.start_seed,
            "status": "worker_bad_stdout",
            "return_code": process.returncode,
            "stdout_tail": text[-2000:],
            "log_path": str(log_path),
        }
    parsed["return_code"] = process.returncode
    parsed["log_path"] = str(log_path)
    if process.returncode != 0 and parsed.get("ok"):
        parsed["ok"] = False
        parsed["status"] = f"worker_return_code_{process.returncode}"
    return parsed


def _worker_one(args: argparse.Namespace) -> None:
    _ensure_worker_env()
    repo_root = Path(args.repo_root).resolve()
    _ensure_repo(repo_root)
    if not args.worker_payload:
        raise ValueError("--worker-payload is required with --worker-one")
    payload = json.loads(args.worker_payload)
    job = CollectionJob(**payload["job"])
    seed_ranges = [(int(a), int(b)) for a, b in payload.get("seed_ranges", [])]
    try:
        with redirect_stdout(sys.stderr):
            result = _collect_one_job(args, repo_root, job, seed_ranges)
    except Exception as exc:
        result = {
            "ok": False,
            "task_name": job.task_name,
            "episode_index": job.episode_index,
            "start_seed": job.start_seed,
            "status": "worker_exception",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=12),
        }
    print(json.dumps(result, ensure_ascii=True), flush=True)


def _collect_one_job(
    args: argparse.Namespace,
    repo_root: Path,
    job: CollectionJob,
    seed_ranges: list[tuple[int, int]],
) -> dict[str, Any]:
    task_dir = Path(args.output_dir) / "raw" / job.task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    last_failure: dict[str, Any] | None = None
    seed = _skip_forbidden(job.start_seed, seed_ranges)
    max_candidates = int(args.max_candidates_per_episode)
    for _ in range(max_candidates):
        seed = _skip_forbidden(seed, seed_ranges)
        started = time.time()
        plan = _try_plan_seed(args, repo_root, job, task_dir, seed)
        attempts.append({"seed": seed, "status": plan["status"], "ok": plan["ok"]})
        if not plan["ok"]:
            last_failure = plan
            if _is_infra_error(plan):
                plan.update(
                    {
                        "task_name": job.task_name,
                        "episode_index": job.episode_index,
                        "start_seed": job.start_seed,
                        "attempts": len(attempts),
                        "next_seed": seed,
                    }
                )
                return plan
            seed += 1
            continue

        collected = _try_collect_seed(args, repo_root, job, task_dir, seed, plan)
        attempts[-1]["collect_status"] = collected["status"]
        attempts[-1]["collect_ok"] = collected["ok"]
        last_failure = collected
        if collected["ok"]:
            result = {
                **collected,
                "task_name": job.task_name,
                "episode_index": job.episode_index,
                "seed": seed,
                "start_seed": job.start_seed,
                "next_seed": seed + 1,
                "attempts": len(attempts),
                "candidate_attempts": attempts[-20:],
                "duration_s": time.time() - started,
            }
            _append_jsonl(Path(args.output_dir) / "episodes.jsonl", result)
            return result
        if _is_infra_error(collected):
            collected.update(
                {
                    "task_name": job.task_name,
                    "episode_index": job.episode_index,
                    "seed": seed,
                    "start_seed": job.start_seed,
                    "next_seed": seed,
                    "attempts": len(attempts),
                }
            )
            return collected
        seed += 1

    return {
        "ok": False,
        "task_name": job.task_name,
        "episode_index": job.episode_index,
        "start_seed": job.start_seed,
        "next_seed": seed,
        "attempts": len(attempts),
        "candidate_attempts": attempts[-50:],
        "last_failure": _compact_failure(last_failure),
        "status": "max_candidates_without_success",
    }


def _try_plan_seed(
    args: argparse.Namespace,
    repo_root: Path,
    job: CollectionJob,
    task_dir: Path,
    seed: int,
) -> dict[str, Any]:
    task_env = _instantiate_task(repo_root, job.task_name)
    task_args = _base_task_args(args, repo_root, job.task_name, task_dir)
    task_args.update(
        {
            "need_plan": True,
            "save_data": False,
            "collect_data": False,
            "render_freq": 0,
        }
    )
    try:
        with _cwd(repo_root):
            task_env.setup_demo(now_ep_num=job.episode_index, seed=seed, **task_args)
            episode_info = task_env.play_once()
            plan_success = bool(getattr(task_env, "plan_success", False))
            task_success = bool(task_env.check_success()) if hasattr(task_env, "check_success") else False
            if plan_success and task_success:
                task_env.save_traj_data(job.episode_index)
                instruction = _episode_instruction(
                    repo_root=repo_root,
                    task_name=job.task_name,
                    episode_info=episode_info,
                    instruction_type=args.instruction_type,
                    language_num=int(args.language_num),
                )
                return {
                    "ok": True,
                    "status": "valid_expert_seed",
                    "instruction": instruction or job.task_instruction,
                    "episode_info": episode_info,
                    "plan_success": plan_success,
                    "task_success": task_success,
                }
            return {
                "ok": False,
                "status": "expert_failed",
                "plan_success": plan_success,
                "task_success": task_success,
            }
    except Exception as exc:
        return {
            "ok": False,
            "status": "plan_exception",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=6),
        }
    finally:
        _close_env(task_env)


def _try_collect_seed(
    args: argparse.Namespace,
    repo_root: Path,
    job: CollectionJob,
    task_dir: Path,
    seed: int,
    plan: dict[str, Any],
) -> dict[str, Any]:
    task_env = _instantiate_task(repo_root, job.task_name)
    task_args = _base_task_args(args, repo_root, job.task_name, task_dir)
    task_args.update(
        {
            "need_plan": False,
            "save_data": True,
            "collect_data": True,
            "render_freq": 0,
            "save_freq": int(args.save_freq),
        }
    )
    segments: list[dict[str, Any]] = []
    cache_task_dir = _episode_cache_task_dir(args, job, task_dir)
    cache_cleaned = False
    _remove_episode_cache(cache_task_dir, job.episode_index, purpose="stale pre-run")
    try:
        with _cwd(repo_root):
            task_env.setup_demo(now_ep_num=job.episode_index, seed=seed, **task_args)
            traj_data = task_env.load_tran_data(job.episode_index)
            task_args["left_joint_path"] = traj_data["left_joint_path"]
            task_args["right_joint_path"] = traj_data["right_joint_path"]
            task_env.set_path_lst(task_args)
            if cache_task_dir != task_dir:
                cache_task_dir.mkdir(parents=True, exist_ok=True)
                task_env.save_dir = str(cache_task_dir)
            task_env._clawvla_merge_hdf5_mode = str(args.merge_hdf5_mode)
            task_env._clawvla_rgb_input_order = str(args.rgb_input_order)
            _install_segment_trace(task_env, repo_root, job, plan, segments)
            episode_info = task_env.play_once()
            plan_success = bool(getattr(task_env, "plan_success", False))
            task_success = bool(task_env.check_success()) if hasattr(task_env, "check_success") else False
            if not (plan_success and task_success):
                return {
                    "ok": False,
                    "status": "collect_replay_failed",
                    "plan_success": plan_success,
                    "task_success": task_success,
                    "segment_count": len(segments),
                }
            hdf5_path = task_dir / "data" / f"episode{job.episode_index}.hdf5"
            _merge_episode_cache(task_env, task_dir, job.episode_index, save_video=bool(args.save_video))
            polish_report = _polish_episode_segments(args, job, plan, episode_info, segments, hdf5_path)
            metadata = _episode_metadata(
                args=args,
                job=job,
                seed=seed,
                plan=plan,
                episode_info=episode_info,
                task_dir=task_dir,
                hdf5_path=hdf5_path,
                segments=segments,
                polish_report=polish_report,
            )
            segment_path = _write_segment_metadata(Path(args.output_dir), job, metadata)
            _attach_hdf5_metadata(hdf5_path, metadata)
            cache_cleanup = None
            if not args.keep_cache:
                cache_cleanup = _remove_episode_cache(cache_task_dir, job.episode_index, purpose="post-collect")
                cache_cleaned = True
                _remove_empty_scratch_dirs(cache_task_dir, Path(args.episode_cache_root) if args.episode_cache_root else None)
            return {
                "ok": True,
                "status": "collected",
                "seed": seed,
                "instruction": metadata["instruction"],
                "hdf5_path": str(hdf5_path),
                "segment_path": str(segment_path),
                "task_dir": str(task_dir),
                "segment_count": len(segments),
                "frame_count": int(metadata["frame_count"]),
                "annotation_counts": metadata["annotation_counts"],
                "cache_cleanup": cache_cleanup,
            }
    except Exception as exc:
        return {
            "ok": False,
            "status": "collect_exception",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=10),
            "segment_count": len(segments),
        }
    finally:
        if not args.keep_cache and not cache_cleaned:
            _remove_episode_cache(cache_task_dir, job.episode_index, purpose="post-failure")
            _remove_empty_scratch_dirs(cache_task_dir, Path(args.episode_cache_root) if args.episode_cache_root else None)
        _close_env(task_env)


def _install_segment_trace(
    task_env: Any,
    repo_root: Path,
    job: CollectionJob,
    plan: dict[str, Any],
    segments: list[dict[str, Any]],
) -> None:
    original_move = task_env.move
    comments_by_line = _load_move_comments(repo_root / "envs" / f"{job.task_name}.py")

    def traced_move(actions_by_arm1: Any, actions_by_arm2: Any = None, save_freq: int = -1) -> Any:
        caller = inspect.getframeinfo(inspect.currentframe().f_back)  # type: ignore[union-attr]
        start_frame = int(getattr(task_env, "FRAME_IDX", 0) or 0)
        segment_index = len(segments)
        source_comment = comments_by_line.get(int(caller.lineno))
        result = original_move(actions_by_arm1, actions_by_arm2, save_freq=save_freq)
        end_frame = int(getattr(task_env, "FRAME_IDX", 0) or 0)
        instruction = _comment_to_instruction(source_comment)
        segments.append(
            {
                "segment_id": f"{job.task_name}_ep{job.episode_index:04d}_seg{segment_index:03d}",
                "task_name": job.task_name,
                "episode_index": job.episode_index,
                "segment_index": segment_index,
                "frame_start": start_frame,
                "frame_end_exclusive": end_frame,
                "num_saved_frames": max(0, end_frame - start_frame),
                "source_file": _relpath(caller.filename, repo_root),
                "source_line": int(caller.lineno),
                "source_code": _source_call_snippet(Path(caller.filename), int(caller.lineno)),
                "source_context": _source_context_snippet(Path(caller.filename), int(caller.lineno)),
                "source_comment": source_comment,
                "raw_canonical_instruction": instruction,
                "canonical_instruction": instruction,
                "annotation_source": "env_comment" if instruction else None,
                "full_task_instruction": plan.get("instruction") or job.task_instruction,
                "primitive_summary": _summarize_move_args(actions_by_arm1, actions_by_arm2),
                "move_return": bool(result) if isinstance(result, bool) else result,
                "plan_success_after_move": bool(getattr(task_env, "plan_success", False)),
            }
        )
        return result

    task_env.move = traced_move


def _polish_episode_segments(
    args: argparse.Namespace,
    job: CollectionJob,
    plan: dict[str, Any],
    episode_info: Any,
    segments: list[dict[str, Any]],
    hdf5_path: Path,
) -> dict[str, Any]:
    if not args.polish_subgoals:
        return {"enabled": False}
    if not segments:
        return {"enabled": True, "status": "skipped_empty_segments"}

    config_path = Path(args.subgoal_polish_config)
    config = _load_polish_config(config_path)
    model = str(config["model"])
    raw_payload = {
        "task_name": job.task_name,
        "task_instruction": plan.get("instruction") or job.task_instruction,
        "episode_info": episode_info,
        "segments": [_segment_polish_input(segment) for segment in segments],
    }
    visual_inputs, visual_report = _build_segment_visual_inputs(args, hdf5_path, segments)
    if visual_report:
        raw_payload["visual_evidence"] = visual_report.get("labels", [])
    parsed = _request_segment_polish(
        config=config,
        model=model,
        payload=raw_payload,
        visual_inputs=visual_inputs,
        n=max(0, int(args.subgoal_polish_n)),
        temperature=float(args.subgoal_polish_temperature),
        max_tokens=int(args.subgoal_polish_max_tokens),
        retries=max(0, int(args.subgoal_polish_retries)),
    )
    polished_segments = parsed.get("segments")
    if not isinstance(polished_segments, list):
        raise RuntimeError("subgoal_polish_failed: model output missing segments list")

    by_index: dict[int, dict[str, Any]] = {}
    for item in polished_segments:
        if not isinstance(item, dict) or item.get("segment_index") is None:
            continue
        by_index[int(item["segment_index"])] = item

    for segment in segments:
        index = int(segment["segment_index"])
        item = by_index.get(index)
        if item is None:
            raise RuntimeError(f"subgoal_polish_failed: missing polished segment {index}")
        instruction = str(item.get("instruction") or "").strip()
        if not instruction:
            raise RuntimeError(f"subgoal_polish_failed: empty instruction for segment {index}")
        segment["polished_instruction"] = instruction
        segment["canonical_instruction"] = instruction
        segment["annotation_source"] = "model_polished_env_comment"
        segment["subgoal_type"] = str(item.get("subgoal_type") or "").strip() or None
        segment["completion_criteria"] = str(item.get("completion_criteria") or "").strip() or None
        segment["paraphrases"] = _string_list(item.get("paraphrases"))
        segment["polish_model"] = model

    return {
        "enabled": True,
        "status": "polished",
        "config_path": str(config_path),
        "model": model,
        "segment_count": len(segments),
        "visual_inputs": visual_report,
        "paraphrases_per_segment": max(0, int(args.subgoal_polish_n)),
        "raw_text_preview": str(parsed.get("_raw_text_preview") or "")[:1200],
    }


def _segment_polish_input(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "segment_index": segment.get("segment_index"),
        "raw_instruction": segment.get("raw_canonical_instruction") or segment.get("canonical_instruction"),
        "source_comment": segment.get("source_comment"),
        "source_file": segment.get("source_file"),
        "source_line": segment.get("source_line"),
        "source_code": segment.get("source_code"),
        "source_context": segment.get("source_context"),
        "primitive_summary": segment.get("primitive_summary"),
        "frame_start": segment.get("frame_start"),
        "frame_end_exclusive": segment.get("frame_end_exclusive"),
    }


def _load_polish_config(path: Path) -> dict[str, Any]:
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise RuntimeError(f"subgoal_polish_failed: config not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in ("model", "base_url", "api_key") if not payload.get(key)]
    if missing:
        raise RuntimeError(f"subgoal_polish_failed: {path} missing {', '.join(missing)}")
    return payload


def _build_segment_visual_inputs(
    args: argparse.Namespace,
    hdf5_path: Path,
    segments: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not args.subgoal_polish_images:
        return [], {"enabled": False}
    try:
        import base64
        from io import BytesIO

        import h5py
        from PIL import Image

        camera = str(args.subgoal_polish_image_camera)
        samples = _split_csv(str(args.subgoal_polish_image_samples))
        if not samples:
            samples = ["mid"]
        max_images = max(0, int(args.subgoal_polish_max_images))
        image_size = max(64, int(args.subgoal_polish_image_size))
        quality = min(95, max(35, int(args.subgoal_polish_image_quality)))
        visual_inputs: list[dict[str, str]] = []
        labels: list[dict[str, Any]] = []
        with h5py.File(hdf5_path, "r") as handle:
            dataset = handle.get(f"observation/{camera}/rgb")
            if dataset is None:
                return [], {"enabled": True, "status": "missing_camera", "camera": camera}
            total_frames = int(dataset.shape[0])
            for segment in segments:
                for sample_name, frame_index in _sample_segment_frames(segment, samples, total_frames):
                    if len(visual_inputs) >= max_images:
                        break
                    raw = bytes(dataset[frame_index]).rstrip(b"\0")
                    image = Image.open(BytesIO(raw)).convert("RGB")
                    image.thumbnail((image_size, image_size))
                    encoded = BytesIO()
                    image.save(encoded, format="JPEG", quality=quality, optimize=True)
                    data_url = "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")
                    label = {
                        "image_id": f"img{len(visual_inputs):02d}",
                        "segment_index": segment.get("segment_index"),
                        "sample": sample_name,
                        "frame_index": frame_index,
                        "camera": camera,
                    }
                    visual_inputs.append({"label": json.dumps(label, ensure_ascii=False), "data_url": data_url})
                    labels.append(label)
                if len(visual_inputs) >= max_images:
                    break
        return visual_inputs, {
            "enabled": True,
            "status": "attached" if visual_inputs else "empty",
            "camera": camera,
            "samples": samples,
            "image_count": len(visual_inputs),
            "max_images": max_images,
            "labels": labels,
        }
    except Exception as exc:
        return [], {
            "enabled": True,
            "status": "image_error",
            "error": f"{type(exc).__name__}: {exc}",
            "camera": str(args.subgoal_polish_image_camera),
        }


def _sample_segment_frames(
    segment: dict[str, Any],
    samples: list[str],
    total_frames: int,
) -> list[tuple[str, int]]:
    start = int(segment.get("frame_start") or 0)
    end = int(segment.get("frame_end_exclusive") or start + 1) - 1
    start = min(max(0, start), max(0, total_frames - 1))
    end = min(max(start, end), max(0, total_frames - 1))
    mid = (start + end) // 2
    frame_by_name = {"start": start, "mid": mid, "end": end}
    result: list[tuple[str, int]] = []
    seen: set[int] = set()
    for sample in samples:
        key = str(sample).strip().lower()
        if key not in frame_by_name:
            continue
        frame_index = frame_by_name[key]
        if frame_index in seen:
            continue
        seen.add(frame_index)
        result.append((key, frame_index))
    return result


def _polish_user_content(prompt_payload: dict[str, Any], visual_inputs: list[dict[str, str]]) -> Any:
    text = (
        "Polish these expert subgoal segments. Return exactly one JSON object.\n"
        "If visual_evidence is present, image_id entries refer to the attached images in order.\n\n"
        f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
    )
    if not visual_inputs:
        return text
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for image in visual_inputs:
        content.append({"type": "text", "text": f"Attached image metadata: {image['label']}"})
        content.append({"type": "image_url", "image_url": {"url": image["data_url"], "detail": "low"}})
    return content


def _request_segment_polish(
    *,
    config: dict[str, Any],
    model: str,
    payload: dict[str, Any],
    visual_inputs: list[dict[str, str]],
    n: int,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> dict[str, Any]:
    from openai import OpenAI

    from clawvla.components.scheduler import _task_plan_instruction
    from clawvla.json_utils import extract_last_json_dict

    prompt_payload = {
        "style_source": _task_plan_instruction(),
        "record": payload,
        "paraphrases_per_segment": n,
        "required_schema": {
            "segments": [
                {
                    "segment_index": "same integer as input",
                    "subgoal_type": "short action label such as approach, grasp, move, place, release, press",
                    "instruction": "one concise VLA-ready robot command, usually 6-14 words and max about 18",
                    "completion_criteria": "one concrete visual success condition for this exact subgoal",
                    "paraphrases": ["short additional commands with exactly the same semantics"],
                }
            ]
        },
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite RoboTwin expert motion comments into short-horizon robot subgoal commands for a "
                "vision-language-action policy. Match the ClawVLA task-planning style in style_source. "
                "The expert segment fields are authoritative for the current subgoal; use the full task "
                "instruction only as context. Do not convert a one-arm segment into a two-arm command unless "
                "the segment itself says both arms act. Do not choose left or right unless the segment, "
                "primitive summary, or source code makes the arm assignment explicit; otherwise say selected arm "
                "or proper arm. For dual-arm segments, spell out each arm-object mapping explicitly, such as "
                "'use the left arm to grasp X and the right arm to grasp Y'. Do not use vague phrases like "
                "'their respective arms', and do not leave a dual-arm command as only 'use both arms' when the "
                "left/right mapping is available. "
                "Use the attached observation images as visual evidence for object colors, object identities, "
                "and which arm is interacting with which object. "
                "Keep object names, colors, left/right arm assignments, tool roles, and target relations exactly. "
                "Keep each instruction compact: usually 6-14 words, max about 18 words unless a dual-arm mapping "
                "truly requires slightly more. Prefer short names such as 'red-cap bottle', 'green bottle', "
                "'left target', and 'right target'. Remove redundant clauses like 'while keeping it grasped', "
                "'keeping both bottles grasped', 'securely', or 'at the same time' unless that wording is needed "
                "to distinguish this segment from another one. A good dual-arm move is: 'Move the red-cap bottle "
                "to the left target and the green bottle to the right target.' "
                "Remove code-maintainer notes, helper-function explanations, simulation wording, and phrases like "
                "'simulate a click'. Do not invent objects or change the action. Do not create confirmation-only "
                "instructions such as check, confirm, verify, or ensure. Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": _polish_user_content(prompt_payload, visual_inputs),
        },
    ]
    client = OpenAI(
        api_key=str(config["api_key"]),
        base_url=str(config["base_url"]),
        timeout=float(config.get("timeout", 120.0)),
    )
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        request_kwargs["temperature"] = temperature

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(**request_kwargs)
            raw_text = response.choices[0].message.content or ""
            parsed = extract_last_json_dict(raw_text, error_prefix="subgoal polish model output")
            parsed["_raw_text_preview"] = raw_text[:1200]
            return parsed
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(10.0, 1.5 * (attempt + 1)))
    raise RuntimeError(f"subgoal_polish_failed: {type(last_error).__name__}: {last_error}")


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _episode_metadata(
    *,
    args: argparse.Namespace,
    job: CollectionJob,
    seed: int,
    plan: dict[str, Any],
    episode_info: Any,
    task_dir: Path,
    hdf5_path: Path,
    segments: list[dict[str, Any]],
    polish_report: dict[str, Any],
) -> dict[str, Any]:
    frame_count = _hdf5_frame_count(hdf5_path)
    annotated = sum(1 for segment in segments if segment.get("canonical_instruction"))
    model_polished = sum(1 for segment in segments if segment.get("polished_instruction"))
    return {
        "schema_version": 1,
        "task_name": job.task_name,
        "task_config": args.task_config,
        "split": args.split,
        "episode_index": job.episode_index,
        "seed": seed,
        "instruction": plan.get("instruction") or job.task_instruction,
        "task_instruction_from_config": job.task_instruction,
        "instruction_type": args.instruction_type,
        "save_freq": int(args.save_freq),
        "image_encoding": {
            "source_array_channel_order": str(args.rgb_input_order).upper(),
            "jpeg_encoder_input_order": "BGR",
            "decoded_training_order": "RGB",
            "explicit_channel_conversion": bool(args.rgb_input_order == "rgb"),
        },
        "hdf5_path": str(hdf5_path),
        "task_dir": str(task_dir),
        "frame_count": frame_count,
        "segments": segments,
        "segment_count": len(segments),
        "subgoal_polish": polish_report,
        "annotation_counts": {
            "annotated": annotated,
            "missing": len(segments) - annotated,
            "source_env_comment": sum(1 for segment in segments if segment.get("raw_canonical_instruction")),
            "model_polished": model_polished,
        },
        "episode_info": episode_info,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


class _Scheduler:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        tasks: list[TaskSpec],
        states: dict[str, dict[str, Any]],
        state_dir: Path,
        run_dir: Path,
    ):
        self.args = args
        self.tasks = tasks
        self.states = states
        self.state_dir = state_dir
        self.run_dir = run_dir
        self.lock = asyncio.Lock()
        self.in_flight: set[str] = set()

    async def next_job(self) -> CollectionJob | None:
        async with self.lock:
            if self._done_locked():
                return None
            target = int(self.args.episodes_per_task)
            for episode_index in range(target):
                for task in self.tasks:
                    state = self.states[task.task_name]
                    if task.task_name in self.in_flight or state.get("blocked"):
                        continue
                    collected = int(state.get("collected", 0))
                    if collected != episode_index:
                        continue
                    job = CollectionJob(
                        task_name=task.task_name,
                        task_instruction=task.instruction,
                        episode_index=episode_index,
                        start_seed=int(state.get("next_seed", self.args.start_seed)),
                    )
                    self.in_flight.add(task.task_name)
                    state["active_episode"] = episode_index
                    state["active_seed"] = job.start_seed
                    state["active_since"] = time.time()
                    return job
            return None

    async def finish_job(self, job: CollectionJob, response: dict[str, Any]) -> None:
        async with self.lock:
            state = self.states[job.task_name]
            state["attempts"] = int(state.get("attempts", 0)) + int(response.get("attempts") or 0)
            state["last_status"] = response.get("status")
            state["last_episode_index"] = job.episode_index
            state["last_seed"] = response.get("seed")
            if response.get("ok"):
                state["collected"] = int(state.get("collected", 0)) + 1
                state["next_seed"] = int(response.get("next_seed") or (int(response["seed"]) + 1))
                state["last_hdf5_path"] = response.get("hdf5_path")
                state["last_segment_path"] = response.get("segment_path")
                state.setdefault("episodes", []).append(
                    {
                        "episode_index": job.episode_index,
                        "seed": response.get("seed"),
                        "hdf5_path": response.get("hdf5_path"),
                        "segment_path": response.get("segment_path"),
                        "segment_count": response.get("segment_count"),
                        "frame_count": response.get("frame_count"),
                    }
                )
            else:
                state["failed_jobs"] = int(state.get("failed_jobs", 0)) + 1
                state["next_seed"] = int(response.get("next_seed") or job.start_seed)
                state["last_error"] = {
                    "status": response.get("status"),
                    "error": response.get("error"),
                    "log_path": response.get("log_path"),
                }
                if response.get("status") == "max_candidates_without_success" or _is_infra_error(response):
                    state["blocked"] = True
                    state["blocked_reason"] = state["last_error"]
            state.pop("active_episode", None)
            state.pop("active_seed", None)
            state.pop("active_since", None)
            self.in_flight.discard(job.task_name)
            _save_state(self.state_dir, job.task_name, state)
            _write_summary(self.run_dir, self.tasks, self.states, self.args)

    def _done_locked(self) -> bool:
        target = int(self.args.episodes_per_task)
        for task in self.tasks:
            state = self.states[task.task_name]
            if int(state.get("collected", 0)) >= target:
                continue
            if state.get("blocked"):
                continue
            return False
        return not self.in_flight


class _Progress:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        tasks: list[TaskSpec],
        states: dict[str, dict[str, Any]],
        run_dir: Path,
    ):
        self.args = args
        self.tasks = tasks
        self.states = states
        self.run_dir = run_dir
        self.started = time.time()
        self.live = None
        self.events: deque[dict[str, Any]] = deque(maxlen=max(1, int(args.progress_event_limit)))

    def start(self) -> None:
        try:
            from rich.live import Live

            self.live = Live(self._render(), refresh_per_second=2, transient=False)
            self.live.start()
        except Exception:
            self.live = None
            print(f"RoboTwin expert collection: {self._collected()}/{self._target()}")

    def advance(self, response: dict[str, Any]) -> None:
        self.events.appendleft(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "task": response.get("task_name"),
                "episode": response.get("episode_index"),
                "seed": response.get("seed"),
                "ok": bool(response.get("ok")),
                "status": response.get("status"),
                "segments": response.get("segment_count"),
                "duration_s": response.get("duration_s"),
            }
        )
        if self.live is not None:
            self.live.update(self._render())
        else:
            print(
                f"collected={self._collected()}/{self._target()} task={response.get('task_name')} "
                f"ep={response.get('episode_index')} seed={response.get('seed')} "
                f"ok={response.get('ok')} status={response.get('status')}"
            )

    async def report_loop(self) -> None:
        while True:
            await asyncio.sleep(max(1.0, float(self.args.status_interval)))
            if self.live is not None:
                self.live.update(self._render())

    def stop(self) -> None:
        if self.live is not None:
            self.live.update(self._render())
            self.live.stop()

    def _collected(self) -> int:
        return sum(
            min(int(state.get("collected", 0)), int(self.args.episodes_per_task))
            for state in self.states.values()
        )

    def _target(self) -> int:
        return len(self.tasks) * int(self.args.episodes_per_task)

    def _render(self) -> Any:
        from rich import box
        from rich.console import Group
        from rich.panel import Panel
        from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn
        from rich.table import Table

        completed = self._collected()
        target = self._target()
        elapsed = max(1e-6, time.time() - self.started)
        rate = completed / elapsed
        eta = (target - completed) / rate if rate > 0 and completed < target else 0.0
        attempts = sum(int(state.get("attempts", 0)) for state in self.states.values())
        active = sum(1 for state in self.states.values() if state.get("active_episode") is not None)
        blocked = sum(1 for state in self.states.values() if state.get("blocked"))

        progress = Progress(
            TextColumn("[bold cyan]Overall"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            expand=True,
        )
        progress.add_task("overall", total=target, completed=completed)

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
            f"[bold]Split[/bold] {self.args.split}",
            f"[bold]Target[/bold] {self.args.episodes_per_task}/task",
            f"[bold]Attempts[/bold] {attempts}",
            f"[bold]Seed[/bold] {self.args.start_seed}+",
        )
        overview.add_row(
            f"[bold]Elapsed[/bold] {self._format_duration(elapsed)}",
            f"[bold]ETA[/bold] {self._format_duration(eta)}",
            f"[bold]Episodes/h[/bold] {rate * 3600:.1f}",
            f"[bold]SaveFreq[/bold] {self.args.save_freq}",
        )

        task_table = Table(title="Interleaved task collection", box=box.SIMPLE_HEAVY, expand=True, show_lines=False)
        task_table.add_column("Task", style="cyan", no_wrap=True)
        task_table.add_column("Episodes", justify="right")
        task_table.add_column("Next seed", justify="right")
        task_table.add_column("Attempts", justify="right")
        task_table.add_column("Status", no_wrap=True)
        task_table.add_column("Last", overflow="fold")
        rows = sorted(self.tasks, key=lambda task: self._task_sort_key(task.task_name))
        for task in rows[: max(1, int(self.args.progress_task_limit))]:
            state = self.states[task.task_name]
            status, style = self._task_status(state)
            task_table.add_row(
                task.task_name,
                f"{int(state.get('collected', 0))}/{int(self.args.episodes_per_task)}",
                str(state.get("next_seed", self.args.start_seed)),
                str(state.get("attempts", 0)),
                f"[{style}]{status}[/{style}]",
                str(state.get("last_status") or "-"),
            )
        if len(rows) > int(self.args.progress_task_limit):
            task_table.add_row(f"... {len(rows) - int(self.args.progress_task_limit)} more", "", "", "", "", "")

        events_table = Table(title="Recent collections", box=box.SIMPLE, expand=True)
        events_table.add_column("Time", no_wrap=True)
        events_table.add_column("Task", style="cyan", overflow="fold")
        events_table.add_column("Ep", justify="right")
        events_table.add_column("Seed", justify="right")
        events_table.add_column("Result", no_wrap=True)
        events_table.add_column("Segments", justify="right")
        events_table.add_column("Sec", justify="right")
        events_table.add_column("Status", overflow="fold")
        for event in list(self.events):
            result_style = "green" if event.get("ok") else "yellow"
            duration = event.get("duration_s")
            events_table.add_row(
                str(event.get("time") or ""),
                str(event.get("task") or ""),
                str(event.get("episode") if event.get("episode") is not None else ""),
                str(event.get("seed") or ""),
                f"[{result_style}]{'ok' if event.get('ok') else 'skip'}[/{result_style}]",
                str(event.get("segments") if event.get("segments") is not None else "-"),
                f"{float(duration):.1f}" if isinstance(duration, (int, float)) else "-",
                str(event.get("status") or ""),
            )

        return Group(
            Panel(Group(overview, progress), title="RoboTwin Expert Subtask Collection", border_style="cyan"),
            task_table,
            events_table,
        )

    def _task_sort_key(self, task_name: str) -> tuple[int, int, str]:
        state = self.states[task_name]
        target = int(self.args.episodes_per_task)
        collected = int(state.get("collected", 0))
        if state.get("blocked"):
            bucket = 3
        elif state.get("active_episode") is not None:
            bucket = 0
        elif collected < target:
            bucket = 1
        else:
            bucket = 2
        return (bucket, -collected, task_name)

    def _task_status(self, state: dict[str, Any]) -> tuple[str, str]:
        if state.get("blocked"):
            return "blocked", "red"
        if state.get("active_episode") is not None:
            active_since = state.get("active_since")
            active_s = time.time() - float(active_since) if isinstance(active_since, (int, float)) else 0.0
            return f"ep{state.get('active_episode')} seed{state.get('active_seed')} ({active_s:.0f}s)", "blue"
        if int(state.get("collected", 0)) >= int(self.args.episodes_per_task):
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


def _base_task_args(args: argparse.Namespace, repo_root: Path, task_name: str, task_dir: Path) -> dict[str, Any]:
    task_args = _task_args(repo_root, task_name, args.task_config, args.camera_profile)
    task_args["save_path"] = str(task_dir)
    task_args["task_name"] = task_name
    task_args["task_config"] = args.task_config
    task_args["language_num"] = int(args.language_num)
    task_args["eval_video_log"] = False
    task_args["eval_mode"] = False
    return task_args


def _load_move_comments(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    comments: dict[int, str] = {}
    for index, line in enumerate(lines):
        if "self.move(" not in line:
            continue
        comment_lines: list[str] = []
        cursor = index - 1
        while cursor >= 0:
            text = lines[cursor].strip()
            if not text:
                cursor -= 1
                continue
            if not text.startswith("#"):
                break
            comment_lines.append(text.lstrip("#").strip())
            cursor -= 1
        comment_lines.reverse()
        if comment_lines:
            comments[index + 1] = " ".join(comment_lines)
        else:
            trailing = line.split("#", 1)[1].strip() if "#" in line else ""
            if trailing:
                comments[index + 1] = trailing
    return comments


def _source_call_snippet(path: Path, line_no: int, max_lines: int = 32) -> str | None:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = max(0, int(line_no) - 1)
    collected: list[str] = []
    balance = 0
    seen_call = False
    for line in lines[start : start + max_lines]:
        stripped = line.rstrip()
        collected.append(stripped.strip())
        if "self.move(" in stripped:
            seen_call = True
        if seen_call:
            balance += stripped.count("(") - stripped.count(")")
            if balance <= 0:
                break
    return "\n".join(collected).strip() or None


def _source_context_snippet(path: Path, line_no: int, before_lines: int = 14) -> str | None:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = max(0, int(line_no) - before_lines - 1)
    call = _source_call_snippet(path, line_no) or ""
    prefix = [line.strip() for line in lines[start : max(0, int(line_no) - 1)]]
    context = "\n".join([*prefix, call]).strip()
    return context or None


def _comment_to_instruction(comment: str | None) -> str | None:
    if not comment:
        return None
    text = " ".join(str(comment).split())
    text = re.sub(r"\bNote:\s*.*$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\bYou must\s+.*$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s+to simulate (?:a |an )?(?:touch/)?click action\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+to simulate (?:a |an )?click\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\((?:no need|optional|based on).*?\)", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^(note:|todo:)\s*", "", text, flags=re.IGNORECASE).strip()
    if not text:
        return None
    return text[:1].lower() + text[1:]


def _summarize_move_args(actions_by_arm1: Any, actions_by_arm2: Any = None) -> dict[str, Any]:
    return {
        "left": _summarize_arm_actions(_select_arm_actions(actions_by_arm1, actions_by_arm2, "left")),
        "right": _summarize_arm_actions(_select_arm_actions(actions_by_arm1, actions_by_arm2, "right")),
        "raw_arms": [_arm_name(actions_by_arm1), _arm_name(actions_by_arm2)],
    }


def _select_arm_actions(actions_by_arm1: Any, actions_by_arm2: Any, arm: str) -> list[Any]:
    for item in (actions_by_arm1, actions_by_arm2):
        if not isinstance(item, tuple) or len(item) < 2:
            continue
        if str(item[0]) == arm:
            return list(item[1] or [])
    return []


def _summarize_arm_actions(actions: list[Any]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for action in actions:
        if action is None:
            continue
        item: dict[str, Any] = {
            "action": str(getattr(action, "action", type(action).__name__)),
            "arm_tag": str(getattr(action, "arm_tag", "")) or None,
        }
        if hasattr(action, "target_gripper_pos"):
            try:
                item["target_gripper_pos"] = float(getattr(action, "target_gripper_pos"))
            except Exception:
                item["target_gripper_pos"] = str(getattr(action, "target_gripper_pos"))
        if hasattr(action, "target_pose"):
            pose = getattr(action, "target_pose")
            item["has_target_pose"] = pose is not None
        args = getattr(action, "args", None)
        if isinstance(args, dict) and args:
            item["args_keys"] = sorted(str(key) for key in args.keys())
        summary.append(item)
    return summary


def _arm_name(value: Any) -> str | None:
    if isinstance(value, tuple) and value:
        return str(value[0])
    return None


def _write_segment_metadata(output_dir: Path, job: CollectionJob, metadata: dict[str, Any]) -> Path:
    task_dir = output_dir / "segments" / job.task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / f"episode{job.episode_index}.json"
    _write_json(path, metadata)
    _append_jsonl(output_dir / "segments.jsonl", _compact_episode_metadata(metadata))
    return path


def _compact_episode_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_name": metadata.get("task_name"),
        "episode_index": metadata.get("episode_index"),
        "seed": metadata.get("seed"),
        "instruction": metadata.get("instruction"),
        "hdf5_path": metadata.get("hdf5_path"),
        "segment_count": metadata.get("segment_count"),
        "frame_count": metadata.get("frame_count"),
        "annotation_counts": metadata.get("annotation_counts"),
    }


def _attach_hdf5_metadata(hdf5_path: Path, metadata: dict[str, Any]) -> None:
    import h5py

    with h5py.File(hdf5_path, "a") as handle:
        meta_group = handle.require_group("clawvla_metadata")
        for name, payload in {
            "episode_metadata_json": _compact_episode_metadata(metadata),
            "segments_json": metadata.get("segments", []),
        }.items():
            if name in meta_group:
                del meta_group[name]
            meta_group.create_dataset(name, data=json.dumps(payload, ensure_ascii=True))


def _merge_episode_cache(task_env: Any, task_dir: Path, episode_index: int, *, save_video: bool) -> None:
    cache_path = Path(task_env.folder_path["cache"])
    hdf5_path = task_dir / "data" / f"episode{episode_index}.hdf5"
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    pkl_files = _numeric_pkl_files(cache_path)
    merge_mode = str(getattr(task_env, "_clawvla_merge_hdf5_mode", "memory"))
    rgb_input_order = str(getattr(task_env, "_clawvla_rgb_input_order", "rgb"))
    if merge_mode == "stream":
        _stream_pkl_files_to_hdf5(pkl_files, hdf5_path, rgb_input_order=rgb_input_order)
    else:
        import h5py

        from envs.utils.pkl2hdf5 import (
            append_data_to_structure,
            create_hdf5_from_dict,
            load_pkl_file,
            parse_dict_structure,
        )

        data = parse_dict_structure(load_pkl_file(str(pkl_files[0])))
        for pkl_file in pkl_files:
            append_data_to_structure(data, load_pkl_file(str(pkl_file)))
        _prepare_rgb_lists_for_cv2(data, rgb_input_order=rgb_input_order)
        with h5py.File(hdf5_path, "w") as handle:
            create_hdf5_from_dict(handle, data)

    if save_video:
        _write_episode_video(
            pkl_files,
            task_dir / "video" / f"episode{episode_index}.mp4",
            rgb_input_order=rgb_input_order,
        )


def _numeric_pkl_files(cache_path: Path) -> list[Path]:
    files: list[tuple[int, Path]] = []
    for path in cache_path.iterdir():
        if path.suffix == ".pkl" and path.stem.isdigit():
            files.append((int(path.stem), path))
    if not files:
        raise FileNotFoundError(f"No valid .pkl files found in {cache_path}")
    files.sort(key=lambda item: item[0])
    for expected, (index, _) in enumerate(files):
        if index != expected:
            raise ValueError(f"Missing file {expected}.pkl in {cache_path}")
    return [path for _, path in files]


def _remove_episode_cache(task_dir: Path, episode_index: int, *, purpose: str) -> dict[str, Any]:
    return _remove_cache_path(task_dir / ".cache" / f"episode{episode_index}", purpose=purpose)


def _episode_cache_task_dir(args: argparse.Namespace, job: CollectionJob, task_dir: Path) -> Path:
    root = str(getattr(args, "episode_cache_root", "") or "").strip()
    if not root:
        return task_dir
    return Path(root) / Path(str(args.output_dir)).name / f"pid{os.getpid()}" / job.task_name


def _remove_empty_scratch_dirs(task_dir: Path, scratch_root: Path | None) -> None:
    if scratch_root is None:
        return
    try:
        root = scratch_root.resolve()
        current = task_dir.resolve()
    except OSError:
        return
    for path in [current, *current.parents]:
        if path == root:
            break
        try:
            path.rmdir()
        except OSError:
            break


def _remove_cache_path(cache_path: Path, *, purpose: str) -> dict[str, Any]:
    if not cache_path.exists():
        return {"status": "missing", "path": str(cache_path), "purpose": purpose}
    last_error: str | None = None
    for attempt in range(1, 5):
        try:
            shutil.rmtree(cache_path)
            return {
                "status": "removed",
                "path": str(cache_path),
                "purpose": purpose,
                "attempts": attempt,
            }
        except FileNotFoundError:
            return {"status": "missing", "path": str(cache_path), "purpose": purpose}
        except OSError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2.0, 0.25 * attempt))
    remaining: int | None = None
    try:
        if cache_path.exists():
            remaining = sum(1 for _ in cache_path.rglob("*"))
    except OSError:
        remaining = None
    print(
        f"Warning: failed to remove {purpose} cache {cache_path}: {last_error}; remaining={remaining}",
        file=sys.stderr,
    )
    return {
        "status": "failed",
        "path": str(cache_path),
        "purpose": purpose,
        "error": last_error,
        "remaining": remaining,
    }


def _stream_pkl_files_to_hdf5(
    pkl_files: list[Path],
    hdf5_path: Path,
    *,
    rgb_input_order: str,
) -> None:
    import cv2
    import h5py
    import numpy as np

    from envs.utils.pkl2hdf5 import load_pkl_file

    leaf_specs: dict[tuple[str, ...], tuple[tuple[int, ...], Any]] = {}
    rgb_max_len: dict[tuple[str, ...], int] = {}
    for pkl_file in pkl_files:
        payload = load_pkl_file(str(pkl_file))
        for path, value in _iter_leaf_items(payload):
            if _is_rgb_leaf(path):
                encoder_input = _image_for_cv2(value, rgb_input_order=rgb_input_order)
                ok, encoded = cv2.imencode(".jpg", encoder_input)
                if not ok:
                    raise ValueError(f"failed to JPEG-encode {'/'.join(path)} from {pkl_file}")
                rgb_max_len[path] = max(rgb_max_len.get(path, 0), len(encoded.tobytes()))
                continue
            array = np.asarray(value)
            spec = (tuple(array.shape), array.dtype)
            if path in leaf_specs and leaf_specs[path] != spec:
                raise ValueError(
                    f"inconsistent shape/dtype for {'/'.join(path)}: {leaf_specs[path]} vs {spec} in {pkl_file}"
                )
            leaf_specs[path] = spec

    frame_count = len(pkl_files)
    with h5py.File(hdf5_path, "w") as handle:
        datasets: dict[tuple[str, ...], Any] = {}
        for path, max_len in rgb_max_len.items():
            datasets[path] = _create_dataset_for_path(handle, path, shape=(frame_count,), dtype=f"S{max_len}")
        for path, (shape, dtype) in leaf_specs.items():
            datasets[path] = _create_dataset_for_path(handle, path, shape=(frame_count, *shape), dtype=dtype)

        for frame_index, pkl_file in enumerate(pkl_files):
            payload = load_pkl_file(str(pkl_file))
            seen: set[tuple[str, ...]] = set()
            for path, value in _iter_leaf_items(payload):
                dataset = datasets[path]
                seen.add(path)
                if _is_rgb_leaf(path):
                    encoder_input = _image_for_cv2(value, rgb_input_order=rgb_input_order)
                    ok, encoded = cv2.imencode(".jpg", encoder_input)
                    if not ok:
                        raise ValueError(f"failed to JPEG-encode {'/'.join(path)} from {pkl_file}")
                    dataset[frame_index] = encoded.tobytes()
                else:
                    dataset[frame_index] = np.asarray(value, dtype=dataset.dtype)
            missing = set(datasets) - seen
            if missing:
                names = ", ".join("/".join(path) for path in sorted(missing)[:5])
                raise ValueError(f"missing fields in {pkl_file}: {names}")


def _image_for_cv2(value: Any, *, rgb_input_order: str) -> Any:
    import cv2
    import numpy as np

    image = np.asarray(value)
    if rgb_input_order == "bgr":
        return np.ascontiguousarray(image)
    if rgb_input_order != "rgb":
        raise ValueError(f"unsupported RGB input order: {rgb_input_order}")
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"expected an RGB/RGBA image, got shape={image.shape}")
    conversion = cv2.COLOR_RGB2BGR if image.shape[-1] == 3 else cv2.COLOR_RGBA2BGRA
    return cv2.cvtColor(image, conversion)


def _prepare_rgb_lists_for_cv2(
    value: Any,
    *,
    rgb_input_order: str,
    path: tuple[str, ...] = (),
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _prepare_rgb_lists_for_cv2(
                child,
                rgb_input_order=rgb_input_order,
                path=(*path, str(key)),
            )
        return
    if isinstance(value, list) and _is_rgb_leaf(path):
        for index, image in enumerate(value):
            value[index] = _image_for_cv2(image, rgb_input_order=rgb_input_order)


def _write_episode_video(
    pkl_files: list[Path],
    video_path: Path,
    *,
    rgb_input_order: str,
) -> None:
    import numpy as np

    from envs.utils.images_to_video import images_to_video
    from envs.utils.pkl2hdf5 import load_pkl_file

    frames = np.stack(
        [load_pkl_file(str(path))["observation"]["head_camera"]["rgb"] for path in pkl_files]
    )
    images_to_video(frames, out_path=str(video_path), is_rgb=rgb_input_order == "rgb")


def _iter_leaf_items(value: Any, prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        items: list[tuple[tuple[str, ...], Any]] = []
        for key, child in value.items():
            items.extend(_iter_leaf_items(child, (*prefix, str(key))))
        return items
    return [(prefix, value)]


def _is_rgb_leaf(path: tuple[str, ...]) -> bool:
    return bool(path) and path[-1] == "rgb"


def _create_dataset_for_path(handle: Any, path: tuple[str, ...], *, shape: tuple[int, ...], dtype: Any) -> Any:
    group = handle
    for name in path[:-1]:
        group = group.require_group(name)
    return group.create_dataset(path[-1], shape=shape, dtype=dtype)


def _hdf5_frame_count(hdf5_path: Path) -> int:
    try:
        import h5py

        with h5py.File(hdf5_path, "r") as handle:
            qpos = handle.get("joint_action/vector")
            if qpos is not None:
                return int(qpos.shape[0])
            images = handle.get("observation/head_camera/rgb")
            if images is not None:
                return int(images.shape[0])
    except Exception:
        return 0
    return 0


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
        tasks = [task for task in tasks if task.task_name in wanted]
        missing = sorted(wanted - {task.task_name for task in tasks})
        if missing:
            raise ValueError(f"unknown RoboTwin task_name(s): {', '.join(missing)}")
    if args.task_limit is not None:
        tasks = tasks[: max(0, int(args.task_limit))]
    if not tasks:
        raise ValueError("task filtering produced an empty task list")
    return tasks


def _load_states(state_dir: Path, tasks: list[TaskSpec], args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for task in tasks:
        path = state_dir / f"{task.task_name}.json"
        if args.resume and path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            state.setdefault("task_name", task.task_name)
            state.setdefault("next_seed", int(args.start_seed))
            state.setdefault("collected", len(state.get("episodes", [])))
            state.setdefault("attempts", 0)
            state.pop("active_episode", None)
            state.pop("active_seed", None)
            state.pop("active_since", None)
        else:
            state = {
                "task_name": task.task_name,
                "task_config": args.task_config,
                "split": args.split,
                "target": int(args.episodes_per_task),
                "start_seed": int(args.start_seed),
                "next_seed": int(args.start_seed),
                "collected": 0,
                "attempts": 0,
                "failed_jobs": 0,
                "episodes": [],
            }
        states[task.task_name] = state
    return states


def _reconcile_states_from_episode_log(
    run_dir: Path,
    tasks: list[TaskSpec],
    states: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[str]:
    log_path = run_dir / "episodes.jsonl"
    if not args.resume or not log_path.exists():
        return []
    task_names = {task.task_name for task in tasks}
    logged: dict[str, dict[int, dict[str, Any]]] = {task_name: {} for task_name in task_names}
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not (item.get("ok") and item.get("status") == "collected"):
                continue
            task_name = str(item.get("task_name") or "")
            if task_name not in logged:
                continue
            try:
                episode_index = int(item["episode_index"])
            except (KeyError, TypeError, ValueError):
                continue
            if episode_index >= int(args.episodes_per_task):
                continue
            logged[task_name][episode_index] = item

    repaired: list[str] = []
    for task_name, by_episode in logged.items():
        if not by_episode:
            continue
        state = states[task_name]
        existing: dict[int, dict[str, Any]] = {}
        for episode in state.get("episodes", []):
            try:
                existing[int(episode["episode_index"])] = episode
            except (KeyError, TypeError, ValueError):
                continue
        changed = False
        for episode_index, item in sorted(by_episode.items()):
            if episode_index in existing:
                continue
            existing[episode_index] = {
                "episode_index": episode_index,
                "seed": item.get("seed"),
                "hdf5_path": item.get("hdf5_path"),
                "segment_path": item.get("segment_path"),
                "segment_count": item.get("segment_count"),
                "frame_count": item.get("frame_count"),
            }
            changed = True
        new_episodes = [existing[index] for index in sorted(existing)]
        new_collected = min(len(new_episodes), int(args.episodes_per_task))
        if int(state.get("collected", 0)) != new_collected:
            changed = True
        if not changed:
            continue

        state["episodes"] = new_episodes
        state["collected"] = new_collected
        latest_index = max(existing)
        latest_episode = existing[latest_index]
        latest_log = by_episode.get(latest_index, {})
        current_next_seed = int(state.get("next_seed", int(args.start_seed)))
        logged_next_seed = latest_log.get("next_seed")
        if logged_next_seed is not None:
            try:
                state["next_seed"] = max(current_next_seed, int(logged_next_seed))
            except (TypeError, ValueError):
                state["next_seed"] = current_next_seed
        state["last_status"] = "collected"
        state["last_episode_index"] = latest_index
        state["last_seed"] = latest_episode.get("seed")
        state["last_hdf5_path"] = latest_episode.get("hdf5_path")
        state["last_segment_path"] = latest_episode.get("segment_path")
        repaired.append(task_name)
    return repaired


def _save_state(state_dir: Path, task_name: str, state: dict[str, Any]) -> None:
    _write_json(state_dir / f"{task_name}.json", state)


def _write_summary(
    run_dir: Path,
    tasks: list[TaskSpec],
    states: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    per_task: dict[str, Any] = {}
    total_collected = 0
    total_attempts = 0
    blocked: list[str] = []
    incomplete: list[str] = []
    for task in tasks:
        state = states[task.task_name]
        collected = min(int(state.get("collected", 0)), int(args.episodes_per_task))
        attempts = int(state.get("attempts", 0))
        total_collected += collected
        total_attempts += attempts
        if collected < int(args.episodes_per_task):
            incomplete.append(task.task_name)
        if state.get("blocked"):
            blocked.append(task.task_name)
        per_task[task.task_name] = {
            "collected": collected,
            "target": int(args.episodes_per_task),
            "attempts": attempts,
            "next_seed": state.get("next_seed"),
            "blocked": bool(state.get("blocked")),
            "last_status": state.get("last_status"),
            "last_seed": state.get("last_seed"),
            "last_hdf5_path": state.get("last_hdf5_path"),
            "last_segment_path": state.get("last_segment_path"),
            "yield": collected / attempts if attempts else None,
        }
    _write_json(
        run_dir / "summary.json",
        {
            "run_id": run_dir.name,
            "split": args.split,
            "task_config": args.task_config,
            "start_seed": int(args.start_seed),
            "episodes_per_task": int(args.episodes_per_task),
            "task_count": len(tasks),
            "total_target": len(tasks) * int(args.episodes_per_task),
            "total_collected": total_collected,
            "total_attempts": total_attempts,
            "complete": not incomplete,
            "incomplete_tasks": incomplete,
            "blocked_tasks": blocked,
            "per_task": per_task,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def _write_task_manifests(run_dir: Path, tasks: list[TaskSpec], states: dict[str, dict[str, Any]]) -> None:
    manifest = []
    for task in tasks:
        state = states[task.task_name]
        for item in state.get("episodes", []):
            manifest.append({"task_name": task.task_name, **item})
    _write_json(run_dir / "manifest.json", {"episodes": manifest, "episode_count": len(manifest)})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _run_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    run_id = args.run_id or (
        f"robotwin_expert_subtasks_{args.split}_{args.task_config}_"
        f"{args.episodes_per_task}x{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    return Path(args.output_root) / run_id


def _resolve_start_seed(args: argparse.Namespace) -> int:
    if args.start_seed is not None:
        return int(args.start_seed)
    if args.split == "train":
        return 200000
    if args.split == "val":
        return 300000
    return 400000


def _parse_seed_ranges(values: list[str] | None) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for value in values or []:
        text = str(value).strip()
        if not text:
            continue
        if ":" in text:
            left, right = text.split(":", 1)
            start, end = int(left), int(right)
        else:
            start = end = int(text)
        if end < start:
            start, end = end, start
        ranges.append((start, end))
    return ranges


def _assert_non_eval_seed_plan(start_seed: int, ranges: list[tuple[int, int]]) -> None:
    for left, right in ranges:
        if left <= start_seed <= right:
            raise ValueError(
                f"start seed {start_seed} is inside forbidden seed range {left}:{right}; "
                "use --start-seed outside official eval seeds"
            )


def _skip_forbidden(seed: int, ranges: list[tuple[int, int]]) -> int:
    changed = True
    while changed:
        changed = False
        for left, right in ranges:
            if left <= seed <= right:
                seed = right + 1
                changed = True
    return seed


def _make_lanes(args: argparse.Namespace) -> list[Lane]:
    gpus = _split_csv(args.gpus)
    workers = max(1, int(args.workers))
    return [Lane(index=index, gpu=gpus[index % len(gpus)] if gpus else None) for index in range(workers)]


def _robotwin_env(args: argparse.Namespace, gpu: str | None) -> dict[str, str]:
    env = dict(os.environ)
    python_paths = [
        *(str(Path(path).expanduser().resolve()) for path in args.pythonpath_prefix if str(path).strip()),
        str(PROJECT_ROOT / "src"),
        *_split_path_list(env.get("PYTHONPATH")),
    ]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(python_paths))
    library_paths = _robotwin_library_paths(args)
    if library_paths:
        existing = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = ":".join([*library_paths, existing] if existing else library_paths)
    env["__EGL_VENDOR_LIBRARY_DIRS"] = "/usr/share/glvnd/egl_vendor.d"
    env["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _validate_collection_inputs(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).expanduser().resolve()
    tasks_config = Path(args.tasks_config).expanduser().resolve()
    task_config = repo_root / "task_config" / f"{args.task_config}.yml"
    required = {
        "RoboTwin repository": repo_root,
        "RoboTwin env directory": repo_root / "envs",
        "RoboTwin task config": task_config,
        "task list": tasks_config,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("collection inputs are missing:\n  " + "\n  ".join(missing))

    executable = str(args.robotwin_python)
    if os.path.sep in executable:
        python_path = Path(executable).expanduser()
        if not python_path.is_file():
            raise FileNotFoundError(f"RoboTwin Python executable not found: {python_path}")
    elif shutil.which(executable) is None:
        raise FileNotFoundError(f"RoboTwin Python executable is not on PATH: {executable}")

    if int(args.episodes_per_task) <= 0:
        raise ValueError("--episodes-per-task must be positive")
    if int(args.workers) <= 0:
        raise ValueError("--workers must be positive")
    if args.polish_subgoals and not args.dry_run:
        polish_config = Path(args.subgoal_polish_config).expanduser()
        if not polish_config.is_absolute():
            polish_config = PROJECT_ROOT / polish_config
        if not polish_config.is_file():
            raise FileNotFoundError(
                f"subgoal polish config not found: {polish_config}; "
                "use --no-polish-subgoals to collect raw expert segments without an API"
            )


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


def _robotwin_library_paths(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []
    python_path = Path(str(args.robotwin_python))
    if python_path.name == "python":
        candidate = python_path.parent.parent / "lib"
        if candidate.exists():
            paths.append(str(candidate))
    return paths


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


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _split_path_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in str(value).split(os.pathsep) if item]


def _last_json_line(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        return payload if isinstance(payload, dict) else None
    return None


def _compact_failure(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    return {
        "status": payload.get("status"),
        "error": payload.get("error"),
        "traceback": str(payload.get("traceback") or "")[-2000:] or None,
        "segment_count": payload.get("segment_count"),
        "plan_success": payload.get("plan_success"),
        "task_success": payload.get("task_success"),
    }


def _is_infra_error(payload: dict[str, Any]) -> bool:
    text = f"{payload.get('status', '')}\n{payload.get('error', '')}\n{payload.get('traceback', '')}".lower()
    markers = (
        "failed to find a rendering device",
        "vulkan",
        "vk_icd",
        "egl",
        "libcuda",
        "cuda driver",
        "segmentation fault",
        "worker_return_code",
        "subgoal_polish_failed",
        "subgoal_polish",
    )
    return any(marker in text for marker in markers)


def _relpath(path: str, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root))
    except Exception:
        return str(path)


def _parent_death_preexec() -> None:
    import ctypes

    os.setsid()
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)


def _terminate_process_group(process: asyncio.subprocess.Process, sig: signal.Signals = signal.SIGTERM) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return


if __name__ == "__main__":
    main()
