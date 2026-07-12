from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import Any, Iterable, Sequence

from clawvla.scripts.robotwin_collect_expert_subtasks import (
    _build_segment_visual_inputs,
    _load_polish_config,
    _request_segment_polish,
    _segment_polish_input,
)


DEFAULT_CONFIG = "configs/krill_gpt55.local.json"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INVALID_LABELS = frozenset(
    {
        "arm_tag",
        "arm tag",
        "none",
        "null",
        "unknown",
    }
)
SCOPES = ("broken", "missing", "invalid", "raw-only", "all-unpolished")
SKIPPABLE_LEDGER_STATUSES = frozenset({"accepted", "rejected"})
CONFIRMATION_ONLY_VERBS = frozenset({"check", "confirm", "ensure", "inspect", "observe", "verify"})
CODE_IDENTIFIER_RE = re.compile(
    r"(?:arm[ _-]?tag|target_pose|self\.|grasp_actor|move_by_displacement)",
    flags=re.IGNORECASE,
)
QUALIFIED_ARM_RE = re.compile(
    r"\b(left|right)[ -]+(?:arm|gripper|hand|wrist|end[- ]?effector)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class EpisodeJob:
    path: Path
    payload: dict[str, Any]
    targets: tuple[dict[str, Any], ...]

    @property
    def task_name(self) -> str:
        return str(self.payload["task_name"])

    @property
    def episode_index(self) -> int:
        return int(self.payload["episode_index"])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair or polish RoboTwin subtask text with the existing ClawVLA "
            "OpenAI-compatible polishing prompt. The script writes a sidecar JSONL ledger and "
            "never mutates source segment JSON/HDF5 files."
        )
    )
    parser.add_argument("--dataset-root", required=True, help="Merged dataset root containing segments/<task>/*.json.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Local model/base_url/api_key config JSON.")
    parser.add_argument("--model", default=None, help="Optional model override; otherwise use config.model.")
    parser.add_argument("--output", default=None, help="Output repair ledger JSONL. A model/scope-specific path is used by default.")
    parser.add_argument(
        "--scope",
        choices=SCOPES,
        default="broken",
        help="Use all-unpolished for all 9,446 eligible non-polished segments; broken selects only hard defects.",
    )
    parser.add_argument(
        "--invalid-label",
        action="append",
        default=[],
        help="Additional exact invalid instruction label; may be repeated.",
    )
    parser.add_argument("--task-name", action="append", default=[], help="Restrict to one or more task names.")
    parser.add_argument("--episode-index", action="append", type=int, default=[], help="Restrict to episode indices.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit affected episodes after filtering.")
    parser.add_argument("--max-targets", type=int, default=None, help="Limit selected target segments after filtering.")
    parser.add_argument("--batch-size", type=int, default=8, help="Target segments per API request; requests never cross episodes.")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of episodes processed concurrently; chunks within one episode remain sequential.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--retry-rejected",
        action="store_true",
        help="When resuming, retry records previously written with status=rejected.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Scan and print a plan without sending API requests or writing files.")
    parser.add_argument("--show-jobs", type=int, default=12, help="Number of planned episode jobs included in dry-run output.")
    parser.add_argument("--images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Allow an image-enabled request to fall back to text if visual extraction fails.",
    )
    parser.add_argument("--image-camera", default="head_camera")
    parser.add_argument("--image-samples", default="mid", help="Comma-separated subset of start,mid,end.")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--image-quality", type=int, default=70)
    parser.add_argument("--max-images", type=int, default=24)
    parser.add_argument("--paraphrases", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--api-retries", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay after each API request.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first API or parsing error.")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")
    if args.max_targets is not None and args.max_targets <= 0:
        parser.error("--max-targets must be positive")
    if args.max_images < 0:
        parser.error("--max-images must be non-negative")
    if args.show_jobs < 0:
        parser.error("--show-jobs must be non-negative")
    if args.image_size <= 0:
        parser.error("--image-size must be positive")
    if not 1 <= args.image_quality <= 100:
        parser.error("--image-quality must be between 1 and 100")
    if args.paraphrases < 0:
        parser.error("--paraphrases must be non-negative")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.api_retries < 0:
        parser.error("--api-retries must be non-negative")
    if args.sleep_seconds < 0:
        parser.error("--sleep-seconds must be non-negative")
    if not math.isfinite(args.temperature):
        parser.error("--temperature must be finite")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    invalid_labels = DEFAULT_INVALID_LABELS | {str(item).strip().lower() for item in args.invalid_label if str(item).strip()}
    scan = scan_dataset(dataset_root, invalid_labels=invalid_labels)

    config_path = _resolve_config_path(Path(args.config))
    config_preview = _read_config_preview(config_path)
    model = str(args.model or config_preview.get("model") or "unknown-model")
    output_path = _resolve_output_path(dataset_root, args.output, model=model, scope=args.scope)
    existing_records = _load_existing_records(output_path) if args.resume and output_path.exists() else {}
    locally_revalidated = 0
    if existing_records and not args.dry_run:
        promotions = _revalidate_existing_rejected(
            existing_records,
            scan["episodes"],
            expected_model=model,
            invalid_labels=invalid_labels,
            expected_paraphrases=max(0, int(args.paraphrases)),
        )
        if promotions:
            _append_jsonl(output_path, promotions)
            existing_records.update({str(record["record_id"]): record for record in promotions})
            locally_revalidated = len(promotions)
            print(f"locally revalidated rejected records as accepted: {locally_revalidated}")
    if output_path.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(f"Output already exists; use --resume or choose another --output: {output_path}")

    jobs, planning = plan_jobs(
        scan["episodes"],
        scope=args.scope,
        invalid_labels=invalid_labels,
        task_names=set(args.task_name),
        episode_indices=set(args.episode_index),
        existing_records=existing_records,
        expected_model=model,
        retry_rejected=bool(args.retry_rejected),
        max_episodes=args.max_episodes,
        max_targets=args.max_targets,
        batch_size=args.batch_size,
    )
    planning["locally_revalidated_records"] = locally_revalidated
    planning["workers"] = min(int(args.workers), len(jobs)) if jobs else 0
    dry_run_report = {
        "dry_run": bool(args.dry_run),
        "would_write": False if args.dry_run else str(output_path),
        "dataset_root": str(dataset_root),
        "config_path": str(config_path),
        "model": model,
        "scope": args.scope,
        "selector": {
            "eligible": "frame_end_exclusive > frame_start and num_saved_frames > 0",
            "polished": "non-empty polished_instruction",
            "raw_only": "eligible and not polished and non-empty raw_canonical_instruction",
            "missing": "eligible and not polished and empty raw_canonical_instruction",
            "invalid": f"current instruction exactly matches one of {sorted(invalid_labels)}",
        },
        "scan": {key: value for key, value in scan.items() if key != "episodes"},
        "plan": planning,
        "sample_jobs": [
            {
                "task_name": job.task_name,
                "episode_index": job.episode_index,
                "target_segment_indices": [int(segment["segment_index"]) for segment in job.targets],
            }
            for job in jobs[: max(0, int(args.show_jobs))]
        ],
    }
    if args.dry_run:
        print(json.dumps(dry_run_report, ensure_ascii=False, indent=2))
        return

    config = _load_polish_config(config_path)
    if args.model:
        config = dict(config)
        config["model"] = args.model
    model = str(config["model"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = _now()
    run_counts = _run_episode_jobs(
        args=args,
        config=config,
        model=model,
        config_path=config_path,
        dataset_root=dataset_root,
        jobs=jobs,
        invalid_labels=invalid_labels,
        output_path=output_path,
    )

    latest_records = _load_existing_records(output_path) if output_path.exists() else {}
    relevant_records = [
        record
        for record in latest_records.values()
        if _text(record.get("model")) == model
        and Path(_text(record.get("source_dataset"))).resolve() == dataset_root
    ]
    ledger_status_counts = Counter(_text(record.get("status")) for record in relevant_records)
    unresolved_records = sum(ledger_status_counts[status] for status in ("api_error", "rejected"))
    summary = {
        **dry_run_report,
        "dry_run": False,
        "would_write": str(output_path),
        "started_at": started_at,
        "completed_at": _now(),
        "run_counts": dict(run_counts),
        "ledger_status_counts": dict(ledger_status_counts),
        "status": "incomplete" if unresolved_records else "complete",
        "unresolved_records": unresolved_records,
        "source_files_mutated": False,
        "ledger": str(output_path),
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    _write_json_atomic(summary_path, summary)
    print(f"repair ledger: {output_path}")
    print(f"summary: {summary_path}")
    if unresolved_records:
        raise SystemExit(
            f"Repair run is incomplete: {unresolved_records} rejected/API-error records remain. "
            "Inspect the ledger and use --retry-rejected when appropriate."
        )


def scan_dataset(dataset_root: Path, *, invalid_labels: set[str] | frozenset[str]) -> dict[str, Any]:
    segment_root = dataset_root / "segments"
    if not segment_root.is_dir():
        raise FileNotFoundError(f"Missing segments directory: {segment_root}")
    episode_files = sorted(segment_root.glob("*/*.json"), key=_episode_file_sort_key)
    episodes: list[tuple[Path, dict[str, Any]]] = []
    counts: Counter[str] = Counter()
    zero_frame_tasks: Counter[str] = Counter()
    affected_tasks: dict[str, set[str]] = {
        "raw_only": set(),
        "missing": set(),
        "invalid": set(),
        "broken": set(),
        "all_unpolished": set(),
    }
    affected_episodes: dict[str, set[tuple[str, int]]] = {key: set() for key in affected_tasks}

    for path in episode_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        segments = payload.get("segments")
        if not isinstance(segments, list):
            raise ValueError(f"{path}: segments must be a list")
        if int(payload.get("segment_count", len(segments))) != len(segments):
            raise ValueError(f"{path}: segment_count does not match segments length")
        task_name = str(payload.get("task_name") or path.parent.name)
        episode_index = int(payload.get("episode_index"))
        episodes.append((path, payload))
        counts["episode_files"] += 1
        for segment in segments:
            counts["segments"] += 1
            span = int(segment.get("frame_end_exclusive") or 0) - int(segment.get("frame_start") or 0)
            saved = int(segment.get("num_saved_frames") or 0)
            if (span > 0) != (saved > 0):
                counts["frame_span_saved_count_mismatches"] += 1
            if span <= 0 or saved <= 0:
                counts["skipped_zero_frame"] += 1
                zero_frame_tasks[task_name] += 1
                continue
            counts["eligible"] += 1
            issue = classify_segment(segment, invalid_labels=invalid_labels)
            counts[issue] += 1
            if issue == "raw_only":
                if _is_invalid_segment(segment, invalid_labels):
                    counts["invalid"] += 1
                    _mark_affected(affected_tasks, affected_episodes, "invalid", task_name, episode_index)
                    _mark_affected(affected_tasks, affected_episodes, "broken", task_name, episode_index)
                _mark_affected(affected_tasks, affected_episodes, "raw_only", task_name, episode_index)
                _mark_affected(affected_tasks, affected_episodes, "all_unpolished", task_name, episode_index)
            elif issue == "missing":
                _mark_affected(affected_tasks, affected_episodes, "missing", task_name, episode_index)
                _mark_affected(affected_tasks, affected_episodes, "broken", task_name, episode_index)
                _mark_affected(affected_tasks, affected_episodes, "all_unpolished", task_name, episode_index)

    if counts["frame_span_saved_count_mismatches"]:
        raise ValueError(
            "Found segments where frame span positivity disagrees with num_saved_frames positivity: "
            f"{counts['frame_span_saved_count_mismatches']}"
        )
    return {
        "episodes": episodes,
        "scanned_episode_files": counts["episode_files"],
        "scanned_segments": counts["segments"],
        "eligible_segments": counts["eligible"],
        "skipped_zero_frame": counts["skipped_zero_frame"],
        "already_polished_valid": counts["polished"],
        "raw_only_valid": counts["raw_only"],
        "missing_valid": counts["missing"],
        "invalid_valid": counts["invalid"],
        "broken_valid": counts["missing"] + counts["invalid"],
        "all_unpolished_valid": counts["raw_only"] + counts["missing"],
        "zero_frame_tasks": dict(zero_frame_tasks),
        "affected": {
            scope: {
                "tasks": len(affected_tasks[scope]),
                "episodes": len(affected_episodes[scope]),
            }
            for scope in affected_tasks
        },
    }


def classify_segment(segment: dict[str, Any], *, invalid_labels: set[str] | frozenset[str]) -> str:
    if _text(segment.get("polished_instruction")):
        return "polished"
    if _text(segment.get("raw_canonical_instruction")):
        return "raw_only"
    return "missing"


def plan_jobs(
    episodes: list[tuple[Path, dict[str, Any]]],
    *,
    scope: str,
    invalid_labels: set[str] | frozenset[str],
    task_names: set[str],
    episode_indices: set[int],
    existing_records: dict[str, dict[str, Any]],
    expected_model: str,
    retry_rejected: bool,
    max_episodes: int | None,
    max_targets: int | None,
    batch_size: int,
) -> tuple[list[EpisodeJob], dict[str, Any]]:
    jobs: list[EpisodeJob] = []
    skipped_existing = 0
    remaining_targets = max_targets
    selected_tasks: set[str] = set()
    chunk_histogram: Counter[str] = Counter()
    max_targets_per_episode = 0
    multi_chunk_episodes = 0
    stale_existing = 0

    for path, payload in episodes:
        task_name = str(payload["task_name"])
        episode_index = int(payload["episode_index"])
        if task_names and task_name not in task_names:
            continue
        if episode_indices and episode_index not in episode_indices:
            continue
        selected: list[dict[str, Any]] = []
        for segment in payload["segments"]:
            if not _is_eligible(segment) or not _selected_by_scope(segment, scope, invalid_labels):
                continue
            record_id = _record_id(task_name, episode_index, int(segment["segment_index"]))
            existing = existing_records.get(record_id)
            status = _text(existing.get("status")) if existing else ""
            if existing and status in SKIPPABLE_LEDGER_STATUSES:
                same_model = _text(existing.get("model")) == expected_model
                same_source = _text(existing.get("source_fingerprint")) == _fingerprint(segment)
                if not (same_model and same_source):
                    stale_existing += 1
                    status = ""
            should_skip = status in SKIPPABLE_LEDGER_STATUSES and not (status == "rejected" and retry_rejected)
            if should_skip:
                skipped_existing += 1
                continue
            selected.append(segment)
        if not selected:
            continue
        if remaining_targets is not None:
            selected = selected[:remaining_targets]
            remaining_targets -= len(selected)
        if not selected:
            break
        jobs.append(EpisodeJob(path=path, payload=payload, targets=tuple(selected)))
        selected_tasks.add(task_name)
        max_targets_per_episode = max(max_targets_per_episode, len(selected))
        chunks = math.ceil(len(selected) / batch_size)
        if chunks > 1:
            multi_chunk_episodes += 1
        for chunk in _chunks(selected, batch_size):
            chunk_histogram[str(len(chunk))] += 1
        if max_episodes is not None and len(jobs) >= max_episodes:
            break
        if remaining_targets is not None and remaining_targets <= 0:
            break

    return jobs, {
        "selected_segments": sum(len(job.targets) for job in jobs),
        "selected_episodes": len(jobs),
        "selected_tasks": len(selected_tasks),
        "chunk_requests": sum(math.ceil(len(job.targets) / batch_size) for job in jobs),
        "chunk_size_histogram": dict(sorted(chunk_histogram.items(), key=lambda item: int(item[0]))),
        "multi_chunk_episodes": multi_chunk_episodes,
        "max_targets_per_episode": max_targets_per_episode,
        "max_chunks_per_episode": max((math.ceil(len(job.targets) / batch_size) for job in jobs), default=0),
        "skipped_existing_records": skipped_existing,
        "stale_existing_records": stale_existing,
        "batch_size": batch_size,
    }


def _run_episode_jobs(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    model: str,
    config_path: Path,
    dataset_root: Path,
    jobs: Sequence[EpisodeJob],
    invalid_labels: set[str] | frozenset[str],
    output_path: Path,
) -> Counter[str]:
    if not jobs:
        return Counter()

    ledger_lock = threading.Lock()
    print_lock = threading.Lock()

    def process_episode(job_index: int, job: EpisodeJob) -> Counter[str]:
        episode_counts: Counter[str] = Counter()
        for chunk_index, chunk in enumerate(_chunks(job.targets, args.batch_size), start=1):
            target_indices = [int(segment["segment_index"]) for segment in chunk]
            request_error: Exception | None = None
            try:
                records = repair_chunk(
                    args=args,
                    config=config,
                    model=model,
                    dataset_root=dataset_root,
                    job=job,
                    chunk=chunk,
                    chunk_index=chunk_index,
                    invalid_labels=invalid_labels,
                )
            except Exception as exc:
                request_error = exc
                records = [
                    _error_record(
                        model=model,
                        config_path=config_path,
                        dataset_root=dataset_root,
                        job=job,
                        segment=segment,
                        chunk_indices=target_indices,
                        error=exc,
                    )
                    for segment in chunk
                ]

            with ledger_lock:
                _append_jsonl(output_path, records)
            chunk_counts = Counter(str(record["status"]) for record in records)
            episode_counts.update(chunk_counts)
            with print_lock:
                print(
                    f"[{job_index}/{len(jobs)}] {job.task_name}/episode{job.episode_index} "
                    f"chunk={chunk_index} targets={target_indices} status={dict(chunk_counts)}"
                )
            if request_error is not None and args.fail_fast:
                raise request_error
            if args.sleep_seconds > 0:
                time.sleep(float(args.sleep_seconds))
        return episode_counts

    workers = min(int(args.workers), len(jobs))
    if workers == 1:
        total_counts: Counter[str] = Counter()
        for job_index, job in enumerate(jobs, start=1):
            total_counts.update(process_episode(job_index, job))
        return total_counts

    total_counts = Counter()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="robotwin-repair") as executor:
        futures = {
            executor.submit(process_episode, job_index, job): job
            for job_index, job in enumerate(jobs, start=1)
        }
        try:
            for future in as_completed(futures):
                total_counts.update(future.result())
        except Exception:
            for future in futures:
                future.cancel()
            raise
    return total_counts


def repair_chunk(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    model: str,
    dataset_root: Path,
    job: EpisodeJob,
    chunk: Sequence[dict[str, Any]],
    chunk_index: int,
    invalid_labels: set[str] | frozenset[str],
) -> list[dict[str, Any]]:
    all_segments = list(job.payload["segments"])
    by_index = {int(segment["segment_index"]): segment for segment in all_segments}
    sequence_outline = [_compact_outline(segment) for segment in all_segments if _is_eligible(segment)]
    detailed_targets = []
    target_indices = [int(segment["segment_index"]) for segment in chunk]
    for segment in chunk:
        index = int(segment["segment_index"])
        record = _segment_polish_input(segment)
        record.update(
            {
                "issue_type": _target_issue(segment, invalid_labels),
                "full_task_instruction": job.payload.get("instruction"),
                "previous_segment": _compact_outline(by_index[index - 1]) if index - 1 in by_index else None,
                "next_segment": _compact_outline(by_index[index + 1]) if index + 1 in by_index else None,
            }
        )
        detailed_targets.append(record)

    image_args = argparse.Namespace(
        subgoal_polish_images=bool(args.images),
        subgoal_polish_image_camera=str(args.image_camera),
        subgoal_polish_image_samples=str(args.image_samples),
        subgoal_polish_max_images=int(args.max_images),
        subgoal_polish_image_size=int(args.image_size),
        subgoal_polish_image_quality=int(args.image_quality),
    )
    hdf5_path = _resolve_hdf5_path(dataset_root, job.payload) if args.images else Path()
    visual_inputs, visual_report = _build_segment_visual_inputs(image_args, hdf5_path, list(chunk))
    if args.images and not args.allow_missing_images:
        visual_target_indices = {
            int(label["segment_index"])
            for label in visual_report.get("labels", [])
            if isinstance(label, dict) and type(label.get("segment_index")) is int
        }
        missing_visuals = sorted(set(target_indices) - visual_target_indices)
        if visual_report.get("status") != "attached" or missing_visuals:
            raise RuntimeError(
                "visual evidence is incomplete; "
                f"status={visual_report.get('status')!r}, missing_target_indices={missing_visuals}. "
                "Use --allow-missing-images only if text-only fallback is intentional."
            )
    payload = {
        "task_name": job.task_name,
        "task_instruction": job.payload.get("instruction"),
        "episode_info": job.payload.get("episode_info"),
        "repair_target_indices": target_indices,
        "repair_directive": (
            "Return annotations only for repair_target_indices. The episode_sequence is context; "
            "do not rewrite non-target segments. Preserve task-level object roles such as size/color/order. "
            "Name the acting arm when it is needed to disambiguate an initial grasp/contact, a handover, or "
            "two arms operating different objects or motions. For a single arm that is already holding an "
            "object and continues moving, lifting, rotating, placing, or releasing it, the instruction and "
            "paraphrases may omit the arm name when the object and action are otherwise unambiguous. When both "
            "arms perform the same symmetric action on one shared object, saying 'both arms' is sufficient. "
            "Never assign an action to the opposite arm."
        ),
        "episode_sequence": sequence_outline,
        "segments": detailed_targets,
    }
    if visual_report:
        payload["visual_evidence"] = visual_report.get("labels", [])
    parsed = _request_segment_polish(
        config=config,
        model=model,
        payload=payload,
        visual_inputs=visual_inputs,
        n=max(0, int(args.paraphrases)),
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        retries=max(0, int(args.api_retries)),
    )
    output_by_index = _validate_model_outputs(parsed.get("segments"), target_indices)

    records = []
    for segment in chunk:
        index = int(segment["segment_index"])
        item = output_by_index.get(index)
        validation = validate_repair(
            segment,
            item,
            invalid_labels=invalid_labels,
            expected_paraphrases=max(0, int(args.paraphrases)),
        )
        status = "accepted" if validation["valid"] else "rejected"
        records.append(
            {
                "schema_version": 1,
                "record_id": _record_id(job.task_name, job.episode_index, index),
                "status": status,
                "generated_at": _now(),
                "model": model,
                "config_path": str(_resolve_config_path(Path(args.config))),
                "source_dataset": str(dataset_root),
                "source_segment_path": str(job.path.relative_to(dataset_root)),
                "source_fingerprint": _fingerprint(segment),
                "task_name": job.task_name,
                "episode_index": job.episode_index,
                "segment_index": index,
                "frame_start": int(segment["frame_start"]),
                "frame_end_exclusive": int(segment["frame_end_exclusive"]),
                "issue_type": _target_issue(segment, invalid_labels),
                "original_instruction": _current_instruction(segment),
                "repair": _clean_model_item(item),
                "validation": validation,
                "request_context": {
                    "chunk_index": chunk_index,
                    "target_segment_indices": target_indices,
                    "episode_sequence_length": len(sequence_outline),
                    "episode_info_included": job.payload.get("episode_info") is not None,
                    "visual_report": visual_report,
                },
                "raw_text_preview": str(parsed.get("_raw_text_preview") or "")[:1200],
            }
        )
    return records


def validate_repair(
    segment: dict[str, Any],
    item: dict[str, Any] | None,
    *,
    invalid_labels: set[str] | frozenset[str],
    expected_paraphrases: int | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if item is None:
        return {"valid": False, "errors": ["missing_model_output_for_segment"], "warnings": []}
    instruction = _text(item.get("instruction"))
    subgoal_type = _text(item.get("subgoal_type"))
    criteria = _text(item.get("completion_criteria"))
    if not instruction:
        errors.append("empty_instruction")
    normalized = instruction.lower()
    if normalized in invalid_labels:
        errors.append("instruction_is_known_invalid_label")
    if CODE_IDENTIFIER_RE.search(instruction):
        errors.append("instruction_contains_code_identifier")
    words = re.findall(r"[A-Za-z0-9'-]+", instruction)
    if instruction and len(words) < 3:
        errors.append("instruction_too_short")
    if len(words) > 24:
        errors.append("instruction_too_long")
    if words and words[0].lower() in CONFIRMATION_ONLY_VERBS:
        errors.append("instruction_is_confirmation_only")
    if not subgoal_type:
        errors.append("empty_subgoal_type")
    if not criteria:
        errors.append("empty_completion_criteria")

    active_arms = _active_arms(segment)
    _validate_arm_text(instruction, active_arms, errors, field="instruction")
    if not active_arms:
        errors.append("ambiguous_source_arm")

    paraphrases = item.get("paraphrases")
    if not isinstance(paraphrases, list):
        errors.append("paraphrases_not_a_list")
        cleaned_paraphrases: list[str] = []
    else:
        cleaned_paraphrases = [_text(value) for value in paraphrases]
        if any(not value for value in cleaned_paraphrases):
            errors.append("empty_paraphrase")
        if len({value.casefold() for value in cleaned_paraphrases if value}) != len(cleaned_paraphrases):
            errors.append("duplicate_paraphrases")
    if expected_paraphrases is not None and len(cleaned_paraphrases) != expected_paraphrases:
        errors.append(f"wrong_paraphrase_count_expected_{expected_paraphrases}")
    for index, paraphrase in enumerate(cleaned_paraphrases):
        if CODE_IDENTIFIER_RE.search(paraphrase):
            errors.append(f"paraphrase_{index}_contains_code_identifier")
    return {"valid": not errors, "errors": errors, "warnings": warnings}


def _validate_model_outputs(output_segments: Any, target_indices: Sequence[int]) -> dict[int, dict[str, Any]]:
    if not isinstance(output_segments, list):
        raise ValueError("model output missing segments list")
    expected = list(target_indices)
    output_by_index: dict[int, dict[str, Any]] = {}
    for position, item in enumerate(output_segments):
        if not isinstance(item, dict):
            raise ValueError(f"model output segments[{position}] is not an object")
        index = item.get("segment_index")
        if type(index) is not int:
            raise ValueError(f"model output segments[{position}].segment_index must be an integer")
        if index in output_by_index:
            raise ValueError(f"model output contains duplicate segment_index {index}")
        output_by_index[index] = item
    actual = list(output_by_index)
    if set(actual) != set(expected) or len(actual) != len(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"model output segment indices do not match targets; missing={missing}, extra={extra}")
    return output_by_index


def _validate_arm_text(
    text: str,
    active_arms: set[str],
    errors: list[str],
    *,
    field: str,
) -> None:
    mentions = {match.group(1).lower() for match in QUALIFIED_ARM_RE.finditer(text)}
    # Omitting an arm is allowed for unambiguous held-object continuations. If
    # the text mentions the actual active arm, other arm mentions can describe
    # context such as a handover ("grasp from the left arm"). Only an exclusive
    # assignment to an inactive arm is a hard error.
    wrong = mentions - active_arms if active_arms and not (mentions & active_arms) else set()
    for arm in sorted(wrong):
        errors.append(f"{field}_mentions_wrong_{arm}_arm")


def _selected_by_scope(
    segment: dict[str, Any], scope: str, invalid_labels: set[str] | frozenset[str]
) -> bool:
    classification = classify_segment(segment, invalid_labels=invalid_labels)
    invalid = _is_invalid_segment(segment, invalid_labels)
    if scope == "broken":
        return classification == "missing" or invalid
    if scope == "missing":
        return classification == "missing"
    if scope == "invalid":
        return invalid
    if scope == "raw-only":
        return classification == "raw_only"
    if scope == "all-unpolished":
        return classification in {"raw_only", "missing"}
    raise ValueError(f"Unknown scope: {scope}")


def _is_invalid_segment(segment: dict[str, Any], invalid_labels: set[str] | frozenset[str]) -> bool:
    if _text(segment.get("polished_instruction")):
        return False
    instruction = _current_instruction(segment).lower()
    return bool(instruction and instruction in invalid_labels)


def _target_issue(segment: dict[str, Any], invalid_labels: set[str] | frozenset[str]) -> str:
    if _is_invalid_segment(segment, invalid_labels):
        return "invalid_subtask_text"
    if classify_segment(segment, invalid_labels=invalid_labels) == "missing":
        return "missing_subtask_text"
    return "raw_unpolished_text"


def _is_eligible(segment: dict[str, Any]) -> bool:
    span = int(segment.get("frame_end_exclusive") or 0) - int(segment.get("frame_start") or 0)
    saved = int(segment.get("num_saved_frames") or 0)
    return span > 0 and saved > 0


def _current_instruction(segment: dict[str, Any]) -> str:
    return (
        _text(segment.get("polished_instruction"))
        or _text(segment.get("canonical_instruction"))
        or _text(segment.get("raw_canonical_instruction"))
    )


def _compact_outline(segment: dict[str, Any]) -> dict[str, Any]:
    primitive = segment.get("primitive_summary") if isinstance(segment.get("primitive_summary"), dict) else {}
    source_code = _text(segment.get("source_code"))
    return {
        "segment_index": int(segment["segment_index"]),
        "frame_start": int(segment.get("frame_start") or 0),
        "frame_end_exclusive": int(segment.get("frame_end_exclusive") or 0),
        "existing_instruction": _current_instruction(segment) or None,
        "source_line": segment.get("source_line"),
        "source_code": source_code[:500] if source_code else None,
        "active_arms": sorted(_active_arms(segment)),
        "raw_arms": primitive.get("raw_arms"),
    }


def _active_arms(segment: dict[str, Any]) -> set[str]:
    primitive = segment.get("primitive_summary")
    if not isinstance(primitive, dict):
        return set()
    return {arm for arm in ("left", "right") if isinstance(primitive.get(arm), list) and primitive[arm]}


def _resolve_hdf5_path(dataset_root: Path, payload: dict[str, Any]) -> Path:
    fallback = dataset_root / "raw" / str(payload["task_name"]) / "data" / f"episode{int(payload['episode_index'])}.hdf5"
    if fallback.is_file():
        return fallback
    configured = payload.get("hdf5_path")
    if configured:
        path = Path(str(configured)).expanduser()
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing HDF5 for {payload['task_name']}/episode{payload['episode_index']}: {fallback}")


def _clean_model_item(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if item is None:
        return None
    return {
        "instruction": _text(item.get("instruction")) or None,
        "subgoal_type": _text(item.get("subgoal_type")) or None,
        "completion_criteria": _text(item.get("completion_criteria")) or None,
        "paraphrases": [_text(value) for value in item.get("paraphrases", []) if _text(value)]
        if isinstance(item.get("paraphrases"), list)
        else [],
    }


def _error_record(
    *,
    model: str,
    config_path: Path,
    dataset_root: Path,
    job: EpisodeJob,
    segment: dict[str, Any],
    chunk_indices: list[int],
    error: Exception,
) -> dict[str, Any]:
    index = int(segment["segment_index"])
    return {
        "schema_version": 1,
        "record_id": _record_id(job.task_name, job.episode_index, index),
        "status": "api_error",
        "generated_at": _now(),
        "model": model,
        "config_path": str(config_path),
        "source_dataset": str(dataset_root),
        "source_segment_path": str(job.path.relative_to(dataset_root)),
        "source_fingerprint": _fingerprint(segment),
        "task_name": job.task_name,
        "episode_index": job.episode_index,
        "segment_index": index,
        "frame_start": int(segment["frame_start"]),
        "frame_end_exclusive": int(segment["frame_end_exclusive"]),
        "original_instruction": _current_instruction(segment),
        "repair": None,
        "validation": {"valid": False, "errors": ["request_failed"], "warnings": []},
        "request_context": {"target_segment_indices": chunk_indices},
        "error": f"{type(error).__name__}: {error}",
    }


def _resolve_config_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    cwd_candidate = (Path.cwd() / expanded).resolve()
    if cwd_candidate.is_file():
        return cwd_candidate
    project_candidate = (PROJECT_ROOT / expanded).resolve()
    if project_candidate.is_file():
        return project_candidate
    return cwd_candidate


def _read_config_preview(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"model": payload.get("model"), "base_url": payload.get("base_url")}


def _resolve_output_path(dataset_root: Path, configured: str | None, *, model: str, scope: str) -> Path:
    if configured:
        return Path(configured).expanduser().resolve()
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._-") or "model"
    safe_scope = scope.replace("-", "_")
    return dataset_root / "annotation_repairs" / f"subtask_repairs_{safe_model}_{safe_scope}.jsonl"


def _load_existing_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        record_id = _text(record.get("record_id"))
        if record_id:
            records[record_id] = record
    return records


def _revalidate_existing_rejected(
    existing_records: dict[str, dict[str, Any]],
    episodes: list[tuple[Path, dict[str, Any]]],
    *,
    expected_model: str,
    invalid_labels: set[str] | frozenset[str],
    expected_paraphrases: int,
) -> list[dict[str, Any]]:
    segment_by_id: dict[str, dict[str, Any]] = {}
    for _, payload in episodes:
        task_name = str(payload["task_name"])
        episode_index = int(payload["episode_index"])
        for segment in payload["segments"]:
            record_id = _record_id(task_name, episode_index, int(segment["segment_index"]))
            segment_by_id[record_id] = segment

    promotions: list[dict[str, Any]] = []
    for record_id, record in existing_records.items():
        if _text(record.get("status")) != "rejected" or _text(record.get("model")) != expected_model:
            continue
        segment = segment_by_id.get(record_id)
        if segment is None or _text(record.get("source_fingerprint")) != _fingerprint(segment):
            continue
        validation = validate_repair(
            segment,
            record.get("repair") if isinstance(record.get("repair"), dict) else None,
            invalid_labels=invalid_labels,
            expected_paraphrases=expected_paraphrases,
        )
        if not validation["valid"]:
            continue
        promoted = dict(record)
        promoted.update(
            {
                "status": "accepted",
                "generated_at": _now(),
                "validation": validation,
                "local_revalidation": {
                    "promoted_from": "rejected",
                    "api_request_sent": False,
                },
            }
        )
        promotions.append(promoted)
    return promotions


def _append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _episode_file_sort_key(path: Path) -> tuple[str, int, str]:
    match = re.search(r"episode(\d+)$", path.stem)
    index = int(match.group(1)) if match else 10**9
    return path.parent.name, index, path.name


def _mark_affected(
    tasks: dict[str, set[str]],
    episodes: dict[str, set[tuple[str, int]]],
    scope: str,
    task_name: str,
    episode_index: int,
) -> None:
    tasks[scope].add(task_name)
    episodes[scope].add((task_name, episode_index))


def _record_id(task_name: str, episode_index: int, segment_index: int) -> str:
    return f"{task_name}/episode{episode_index}/segment{segment_index}"


def _fingerprint(segment: dict[str, Any]) -> str:
    payload = json.dumps(segment, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
