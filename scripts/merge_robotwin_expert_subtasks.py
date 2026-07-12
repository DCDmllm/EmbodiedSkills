#!/usr/bin/env python3
"""Merge several RoboTwin expert-subtask runs into one self-contained run.

The source runs are never modified. Episode indices are reassigned per task,
JSON metadata is rewritten, and the embedded ClawVLA metadata in every HDF5
file is updated to match the merged layout. Work is performed in a resumable
staging directory and published with an atomic rename after validation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable


@dataclass(frozen=True)
class SourceRun:
    root: Path
    run_id: str
    summary: dict[str, Any]
    records_by_task: dict[str, tuple[dict[str, Any], ...]]


@dataclass(frozen=True)
class PlannedEpisode:
    task_name: str
    episode_index: int
    source_run_id: str
    source_root: Path
    source_episode_index: int
    source_hdf5: Path
    source_segment: Path
    source_pkl: Path
    final_hdf5: Path
    final_segment: Path
    final_pkl: Path
    episode_record: dict[str, Any]
    segment_record: dict[str, Any]
    manifest_record: dict[str, Any]
    provenance_record: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="Source run directory; order defines precedence.")
    parser.add_argument("--output-dir", required=True, help="Final merged run directory.")
    parser.add_argument("--target-per-task", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    temporary.replace(path)


def load_source(root_text: str) -> SourceRun:
    root = Path(root_text).expanduser().resolve()
    summary_path = root / "summary.json"
    episodes_path = root / "episodes.jsonl"
    if not summary_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(f"source run lacks summary.json or episodes.jsonl: {root}")

    summary = read_json(summary_path)
    run_id = str(summary.get("run_id") or root.name)
    deduplicated: dict[tuple[str, int], dict[str, Any]] = {}
    for record in read_jsonl(episodes_path):
        if not (record.get("ok") is True and record.get("status") == "collected"):
            continue
        task_name = str(record.get("task_name") or "")
        episode_index = int(record["episode_index"])
        if not task_name:
            raise ValueError(f"source record has no task_name: {root}")
        key = (task_name, episode_index)
        previous = deduplicated.get(key)
        if previous is not None and (
            previous.get("seed") != record.get("seed")
            or previous.get("frame_count") != record.get("frame_count")
        ):
            raise ValueError(f"conflicting duplicate source record: {run_id}:{task_name}:{episode_index}")
        deduplicated[key] = record

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (task_name, _), record in deduplicated.items():
        grouped[task_name].append(record)
    records_by_task = {
        task_name: tuple(sorted(records, key=lambda item: int(item["episode_index"])))
        for task_name, records in grouped.items()
    }
    expected = int(summary.get("total_collected", sum(map(len, records_by_task.values()))))
    actual = sum(map(len, records_by_task.values()))
    if actual != expected:
        raise ValueError(f"source count mismatch for {run_id}: summary={expected}, episodes.jsonl={actual}")
    return SourceRun(root=root, run_id=run_id, summary=summary, records_by_task=records_by_task)


def source_paths(source: SourceRun, record: dict[str, Any]) -> tuple[Path, Path, Path]:
    task_name = str(record["task_name"])
    episode_index = int(record["episode_index"])
    task_root = source.root / "raw" / task_name
    return (
        task_root / "data" / f"episode{episode_index}.hdf5",
        source.root / "segments" / task_name / f"episode{episode_index}.json",
        task_root / "_traj_data" / f"episode{episode_index}.pkl",
    )


def merge_provenance(
    source: SourceRun,
    record: dict[str, Any],
    source_hdf5: Path,
    source_segment: Path,
    source_pkl: Path,
) -> dict[str, Any]:
    return {
        "source_run_id": source.run_id,
        "source_root": str(source.root),
        "source_episode_index": int(record["episode_index"]),
        "source_seed": record.get("seed"),
        "source_hdf5_path": str(source_hdf5),
        "source_segment_path": str(source_segment),
        "source_pkl_path": str(source_pkl),
    }


def rewrite_segment_record(
    payload: dict[str, Any],
    *,
    task_name: str,
    episode_index: int,
    final_hdf5: Path,
    final_task_dir: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    rewritten = deepcopy(payload)
    rewritten["task_name"] = task_name
    rewritten["episode_index"] = episode_index
    rewritten["hdf5_path"] = str(final_hdf5)
    rewritten["task_dir"] = str(final_task_dir)
    rewritten["merge_provenance"] = deepcopy(provenance)
    segments = rewritten.get("segments")
    if not isinstance(segments, list):
        raise ValueError(f"segment metadata lacks a segments list: {task_name}:{episode_index}")
    for offset, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"invalid segment item: {task_name}:{episode_index}:{offset}")
        segment_index = int(segment.get("segment_index", offset))
        segment["task_name"] = task_name
        segment["episode_index"] = episode_index
        segment["segment_index"] = segment_index
        segment["segment_id"] = f"{task_name}_ep{episode_index:04d}_seg{segment_index:03d}"
    rewritten["segment_count"] = len(segments)
    return rewritten


def compact_segment_record(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_name": payload.get("task_name"),
        "episode_index": payload.get("episode_index"),
        "seed": payload.get("seed"),
        "instruction": payload.get("instruction"),
        "hdf5_path": payload.get("hdf5_path"),
        "segment_count": payload.get("segment_count"),
        "frame_count": payload.get("frame_count"),
        "annotation_counts": payload.get("annotation_counts"),
    }


def build_plan(
    sources: list[SourceRun],
    final_root: Path,
    target_per_task: int,
) -> tuple[list[str], list[PlannedEpisode]]:
    first_per_task = sources[0].summary.get("per_task")
    if not isinstance(first_per_task, dict) or not first_per_task:
        raise ValueError(f"first source has no per_task summary: {sources[0].root}")
    task_order = [str(task_name) for task_name in first_per_task]
    canonical_tasks = set(task_order)
    for source in sources:
        unknown = set(source.records_by_task) - canonical_tasks
        if unknown:
            raise ValueError(f"source {source.run_id} has unknown tasks: {sorted(unknown)}")

    plan: list[PlannedEpisode] = []
    for task_name in task_order:
        selected: list[tuple[SourceRun, dict[str, Any]]] = []
        for source in sources:
            selected.extend((source, record) for record in source.records_by_task.get(task_name, ()))
        selected = selected[:target_per_task]
        seen_seeds: set[int] = set()
        for new_index, (source, record) in enumerate(selected):
            seed = int(record["seed"])
            if seed in seen_seeds:
                raise ValueError(f"duplicate seed within task {task_name}: {seed}")
            seen_seeds.add(seed)
            source_hdf5, source_segment, source_pkl = source_paths(source, record)
            for path in (source_hdf5, source_segment, source_pkl):
                if not path.is_file():
                    raise FileNotFoundError(f"source episode file is missing: {path}")

            final_task_dir = final_root / "raw" / task_name
            final_hdf5 = final_task_dir / "data" / f"episode{new_index}.hdf5"
            final_segment = final_root / "segments" / task_name / f"episode{new_index}.json"
            final_pkl = final_task_dir / "_traj_data" / f"episode{new_index}.pkl"
            provenance = merge_provenance(source, record, source_hdf5, source_segment, source_pkl)
            segment_record = rewrite_segment_record(
                read_json(source_segment),
                task_name=task_name,
                episode_index=new_index,
                final_hdf5=final_hdf5,
                final_task_dir=final_task_dir,
                provenance=provenance,
            )
            episode_record = deepcopy(record)
            episode_record.update(
                {
                    "task_name": task_name,
                    "episode_index": new_index,
                    "hdf5_path": str(final_hdf5),
                    "segment_path": str(final_segment),
                    "task_dir": str(final_task_dir),
                    "merge_provenance": deepcopy(provenance),
                }
            )
            manifest_record = {
                "task_name": task_name,
                "episode_index": new_index,
                "seed": seed,
                "hdf5_path": str(final_hdf5),
                "segment_path": str(final_segment),
                "segment_count": int(segment_record["segment_count"]),
                "frame_count": int(segment_record["frame_count"]),
            }
            provenance_record = {
                "task_name": task_name,
                "episode_index": new_index,
                "hdf5_path": str(final_hdf5),
                **provenance,
            }
            plan.append(
                PlannedEpisode(
                    task_name=task_name,
                    episode_index=new_index,
                    source_run_id=source.run_id,
                    source_root=source.root,
                    source_episode_index=int(record["episode_index"]),
                    source_hdf5=source_hdf5,
                    source_segment=source_segment,
                    source_pkl=source_pkl,
                    final_hdf5=final_hdf5,
                    final_segment=final_segment,
                    final_pkl=final_pkl,
                    episode_record=episode_record,
                    segment_record=segment_record,
                    manifest_record=manifest_record,
                    provenance_record=provenance_record,
                )
            )
    return task_order, plan


def stage_path(final_path: Path, final_root: Path, stage_root: Path) -> Path:
    return stage_root / final_path.relative_to(final_root)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".copying.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def decode_json_dataset(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def hdf5_metadata_matches(path: Path, episode: PlannedEpisode) -> bool:
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            metadata = decode_json_dataset(handle["clawvla_metadata/episode_metadata_json"][()])
            segments = decode_json_dataset(handle["clawvla_metadata/segments_json"][()])
            return (
                metadata.get("task_name") == episode.task_name
                and int(metadata.get("episode_index", -1)) == episode.episode_index
                and metadata.get("hdf5_path") == str(episode.final_hdf5)
                and isinstance(segments, list)
                and len(segments) == int(episode.manifest_record["segment_count"])
                and all(int(item.get("episode_index", -1)) == episode.episode_index for item in segments)
            )
    except Exception:
        return False


def rewrite_hdf5_metadata(path: Path, episode: PlannedEpisode) -> None:
    import h5py

    embedded_episode = compact_segment_record(episode.segment_record)
    embedded_episode["merge_provenance"] = deepcopy(episode.provenance_record)
    with h5py.File(path, "a") as handle:
        group = handle.require_group("clawvla_metadata")
        for name, payload in {
            "episode_metadata_json": embedded_episode,
            "segments_json": episode.segment_record["segments"],
        }.items():
            if name in group:
                del group[name]
            group.create_dataset(name, data=json.dumps(payload, ensure_ascii=True))


def materialize_episode(episode: PlannedEpisode, final_root: Path, stage_root: Path) -> str:
    stage_hdf5 = stage_path(episode.final_hdf5, final_root, stage_root)
    stage_segment = stage_path(episode.final_segment, final_root, stage_root)
    stage_pkl = stage_path(episode.final_pkl, final_root, stage_root)

    if not hdf5_metadata_matches(stage_hdf5, episode):
        atomic_copy(episode.source_hdf5, stage_hdf5)
        rewrite_hdf5_metadata(stage_hdf5, episode)
    if not stage_pkl.is_file() or stage_pkl.stat().st_size != episode.source_pkl.stat().st_size:
        atomic_copy(episode.source_pkl, stage_pkl)
    atomic_write_json(stage_segment, episode.segment_record)
    return f"{episode.task_name}:{episode.episode_index}"


def validate_episode(episode: PlannedEpisode, final_root: Path, stage_root: Path) -> None:
    stage_hdf5 = stage_path(episode.final_hdf5, final_root, stage_root)
    stage_segment = stage_path(episode.final_segment, final_root, stage_root)
    stage_pkl = stage_path(episode.final_pkl, final_root, stage_root)
    if not stage_hdf5.is_file() or not stage_segment.is_file() or not stage_pkl.is_file():
        raise FileNotFoundError(f"merged episode is incomplete: {episode.task_name}:{episode.episode_index}")
    if stage_pkl.stat().st_size != episode.source_pkl.stat().st_size:
        raise ValueError(f"PKL size mismatch: {stage_pkl}")
    if not hdf5_metadata_matches(stage_hdf5, episode):
        raise ValueError(f"HDF5 metadata mismatch: {stage_hdf5}")

    segment = read_json(stage_segment)
    if segment.get("hdf5_path") != str(episode.final_hdf5):
        raise ValueError(f"segment hdf5_path mismatch: {stage_segment}")
    if int(segment.get("episode_index", -1)) != episode.episode_index:
        raise ValueError(f"segment episode_index mismatch: {stage_segment}")
    segments = segment.get("segments")
    if not isinstance(segments, list) or len(segments) != int(episode.manifest_record["segment_count"]):
        raise ValueError(f"segment count mismatch: {stage_segment}")

    import h5py

    with h5py.File(stage_hdf5, "r") as handle:
        required = {"clawvla_metadata", "endpose", "joint_action", "observation", "pointcloud"}
        missing = required - set(handle.keys())
        if missing:
            raise ValueError(f"HDF5 groups missing in {stage_hdf5}: {sorted(missing)}")
        frame_count = int(handle["joint_action/vector"].shape[0])
        if frame_count != int(episode.manifest_record["frame_count"]):
            raise ValueError(f"HDF5 frame_count mismatch: {stage_hdf5}: {frame_count}")


def source_snapshot(plan: list[PlannedEpisode]) -> dict[Path, tuple[int, int]]:
    snapshot: dict[Path, tuple[int, int]] = {}
    for episode in plan:
        for path in (episode.source_hdf5, episode.source_segment, episode.source_pkl):
            stat = path.stat()
            snapshot[path] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def verify_source_snapshot(snapshot: dict[Path, tuple[int, int]]) -> None:
    changed = []
    for path, expected in snapshot.items():
        stat = path.stat()
        current = (stat.st_size, stat.st_mtime_ns)
        if current != expected:
            changed.append(str(path))
    if changed:
        raise RuntimeError(f"source files changed during merge: {changed[:10]}")


def build_metadata(
    *,
    sources: list[SourceRun],
    task_order: list[str],
    plan: list[PlannedEpisode],
    final_root: Path,
    target_per_task: int,
) -> dict[str, Any]:
    by_task: dict[str, list[PlannedEpisode]] = defaultdict(list)
    for episode in plan:
        by_task[episode.task_name].append(episode)

    source_attempts: dict[str, int] = defaultdict(int)
    source_failed_jobs: dict[str, int] = defaultdict(int)
    for source in sources:
        per_task = source.summary.get("per_task") or {}
        for task_name in task_order:
            item = per_task.get(task_name) or {}
            source_attempts[task_name] += int(item.get("attempts", 0))
            state_path = source.root / "states" / f"{task_name}.json"
            if state_path.is_file():
                source_failed_jobs[task_name] += int(read_json(state_path).get("failed_jobs", 0))

    states: dict[str, dict[str, Any]] = {}
    per_task_summary: dict[str, dict[str, Any]] = {}
    incomplete_tasks: list[str] = []
    for task_name in task_order:
        episodes = sorted(by_task[task_name], key=lambda item: item.episode_index)
        count = len(episodes)
        if count < target_per_task:
            incomplete_tasks.append(task_name)
        state_episodes = [deepcopy(item.manifest_record) for item in episodes]
        last = episodes[-1] if episodes else None
        states[task_name] = {
            "task_name": task_name,
            "task_config": "demo_clean",
            "split": "train",
            "target": target_per_task,
            "collected": count,
            "attempts": source_attempts[task_name],
            "failed_jobs": source_failed_jobs[task_name],
            "episodes": state_episodes,
            "last_episode_index": last.episode_index if last else None,
            "last_seed": last.manifest_record.get("seed") if last else None,
            "last_hdf5_path": str(last.final_hdf5) if last else None,
            "last_segment_path": str(last.final_segment) if last else None,
            "last_status": "merged",
            "resume_supported": False,
            "source_run_ids": [source.run_id for source in sources],
        }
        attempts = source_attempts[task_name]
        per_task_summary[task_name] = {
            "collected": count,
            "target": target_per_task,
            "attempts": attempts,
            "blocked": False,
            "last_status": "merged",
            "last_seed": last.manifest_record.get("seed") if last else None,
            "last_hdf5_path": str(last.final_hdf5) if last else None,
            "last_segment_path": str(last.final_segment) if last else None,
            "yield": count / attempts if attempts else None,
        }

    now = datetime.now().isoformat(timespec="seconds")
    total_collected = len(plan)
    total_attempts = sum(source_attempts.values())
    summary = {
        "run_id": final_root.name,
        "split": "train",
        "task_config": "demo_clean",
        "episodes_per_task": target_per_task,
        "task_count": len(task_order),
        "total_target": len(task_order) * target_per_task,
        "total_collected": total_collected,
        "total_attempts": total_attempts,
        "complete": not incomplete_tasks,
        "incomplete_tasks": incomplete_tasks,
        "blocked_tasks": [],
        "per_task": per_task_summary,
        "merge_source_run_ids": [source.run_id for source in sources],
        "updated_at": now,
    }
    source_descriptions = [
        {
            "run_id": source.run_id,
            "root": str(source.root),
            "total_collected": int(source.summary.get("total_collected", 0)),
            "episodes_per_task": int(source.summary.get("episodes_per_task", 0)),
            "complete": bool(source.summary.get("complete")),
        }
        for source in sources
    ]
    selected_by_source = Counter(episode.source_run_id for episode in plan)
    merge_config = {
        "schema_version": 1,
        "mode": "copy_and_reindex",
        "output_dir": str(final_root),
        "target_per_task": target_per_task,
        "task_count": len(task_order),
        "episode_count": total_collected,
        "sources": source_descriptions,
        "selected_episode_count_by_source": dict(selected_by_source),
        "source_runs_preserved": True,
        "resume_collection_supported": False,
        "created_at": now,
    }
    run_config = {
        "mode": "merged_dataset",
        "args": {
            "target_per_task": target_per_task,
            "output_dir": str(final_root),
            "source": [str(source.root) for source in sources],
        },
        "task_count": len(task_order),
        "merge_config": merge_config,
    }
    return {
        "states": states,
        "summary": summary,
        "merge_config": merge_config,
        "run_config": run_config,
    }


def write_run_metadata(
    stage_root: Path,
    plan: list[PlannedEpisode],
    metadata: dict[str, Any],
) -> None:
    ordered = sorted(plan, key=lambda item: (item.task_name, item.episode_index))
    atomic_write_jsonl(stage_root / "episodes.jsonl", (item.episode_record for item in ordered))
    atomic_write_jsonl(stage_root / "segments.jsonl", (compact_segment_record(item.segment_record) for item in ordered))
    atomic_write_jsonl(stage_root / "provenance.jsonl", (item.provenance_record for item in ordered))
    manifest = [item.manifest_record for item in ordered]
    atomic_write_json(stage_root / "manifest.json", {"episodes": manifest, "episode_count": len(manifest)})
    atomic_write_json(stage_root / "summary.json", metadata["summary"])
    atomic_write_json(stage_root / "merge_config.json", metadata["merge_config"])
    atomic_write_json(stage_root / "run_config.json", metadata["run_config"])
    for task_name, state in metadata["states"].items():
        atomic_write_json(stage_root / "states" / f"{task_name}.json", state)
    (stage_root / "logs").mkdir(parents=True, exist_ok=True)
    (stage_root / "summaries").mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    if args.target_per_task <= 0:
        raise ValueError("--target-per-task must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    sources = [load_source(value) for value in args.source]
    final_root = Path(args.output_dir).expanduser().resolve()
    stage_root = final_root.parent / f".{final_root.name}.partial"
    if final_root.exists():
        raise FileExistsError(f"output already exists: {final_root}")
    task_order, plan = build_plan(sources, final_root, args.target_per_task)
    counts = Counter(item.task_name for item in plan)
    distribution = dict(sorted(Counter(counts.values()).items()))
    selected_by_source = Counter(item.source_run_id for item in plan)
    print(f"output={final_root}", flush=True)
    print(f"stage={stage_root}", flush=True)
    print(f"tasks={len(task_order)} episodes={len(plan)} distribution={distribution}", flush=True)
    print(f"selected_by_source={dict(selected_by_source)}", flush=True)
    if args.dry_run:
        return 0

    stage_root.mkdir(parents=True, exist_ok=True)
    snapshot = source_snapshot(plan)
    started = time.monotonic()
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(materialize_episode, episode, final_root, stage_root): episode
            for episode in plan
        }
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == len(plan):
                elapsed = time.monotonic() - started
                rate = completed / elapsed if elapsed else 0.0
                print(f"materialized={completed}/{len(plan)} rate={rate:.2f}_episodes/s", flush=True)

    metadata = build_metadata(
        sources=sources,
        task_order=task_order,
        plan=plan,
        final_root=final_root,
        target_per_task=args.target_per_task,
    )
    write_run_metadata(stage_root, plan, metadata)
    print("validating merged episodes", flush=True)
    validated = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(validate_episode, episode, final_root, stage_root): episode
            for episode in plan
        }
        for future in as_completed(futures):
            future.result()
            validated += 1
            if validated == 1 or validated % 100 == 0 or validated == len(plan):
                print(f"validated={validated}/{len(plan)}", flush=True)

    verify_source_snapshot(snapshot)
    completion = {
        "status": "complete",
        "episode_count": len(plan),
        "task_count": len(task_order),
        "count_distribution": distribution,
        "source_files_unchanged": True,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_json(stage_root / "MERGE_COMPLETE.json", completion)
    stage_root.replace(final_root)
    print(f"published={final_root}", flush=True)
    print(json.dumps(completion, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("merge interrupted; rerun the same command to resume the staging directory", file=sys.stderr)
        raise SystemExit(130)
