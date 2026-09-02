from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one lightweight RoboTwin pi0.5 dataset root per task. "
            "Only segment metadata is copied; metadata continues to reference "
            "the original HDF5 trajectories."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Prepare only this task. Repeat for multiple tasks; omit for all tasks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    split_path = args.split_manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    task_splits = split_payload.get("tasks")
    if not isinstance(task_splits, dict) or not task_splits:
        raise ValueError(f"split manifest has no tasks: {split_path}")

    requested_tasks = list(dict.fromkeys(str(task).strip() for task in args.task if str(task).strip()))
    tasks = requested_tasks or sorted(task_splits)
    unknown = sorted(set(tasks) - set(task_splits))
    if unknown:
        raise ValueError(f"tasks are absent from split manifest: {unknown}")

    output_root.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    for task_name in tasks:
        task_info = task_splits[task_name]
        source_segment_dir = source_root / "segments" / task_name
        metadata_paths = sorted(source_segment_dir.glob("episode*.json"), key=_episode_index)
        expected_episode_count = int(task_info["num_episodes"])
        if len(metadata_paths) != expected_episode_count:
            raise ValueError(
                f"{task_name}: expected {expected_episode_count} metadata files, "
                f"found {len(metadata_paths)} in {source_segment_dir}"
            )

        train_indices = {int(index) for index in task_info["train_episode_indices"]}
        val_indices = {int(index) for index in task_info["val_episode_indices"]}
        if train_indices & val_indices:
            raise ValueError(f"{task_name}: train/val episode overlap")
        expected_indices = train_indices | val_indices
        actual_indices = {_episode_index(path) for path in metadata_paths}
        if actual_indices != expected_indices:
            raise ValueError(
                f"{task_name}: metadata/split episode mismatch; "
                f"missing={sorted(expected_indices - actual_indices)}, "
                f"extra={sorted(actual_indices - expected_indices)}"
            )

        task_root = output_root / task_name
        output_segment_dir = task_root / "segments" / task_name
        output_raw_dir = task_root / "raw"
        output_segment_dir.mkdir(parents=True, exist_ok=True)
        output_raw_dir.mkdir(parents=True, exist_ok=True)

        copied_names: set[str] = set()
        segment_count = 0
        frame_count = 0
        for metadata_path in metadata_paths:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            episode_index = int(payload["episode_index"])
            if str(payload.get("task_name") or "") != task_name:
                raise ValueError(f"{metadata_path}: task_name does not match {task_name}")
            hdf5_path = Path(str(payload.get("hdf5_path") or ""))
            if not hdf5_path.is_file():
                raise FileNotFoundError(f"{metadata_path}: missing HDF5 {hdf5_path}")
            destination = output_segment_dir / metadata_path.name
            shutil.copy2(metadata_path, destination)
            copied_names.add(metadata_path.name)
            segments = payload.get("segments")
            if not isinstance(segments, list) or not segments:
                raise ValueError(f"{metadata_path}: missing segments")
            segment_count += len(segments)
            frame_count += sum(int(segment.get("num_saved_frames") or 0) for segment in segments)

        stale_files = [
            path for path in output_segment_dir.glob("episode*.json") if path.name not in copied_names
        ]
        if stale_files:
            raise ValueError(
                f"{task_name}: output contains stale metadata files: "
                f"{[path.name for path in stale_files]}"
            )

        task_summary = {
            "task_name": task_name,
            "dataset_root": str(task_root),
            "source_root": str(source_root),
            "split_manifest": str(split_path),
            "num_episodes": len(metadata_paths),
            "num_train_episodes": len(train_indices),
            "num_val_episodes": len(val_indices),
            "num_segments": segment_count,
            "num_frame_windows_before_horizon_padding": frame_count,
            "hdf5_storage_copied": False,
        }
        (task_root / "specialist_manifest.json").write_text(
            json.dumps(task_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summaries[task_name] = task_summary

    summary = {
        "schema": "clawvla-robotwin-pi05-task-specialists-v1",
        "source_root": str(source_root),
        "split_manifest": str(split_path),
        "output_root": str(output_root),
        "num_tasks": len(summaries),
        "num_episodes": sum(item["num_episodes"] for item in summaries.values()),
        "num_train_episodes": sum(item["num_train_episodes"] for item in summaries.values()),
        "num_val_episodes": sum(item["num_val_episodes"] for item in summaries.values()),
        "tasks": summaries,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _episode_index(path: Path) -> int:
    return int(path.stem.removeprefix("episode"))


if __name__ == "__main__":
    main()
