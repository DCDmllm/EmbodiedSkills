from __future__ import annotations

"""Build the local JSON+HDF5 layout consumed by the Pi0.5 subtask loader.

The RMBench annotations already provide the executor segments.  This script
only validates those boundaries and records them; it does not merge, rewrite,
or synthesize instructions.  HDF5 files stay in the official download and are
referenced by absolute path from each episode manifest.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
from typing import Any


REQUIRED_DATASETS = (
    "joint_action/vector",
    "observation/head_camera/rgb",
    "observation/left_camera/rgb",
    "observation/right_camera/rgb",
)


def _episode_index(key: str) -> int:
    if not key.startswith("episode_"):
        raise ValueError(f"invalid episode key: {key!r}")
    return int(key.removeprefix("episode_"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_dataset(
    *, source_root: Path, output_root: Path, val_episodes: int, split_seed: int, overwrite: bool
) -> dict[str, Any]:
    annotation_path = source_root / "language_annotation.json"
    data_root = source_root / "data"
    if not annotation_path.is_file() or not data_root.is_dir():
        raise FileNotFoundError(f"Expected language_annotation.json and data/ under {source_root}")
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(annotations, dict) or not annotations:
        raise ValueError(f"{annotation_path}: expected a non-empty episode mapping")
    episode_indices = sorted(_episode_index(key) for key in annotations)
    if episode_indices != list(range(len(episode_indices))):
        raise ValueError(f"episode indices are not dense: {episode_indices}")
    if not 0 < val_episodes < len(episode_indices):
        raise ValueError(f"val_episodes must be in [1, {len(episode_indices) - 1}]")

    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "segments").mkdir(parents=True, exist_ok=True)
    # The loader uses this marker to identify the local subtask layout.  The
    # actual HDF5 data remains at the official RMBench path.
    (output_root / "raw").mkdir(parents=True, exist_ok=True)

    import h5py

    task_name = source_root.parent.name
    prompt_counts: Counter[str] = Counter()
    prompt_frames: Counter[str] = Counter()
    total_frames = 0
    total_segments = 0
    for episode_index in episode_indices:
        episode_key = f"episode_{episode_index}"
        raw_segments = annotations[episode_key]
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError(f"{annotation_path}: {episode_key} has no segments")
        hdf5_path = (data_root / f"episode{episode_index}.hdf5").resolve()
        if not hdf5_path.is_file():
            raise FileNotFoundError(hdf5_path)
        with h5py.File(hdf5_path, "r") as handle:
            missing = [key for key in REQUIRED_DATASETS if key not in handle]
            if missing:
                raise ValueError(f"{hdf5_path}: missing required datasets {missing}")
            action = handle["joint_action/vector"]
            if action.ndim != 2 or action.shape[1] != 14:
                raise ValueError(f"{hdf5_path}: expected 14D action, got {action.shape}")
            frame_count = int(action.shape[0])
            for camera_key in REQUIRED_DATASETS[1:]:
                if int(handle[camera_key].shape[0]) != frame_count:
                    raise ValueError(f"{hdf5_path}: {camera_key} length mismatch")

        frame_start = 0
        segments: list[dict[str, Any]] = []
        for segment_index, raw_segment in enumerate(raw_segments):
            if not isinstance(raw_segment, list) or len(raw_segment) != 2:
                raise ValueError(f"{episode_key}: malformed segment {segment_index}: {raw_segment!r}")
            prompt = str(raw_segment[0]).strip()
            duration = int(raw_segment[1])
            if not prompt or duration <= 0:
                raise ValueError(f"{episode_key}: invalid segment {segment_index}: {raw_segment!r}")
            frame_end = frame_start + duration
            if frame_end > frame_count:
                raise ValueError(f"{episode_key}: segment {segment_index} ends at {frame_end}, HDF5 has {frame_count}")
            segments.append(
                {
                    "segment_id": f"{task_name}_ep{episode_index:04d}_seg{segment_index:03d}",
                    "task_name": task_name,
                    "episode_index": episode_index,
                    "segment_index": segment_index,
                    "frame_start": frame_start,
                    "frame_end_exclusive": frame_end,
                    "sample_frame_end_exclusive": frame_end,
                    "sample_stride": 1,
                    "num_saved_frames": duration,
                    "canonical_instruction": prompt,
                    "raw_canonical_instruction": prompt,
                    "polished_instruction": prompt,
                    "annotation_source": "rmbench_official_language_annotation",
                }
            )
            prompt_counts[prompt] += 1
            prompt_frames[prompt] += duration
            frame_start = frame_end
        if frame_start != frame_count:
            raise ValueError(f"{episode_key}: annotation durations sum to {frame_start}, HDF5 has {frame_count}")
        _write_json(
            output_root / "segments" / task_name / f"episode{episode_index}.json",
            {
                "schema": "clawvla-rmbench-pi05-segments-v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "task_name": task_name,
                "task_config": "demo_clean",
                "episode_index": episode_index,
                "frame_count": frame_count,
                "segment_count": len(segments),
                "hdf5_path": str(hdf5_path),
                "source_language_annotation": str(annotation_path.resolve()),
                "segments": segments,
            },
        )
        total_frames += frame_count
        total_segments += len(segments)

    shuffled = episode_indices.copy()
    random.Random(split_seed).shuffle(shuffled)
    val_indices = sorted(shuffled[:val_episodes])
    train_indices = sorted(shuffled[val_episodes:])
    split_path = output_root / "splits" / f"episode_seed{split_seed}_val{val_episodes}.json"
    _write_json(
        split_path,
        {
            "schema": "clawvla-rmbench-task-split-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": split_seed,
            "dataset_root": str(output_root.resolve()),
            "tasks": {
                task_name: {
                    "episode_count": len(episode_indices),
                    "train_episode_indices": train_indices,
                    "val_episode_indices": val_indices,
                }
            },
        },
    )
    summary = {
        "schema": "clawvla-rmbench-pi05-summary-v1",
        "status": "PASS",
        "task_name": task_name,
        "task_config": "demo_clean",
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "episode_count": len(episode_indices),
        "train_episode_count": len(train_indices),
        "val_episode_count": len(val_indices),
        "train_episode_indices": train_indices,
        "val_episode_indices": val_indices,
        "segment_count": total_segments,
        "frame_count": total_frames,
        "prompt_segment_counts": dict(sorted(prompt_counts.items())),
        "prompt_frame_counts": dict(sorted(prompt_frames.items())),
        "action_horizon": 32,
        "split_manifest": str(split_path.resolve()),
        "annotation_source": str(annotation_path.resolve()),
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--val-episodes", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_dataset(
        source_root=args.source_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        val_episodes=args.val_episodes,
        split_seed=args.split_seed,
        overwrite=args.overwrite,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
