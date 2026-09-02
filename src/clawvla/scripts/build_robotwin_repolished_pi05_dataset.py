from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


SCHEMA = "clawvla-robotwin-repolished-pi05-segments-v1"
CAMERAS = ("head_camera", "left_camera", "right_camera", "front_camera")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a RoboTwin pi0.5 dataset whose segment boundaries and prompts come "
            "from the accepted repolished/merged 2486-episode plan mapping."
        )
    )
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-hdf5-audit",
        action="store_true",
        help="Skip HDF5 dataset-length checks. Metadata/frame-range checks still run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        mapping_path=args.mapping.expanduser().resolve(),
        source_root=args.source_root.expanduser().resolve(),
        split_manifest_path=args.split_manifest.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        overwrite=bool(args.overwrite),
        audit_hdf5=not bool(args.skip_hdf5_audit),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_dataset(
    *,
    mapping_path: Path,
    source_root: Path,
    split_manifest_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    audit_hdf5: bool = True,
) -> dict[str, Any]:
    for path in (mapping_path, split_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (source_root / "segments").is_dir():
        raise FileNotFoundError(source_root / "segments")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # The OpenPI loader recognizes a local RoboTwin dataset by the presence of
    # both segments/ and raw/. HDF5 files remain referenced by their audited
    # absolute source paths, so this directory is intentionally only a marker.
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)

    mappings = _load_jsonl(mapping_path)
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    split_identities = _split_identities(split_manifest)
    expected_identities = set(split_identities["train"]) | set(split_identities["val"])

    seen: set[tuple[str, int]] = set()
    task_counts: Counter[str] = Counter()
    old_segment_count = 0
    merged_segment_count = 0
    multi_source_merged_count = 0
    total_frame_count = 0
    merged_lengths: list[int] = []
    hdf5_audited: set[Path] = set()

    for mapping in sorted(
        mappings,
        key=lambda item: (str(item.get("task_name") or ""), int(item.get("episode_index", -1))),
    ):
        task_name = str(mapping.get("task_name") or "").strip()
        episode_index = int(mapping.get("episode_index", -1))
        identity = (task_name, episode_index)
        if not task_name or episode_index < 0:
            raise ValueError(f"invalid mapping identity: {identity}")
        if identity in seen:
            raise ValueError(f"duplicate mapping identity: {identity}")
        seen.add(identity)

        source_metadata_path = Path(str(mapping.get("source_segment_path") or "")).resolve()
        expected_source_path = source_root / "segments" / task_name / f"episode{episode_index}.json"
        if source_metadata_path != expected_source_path.resolve():
            raise ValueError(
                f"source metadata path mismatch for {identity}: "
                f"mapping={source_metadata_path}, expected={expected_source_path.resolve()}"
            )
        if not source_metadata_path.is_file():
            raise FileNotFoundError(source_metadata_path)
        source = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        _validate_source_identity(source, identity, source_metadata_path)

        source_segments = source.get("segments")
        if not isinstance(source_segments, list) or not source_segments:
            raise ValueError(f"{source_metadata_path}: missing source segments")
        source_by_index = {int(segment["segment_index"]): segment for segment in source_segments}
        if sorted(source_by_index) != list(range(len(source_segments))):
            raise ValueError(f"{source_metadata_path}: source segment indices are not dense and ordered")

        hdf5_path = Path(str(mapping.get("source_hdf5_path") or "")).resolve()
        if not hdf5_path.is_file():
            raise FileNotFoundError(hdf5_path)
        source_hdf5_path = Path(str(source.get("hdf5_path") or "")).resolve()
        if hdf5_path != source_hdf5_path:
            raise ValueError(
                f"HDF5 path mismatch for {identity}: mapping={hdf5_path}, source={source_hdf5_path}"
            )
        frame_count = int(source.get("frame_count") or 0)
        if frame_count <= 0:
            raise ValueError(f"{source_metadata_path}: invalid frame_count={frame_count}")
        if audit_hdf5 and hdf5_path not in hdf5_audited:
            _audit_hdf5(hdf5_path, expected_frames=frame_count)
            hdf5_audited.add(hdf5_path)

        merged_source = mapping.get("merged_subtasks")
        if not isinstance(merged_source, list) or not merged_source:
            raise ValueError(f"mapping has no merged subtasks: {identity}")
        merged_segments: list[dict[str, Any]] = []
        covered_indices: list[int] = []
        previous_frame_end: int | None = None
        for merged_index, merged in enumerate(merged_source):
            segment = _build_merged_segment(
                identity=identity,
                merged_index=merged_index,
                merged=merged,
                source_by_index=source_by_index,
                hdf5_path=hdf5_path,
            )
            indices = [int(value) for value in segment["source_segment_indices"]]
            if indices[0] != len(covered_indices):
                raise ValueError(
                    f"{identity}: merged segment {merged_index} begins at source index "
                    f"{indices[0]}, expected {len(covered_indices)}"
                )
            covered_indices.extend(indices)
            if previous_frame_end is not None and segment["frame_start"] != previous_frame_end:
                raise ValueError(
                    f"{identity}: gap/overlap between merged segments: "
                    f"previous_end={previous_frame_end}, current_start={segment['frame_start']}"
                )
            previous_frame_end = int(segment["frame_end_exclusive"])
            merged_segments.append(segment)
            length = int(segment["num_saved_frames"])
            merged_lengths.append(length)
            total_frame_count += length
            multi_source_merged_count += int(len(indices) > 1)

        expected_source_indices = list(range(len(source_segments)))
        if covered_indices != expected_source_indices:
            raise ValueError(
                f"{identity}: source coverage mismatch: expected={expected_source_indices}, "
                f"actual={covered_indices}"
            )
        if int(merged_segments[0]["frame_start"]) != int(source_segments[0]["frame_start"]):
            raise ValueError(f"{identity}: first merged segment does not start at first source frame")
        if int(merged_segments[-1]["frame_end_exclusive"]) != int(
            source_segments[-1]["frame_end_exclusive"]
        ):
            raise ValueError(f"{identity}: last merged segment does not end at last source frame")
        if sum(int(item["num_saved_frames"]) for item in merged_segments) != frame_count:
            raise ValueError(
                f"{identity}: merged frame coverage does not equal episode frame_count={frame_count}"
            )

        output_payload = _build_episode_payload(
            source=source,
            source_metadata_path=source_metadata_path,
            mapping_path=mapping_path,
            hdf5_path=hdf5_path,
            merged_segments=merged_segments,
        )
        output_path = output_dir / "segments" / task_name / f"episode{episode_index}.json"
        _write_json(output_path, output_payload)

        task_counts[task_name] += 1
        old_segment_count += len(source_segments)
        merged_segment_count += len(merged_segments)

    if seen != expected_identities:
        missing = sorted(expected_identities - seen)[:20]
        extra = sorted(seen - expected_identities)[:20]
        raise ValueError(f"mapping/split identity mismatch: missing={missing}, extra={extra}")

    output_split = deepcopy(split_manifest)
    output_split["dataset_root"] = str(output_dir)
    output_split["source_dataset_root"] = str(source_root)
    output_split["source_split_manifest"] = str(split_manifest_path)
    output_split["repolished_mapping"] = str(mapping_path)
    output_split_path = output_dir / "splits" / split_manifest_path.name
    _write_json(output_split_path, output_split)

    summary = {
        "schema": SCHEMA,
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "source_root": str(source_root),
        "source_root_metadata_sha256": _tree_metadata_sha(source_root / "segments"),
        "mapping": str(mapping_path),
        "mapping_sha256": _file_sha(mapping_path),
        "split_manifest": str(output_split_path),
        "source_split_manifest_sha256": _file_sha(split_manifest_path),
        "episode_count": len(seen),
        "task_count": len(task_counts),
        "task_counts": dict(sorted(task_counts.items())),
        "train_episode_count": len(split_identities["train"]),
        "val_episode_count": len(split_identities["val"]),
        "source_segment_count": old_segment_count,
        "merged_segment_count": merged_segment_count,
        "multi_source_merged_segment_count": multi_source_merged_count,
        "total_frame_count": total_frame_count,
        "merged_length_min": min(merged_lengths),
        "merged_length_max": max(merged_lengths),
        "merged_length_histogram": _length_histogram(merged_lengths),
        "action_horizon_note": (
            "Segment lengths are the true merged expert ranges. The pi0.5 action horizon "
            "is applied later by the loader and is not a segment-length assumption."
        ),
        "hdf5_audit_enabled": audit_hdf5,
        "hdf5_audited_count": len(hdf5_audited),
        "source_coverage": "every source segment exactly once, in order, with no frame gaps/overlaps",
        "prompt_field": "segments[].polished_instruction",
        "repair_ledger_required": False,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _build_merged_segment(
    *,
    identity: tuple[str, int],
    merged_index: int,
    merged: dict[str, Any],
    source_by_index: dict[int, dict[str, Any]],
    hdf5_path: Path,
) -> dict[str, Any]:
    task_name, episode_index = identity
    if int(merged.get("merged_subtask_index", -1)) != merged_index:
        raise ValueError(f"{identity}: merged_subtask_index mismatch at {merged_index}")
    instruction = str(merged.get("instruction") or "").strip()
    success_condition = str(merged.get("success_condition") or "").strip()
    if not instruction or not success_condition:
        raise ValueError(f"{identity}: empty merged instruction/success condition at {merged_index}")

    indices = [int(value) for value in (merged.get("source_segment_indices") or [])]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError(f"{identity}: merged source indices are not non-empty contiguous: {indices}")
    try:
        sources = [source_by_index[index] for index in indices]
    except KeyError as exc:
        raise ValueError(f"{identity}: merged mapping references missing source segment {exc}") from exc
    for left, right in zip(sources, sources[1:], strict=False):
        if int(left["frame_end_exclusive"]) != int(right["frame_start"]):
            raise ValueError(
                f"{identity}: source frame gap/overlap in merged segment {merged_index}: "
                f"{left['segment_index']}->{right['segment_index']}"
            )

    source_ids = [str(source["segment_id"]) for source in sources]
    mapped_ids = [str(value) for value in (merged.get("source_segment_ids") or [])]
    if mapped_ids != source_ids:
        raise ValueError(
            f"{identity}: source_segment_ids mismatch at merged segment {merged_index}: "
            f"mapping={mapped_ids}, source={source_ids}"
        )
    frame_start = int(sources[0]["frame_start"])
    frame_end = int(sources[-1]["frame_end_exclusive"])
    frame_count = frame_end - frame_start
    mapped_hdf5 = Path(str(merged.get("source_hdf5_path") or "")).resolve()
    checks = {
        "source_frame_start": (int(merged.get("source_frame_start", -1)), frame_start),
        "source_frame_end_exclusive": (int(merged.get("source_frame_end_exclusive", -1)), frame_end),
        "source_frame_count": (int(merged.get("source_frame_count", -1)), frame_count),
    }
    for field, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                f"{identity}: {field} mismatch at merged segment {merged_index}: "
                f"mapping={actual}, expected={expected}"
            )
    if mapped_hdf5 != hdf5_path:
        raise ValueError(
            f"{identity}: merged HDF5 mismatch at {merged_index}: {mapped_hdf5} != {hdf5_path}"
        )

    return {
        "segment_id": f"{task_name}_ep{episode_index:04d}_merged{merged_index:03d}",
        "task_name": task_name,
        "episode_index": episode_index,
        "segment_index": merged_index,
        "frame_start": frame_start,
        "frame_end_exclusive": frame_end,
        "num_saved_frames": frame_count,
        "polished_instruction": instruction,
        "canonical_instruction": instruction,
        "raw_canonical_instruction": instruction,
        "completion_criteria": success_condition,
        "annotation_source": "accepted_gpt-5.6-luna_repolished_plan",
        "source_segment_indices": indices,
        "source_segment_ids": source_ids,
        "source_frame_start": frame_start,
        "source_frame_end_exclusive": frame_end,
        "source_frame_count": frame_count,
        "source_hdf5_path": str(hdf5_path),
    }


def _build_episode_payload(
    *,
    source: dict[str, Any],
    source_metadata_path: Path,
    mapping_path: Path,
    hdf5_path: Path,
    merged_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "task_name": str(source["task_name"]),
        "task_config": source.get("task_config"),
        "split": source.get("split"),
        "episode_index": int(source["episode_index"]),
        "seed": int(source.get("seed", source["episode_index"])),
        "instruction": str(source.get("instruction") or ""),
        "task_instruction_from_config": str(source.get("task_instruction_from_config") or ""),
        "instruction_type": source.get("instruction_type"),
        "save_freq": source.get("save_freq"),
        "hdf5_path": str(hdf5_path),
        "task_dir": source.get("task_dir"),
        "frame_count": int(source["frame_count"]),
        "segments": merged_segments,
        "segment_count": len(merged_segments),
        "episode_info": source.get("episode_info"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repolish_provenance": {
            "source_metadata_path": str(source_metadata_path),
            "source_metadata_sha256": _file_sha(source_metadata_path),
            "expert_trajectory_mapping": str(mapping_path),
            "mapping_policy": (
                "contiguous source segments are merged; every original source segment is covered "
                "exactly once; merged frame ranges are copied without resampling"
            ),
        },
    }


def _audit_hdf5(path: Path, *, expected_frames: int) -> None:
    import h5py

    with h5py.File(path, "r") as handle:
        required = ["joint_action/vector"] + [f"observation/{camera}/rgb" for camera in CAMERAS]
        for key in required:
            if key not in handle:
                raise ValueError(f"{path}: missing HDF5 dataset {key}")
            if int(handle[key].shape[0]) < expected_frames:
                raise ValueError(
                    f"{path}: {key} length {handle[key].shape[0]} is shorter than {expected_frames}"
                )
        action = handle["joint_action/vector"]
        if len(action.shape) != 2 or int(action.shape[1]) < 14:
            raise ValueError(f"{path}: expected at least 14-D joint_action/vector, got {action.shape}")


def _split_identities(manifest: dict[str, Any]) -> dict[str, set[tuple[str, int]]]:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("split manifest has no tasks")
    result = {"train": set(), "val": set()}
    for task_name, info in tasks.items():
        for split in result:
            values = info.get(f"{split}_episode_indices")
            if not isinstance(values, list):
                raise ValueError(f"split manifest missing {split} episodes for {task_name}")
            result[split].update((str(task_name), int(value)) for value in values)
    if result["train"] & result["val"]:
        raise ValueError("train and val episode identities overlap")
    if len(result["train"]) != int(manifest.get("num_train_episodes", -1)):
        raise ValueError("split manifest train count mismatch")
    if len(result["val"]) != int(manifest.get("num_val_episodes", -1)):
        raise ValueError("split manifest val count mismatch")
    return result


def _validate_source_identity(
    source: dict[str, Any], identity: tuple[str, int], path: Path
) -> None:
    actual = (str(source.get("task_name") or ""), int(source.get("episode_index", -1)))
    if actual != identity:
        raise ValueError(f"{path}: source identity {actual} does not match mapping {identity}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"no records in {path}")
    return rows


def _length_histogram(lengths: list[int]) -> dict[str, int]:
    buckets = Counter()
    for value in lengths:
        if value <= 15:
            buckets["1-15"] += 1
        elif value <= 32:
            buckets["16-32"] += 1
        elif value <= 64:
            buckets["33-64"] += 1
        elif value <= 128:
            buckets["65-128"] += 1
        else:
            buckets["129+"] += 1
    return {key: buckets[key] for key in ("1-15", "16-32", "33-64", "65-128", "129+")}


def _file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_metadata_sha(root: Path) -> str:
    digest = sha256()
    for path in sorted(root.glob("*/*.json")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(_file_sha(path).encode("ascii"))
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
