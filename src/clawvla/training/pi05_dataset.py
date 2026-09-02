from __future__ import annotations

import bisect
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, SupportsIndex

import numpy as np


INVALID_PROMPTS = frozenset({"", "arm_tag", "arm tag", "none", "null", "unknown"})


@dataclass(frozen=True)
class SegmentRecord:
    record_id: str
    hdf5_path: Path
    task_name: str
    episode_index: int
    segment_index: int
    frame_start: int
    frame_end_exclusive: int
    sample_frame_end_exclusive: int
    sample_stride: int
    prompt: str
    prompt_variants: tuple[str, ...]


class RoboTwinSubtaskDataset:
    """Read frame-aligned RoboTwin/RMBench subtasks for OpenPI training.

    Every source frame is conditioned only on its current subtask. Action
    chunks stop at that subtask's boundary and are padded with the final action,
    preventing labels from leaking into the next subtask.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        action_horizon: int,
        *,
        repair_ledger: str | Path | None = None,
        strict_prompts: bool = True,
        episode_split_manifest: str | Path | None = None,
        episode_split_name: str | None = None,
        prompt_variant_manifest: str | Path | None = None,
        prompt_variant_probability: float = 0.0,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.action_horizon = int(action_horizon)
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")
        self.prompt_variant_probability = float(prompt_variant_probability)
        if not 0.0 <= self.prompt_variant_probability <= 1.0:
            raise ValueError("prompt_variant_probability must be in [0, 1]")
        self.skip_images = os.environ.get("ROBOTWIN_SUBTASK_SKIP_IMAGES", "0") == "1"

        segment_root = self.dataset_root / "segments"
        if not segment_root.is_dir():
            raise FileNotFoundError(f"missing segment directory: {segment_root}")
        repairs = self._load_jsonl_index(repair_ledger)
        variants = self._load_jsonl_groups(prompt_variant_manifest)
        allowed = self._load_split(episode_split_manifest, episode_split_name)
        repair_required = repair_ledger is not None and bool(str(repair_ledger).strip())

        self.segments: list[SegmentRecord] = []
        included_episodes: set[tuple[str, int]] = set()
        invalid: list[str] = []
        included_ids: set[str] = set()
        for metadata_path in sorted(segment_root.glob("*/*.json"), key=self._metadata_sort_key):
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            task_name = str(payload.get("task_name") or metadata_path.parent.name)
            episode_index = int(payload["episode_index"])
            if allowed is not None:
                if task_name not in allowed:
                    raise ValueError(f"split manifest has no task {task_name!r}")
                if episode_index not in allowed[task_name]:
                    continue
            included_episodes.add((task_name, episode_index))
            hdf5_path = self._resolve_hdf5_path(payload, task_name, episode_index)
            raw_segments = payload.get("segments")
            if not isinstance(raw_segments, list):
                raise ValueError(f"{metadata_path}: segments must be a list")
            for raw in raw_segments:
                frame_start = int(raw.get("frame_start") or 0)
                frame_end = int(raw.get("frame_end_exclusive") or 0)
                if frame_end <= frame_start or int(raw.get("num_saved_frames") or 0) <= 0:
                    continue
                sample_end = int(raw.get("sample_frame_end_exclusive") or frame_end)
                stride = int(raw.get("sample_stride") or 1)
                if not frame_start < sample_end <= frame_end:
                    raise ValueError(f"{metadata_path}: invalid sample frame range")
                if stride <= 0:
                    raise ValueError(f"{metadata_path}: sample_stride must be positive")
                segment_index = int(raw["segment_index"])
                record_id = f"{task_name}/episode{episode_index}/segment{segment_index}"
                included_ids.add(record_id)
                prompt = self._segment_prompt(
                    raw,
                    repairs.get(record_id),
                    record_id,
                    require_accepted_repair=repair_required
                    and not bool(str(raw.get("polished_instruction") or "").strip()),
                )
                if prompt.casefold() in INVALID_PROMPTS:
                    invalid.append(record_id)
                    continue
                prompt_variants = self._validate_variants(
                    variants.get(record_id, []), prompt, record_id
                )
                self.segments.append(
                    SegmentRecord(
                        record_id=record_id,
                        hdf5_path=hdf5_path,
                        task_name=task_name,
                        episode_index=episode_index,
                        segment_index=segment_index,
                        frame_start=frame_start,
                        frame_end_exclusive=frame_end,
                        sample_frame_end_exclusive=sample_end,
                        sample_stride=stride,
                        prompt=prompt,
                        prompt_variants=prompt_variants,
                    )
                )

        unexpected_variants = sorted(set(variants) - included_ids)
        if unexpected_variants:
            raise ValueError(
                f"prompt variants contain records outside the selected split: {unexpected_variants[:8]}"
            )
        if strict_prompts and invalid:
            raise ValueError(f"{len(invalid)} segments have invalid prompts: {invalid[:8]}")
        if not self.segments:
            raise ValueError(f"no usable subtask segments under {self.dataset_root}")

        self.cumulative = np.cumsum(
            [
                (segment.sample_frame_end_exclusive - segment.frame_start + segment.sample_stride - 1)
                // segment.sample_stride
                for segment in self.segments
            ],
            dtype=np.int64,
        )
        self.num_episodes = len(included_episodes)
        self._hdf5_cache: OrderedDict[str, Any] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_hdf5_cache"] = OrderedDict()
        return state

    def __del__(self) -> None:
        for handle in getattr(self, "_hdf5_cache", {}).values():
            try:
                handle.close()
            except Exception:
                pass

    def __len__(self) -> int:
        return int(self.cumulative[-1])

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        idx = index.__index__()
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        segment_position = bisect.bisect_right(self.cumulative, idx)
        previous = 0 if segment_position == 0 else int(self.cumulative[segment_position - 1])
        segment = self.segments[segment_position]
        frame_index = segment.frame_start + (idx - previous) * segment.sample_stride
        handle = self._get_hdf5(segment.hdf5_path)
        action_dataset = handle["joint_action/vector"]
        if segment.frame_end_exclusive > int(action_dataset.shape[0]):
            raise ValueError(f"{segment.record_id}: segment exceeds HDF5 action length")
        state = np.asarray(action_dataset[frame_index], dtype=np.float32)
        chunk_end = min(frame_index + self.action_horizon, segment.frame_end_exclusive)
        actions = np.asarray(action_dataset[frame_index:chunk_end], dtype=np.float32)
        if not len(actions):
            raise ValueError(f"{segment.record_id}: empty action chunk at frame {frame_index}")
        if len(actions) < self.action_horizon:
            padding = np.repeat(actions[-1:], self.action_horizon - len(actions), axis=0)
            actions = np.concatenate((actions, padding), axis=0)
        return {
            "observation.state": state,
            "action": actions,
            "observation.images.cam_high": self._image_or_zero(handle, "head_camera", frame_index),
            "observation.images.cam_left_wrist": self._image_or_zero(handle, "left_camera", frame_index),
            "observation.images.cam_right_wrist": self._image_or_zero(handle, "right_camera", frame_index),
            "prompt": self._sample_prompt(segment, frame_index),
        }

    def _sample_prompt(self, segment: SegmentRecord, frame_index: int) -> str:
        if not segment.prompt_variants or self.prompt_variant_probability <= 0.0:
            return segment.prompt
        digest = hashlib.sha256(f"{segment.record_id}:{frame_index}:prompt-variant-v1".encode()).digest()
        draw = int.from_bytes(digest[:8], "big") / float(2**64)
        if draw >= self.prompt_variant_probability:
            return segment.prompt
        return segment.prompt_variants[int.from_bytes(digest[8:16], "big") % len(segment.prompt_variants)]

    def _image_or_zero(self, handle: Any, camera: str, frame_index: int) -> np.ndarray:
        if self.skip_images:
            return np.zeros((3, 8, 8), dtype=np.uint8)
        import cv2

        encoded = bytes(handle[f"observation/{camera}/rgb"][frame_index]).rstrip(b"\0")
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to decode {camera} frame {frame_index}")
        return np.transpose(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (2, 0, 1))

    def _get_hdf5(self, path: Path):
        import h5py

        key = str(path)
        handle = self._hdf5_cache.get(key)
        if handle is None:
            handle = h5py.File(path, "r")
            self._hdf5_cache[key] = handle
        self._hdf5_cache.move_to_end(key)
        while len(self._hdf5_cache) > 2:
            _, old = self._hdf5_cache.popitem(last=False)
            old.close()
        return handle

    def _resolve_hdf5_path(self, payload: dict[str, Any], task: str, episode: int) -> Path:
        local = self.dataset_root / "raw" / task / "data" / f"episode{episode}.hdf5"
        if local.is_file():
            return local
        configured = Path(str(payload.get("hdf5_path") or "")).expanduser()
        if configured.is_file():
            return configured.resolve()
        raise FileNotFoundError(f"missing HDF5 for {task}/episode{episode}: {local}")

    @staticmethod
    def _segment_prompt(
        segment: dict[str, Any],
        repair: dict[str, Any] | None,
        record_id: str,
        *,
        require_accepted_repair: bool,
    ) -> str:
        if repair is not None and str(repair.get("status") or "") == "accepted":
            fingerprint = hashlib.sha256(
                json.dumps(segment, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest()
            if str(repair.get("source_fingerprint") or "") != fingerprint:
                raise ValueError(f"stale repair ledger record: {record_id}")
            replacement = repair.get("repair")
            if not isinstance(replacement, dict) or not str(replacement.get("instruction") or "").strip():
                raise ValueError(f"accepted repair has no instruction: {record_id}")
            return str(replacement["instruction"]).strip()
        if require_accepted_repair:
            return ""
        return next(
            (
                str(segment.get(field) or "").strip()
                for field in ("polished_instruction", "canonical_instruction", "raw_canonical_instruction")
                if str(segment.get(field) or "").strip()
            ),
            "",
        )

    @staticmethod
    def _load_split(path: str | Path | None, split: str | None) -> dict[str, set[int]] | None:
        if path is None and split is None:
            return None
        if path is None or split not in {"train", "val"}:
            raise ValueError("split manifest and split name ('train' or 'val') are required together")
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        field = f"{split}_episode_indices"
        tasks = payload.get("tasks")
        if not isinstance(tasks, dict) or not tasks:
            raise ValueError("split manifest must contain a non-empty tasks object")
        return {str(task): {int(index) for index in info[field]} for task, info in tasks.items()}

    @staticmethod
    def _load_jsonl_index(path: str | Path | None) -> dict[str, dict[str, Any]]:
        groups = RoboTwinSubtaskDataset._load_jsonl_groups(path)
        return {key: records[-1] for key, records in groups.items()}

    @staticmethod
    def _load_jsonl_groups(path: str | Path | None) -> dict[str, list[dict[str, Any]]]:
        if path is None or not str(path).strip():
            return {}
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        result: dict[str, list[dict[str, Any]]] = {}
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = str(record.get("record_id") or "").strip()
            if not record_id:
                raise ValueError(f"missing record_id at {source}:{line_number}")
            result.setdefault(record_id, []).append(record)
        return result

    @staticmethod
    def _validate_variants(
        records: list[dict[str, Any]], canonical: str, record_id: str
    ) -> tuple[str, ...]:
        expected = hashlib.sha256(canonical.encode()).hexdigest()
        result: list[str] = []
        for record in records:
            if str(record.get("canonical_instruction_sha256") or "") != expected:
                raise ValueError(f"stale prompt variant: {record_id}")
            value = str(record.get("planner_instruction") or "").strip()
            if value.casefold() in INVALID_PROMPTS:
                raise ValueError(f"invalid prompt variant: {record_id}")
            if value != canonical and value not in result:
                result.append(value)
        return tuple(result)

    @staticmethod
    def _metadata_sort_key(path: Path) -> tuple[str, int, str]:
        match = re.search(r"episode(\d+)$", path.stem)
        return path.parent.name, int(match.group(1)) if match else 10**9, path.name
