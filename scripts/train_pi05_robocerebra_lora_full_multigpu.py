#!/usr/bin/env python3
"""Full-data RoboCerebra LeRobot pi0.5 LoRA trainer with data sharding.

This smoke trainer reads LeRobot parquet + mp4 shards directly. It does not use
pre-exported PNG samples, does not download raw RoboCerebra, and does not do
full-parameter fine-tuning.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import dataclasses
import functools
import gc
import json
import os
import pickle
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_pi05_robocerebra_lora_minimal import (  # noqa: E402
    DEFAULT_BASE_PARAMS,
    DEFAULT_OPENPI_ROOT,
    add_openpi_paths,
    load_norm_stats,
    normalize,
    pad_dim,
    save_lora_checkpoint,
    unnormalize,
)


DEFAULT_DATA_ROOT = Path("/mnt/raid1/mjh/datasets/robocerebra_lerobot_unified")
DEFAULT_INDEX_JSONL = PROJECT_ROOT / "outputs/robocerebra_lerobot_full_index.jsonl"
DEFAULT_NORM_STATS = PROJECT_ROOT / "outputs/openpi_assets/robocerebra_unified_full/norm_stats.json"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "outputs/pi05_robocerebra_lora_full_6gpu_smoke_20step"


def make_local_train_state_class():
    from flax import struct

    @struct.dataclass
    class LocalTrainState:
        step: Any
        params: Any
        model_def: Any
        opt_state: Any
        tx: Any = struct.field(pytree_node=False)
        ema_decay: float | None = struct.field(pytree_node=False, default=None)
        ema_params: Any = None

    return LocalTrainState


@dataclass
class EpisodeRecord:
    episode_index: int
    task_index: int
    task_text: str
    dataset_from_index: int
    dataset_to_index: int
    num_frames: int
    front_video_path: Path
    wrist_video_path: Path
    front_video_timestamp_start: float
    wrist_video_timestamp_start: float
    transition_frames: list[int] | None = None
    late_start_frame: int = 0


class FullLeRobotDataset:
    def __init__(
        self,
        data_root: Path,
        index_jsonl: Path,
        *,
        action_horizon: int,
        fps: float = 20.0,
    ) -> None:
        self.data_root = data_root
        self.index_jsonl = index_jsonl
        self.action_horizon = action_horizon
        self.fps = fps
        self.records = self._load_records(index_jsonl)
        parquet_start = time.perf_counter()
        self.states, self.actions = self._load_arrays(data_root)
        self.parquet_read_time = time.perf_counter() - parquet_start
        if self.states.shape[-1] != 8:
            raise ValueError(f"Expected state dim 8, got {self.states.shape}")
        if self.actions.shape[-1] != 7:
            raise ValueError(f"Expected action dim 7, got {self.actions.shape}")
        self._build_phase_indices()

    def _load_records(self, index_jsonl: Path) -> list[EpisodeRecord]:
        records: list[EpisodeRecord] = []
        with index_jsonl.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                records.append(
                    EpisodeRecord(
                        episode_index=int(row["episode_index"]),
                        task_index=int(row["task_index"]),
                        task_text=str(row["task_text"]),
                        dataset_from_index=int(row["dataset_from_index"]),
                        dataset_to_index=int(row["dataset_to_index"]),
                        num_frames=int(row["num_frames"]),
                        front_video_path=Path(row["front_video_path"]),
                        wrist_video_path=Path(row["wrist_video_path"]),
                        front_video_timestamp_start=float(row["front_video_timestamp_start"]),
                        wrist_video_timestamp_start=float(row["wrist_video_timestamp_start"]),
                    )
                )
        if not records:
            raise FileNotFoundError(f"No records in {index_jsonl}")
        return records

    def _load_arrays(self, data_root: Path) -> tuple[np.ndarray, np.ndarray]:
        parquet_paths = sorted((data_root / "data").glob("chunk-*/file-*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(f"No data parquet under {data_root / 'data'}")
        frames = [pd.read_parquet(path) for path in parquet_paths]
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        df = df.sort_values("index").reset_index(drop=True)
        states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
        actions = np.stack(df["action"].to_numpy()).astype(np.float32)
        return states, actions

    def _build_phase_indices(self, *, transition_threshold: float = 0.5, transition_window: int = 8) -> None:
        for record in self.records:
            action_slice = self.actions[record.dataset_from_index : record.dataset_to_index]
            gripper = action_slice[:, 6]
            centers = np.where(np.abs(np.diff(gripper)) >= transition_threshold)[0] + 1
            frames: set[int] = set()
            for center in centers.tolist():
                lo = max(0, int(center) - transition_window)
                hi = min(record.num_frames, int(center) + transition_window + 1)
                frames.update(range(lo, hi))
            record.transition_frames = sorted(frames)
            record.late_start_frame = max(0, int(record.num_frames * 0.65))

    def split_records(
        self,
        val_ratio: float,
        split_seed: int,
        *,
        split_mode: str = "task",
    ) -> tuple[list[EpisodeRecord], list[EpisodeRecord], dict[str, Any]]:
        if split_mode == "episode":
            train, val, metadata = self._split_records_by_episode(val_ratio, split_seed)
        elif split_mode == "task":
            train, val, metadata = self._split_records_by_task(val_ratio, split_seed)
        else:
            raise ValueError(f"Unsupported split_mode: {split_mode}")
        return train, val, metadata

    def _split_records_by_episode(
        self,
        val_ratio: float,
        split_seed: int,
    ) -> tuple[list[EpisodeRecord], list[EpisodeRecord], dict[str, Any]]:
        indices = list(range(len(self.records)))
        rng = random.Random(split_seed)
        rng.shuffle(indices)
        num_val = max(1, int(round(len(indices) * val_ratio))) if val_ratio > 0 else 0
        num_val = min(num_val, len(indices) - 1)
        val_ids = set(indices[:num_val])
        train = [record for i, record in enumerate(self.records) if i not in val_ids]
        val = [record for i, record in enumerate(self.records) if i in val_ids]
        metadata = {
            "split_mode": "episode",
            "num_train_tasks": len({record.task_index for record in train}),
            "num_val_tasks": len({record.task_index for record in val}),
            "train_task_indices": sorted({record.task_index for record in train}),
            "val_task_indices": sorted({record.task_index for record in val}),
        }
        return train, val, metadata

    def _split_records_by_task(
        self,
        val_ratio: float,
        split_seed: int,
    ) -> tuple[list[EpisodeRecord], list[EpisodeRecord], dict[str, Any]]:
        task_indices = sorted({record.task_index for record in self.records})
        rng = random.Random(split_seed)
        rng.shuffle(task_indices)
        if len(task_indices) <= 1:
            return self._split_records_by_episode(val_ratio, split_seed)
        num_val = max(1, int(round(len(task_indices) * val_ratio))) if val_ratio > 0 else 0
        num_val = min(num_val, len(task_indices) - 1)
        val_tasks = set(task_indices[:num_val])
        train = [record for record in self.records if record.task_index not in val_tasks]
        val = [record for record in self.records if record.task_index in val_tasks]
        metadata = {
            "split_mode": "task",
            "num_total_tasks": len(task_indices),
            "num_train_tasks": len({record.task_index for record in train}),
            "num_val_tasks": len({record.task_index for record in val}),
            "train_task_indices": sorted({record.task_index for record in train}),
            "val_task_indices": sorted(val_tasks),
        }
        return train, val, metadata


class VideoDecoder:
    def __init__(self, image_size: int, *, fps: float = 20.0, frame_cache_size: int = 512) -> None:
        self.image_size = image_size
        self.fps = fps
        self.frame_cache_size = frame_cache_size
        self._containers: dict[Path, tuple[Any, Any]] = {}
        self._frame_cache: OrderedDict[tuple[Path, int], np.ndarray] = OrderedDict()
        self.reset_stats()

    def reset_stats(self) -> None:
        self.stats = {
            "video_decode_time": 0.0,
            "container_open_time": 0.0,
            "decode_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "containers_opened": 0,
        }

    def snapshot_stats(self) -> dict[str, Any]:
        return dict(self.stats)

    def _get_container(self, video_path: Path):
        cached = self._containers.get(video_path)
        if cached is not None:
            return cached
        start = time.perf_counter()
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        self.stats["container_open_time"] += time.perf_counter() - start
        self.stats["containers_opened"] += 1
        self._containers[video_path] = (container, stream)
        return container, stream

    def _cache_get(self, key: tuple[Path, int]) -> np.ndarray | None:
        value = self._frame_cache.get(key)
        if value is None:
            return None
        self._frame_cache.move_to_end(key)
        return value

    def _cache_put(self, key: tuple[Path, int], value: np.ndarray) -> None:
        if self.frame_cache_size <= 0:
            return
        self._frame_cache[key] = value
        self._frame_cache.move_to_end(key)
        while len(self._frame_cache) > self.frame_cache_size:
            self._frame_cache.popitem(last=False)

    def decode_rgb(self, video_path: Path, timestamp: float) -> np.ndarray:
        self.stats["decode_calls"] += 1
        frame_key = int(round(timestamp * self.fps))
        cache_key = (video_path, frame_key)
        cached = self._cache_get(cache_key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached

        self.stats["cache_misses"] += 1
        start = time.perf_counter()
        container, stream = self._get_container(video_path)
        try:
            seek_pts = int(timestamp / float(stream.time_base))
            container.seek(seek_pts, any_frame=False, backward=True, stream=stream)
        except Exception:
            container.seek(0)
        selected = None
        for i, frame in enumerate(container.decode(stream)):
            selected = frame
            if frame.pts is not None:
                frame_time = float(frame.pts * stream.time_base)
                if frame_time + 1e-6 >= timestamp:
                    break
            if i > 600:
                break
        if selected is None:
            raise RuntimeError(f"No frame decoded from {video_path} at {timestamp}")
        rgb = selected.to_ndarray(format="rgb24")
        if rgb.shape[0] != self.image_size or rgb.shape[1] != self.image_size:
            rgb = np.asarray(
                Image.fromarray(rgb).resize((self.image_size, self.image_size), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
        out = rgb.astype(np.uint8, copy=False)
        self._cache_put(cache_key, out)
        self.stats["video_decode_time"] += time.perf_counter() - start
        return out

    def close(self) -> None:
        for container, _ in self._containers.values():
            container.close()
        self._containers.clear()
        self._frame_cache.clear()


def gpu_memory() -> dict[str, Any]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        rows = []
        for line in out.strip().splitlines():
            idx, used, total = [part.strip() for part in line.split(",")]
            rows.append({"index": int(idx), "used_mb": int(used), "total_mb": int(total)})
        return {"gpus": rows, "max_used_mb": max((r["used_mb"] for r in rows), default=0)}
    except Exception as exc:
        return {"error": repr(exc)}


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in {"1", "true", "yes", "y"}:
        return True
    if value.lower() in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool: {value}")


def sample_indices(
    records: list[EpisodeRecord],
    batch_size: int,
    rng: random.Random,
    *,
    sampler_mode: str = "uniform",
    ordinary_ratio: float = 0.5,
    transition_ratio: float = 0.25,
) -> list[tuple[EpisodeRecord, int]]:
    out = []
    for _ in range(batch_size):
        record = records[rng.randrange(len(records))]
        if sampler_mode == "phase_balanced":
            p = rng.random()
            if p < transition_ratio and record.transition_frames:
                frame_i = record.transition_frames[rng.randrange(len(record.transition_frames))]
            elif p < transition_ratio + ordinary_ratio:
                frame_i = rng.randrange(record.num_frames)
            else:
                lo = min(record.late_start_frame, max(record.num_frames - 1, 0))
                frame_i = rng.randrange(lo, record.num_frames)
        else:
            frame_i = rng.randrange(record.num_frames)
        out.append((record, frame_i))
    return out


def sample_phase_indices(
    records: list[EpisodeRecord],
    batch_size: int,
    rng: random.Random,
    phase: str,
) -> list[tuple[EpisodeRecord, int]]:
    out = []
    for _ in range(batch_size):
        for _attempt in range(20):
            record = records[rng.randrange(len(records))]
            if phase == "gripper_transition":
                if not record.transition_frames:
                    continue
                frame_i = record.transition_frames[rng.randrange(len(record.transition_frames))]
            elif phase == "late":
                lo = min(record.late_start_frame, max(record.num_frames - 1, 0))
                frame_i = rng.randrange(lo, record.num_frames)
            elif phase == "approach":
                hi = record.transition_frames[0] if record.transition_frames else max(1, int(record.num_frames * 0.6))
                hi = max(1, min(hi, record.num_frames))
                frame_i = rng.randrange(0, hi)
            else:
                raise ValueError(f"Unknown phase: {phase}")
            out.append((record, frame_i))
            break
        else:
            record = records[rng.randrange(len(records))]
            out.append((record, rng.randrange(record.num_frames)))
    return out


def build_full_batch(
    dataset: FullLeRobotDataset,
    records_and_frames: list[tuple[EpisodeRecord, int]],
    *,
    decoder: VideoDecoder,
    tokenizer,
    norm_stats: dict[str, Any],
    model_config,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    from openpi.models import model as openpi_model

    images = {"base_0_rgb": [], "left_wrist_0_rgb": [], "right_wrist_0_rgb": []}
    image_masks = {"base_0_rgb": [], "left_wrist_0_rgb": [], "right_wrist_0_rgb": []}
    states = []
    tokenized = []
    token_masks = []
    actions = []
    raw_actions = []
    action_masks = []
    items = []

    for record, frame_i in records_and_frames:
        data_i = record.dataset_from_index + frame_i
        state = dataset.states[data_i]
        chunk = np.zeros((dataset.action_horizon, dataset.actions.shape[-1]), dtype=np.float32)
        raw_chunk = np.zeros_like(chunk)
        mask = np.zeros((dataset.action_horizon,), dtype=np.float32)
        end_i = min(data_i + dataset.action_horizon, record.dataset_to_index)
        valid = end_i - data_i
        chunk[:valid] = dataset.actions[data_i:end_i]
        raw_chunk[:valid] = dataset.actions[data_i:end_i]
        mask[:valid] = 1.0

        norm_state = normalize(state, norm_stats["state"]).astype(np.float32)
        norm_actions = normalize(chunk, norm_stats["actions"]).astype(np.float32)
        padded_state = pad_dim(norm_state[None, :], model_config.action_dim)[0].astype(np.float32)
        padded_actions = pad_dim(norm_actions, model_config.action_dim).astype(np.float32)
        tokens, masks = tokenizer.tokenize(record.task_text, norm_state)

        offset_sec = frame_i / dataset.fps
        front = decoder.decode_rgb(record.front_video_path, record.front_video_timestamp_start + offset_sec)
        wrist = decoder.decode_rgb(record.wrist_video_path, record.wrist_video_timestamp_start + offset_sec)
        images["base_0_rgb"].append(front)
        images["left_wrist_0_rgb"].append(wrist)
        images["right_wrist_0_rgb"].append(np.zeros_like(front))
        image_masks["base_0_rgb"].append(True)
        image_masks["left_wrist_0_rgb"].append(True)
        image_masks["right_wrist_0_rgb"].append(False)
        states.append(padded_state)
        tokenized.append(tokens.astype(np.int32))
        token_masks.append(masks.astype(bool))
        actions.append(padded_actions)
        raw_actions.append(raw_chunk)
        action_masks.append(mask)
        items.append(
            {
                "episode_index": record.episode_index,
                "frame_index": frame_i,
                "dataset_index": data_i,
                "valid_actions": valid,
            }
        )

    batch_dict = {
        "image": {key: np.stack(value, axis=0) for key, value in images.items()},
        "image_mask": {key: np.asarray(value, dtype=bool) for key, value in image_masks.items()},
        "state": np.stack(states, axis=0),
        "tokenized_prompt": np.stack(tokenized, axis=0),
        "tokenized_prompt_mask": np.stack(token_masks, axis=0),
    }
    observation = openpi_model.Observation.from_dict(batch_dict)
    meta = {
        "items": items,
        "raw_actions": np.stack(raw_actions, axis=0),
        "action_mask": np.stack(action_masks, axis=0),
    }
    return observation, np.stack(actions, axis=0), meta


def put_observation_sharded(observation, mesh, batch_spec):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from openpi.models import model as openpi_model

    image_sharding = NamedSharding(mesh, P(batch_spec, None, None, None))
    vector_sharding = NamedSharding(mesh, P(batch_spec, None))
    return openpi_model.Observation(
        images={key: jax.device_put(value, image_sharding) for key, value in observation.images.items()},
        image_masks={key: jax.device_put(value, NamedSharding(mesh, P(batch_spec))) for key, value in observation.image_masks.items()},
        state=jax.device_put(observation.state, vector_sharding),
        tokenized_prompt=jax.device_put(observation.tokenized_prompt, vector_sharding),
        tokenized_prompt_mask=jax.device_put(observation.tokenized_prompt_mask, vector_sharding),
        token_ar_mask=None if observation.token_ar_mask is None else jax.device_put(observation.token_ar_mask, vector_sharding),
        token_loss_mask=None
        if observation.token_loss_mask is None
        else jax.device_put(observation.token_loss_mask, vector_sharding),
    )


def put_actions_sharded(actions: np.ndarray, mesh, batch_spec):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    return jax.device_put(actions, NamedSharding(mesh, P(batch_spec, None, None)))


def put_action_mask_sharded(action_mask: np.ndarray, mesh, batch_spec):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    return jax.device_put(action_mask, NamedSharding(mesh, P(batch_spec, None)))


def sharding_report(array) -> dict[str, Any]:
    shards = getattr(array, "addressable_shards", [])
    return {
        "sharding": str(getattr(array, "sharding", None)),
        "num_addressable_shards": len(shards),
        "devices": [str(shard.device) for shard in shards],
        "shard_shapes": [list(shard.data.shape) for shard in shards],
    }


def masked_raw_mse(pred_raw: np.ndarray, target_raw: np.ndarray, mask: np.ndarray) -> tuple[float, list[float]]:
    sq = (pred_raw - target_raw) ** 2
    mask3 = mask[..., None]
    denom = np.maximum(mask3.sum(axis=(0, 1)), 1.0)
    per_dim = (sq * mask3).sum(axis=(0, 1)) / denom
    total = float(per_dim.mean())
    return total, per_dim.astype(float).tolist()


def max_gpu_memory_gb(mem: dict[str, Any]) -> float | None:
    value = mem.get("max_used_mb")
    if value is None:
        return None
    return float(value) / 1024.0


def parse_float_list(value: str) -> list[float]:
    try:
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Invalid comma-separated float list: {value}") from exc


def build_action_dim_weights(raw_weights: list[float], *, model_action_dim: int, padded_weight: float) -> np.ndarray:
    if len(raw_weights) != 7:
        raise ValueError(f"Expected 7 raw action weights, got {len(raw_weights)}: {raw_weights}")
    weights = np.full((model_action_dim,), float(padded_weight), dtype=np.float32)
    weights[:7] = np.asarray(raw_weights, dtype=np.float32)
    return weights


def param_state_sharding_report(params, limit: int = 12) -> list[dict[str, Any]]:
    import jax

    leaves = []
    for path, value in jax.tree_util.tree_flatten_with_path(params)[0]:
        array = getattr(value, "value", value)
        if not hasattr(array, "shape"):
            continue
        leaves.append(
            {
                "path": jax.tree_util.keystr(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sharding": str(getattr(array, "sharding", None)),
            }
        )
        if len(leaves) >= limit:
            break
    return leaves


def train_state_param_sharding_report(train_state, limit: int = 12) -> list[dict[str, Any]]:
    return param_state_sharding_report(train_state.params, limit=limit)


def init_fsdp_train_state(config, init_rng, mesh, *, init_lora_path: Path | None = None):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import flax.traverse_util as traverse_util
    import optax
    import openpi.shared.nnx_utils as nnx_utils
    import openpi.training.sharding as sharding

    tx = optax.adamw(config.lr_schedule.create())
    LocalTrainState = make_local_train_state_class()

    def init(rng, partial_params=None):
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )
        return LocalTrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=None,
            ema_params=None,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    train_state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)
    partial_params = config.weight_loader.load(train_state_shape.params.to_pure_dict())
    if init_lora_path is not None:
        if not init_lora_path.exists():
            raise FileNotFoundError(f"init_lora_path not found: {init_lora_path}")
        with init_lora_path.open("rb") as f:
            init_lora_params = pickle.load(f)
        partial_params = traverse_util.unflatten_dict(
            {
                **traverse_util.flatten_dict(partial_params),
                **traverse_util.flatten_dict(init_lora_params),
            }
        )
    partial_params = traverse_util.unflatten_dict(
        {
            key: value
            for key, value in traverse_util.flatten_dict(partial_params).items()
            if not isinstance(value, jax.ShapeDtypeStruct)
        }
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=train_state_sharding,
    )(init_rng, partial_params)
    return train_state, train_state_sharding


def flow_loss_per_dim(model, rng_key, observation, actions, *, train: bool):
    import jax
    import jax.numpy as jnp
    from openpi.models import model as openpi_model
    from openpi.models.pi0 import make_attn_mask

    preprocess_rng, noise_rng, time_rng = jax.random.split(rng_key, 3)
    observation = openpi_model.preprocess_observation(preprocess_rng, observation, train=train)

    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
    )
    v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
    return jnp.square(v_t - u_t)


def reduce_flow_loss(loss_per_dim, action_mask, action_dim_weights):
    import jax.numpy as jnp

    dim_weights = jnp.asarray(action_dim_weights, dtype=loss_per_dim.dtype)
    valid = action_mask[..., None].astype(loss_per_dim.dtype)
    weighted_mask = valid * dim_weights[None, None, :]
    loss = jnp.sum(loss_per_dim * weighted_mask) / jnp.clip(jnp.sum(weighted_mask), 1.0)

    real_mask = valid * (jnp.arange(loss_per_dim.shape[-1]) < 7)[None, None, :]
    pad_mask = valid * (jnp.arange(loss_per_dim.shape[-1]) >= 7)[None, None, :]
    real7_loss = jnp.sum(loss_per_dim * real_mask) / jnp.clip(jnp.sum(real_mask), 1.0)
    padded_loss = jnp.sum(loss_per_dim * pad_mask) / jnp.clip(jnp.sum(pad_mask), 1.0)
    return loss, real7_loss, padded_loss


def train_step(config, action_dim_weights, rng, state, batch):
    import dataclasses as _dataclasses

    import jax
    import jax.numpy as jnp
    from flax import nnx
    import optax

    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(train_model, rng_key, observation, actions, action_mask):
        loss_per_dim = flow_loss_per_dim(train_model, rng_key, observation, actions, train=True)
        loss, real7_loss, padded_loss = reduce_flow_loss(loss_per_dim, action_mask, action_dim_weights)
        return loss, (real7_loss, padded_loss)

    observation, actions, action_mask = batch
    train_rng = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, (real7_loss, padded_loss)), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions, action_mask
    )
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_trainable_params = optax.apply_updates(params, updates)
    nnx.update(model, new_trainable_params)
    new_state = _dataclasses.replace(
        state,
        step=state.step + 1,
        params=nnx.state(model),
        opt_state=new_opt_state,
    )
    return new_state, {"loss": loss, "real7_loss": real7_loss, "padded_loss": padded_loss}


def eval_step(config, action_dim_weights, rng, state, batch, *, sample_action_steps: int):
    from flax import nnx

    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, actions, action_mask = batch
    loss_per_dim = flow_loss_per_dim(model, rng, observation, actions, train=False)
    loss, real7_loss, padded_loss = reduce_flow_loss(loss_per_dim, action_mask, action_dim_weights)
    pred_actions = model.sample_actions(rng, observation, num_steps=sample_action_steps)
    return {"loss": loss, "real7_loss": real7_loss, "padded_loss": padded_loss, "pred_actions": pred_actions}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--index_jsonl", type=Path, default=DEFAULT_INDEX_JSONL)
    parser.add_argument("--norm_stats", type=Path, default=DEFAULT_NORM_STATS)
    parser.add_argument("--base_ckpt", type=Path, default=DEFAULT_BASE_PARAMS)
    parser.add_argument("--init_lora_path", type=Path, default=None)
    parser.add_argument("--save_dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--openpi_root", type=Path, default=DEFAULT_OPENPI_ROOT)
    parser.add_argument("--num_devices", type=int, default=6)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_mode", choices=["task", "episode"], default="task")
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_batches", type=int, default=2)
    parser.add_argument(
        "--phase_eval_batches",
        type=int,
        default=0,
        help="Extra eval batches per phase for approach/gripper_transition/late. 0 disables phase-wise eval.",
    )
    parser.add_argument("--save_final_only", type=parse_bool, default=True)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--action_horizon", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--frame_cache_size", type=int, default=512)
    parser.add_argument("--sample_action_steps", type=int, default=1)
    parser.add_argument("--sampler_mode", choices=["uniform", "phase_balanced"], default="uniform")
    parser.add_argument("--ordinary_ratio", type=float, default=0.5)
    parser.add_argument("--transition_ratio", type=float, default=0.25)
    parser.add_argument(
        "--action_dim_weights",
        type=parse_float_list,
        default=parse_float_list("1,1,1,1,1,1,1"),
        help="Comma-separated weights for RoboCerebra raw 7D action dims.",
    )
    parser.add_argument(
        "--padded_action_weight",
        type=float,
        default=0.0,
        help="Loss weight for padded pi0.5 action dims 7:32. Default excludes padding from the training loss.",
    )
    parser.add_argument(
        "--clear_jax_caches",
        type=parse_bool,
        default=False,
        help="Debug-only. Clearing JAX caches every step can force recompilation and destroy steady-state speed.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    add_openpi_paths(args.openpi_root)
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from flax import nnx
    import openpi.shared.nnx_utils as nnx_utils
    from openpi.models.tokenizer import PaligemmaTokenizer
    from scripts.openpi_robocerebra_config import make_pi05_robocerebra_lora_config
    import openpi.training.optimizer as openpi_optimizer
    import openpi.training.sharding as sharding
    import openpi.training.weight_loaders as weight_loaders

    jax_devices = jax.devices()
    if len(jax_devices) < args.num_devices:
        raise RuntimeError(f"Requested {args.num_devices} devices, but JAX sees {len(jax_devices)}: {jax_devices}")
    used_devices = jax_devices[: args.num_devices]
    global_batch_size = args.num_devices * args.per_device_batch_size
    if global_batch_size % args.num_devices != 0:
        raise ValueError("global batch size must be divisible by num_devices")
    if not args.base_ckpt.exists():
        raise FileNotFoundError(f"Official pi0.5 base checkpoint not found: {args.base_ckpt}")
    if len(jax_devices) != args.num_devices:
        raise RuntimeError(
            "This smoke expects exactly the requested devices to be visible. "
            f"JAX sees {len(jax_devices)} devices; set CUDA_VISIBLE_DEVICES=0,1,2,3,4,5."
        )
    mesh = sharding.make_mesh(args.num_devices)
    data_sharding = NamedSharding(mesh, P(sharding.DATA_AXIS))
    replicated_sharding = NamedSharding(mesh, P())

    args.save_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.save_dir / "train_log.jsonl"
    summary_path = args.save_dir / "summary.json"
    config_path = args.save_dir / "config.json"
    lora_path = args.save_dir / "lora_params.pkl"

    dataset = FullLeRobotDataset(args.data_root, args.index_jsonl, action_horizon=args.action_horizon)
    train_records, val_records, split_metadata = dataset.split_records(
        args.val_ratio,
        args.split_seed,
        split_mode=args.split_mode,
    )
    norm_stats = load_norm_stats(args.norm_stats)
    decoder = VideoDecoder(args.image_size, fps=dataset.fps, frame_cache_size=args.frame_cache_size)

    config = make_pi05_robocerebra_lora_config(
        assets_base_dir=str(args.norm_stats.parent.parent),
        checkpoint_base_dir=str(args.save_dir / "openpi_checkpoints_unused"),
        batch_size=global_batch_size,
        num_train_steps=args.steps,
    )
    config = dataclasses.replace(
        config,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(args.base_ckpt)),
        lr_schedule=openpi_optimizer.CosineDecaySchedule(
            warmup_steps=1,
            peak_lr=args.lr,
            decay_steps=max(args.steps, 2),
            decay_lr=args.lr,
        ),
        freeze_filter=nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(".*lora.*"))),
        ema_decay=None,
        fsdp_devices=args.num_devices,
    )
    tokenizer = PaligemmaTokenizer(config.model.max_token_len)
    action_dim_weights = build_action_dim_weights(
        args.action_dim_weights,
        model_action_dim=config.model.action_dim,
        padded_weight=args.padded_action_weight,
    )
    train_state, train_state_sharding = init_fsdp_train_state(
        config,
        jax.random.key(args.seed + 123),
        mesh,
        init_lora_path=args.init_lora_path,
    )
    jax.block_until_ready(train_state)
    ptrain_step = jax.jit(
        functools.partial(train_step, config, action_dim_weights),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    peval_step = jax.jit(
        functools.partial(eval_step, config, action_dim_weights, sample_action_steps=args.sample_action_steps),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    rng = random.Random(args.seed)
    eval_rng = random.Random(args.seed + 10_000)
    train_losses: list[float] = []
    val_losses: list[float] = []
    raw_mses: list[float] = []
    raw_mse_per_dim_values: list[list[float]] = []
    data_times: list[float] = []
    step_times: list[float] = []
    total_step_times: list[float] = []
    sample_times: list[float] = []
    batch_build_times: list[float] = []
    shard_times: list[float] = []
    video_decode_times: list[float] = []
    train_step_times: list[float] = []
    eval_times: list[float] = []
    eval_compute_times: list[float] = []
    phase_eval_history: list[dict[str, Any]] = []
    nan_inf_detected = False
    first_sharding: dict[str, Any] | None = None
    failure_reason: str | None = None

    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run_config = {
        **serializable_args,
        "data_root": str(args.data_root),
        "index_jsonl": str(args.index_jsonl),
        "norm_stats": str(args.norm_stats),
        "base_ckpt": str(args.base_ckpt),
        "init_lora_path": str(args.init_lora_path) if args.init_lora_path else None,
        "save_dir": str(args.save_dir),
        "jax_devices": [str(d) for d in jax_devices],
        "actual_used_devices": [str(d) for d in used_devices],
        "mesh_shape": dict(mesh.shape),
        "mesh_axis_names": list(mesh.axis_names),
        "global_batch_size": global_batch_size,
        "action_dim_weights": action_dim_weights.astype(float).tolist(),
        "raw_action_dim_weights": args.action_dim_weights,
        "padded_action_weight": args.padded_action_weight,
        "sampler_mode": args.sampler_mode,
        "ordinary_ratio": args.ordinary_ratio,
        "transition_ratio": args.transition_ratio,
        "phase_eval_batches": args.phase_eval_batches,
        "phase_eval_batches": args.phase_eval_batches,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "train_episodes": len(train_records),
        "val_episodes": len(val_records),
        "split_metadata": split_metadata,
        "total_frames": int(dataset.states.shape[0]),
        "parquet_read_time": dataset.parquet_read_time,
        "train_state_param_sharding_sample": train_state_param_sharding_report(train_state),
        "trainable_param_sharding_sample": param_state_sharding_report(train_state.params.filter(config.trainable_filter)),
    }
    config_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def make_sharded_batch(records: list[EpisodeRecord], local_rng: random.Random, *, phase: str | None = None):
        data_start = time.perf_counter()
        sample_start = time.perf_counter()
        if phase is None:
            batch_keys = sample_indices(
                records,
                global_batch_size,
                local_rng,
                sampler_mode=args.sampler_mode,
                ordinary_ratio=args.ordinary_ratio,
                transition_ratio=args.transition_ratio,
            )
        else:
            batch_keys = sample_phase_indices(records, global_batch_size, local_rng, phase)
        sample_time = time.perf_counter() - sample_start
        decoder.reset_stats()
        build_start = time.perf_counter()
        obs_np, actions_np, meta = build_full_batch(
            dataset,
            batch_keys,
            decoder=decoder,
            tokenizer=tokenizer,
            norm_stats=norm_stats,
            model_config=config.model,
        )
        batch_build_time = time.perf_counter() - build_start
        decoder_stats = decoder.snapshot_stats()
        shard_start = time.perf_counter()
        obs = put_observation_sharded(obs_np, mesh, sharding.DATA_AXIS)
        actions = put_actions_sharded(actions_np, mesh, sharding.DATA_AXIS)
        action_mask = put_action_mask_sharded(meta["action_mask"], mesh, sharding.DATA_AXIS)
        shard_time = time.perf_counter() - shard_start
        data_time = time.perf_counter() - data_start
        timing = {
            "sample_time": sample_time,
            "parquet_read_time": 0.0,
            "video_decode_time": decoder_stats["video_decode_time"],
            "container_open_time": decoder_stats["container_open_time"],
            "video_decode_calls": decoder_stats["decode_calls"],
            "video_cache_hits": decoder_stats["cache_hits"],
            "video_cache_misses": decoder_stats["cache_misses"],
            "containers_opened": decoder_stats["containers_opened"],
            "batch_build_time": batch_build_time,
            "device_put_time": shard_time,
            "shard_time": shard_time,
            "data_time": data_time,
        }
        return obs, actions, action_mask, meta, timing

    def eval_batches(step: int, *, phase: str | None, num_batches: int) -> tuple[float, float, list[float], dict[str, float]]:
        eval_start = time.perf_counter()
        losses = []
        real7_losses = []
        padded_losses = []
        total_raw_mses = []
        per_dim_mses = []
        timing_totals = {
            "eval_time": 0.0,
            "eval_data_time": 0.0,
            "eval_compute_time": 0.0,
            "eval_raw_mse_time": 0.0,
            "eval_video_decode_time": 0.0,
        }
        for batch_i in range(num_batches):
            obs, actions, action_mask, meta, batch_timing = make_sharded_batch(val_records, eval_rng, phase=phase)
            timing_totals["eval_data_time"] += batch_timing["data_time"]
            timing_totals["eval_video_decode_time"] += batch_timing["video_decode_time"]
            compute_start = time.perf_counter()
            with sharding.set_mesh(mesh):
                eval_info = peval_step(
                    jax.random.key(args.seed + 200_000 + step * 100 + batch_i),
                    train_state,
                    (obs, actions, action_mask),
                )
            loss = float(jax.device_get(eval_info["loss"]))
            eval_real7_loss = float(jax.device_get(eval_info["real7_loss"]))
            eval_padded_loss = float(jax.device_get(eval_info["padded_loss"]))
            timing_totals["eval_compute_time"] += time.perf_counter() - compute_start
            losses.append(loss)
            real7_losses.append(eval_real7_loss)
            padded_losses.append(eval_padded_loss)
            raw_start = time.perf_counter()
            pred_norm_np = np.asarray(jax.device_get(eval_info["pred_actions"]))[..., :7]
            pred_raw = unnormalize(pred_norm_np, norm_stats["actions"])
            raw_mse, raw_mse_per_dim = masked_raw_mse(pred_raw, meta["raw_actions"], meta["action_mask"])
            total_raw_mses.append(raw_mse)
            per_dim_mses.append(raw_mse_per_dim)
            timing_totals["eval_raw_mse_time"] += time.perf_counter() - raw_start
            del pred_norm_np, pred_raw
            gc.collect()
            if args.clear_jax_caches:
                jax.clear_caches()
            del obs, actions, action_mask, eval_info
        timing_totals["eval_time"] = time.perf_counter() - eval_start
        timing_totals["eval_real7_loss"] = float(np.mean(real7_losses)) if real7_losses else None
        timing_totals["eval_padded_loss"] = float(np.mean(padded_losses)) if padded_losses else None
        return (
            float(np.mean(losses)),
            float(np.mean(total_raw_mses)),
            np.mean(np.asarray(per_dim_mses), axis=0).astype(float).tolist(),
            timing_totals,
        )

    def eval_random(step: int) -> tuple[float, float, list[float], dict[str, float]]:
        return eval_batches(step, phase=None, num_batches=args.eval_batches)

    def eval_phases(step: int) -> dict[str, Any]:
        if args.phase_eval_batches <= 0:
            return {}
        out: dict[str, Any] = {}
        for phase in ["approach", "gripper_transition", "late"]:
            loss, raw_mse, raw_mse_per_dim, timing = eval_batches(
                step,
                phase=phase,
                num_batches=args.phase_eval_batches,
            )
            out[phase] = {
                "loss": loss,
                "real7_loss": timing.get("eval_real7_loss"),
                "padded_loss": timing.get("eval_padded_loss"),
                "raw_action_mse": raw_mse,
                "raw_action_mse_per_dim": raw_mse_per_dim,
                "eval_time": timing.get("eval_time"),
                "eval_data_time": timing.get("eval_data_time"),
                "eval_compute_time": timing.get("eval_compute_time"),
                "eval_video_decode_time": timing.get("eval_video_decode_time"),
            }
        phase_eval_history.append({"step": step, "phase_eval": out})
        return out

    start_time = time.perf_counter()
    try:
        with log_path.open("w", encoding="utf-8") as log_f:
            for step in range(1, args.steps + 1):
                total_step_start = time.perf_counter()
                obs, actions, action_mask, _, batch_timing = make_sharded_batch(train_records, rng)
                data_time = batch_timing["data_time"]
                if first_sharding is None:
                    first_sharding = {
                        "base_0_rgb": sharding_report(obs.images["base_0_rgb"]),
                        "state": sharding_report(obs.state),
                        "actions": sharding_report(actions),
                        "action_mask": sharding_report(action_mask),
                    }
                    if first_sharding["actions"]["num_addressable_shards"] != args.num_devices:
                        raise RuntimeError(f"Batch was not sharded to {args.num_devices} devices: {first_sharding}")
                step_start = time.perf_counter()
                with sharding.set_mesh(mesh):
                    train_state, train_info = ptrain_step(
                        jax.random.key(args.seed + step),
                        train_state,
                        (obs, actions, action_mask),
                    )
                loss_value = float(jax.device_get(train_info["loss"]))
                real7_loss_value = float(jax.device_get(train_info["real7_loss"]))
                padded_loss_value = float(jax.device_get(train_info["padded_loss"]))
                train_step_time = time.perf_counter() - step_start
                step_time = train_step_time
                train_losses.append(loss_value)
                data_times.append(data_time)
                step_times.append(step_time)
                sample_times.append(batch_timing["sample_time"])
                batch_build_times.append(batch_timing["batch_build_time"])
                shard_times.append(batch_timing["shard_time"])
                video_decode_times.append(batch_timing["video_decode_time"])
                train_step_times.append(train_step_time)
                if not np.isfinite(loss_value):
                    nan_inf_detected = True

                val_loss = None
                raw_mse = None
                raw_mse_per_dim = None
                eval_data_time = 0.0
                eval_time = 0.0
                eval_compute_time = 0.0
                eval_raw_mse_time = 0.0
                eval_video_decode_time = 0.0
                eval_real7_loss = None
                eval_padded_loss = None
                phase_eval = None
                if step % args.eval_interval == 0 or step == args.steps:
                    val_loss, raw_mse, raw_mse_per_dim, eval_timing = eval_random(step)
                    eval_time = eval_timing["eval_time"]
                    eval_data_time = eval_timing["eval_data_time"]
                    eval_compute_time = eval_timing["eval_compute_time"]
                    eval_raw_mse_time = eval_timing["eval_raw_mse_time"]
                    eval_video_decode_time = eval_timing["eval_video_decode_time"]
                    eval_real7_loss = eval_timing.get("eval_real7_loss")
                    eval_padded_loss = eval_timing.get("eval_padded_loss")
                    eval_times.append(eval_time)
                    eval_compute_times.append(eval_compute_time)
                    val_losses.append(val_loss)
                    raw_mses.append(raw_mse)
                    raw_mse_per_dim_values.append(raw_mse_per_dim)
                    if not np.isfinite(val_loss) or not np.isfinite(raw_mse):
                        nan_inf_detected = True
                    phase_eval = eval_phases(step)

                total_step_time = time.perf_counter() - total_step_start
                total_step_times.append(total_step_time)

                mem = gpu_memory()
                row = {
                    "step": step,
                    "train_loss": loss_value,
                    "train_real7_loss": real7_loss_value,
                    "train_padded_loss": padded_loss_value,
                    "val_loss": val_loss,
                    "val_real7_loss": eval_real7_loss,
                    "val_padded_loss": eval_padded_loss,
                    "raw_action_mse": raw_mse,
                    "raw_action_mse_per_dim": raw_mse_per_dim,
                    "phase_eval": phase_eval,
                    "lr": args.lr,
                    "data_time": data_time,
                    "sample_time": batch_timing["sample_time"],
                    "parquet_read_time": 0.0,
                    "video_decode_time": batch_timing["video_decode_time"],
                    "container_open_time": batch_timing["container_open_time"],
                    "video_decode_calls": batch_timing["video_decode_calls"],
                    "video_cache_hits": batch_timing["video_cache_hits"],
                    "video_cache_misses": batch_timing["video_cache_misses"],
                    "containers_opened": batch_timing["containers_opened"],
                    "batch_build_time": batch_timing["batch_build_time"],
                    "device_put_time": batch_timing["device_put_time"],
                    "shard_time": batch_timing["shard_time"],
                    "eval_data_time": eval_data_time,
                    "eval_time": eval_time,
                    "eval_compute_time": eval_compute_time,
                    "eval_raw_mse_time": eval_raw_mse_time,
                    "eval_video_decode_time": eval_video_decode_time,
                    "train_step_time": train_step_time,
                    "step_time": step_time,
                    "total_step_time": total_step_time,
                    "train_update_time": train_step_time,
                    "train_update_minus_data_time": max(train_step_time - data_time, 0.0),
                    "max_gpu_memory_gb": max_gpu_memory_gb(mem),
                    "gpu_memory": mem,
                    "nan_inf_detected": nan_inf_detected,
                    "jax_devices": [str(d) for d in jax_devices],
                    "actual_used_devices": [str(d) for d in used_devices],
                    "global_batch_size": global_batch_size,
                    "checkpoint_path": str(lora_path),
                }
                log_sharding_details = step == 1 or step % args.eval_interval == 0 or step == args.steps
                if log_sharding_details:
                    row.update(
                        {
                            "batch_sharding": first_sharding,
                            "train_state_param_sharding_sample": run_config["train_state_param_sharding_sample"],
                            "trainable_param_sharding_sample": run_config["trainable_param_sharding_sample"],
                        }
                    )
                print(json.dumps(row), flush=True)
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()
                del obs, actions, action_mask
                gc.collect()
                if args.clear_jax_caches:
                    jax.clear_caches()

                if (
                    not args.save_final_only
                    and args.save_interval > 0
                    and step % args.save_interval == 0
                    and step != args.steps
                ):
                    save_lora_checkpoint(
                        args.save_dir / f"lora_params_step_{step:06d}.pkl",
                        train_state.params.filter(config.trainable_filter),
                    )
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"failure_reason": failure_reason}), flush=True)

    if failure_reason is None:
        save_lora_checkpoint(lora_path, train_state.params.filter(config.trainable_filter))

    compile_skip_steps = min(3, len(train_step_times))
    steady_train_step_times = train_step_times[compile_skip_steps:]
    steady_data_times = data_times[compile_skip_steps:]
    steady_video_decode_times = video_decode_times[compile_skip_steps:]
    steady_batch_build_times = batch_build_times[compile_skip_steps:]
    steady_shard_times = shard_times[compile_skip_steps:]
    avg_steady_train_step_time = float(np.mean(steady_train_step_times)) if steady_train_step_times else None
    avg_steady_data_time = float(np.mean(steady_data_times)) if steady_data_times else None
    avg_steady_video_decode_time = float(np.mean(steady_video_decode_times)) if steady_video_decode_times else None
    avg_steady_batch_build_time = float(np.mean(steady_batch_build_times)) if steady_batch_build_times else None
    avg_steady_shard_time = float(np.mean(steady_shard_times)) if steady_shard_times else None
    if avg_steady_train_step_time is not None and avg_steady_data_time is not None:
        if avg_steady_train_step_time > max(avg_steady_data_time * 10.0, 5.0):
            bottleneck = "train_step_compute_fsdp"
        elif avg_steady_video_decode_time is not None and avg_steady_video_decode_time > avg_steady_train_step_time:
            bottleneck = "video_decode"
        else:
            bottleneck = "data_pipeline"
    else:
        bottleneck = "unknown"

    summary = {
        "failure_reason": failure_reason,
        "data_root": str(args.data_root),
        "index_jsonl": str(args.index_jsonl),
        "norm_stats": str(args.norm_stats),
        "base_ckpt": str(args.base_ckpt),
        "init_lora_path": str(args.init_lora_path) if args.init_lora_path else None,
        "steps": args.steps,
        "completed_steps": len(train_losses),
        "num_devices": args.num_devices,
        "per_device_batch_size": args.per_device_batch_size,
        "global_batch_size": global_batch_size,
        "action_dim_weights": action_dim_weights.astype(float).tolist(),
        "raw_action_dim_weights": args.action_dim_weights,
        "padded_action_weight": args.padded_action_weight,
        "sampler_mode": args.sampler_mode,
        "ordinary_ratio": args.ordinary_ratio,
        "transition_ratio": args.transition_ratio,
        "jax_devices": [str(d) for d in jax_devices],
        "actual_used_devices": [str(d) for d in used_devices],
        "mesh_shape": dict(mesh.shape),
        "mesh_axis_names": list(mesh.axis_names),
        "batch_sharding": first_sharding,
        "train_state_param_sharding_sample": run_config["train_state_param_sharding_sample"],
        "trainable_param_sharding_sample": run_config["trainable_param_sharding_sample"],
        "train_episodes": len(train_records),
        "val_episodes": len(val_records),
        "split_metadata": split_metadata,
        "total_frames": int(dataset.states.shape[0]),
        "parquet_read_time": dataset.parquet_read_time,
        "initial_train_loss": train_losses[0] if train_losses else None,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "initial_val_loss": val_losses[0] if val_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "initial_raw_action_mse": raw_mses[0] if raw_mses else None,
        "final_raw_action_mse": raw_mses[-1] if raw_mses else None,
        "final_raw_action_mse_per_dim": raw_mse_per_dim_values[-1] if raw_mse_per_dim_values else None,
        "phase_eval_history": phase_eval_history,
        "final_phase_eval": phase_eval_history[-1]["phase_eval"] if phase_eval_history else None,
        "nan_inf_detected": nan_inf_detected,
        "compile_skip_steps_for_steady_metrics": compile_skip_steps,
        "avg_total_step_time": float(np.mean(total_step_times)) if total_step_times else None,
        "avg_data_time": float(np.mean(data_times)) if data_times else None,
        "avg_sample_time": float(np.mean(sample_times)) if sample_times else None,
        "avg_batch_build_time": float(np.mean(batch_build_times)) if batch_build_times else None,
        "avg_video_decode_time": float(np.mean(video_decode_times)) if video_decode_times else None,
        "avg_shard_time": float(np.mean(shard_times)) if shard_times else None,
        "avg_device_put_time": float(np.mean(shard_times)) if shard_times else None,
        "avg_train_step_time": float(np.mean(train_step_times)) if train_step_times else None,
        "avg_step_time": float(np.mean(step_times)) if step_times else None,
        "avg_eval_time": float(np.mean(eval_times)) if eval_times else None,
        "avg_eval_compute_time": float(np.mean(eval_compute_times)) if eval_compute_times else None,
        "avg_steady_train_step_time_excluding_first_compile_steps": avg_steady_train_step_time,
        "avg_steady_data_time_excluding_first_compile_steps": avg_steady_data_time,
        "avg_steady_video_decode_time_excluding_first_compile_steps": avg_steady_video_decode_time,
        "avg_steady_batch_build_time_excluding_first_compile_steps": avg_steady_batch_build_time,
        "avg_steady_shard_time_excluding_first_compile_steps": avg_steady_shard_time,
        "bottleneck": bottleneck,
        "max_gpu_memory_gb": max_gpu_memory_gb(gpu_memory()),
        "checkpoint_path": str(lora_path) if failure_reason is None else None,
        "config_path": str(config_path),
        "train_log": str(log_path),
        "total_time_sec": time.perf_counter() - start_time,
    }
    decoder.close()
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if failure_reason is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
