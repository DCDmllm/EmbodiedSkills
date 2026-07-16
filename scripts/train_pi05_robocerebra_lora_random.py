#!/usr/bin/env python3
"""Random-batch RoboCerebra LoRA trainer for official pi0.5/OpenPI.

This is the small-scale training path for exported local RoboCerebra visual
episodes. It intentionally bypasses OpenPI's official train.py and LeRobot
loader while preserving the OpenPI/pi0.5 batch schema already smoke-tested in
this project.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_pi05_robocerebra_lora_minimal import (  # noqa: E402
    DEFAULT_BASE_PARAMS,
    DEFAULT_NORM_STATS,
    DEFAULT_OPENPI_ROOT,
    LocalVisualDataset,
    add_openpi_paths,
    build_batch,
    gpu_memory,
    load_norm_stats,
    masked_raw_mse,
    observation_to_jax,
    save_lora_checkpoint,
    unnormalize,
)


DEFAULT_DATA_DIR = PROJECT_ROOT / "outputs/robocerebra_lerobot_visual_sample_200"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "outputs/pi05_robocerebra_lora_random_200ep_1kstep"


class DatasetView:
    def __init__(self, base: LocalVisualDataset, episode_indices: list[int]) -> None:
        self.data_dir = base.data_dir
        self.action_horizon = base.action_horizon
        self.episodes = [base.episodes[i] for i in episode_indices]
        self.index: list[tuple[int, int]] = []
        for ep_i, episode in enumerate(self.episodes):
            for frame_i in range(episode.num_frames):
                self.index.append((ep_i, frame_i))

    def sample_indices(self, batch_size: int, rng: random.Random) -> list[tuple[int, int]]:
        return [self.index[rng.randrange(len(self.index))] for _ in range(batch_size)]

    def first_indices(self, batch_size: int) -> list[tuple[int, int]]:
        return self.index[:batch_size]

    def summary(self) -> dict[str, Any]:
        return {
            "num_episodes": len(self.episodes),
            "num_samples": len(self.index),
            "total_frames": len(self.index),
            "episode_indices": [int(ep.meta["episode_index"]) for ep in self.episodes],
        }


def split_dataset(
    dataset: LocalVisualDataset,
    *,
    val_ratio: float,
    split_seed: int,
) -> tuple[DatasetView, DatasetView]:
    all_indices = list(range(len(dataset.episodes)))
    rng = random.Random(split_seed)
    rng.shuffle(all_indices)
    num_val = max(1, int(round(len(all_indices) * val_ratio))) if val_ratio > 0 else 0
    num_val = min(num_val, max(len(all_indices) - 1, 0))
    val_episode_indices = sorted(all_indices[:num_val])
    train_episode_indices = sorted(all_indices[num_val:])
    if not train_episode_indices:
        raise ValueError("Train split is empty; reduce --val_ratio")
    if not val_episode_indices:
        val_episode_indices = train_episode_indices[:1]
    return DatasetView(dataset, train_episode_indices), DatasetView(dataset, val_episode_indices)


def max_gpu_memory_gb(mem: dict[str, Any]) -> float | None:
    value = mem.get("max_used_mb")
    if value is None:
        return None
    return float(value) / 1024.0


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in {"1", "true", "yes", "y"}:
        return True
    if value.lower() in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool: {value}")


def save_split(path: Path, train_view: DatasetView, val_view: DatasetView) -> None:
    payload = {
        "train_episode_indices": train_view.summary()["episode_indices"],
        "val_episode_indices": val_view.summary()["episode_indices"],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--save_dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--openpi_root", type=Path, default=DEFAULT_OPENPI_ROOT)
    parser.add_argument("--base_params", type=Path, default=DEFAULT_BASE_PARAMS)
    parser.add_argument("--norm_stats", type=Path, default=DEFAULT_NORM_STATS)
    parser.add_argument("--action_horizon", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--eval_batches", type=int, default=2)
    parser.add_argument("--raw_mse_batches", type=int, default=1)
    parser.add_argument("--sample_action_steps", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--save_final_only", type=parse_bool, default=True)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()

    add_openpi_paths(args.openpi_root)
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import optax
    from openpi.models.tokenizer import PaligemmaTokenizer
    from scripts.openpi_robocerebra_config import make_pi05_robocerebra_lora_config

    if not args.base_params.exists():
        raise FileNotFoundError(f"Official pi0.5 base params not found: {args.base_params}")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.save_dir / "train_log.jsonl"
    summary_path = args.save_dir / "summary.json"
    lora_path = args.save_dir / "lora_params.pkl"

    dataset = LocalVisualDataset(args.data_dir, action_horizon=args.action_horizon, max_episodes=args.max_episodes)
    train_set, val_set = split_dataset(dataset, val_ratio=args.val_ratio, split_seed=args.split_seed)
    save_split(args.save_dir / "split.json", train_set, val_set)

    norm_stats = load_norm_stats(args.norm_stats)
    config = make_pi05_robocerebra_lora_config(
        assets_base_dir=str(args.norm_stats.parent.parent),
        checkpoint_base_dir=str(args.save_dir / "openpi_checkpoints_unused"),
        batch_size=args.batch_size,
        num_train_steps=args.steps,
    )
    tokenizer = PaligemmaTokenizer(config.model.max_token_len)

    model = config.model.create(jax.random.key(args.seed))
    graphdef, state = nnx.split(model)
    merged_params = config.weight_loader.load(state.to_pure_dict())
    state.replace_by_pure_dict(merged_params)
    model = nnx.merge(graphdef, state)

    tx = optax.adamw(args.lr)
    opt_state = tx.init(nnx.state(model).filter(config.trainable_filter))

    def loss_fn(train_model, rng_key, observation, actions):
        return jnp.mean(train_model.compute_loss(rng_key, observation, actions, train=False))

    diff_state = nnx.DiffState(0, config.trainable_filter)

    @nnx.jit
    def train_step(train_model, opt_state, rng_key, observation, actions):
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(train_model, rng_key, observation, actions)
        params = nnx.state(train_model).filter(config.trainable_filter)
        updates, opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(train_model, new_params)
        return loss, opt_state

    @nnx.jit
    def eval_step(train_model, rng_key, observation, actions):
        return loss_fn(train_model, rng_key, observation, actions)

    rng = random.Random(args.seed)
    eval_rng = random.Random(args.seed + 10_000)
    train_losses: list[float] = []
    val_losses: list[float] = []
    raw_mses: list[float] = []
    step_times: list[float] = []
    max_gpu_gb_seen = 0.0
    nan_detected = False

    def eval_random(step: int) -> tuple[float, float | None]:
        losses = []
        raw_mse_values = []
        total_batches = max(args.eval_batches, args.raw_mse_batches)
        for batch_i in range(total_batches):
            indices = val_set.sample_indices(args.batch_size, eval_rng)
            obs_np, actions_np, meta = build_batch(
                val_set,
                indices,
                tokenizer=tokenizer,
                norm_stats=norm_stats,
                model_config=config.model,
                image_size=args.image_size,
            )
            obs = observation_to_jax(obs_np)
            actions = jnp.asarray(actions_np)
            if batch_i < args.eval_batches:
                loss = float(
                    jax.device_get(
                        eval_step(model, jax.random.key(args.seed + 200_000 + step * 100 + batch_i), obs, actions)
                    )
                )
                losses.append(loss)
            if batch_i < args.raw_mse_batches:
                pred_norm = model.sample_actions(
                    jax.random.key(args.seed + 300_000 + step * 100 + batch_i),
                    obs,
                    num_steps=args.sample_action_steps,
                )
                pred_norm_np = np.asarray(jax.device_get(pred_norm))[..., :7]
                pred_raw = unnormalize(pred_norm_np, norm_stats["actions"])
                raw_mse_values.append(masked_raw_mse(pred_raw, meta["raw_actions"], meta["action_mask"]))
                del pred_norm, pred_norm_np, pred_raw
                gc.collect()
                jax.clear_caches()
        val_loss = float(np.mean(losses)) if losses else float("nan")
        raw_mse = float(np.mean(raw_mse_values)) if raw_mse_values else None
        return val_loss, raw_mse

    with log_path.open("w", encoding="utf-8") as log_f:
        start_time = time.perf_counter()
        initial_val_loss, initial_raw_mse = eval_random(0)
        val_losses.append(initial_val_loss)
        if initial_raw_mse is not None:
            raw_mses.append(initial_raw_mse)
        mem = gpu_memory()
        max_gpu_gb_seen = max(max_gpu_gb_seen, max_gpu_memory_gb(mem) or 0.0)
        row = {
            "step": 0,
            "train_loss": None,
            "val_loss": initial_val_loss,
            "raw_action_mse": initial_raw_mse,
            "lr": args.lr,
            "step_time": 0.0,
            "max_gpu_memory_gb": max_gpu_memory_gb(mem),
            "nan_detected": nan_detected,
        }
        print(json.dumps(row), flush=True)
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()

        for step in range(1, args.steps + 1):
            indices = train_set.sample_indices(args.batch_size, rng)
            obs_np, actions_np, _ = build_batch(
                train_set,
                indices,
                tokenizer=tokenizer,
                norm_stats=norm_stats,
                model_config=config.model,
                image_size=args.image_size,
            )
            obs = observation_to_jax(obs_np)
            actions = jnp.asarray(actions_np)
            step_start = time.perf_counter()
            loss, opt_state = train_step(model, opt_state, jax.random.key(args.seed + step), obs, actions)
            loss_value = float(jax.device_get(loss))
            elapsed = time.perf_counter() - step_start
            train_losses.append(loss_value)
            step_times.append(elapsed)
            if not np.isfinite(loss_value):
                nan_detected = True

            should_eval = step == 1 or step % args.eval_interval == 0 or step == args.steps
            should_log = step == 1 or step % args.log_interval == 0 or step == args.steps
            val_loss = None
            raw_mse = None
            if should_eval:
                val_loss, raw_mse = eval_random(step)
                val_losses.append(val_loss)
                if raw_mse is not None:
                    raw_mses.append(raw_mse)
                if not np.isfinite(val_loss) or (raw_mse is not None and not np.isfinite(raw_mse)):
                    nan_detected = True

            if should_log:
                mem = gpu_memory()
                max_gpu_gb_seen = max(max_gpu_gb_seen, max_gpu_memory_gb(mem) or 0.0)
                row = {
                    "step": step,
                    "train_loss": loss_value,
                    "val_loss": val_loss,
                    "raw_action_mse": raw_mse,
                    "lr": args.lr,
                    "step_time": elapsed,
                    "max_gpu_memory_gb": max_gpu_memory_gb(mem),
                    "nan_detected": nan_detected,
                }
                print(json.dumps(row), flush=True)
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()

            if (
                not args.save_final_only
                and args.save_interval > 0
                and step % args.save_interval == 0
                and step != args.steps
            ):
                save_lora_checkpoint(
                    args.save_dir / f"lora_params_step_{step:06d}.pkl",
                    nnx.state(model).filter(config.trainable_filter),
                )

    save_lora_checkpoint(lora_path, nnx.state(model).filter(config.trainable_filter))
    total_time = time.perf_counter() - start_time
    summary = {
        "data_dir": str(args.data_dir),
        "num_episodes": len(dataset.episodes),
        "num_train_episodes": len(train_set.episodes),
        "num_val_episodes": len(val_set.episodes),
        "total_frames": len(dataset.index),
        "train_frames": len(train_set.index),
        "val_frames": len(val_set.index),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_ratio": args.val_ratio,
        "split_seed": args.split_seed,
        "eval_interval": args.eval_interval,
        "eval_batches": args.eval_batches,
        "raw_mse_batches": args.raw_mse_batches,
        "sample_action_steps": args.sample_action_steps,
        "save_final_only": args.save_final_only,
        "initial_train_loss": train_losses[0] if train_losses else None,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "initial_val_loss": val_losses[0] if val_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "initial_raw_mse": raw_mses[0] if raw_mses else None,
        "final_raw_mse": raw_mses[-1] if raw_mses else None,
        "loss_decreased": train_losses[-1] < train_losses[0] if len(train_losses) >= 2 else None,
        "val_loss_decreased": val_losses[-1] < val_losses[0] if len(val_losses) >= 2 else None,
        "raw_mse_decreased": raw_mses[-1] < raw_mses[0] if len(raw_mses) >= 2 else None,
        "nan_detected": nan_detected,
        "max_gpu_memory_gb": max_gpu_gb_seen,
        "avg_step_time": float(np.mean(step_times)) if step_times else None,
        "total_time_sec": total_time,
        "lora_checkpoint": str(lora_path),
        "train_log": str(log_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
