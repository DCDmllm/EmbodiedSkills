#!/usr/bin/env python3
"""Minimal RoboCerebra LoRA trainer for official pi0.5/OpenPI.

This intentionally bypasses OpenPI's official train.py and LeRobot loader. It
trains LoRA parameters from local exported visual episodes:

episode_xxxxxx/
  images/*.png
  wrist_images/*.png
  actions.npy
  states.npy
  meta.json
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENPI_ROOT = Path("/mnt/raid1/mjh/RoboTwin/policy/pi05")
DEFAULT_BASE_PARAMS = Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_base/params"
DEFAULT_NORM_STATS = PROJECT_ROOT / "outputs/openpi_assets/robocerebra_unified/norm_stats.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "outputs/robocerebra_lerobot_visual_sample_50"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "outputs/pi05_robocerebra_lora_minimal"


def add_openpi_paths(openpi_root: Path) -> None:
    for path in [
        PROJECT_ROOT,
        openpi_root / "packages/openpi-client/src",
        openpi_root / "src",
    ]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def load_norm_stats(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())["norm_stats"]


def normalize(x: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return (x - mean[: x.shape[-1]]) / (std[: x.shape[-1]] + 1e-6)


def unnormalize(x: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return x * (std[: x.shape[-1]] + 1e-6) + mean[: x.shape[-1]]


def pad_dim(x: np.ndarray, dim: int) -> np.ndarray:
    if x.shape[-1] >= dim:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (0, dim - x.shape[-1])
    return np.pad(x, pad_width, constant_values=0.0)


def sorted_pngs(path: Path) -> list[Path]:
    return sorted(path.glob("*.png"), key=lambda p: int(p.stem))


def load_rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


@dataclass
class Episode:
    episode_dir: Path
    meta: dict[str, Any]
    actions: np.ndarray
    states: np.ndarray
    images: list[Path]
    wrist_images: list[Path]

    @property
    def prompt(self) -> str:
        return str(self.meta["subgoal_instruction"])

    @property
    def num_frames(self) -> int:
        return int(self.actions.shape[0])


class LocalVisualDataset:
    def __init__(self, data_dir: Path, *, action_horizon: int, max_episodes: int | None = None) -> None:
        self.data_dir = data_dir
        self.action_horizon = action_horizon
        self.episodes = self._load_episodes(max_episodes)
        self.index: list[tuple[int, int]] = []
        for ep_i, episode in enumerate(self.episodes):
            for frame_i in range(episode.num_frames):
                self.index.append((ep_i, frame_i))

    def _load_episodes(self, max_episodes: int | None) -> list[Episode]:
        episode_dirs = sorted(self.data_dir.glob("episode_*"))
        if max_episodes is not None:
            episode_dirs = episode_dirs[:max_episodes]
        if not episode_dirs:
            raise FileNotFoundError(f"No episode_* directories under {self.data_dir}")

        episodes: list[Episode] = []
        for ep_dir in episode_dirs:
            meta = json.loads((ep_dir / "meta.json").read_text())
            actions = np.load(ep_dir / "actions.npy").astype(np.float32)
            states = np.load(ep_dir / "states.npy").astype(np.float32)
            images = sorted_pngs(ep_dir / "images")
            wrist_images = sorted_pngs(ep_dir / "wrist_images")
            lengths = {
                "actions": len(actions),
                "states": len(states),
                "images": len(images),
                "wrist_images": len(wrist_images),
            }
            if len(set(lengths.values())) != 1:
                raise ValueError(f"{ep_dir.name} length mismatch: {lengths}")
            episodes.append(Episode(ep_dir, meta, actions, states, images, wrist_images))
        return episodes

    def sample_indices(self, batch_size: int, rng: random.Random) -> list[tuple[int, int]]:
        return [self.index[rng.randrange(len(self.index))] for _ in range(batch_size)]

    def first_indices(self, batch_size: int) -> list[tuple[int, int]]:
        return self.index[:batch_size]

    def summary(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.data_dir),
            "num_episodes": len(self.episodes),
            "num_samples": len(self.index),
            "action_horizon": self.action_horizon,
            "episodes": [
                {
                    "episode_index": int(ep.meta["episode_index"]),
                    "task": ep.prompt,
                    "num_frames": ep.num_frames,
                    "action_shape": list(ep.actions.shape),
                    "state_shape": list(ep.states.shape),
                    "image_count": len(ep.images),
                    "wrist_image_count": len(ep.wrist_images),
                    "aligned": len(ep.images) == len(ep.wrist_images) == len(ep.actions) == len(ep.states),
                }
                for ep in self.episodes
            ],
        }


def build_batch(
    dataset: LocalVisualDataset,
    indices: list[tuple[int, int]],
    *,
    tokenizer,
    norm_stats: dict[str, Any],
    model_config,
    image_size: int,
) -> tuple[Any, Any, dict[str, Any]]:
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

    for ep_i, frame_i in indices:
        ep = dataset.episodes[ep_i]
        state = ep.states[frame_i]
        chunk = np.zeros((dataset.action_horizon, ep.actions.shape[-1]), dtype=np.float32)
        raw_chunk = np.zeros_like(chunk)
        mask = np.zeros((dataset.action_horizon,), dtype=np.float32)
        end_i = min(frame_i + dataset.action_horizon, ep.num_frames)
        valid = end_i - frame_i
        chunk[:valid] = ep.actions[frame_i:end_i]
        raw_chunk[:valid] = ep.actions[frame_i:end_i]
        mask[:valid] = 1.0

        norm_state = normalize(state, norm_stats["state"]).astype(np.float32)
        norm_actions = normalize(chunk, norm_stats["actions"]).astype(np.float32)
        padded_state = pad_dim(norm_state[None, :], model_config.action_dim)[0].astype(np.float32)
        padded_actions = pad_dim(norm_actions, model_config.action_dim).astype(np.float32)
        tokens, masks = tokenizer.tokenize(ep.prompt, norm_state)

        front = load_rgb(ep.images[frame_i], image_size)
        wrist = load_rgb(ep.wrist_images[frame_i], image_size)
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
                "episode_index": int(ep.meta["episode_index"]),
                "frame_index": frame_i,
                "prompt": ep.prompt,
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


def observation_to_jax(observation):
    import jax.numpy as jnp
    from openpi.models import model as openpi_model

    return openpi_model.Observation(
        images={key: jnp.asarray(value) for key, value in observation.images.items()},
        image_masks={key: jnp.asarray(value) for key, value in observation.image_masks.items()},
        state=jnp.asarray(observation.state),
        tokenized_prompt=None if observation.tokenized_prompt is None else jnp.asarray(observation.tokenized_prompt),
        tokenized_prompt_mask=None
        if observation.tokenized_prompt_mask is None
        else jnp.asarray(observation.tokenized_prompt_mask),
        token_ar_mask=None if observation.token_ar_mask is None else jnp.asarray(observation.token_ar_mask),
        token_loss_mask=None if observation.token_loss_mask is None else jnp.asarray(observation.token_loss_mask),
    )


def gpu_memory() -> dict[str, Any]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
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


def masked_raw_mse(pred_raw: np.ndarray, target_raw: np.ndarray, mask: np.ndarray) -> float:
    per_step = np.mean((pred_raw - target_raw) ** 2, axis=-1)
    return float(np.sum(per_step * mask) / max(float(np.sum(mask)), 1.0))


def save_lora_checkpoint(path: Path, params) -> None:
    import jax

    path.parent.mkdir(parents=True, exist_ok=True)
    params_cpu = jax.device_get(params.to_pure_dict())
    with path.open("wb") as f:
        pickle.dump(params_cpu, f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--openpi_root", type=Path, default=DEFAULT_OPENPI_ROOT)
    parser.add_argument("--base_params", type=Path, default=DEFAULT_BASE_PARAMS)
    parser.add_argument("--norm_stats", type=Path, default=DEFAULT_NORM_STATS)
    parser.add_argument("--action_horizon", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--raw_mse_interval", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--sample_action_steps", type=int, default=5)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument(
        "--overfit_fixed_batch",
        action="store_true",
        help="Repeat the first batch every step. Useful for optimizer/data-path sanity checks.",
    )
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

    def loss_fn(train_model, rng, observation, actions):
        return jnp.mean(train_model.compute_loss(rng, observation, actions, train=False))

    diff_state = nnx.DiffState(0, config.trainable_filter)

    @nnx.jit
    def train_step(train_model, opt_state, rng, observation, actions):
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(train_model, rng, observation, actions)
        params = nnx.state(train_model).filter(config.trainable_filter)
        updates, opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(train_model, new_params)
        return loss, opt_state

    @nnx.jit
    def eval_step(train_model, rng, observation, actions):
        return loss_fn(train_model, rng, observation, actions)

    rng = random.Random(args.seed)
    eval_indices = dataset.first_indices(args.batch_size)
    eval_obs_np, eval_actions_np, eval_meta = build_batch(
        dataset,
        eval_indices,
        tokenizer=tokenizer,
        norm_stats=norm_stats,
        model_config=config.model,
        image_size=args.image_size,
    )
    eval_obs = observation_to_jax(eval_obs_np)
    eval_actions = jnp.asarray(eval_actions_np)
    fixed_eval_rng = jax.random.key(args.seed + 999_999)
    train_losses: list[float] = []
    eval_losses: list[float] = []
    raw_mses: list[float] = []
    step_times: list[float] = []
    nan_seen = False

    with log_path.open("w", encoding="utf-8") as log_f:
        start_time = time.perf_counter()
        initial_eval_loss = float(jax.device_get(eval_step(model, fixed_eval_rng, eval_obs, eval_actions)))
        eval_losses.append(initial_eval_loss)
        initial_row = {
            "step": 0,
            "train_loss": None,
            "eval_loss": initial_eval_loss,
            "raw_action_mse": None,
            "step_time_sec": 0.0,
            "avg_step_time_sec": None,
            "nan_seen": nan_seen,
            "gpu_memory": gpu_memory(),
        }
        print(
            "step=0 eval_loss={eval_loss:.6f} max_gpu_mem={max_mem}".format(
                **initial_row,
                max_mem=initial_row["gpu_memory"].get("max_used_mb", "unknown"),
            ),
            flush=True,
        )
        log_f.write(json.dumps(initial_row) + "\n")
        log_f.flush()

        for step in range(1, args.steps + 1):
            indices = eval_indices if args.overfit_fixed_batch else dataset.sample_indices(args.batch_size, rng)
            obs_np, actions_np, _ = build_batch(
                dataset,
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
                nan_seen = True

            should_log = step == 1 or step % args.log_interval == 0 or step == args.steps
            should_eval = step == 1 or step % args.eval_interval == 0 or step == args.steps
            should_raw_mse = (
                args.raw_mse_interval > 0
                and (step == 1 or step % args.raw_mse_interval == 0 or step == args.steps)
            )
            if should_log:
                eval_loss = None
                if should_eval:
                    eval_loss = float(jax.device_get(eval_step(model, fixed_eval_rng, eval_obs, eval_actions)))
                    eval_losses.append(eval_loss)

                raw_mse = None
                if should_raw_mse:
                    pred_norm = model.sample_actions(
                        jax.random.key(args.seed + 10_000 + step),
                        eval_obs,
                        num_steps=args.sample_action_steps,
                    )
                    pred_norm_np = np.asarray(jax.device_get(pred_norm))[..., :7]
                    pred_raw = unnormalize(pred_norm_np, norm_stats["actions"])
                    raw_mse = masked_raw_mse(pred_raw, eval_meta["raw_actions"], eval_meta["action_mask"])
                    raw_mses.append(raw_mse)
                    del pred_norm, pred_norm_np, pred_raw
                    gc.collect()
                    jax.clear_caches()

                mem = gpu_memory()
                row = {
                    "step": step,
                    "train_loss": loss_value,
                    "eval_loss": eval_loss,
                    "raw_action_mse": raw_mse,
                    "step_time_sec": elapsed,
                    "avg_step_time_sec": float(np.mean(step_times)),
                    "nan_seen": nan_seen,
                    "gpu_memory": mem,
                }
                parts = [
                    f"step={step}",
                    f"train_loss={loss_value:.6f}",
                ]
                if eval_loss is not None:
                    parts.append(f"eval_loss={eval_loss:.6f}")
                if raw_mse is not None:
                    parts.append(f"raw_action_mse={raw_mse:.6f}")
                parts.extend(
                    [
                        f"step_time={elapsed:.3f}s",
                        f"max_gpu_mem={mem.get('max_used_mb', 'unknown')}",
                    ]
                )
                print(" ".join(parts), flush=True)
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()

            if args.save_interval > 0 and step % args.save_interval == 0:
                save_lora_checkpoint(args.save_dir / f"lora_params_step_{step:06d}.pkl", nnx.state(model).filter(config.trainable_filter))

    save_lora_checkpoint(lora_path, nnx.state(model).filter(config.trainable_filter))
    summary = {
        "data": dataset.summary(),
        "base_params": str(args.base_params),
        "norm_stats": str(args.norm_stats),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "eval_interval": args.eval_interval,
        "raw_mse_interval": args.raw_mse_interval,
        "overfit_fixed_batch": args.overfit_fixed_batch,
        "initial_loss": train_losses[0] if train_losses else None,
        "final_loss": train_losses[-1] if train_losses else None,
        "loss_decreased": train_losses[-1] < train_losses[0] if len(train_losses) >= 2 else None,
        "min_loss": min(train_losses) if train_losses else None,
        "initial_eval_loss": eval_losses[0] if eval_losses else None,
        "final_eval_loss": eval_losses[-1] if eval_losses else None,
        "eval_loss_decreased": eval_losses[-1] < eval_losses[0] if len(eval_losses) >= 2 else None,
        "min_eval_loss": min(eval_losses) if eval_losses else None,
        "initial_raw_action_mse": raw_mses[0] if raw_mses else None,
        "final_raw_action_mse": raw_mses[-1] if raw_mses else None,
        "raw_action_mse_decreased": raw_mses[-1] < raw_mses[0] if len(raw_mses) >= 2 else None,
        "avg_step_time_sec": float(np.mean(step_times)) if step_times else None,
        "nan_seen": nan_seen,
        "final_gpu_memory": gpu_memory(),
        "lora_checkpoint": str(lora_path),
        "train_log": str(log_path),
        "total_time_sec": time.perf_counter() - start_time,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
