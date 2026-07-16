#!/usr/bin/env python3
"""HTTP inference server for RoboCerebra pi0.5 rollouts.

Run this in the OpenPI environment. The RoboCerebra/LIBERO eval process calls
it over localhost so we do not need one conda env containing both stacks.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import pickle
import sys
import time
import traceback
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENPI_ROOT = Path("/mnt/raid1/mjh/RoboTwin/policy/pi05")
DEFAULT_BASE = Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_base/params"
DEFAULT_LORA = PROJECT_ROOT / "outputs/pi05_robocerebra_lora_random_200ep_1kstep/lora_params.pkl"
DEFAULT_NORM = PROJECT_ROOT / "outputs/openpi_assets/robocerebra_unified_full/norm_stats.json"


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
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("norm_stats", payload)


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


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32)
    return {
        "shape": list(arr.shape),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "nan": bool(np.isnan(arr).any()),
        "inf": bool(np.isinf(arr).any()),
    }


def image_from_b64_png(encoded: str, image_size: int) -> np.ndarray:
    raw = base64.b64decode(encoded.encode("ascii"))
    with Image.open(BytesIO(raw)) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


class Pi05RoboCerebraPolicy:
    def __init__(self, args: argparse.Namespace):
        add_openpi_paths(args.openpi_root)
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from openpi.models import model as openpi_model
        from openpi.models.tokenizer import PaligemmaTokenizer
        import openpi.training.weight_loaders as weight_loaders
        from scripts.openpi_robocerebra_config import make_pi05_robocerebra_lora_config

        if not args.base_ckpt.exists():
            raise FileNotFoundError(f"official pi0.5 base params not found: {args.base_ckpt}")
        if args.lora_path is not None and not args.lora_path.exists():
            raise FileNotFoundError(f"LoRA params not found: {args.lora_path}")

        self.args = args
        self.jax = jax
        self.jnp = jnp
        self.nnx = nnx
        self.openpi_model = openpi_model
        self.norm_stats = load_norm_stats(args.norm_stats)
        self.config = make_pi05_robocerebra_lora_config(
            assets_base_dir=str(args.norm_stats.parent.parent),
            checkpoint_base_dir=str(args.output_dir / "openpi_checkpoints_unused"),
            batch_size=1,
            num_train_steps=1,
        )
        self.config = dataclasses.replace(
            self.config,
            weight_loader=weight_loaders.CheckpointWeightLoader(str(args.base_ckpt))
        )
        self.tokenizer = PaligemmaTokenizer(self.config.model.max_token_len)

        model = self.config.model.create(jax.random.key(args.seed))
        graphdef, state = nnx.split(model)
        merged_params = self.config.weight_loader.load(state.to_pure_dict())
        state.replace_by_pure_dict(merged_params)
        self.loaded_lora = False
        if args.lora_path is not None:
            lora_params = pickle.loads(args.lora_path.read_bytes())
            state.replace_by_pure_dict(lora_params)
            self.loaded_lora = True
        self.model = nnx.merge(graphdef, state)
        # Keep this un-jitted. pi0.5's token embedding path indexes a NumPy
        # table with tokenized prompts, which fails under a direct nnx.jit
        # wrapper with TracerArrayConversionError.
        self.sample_actions = self.model.sample_actions
        self.rng = jax.random.key(args.seed)
        self.request_count = 0

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "pi05_http_server",
            "mode": "lora" if self.loaded_lora else "base_only",
            "base_ckpt": str(self.args.base_ckpt),
            "lora_path": str(self.args.lora_path) if self.args.lora_path else None,
            "norm_stats": str(self.args.norm_stats),
            "jax_devices": [str(device) for device in self.jax.devices()],
            "default_device": str(self.jax.devices()[0]) if self.jax.devices() else None,
            "action_horizon": self.config.model.action_horizon,
            "model_action_dim": self.config.model.action_dim,
            "raw_state_dim": 8,
            "raw_action_dim": 7,
            "sample_steps": self.args.sample_steps,
        }

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng = self.jax.random.key(int(seed))
        return {"status": "ok", "metadata": self.metadata}

    def _observation_to_openpi(self, prompt: str, front: np.ndarray, wrist: np.ndarray, state8: np.ndarray):
        norm_state = normalize(state8.astype(np.float32), self.norm_stats["state"]).astype(np.float32)
        padded_state = pad_dim(norm_state[None, :], self.config.model.action_dim).astype(np.float32)
        tokens, masks = self.tokenizer.tokenize(prompt, norm_state)
        batch = {
            "image": {
                "base_0_rgb": front[None, ...].astype(np.uint8),
                "left_wrist_0_rgb": wrist[None, ...].astype(np.uint8),
                "right_wrist_0_rgb": np.zeros_like(front[None, ...], dtype=np.uint8),
            },
            "image_mask": {
                "base_0_rgb": np.asarray([True]),
                "left_wrist_0_rgb": np.asarray([True]),
                "right_wrist_0_rgb": np.asarray([False]),
            },
            "state": padded_state,
            "tokenized_prompt": tokens[None, ...].astype(np.int32),
            "tokenized_prompt_mask": masks[None, ...].astype(bool),
        }
        obs = self.openpi_model.Observation.from_dict(batch)
        return self.openpi_model.Observation(
            images={key: self.jnp.asarray(value) for key, value in obs.images.items()},
            image_masks={key: self.jnp.asarray(value) for key, value in obs.image_masks.items()},
            state=self.jnp.asarray(obs.state),
            tokenized_prompt=self.jnp.asarray(obs.tokenized_prompt),
            tokenized_prompt_mask=self.jnp.asarray(obs.tokenized_prompt_mask),
            token_ar_mask=None,
            token_loss_mask=None,
        )

    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        prompt = str(request["prompt"])
        image_size = int(request.get("image_size") or self.args.image_size)
        front = image_from_b64_png(str(request["front_image_png_b64"]), image_size)
        wrist = image_from_b64_png(str(request["wrist_image_png_b64"]), image_size)
        state8 = np.asarray(request["state"], dtype=np.float32)
        if state8.shape != (8,):
            raise ValueError(f"state must be 8D, got shape {state8.shape}")

        self.rng, sample_rng = self.jax.random.split(self.rng)
        observation = self._observation_to_openpi(prompt, front, wrist, state8)
        start = time.perf_counter()
        pred_norm = self.sample_actions(sample_rng, observation, num_steps=self.args.sample_steps)
        pred_norm_np = np.asarray(self.jax.device_get(pred_norm))[0, :, :7].astype(np.float32)
        pred_raw = unnormalize(pred_norm_np, self.norm_stats["actions"]).astype(np.float32)
        elapsed = time.perf_counter() - start
        self.request_count += 1
        return {
            "status": "ok",
            "request_count": self.request_count,
            "prompt": prompt,
            "action_chunk": pred_raw.tolist(),
            "normalized_action_stats": finite_stats(pred_norm_np),
            "raw_action_stats": finite_stats(pred_raw),
            "nan": bool(np.isnan(pred_raw).any()),
            "inf": bool(np.isinf(pred_raw).any()),
            "inference_time_sec": elapsed,
            "metadata": self.metadata,
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "Pi05RoboCerebraHTTP/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8")) if raw else {}

    def _write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._write_json({"status": "not_found"}, status=404)
            return
        self._write_json({"status": "ok", "metadata": self.server.policy.metadata})  # type: ignore[attr-defined]

    def do_POST(self) -> None:
        try:
            request = self._read_json()
            if self.path == "/reset":
                payload = self.server.policy.reset(request.get("seed"))  # type: ignore[attr-defined]
            elif self.path == "/predict":
                payload = self.server.policy.predict(request)  # type: ignore[attr-defined]
            else:
                self._write_json({"status": "not_found"}, status=404)
                return
            self._write_json(payload)
        except Exception as exc:
            self._write_json(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
                status=500,
            )


class Server(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], policy: Pi05RoboCerebraPolicy):
        super().__init__(address, Handler)
        self.policy = policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--openpi-root", type=Path, default=DEFAULT_OPENPI_ROOT)
    parser.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--lora-path", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--norm-stats", type=Path, default=DEFAULT_NORM)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/pi05_robocerebra_http_server")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-steps", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.base_only:
        args.lora_path = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = Pi05RoboCerebraPolicy(args)
    server = Server((args.host, args.port), policy)
    print(
        json.dumps(
            {
                "status": "pi05_robocerebra_server_ready",
                "host": args.host,
                "port": args.port,
                "metadata": policy.metadata,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
