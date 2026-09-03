#!/usr/bin/env python3
"""Run/aggregate RoboCerebra Ideal/case1 full-task model comparisons.

The rollout path is intentionally small and explicit. It uses the official
RoboCerebra LIBERO task environment and the original case1 task files, but it
does not use planner-generated subgoals or paraphrases. Every policy receives
the full task instruction.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, deque
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback
from typing import Any
from io import BytesIO
import urllib.request

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH_ROOT = Path("/mnt/raid1/mjh/datasets/RoboCerebraBench_case1")
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs/robocerebra_probe_logs/full_task_comparison"
DEFAULT_REPORT = PROJECT_ROOT / "outputs/robocerebra_full_task_model_comparison.md"
DEFAULT_OPENVLA_OFT = Path("/mnt/raid1/mjh/RoboTwin/RoboTwin/policy/openvla-oft")
DEFAULT_ROBOCEREBRA_EVAL = PROJECT_ROOT / ".deps/RoboCerebra_zip/evaluation"
DEFAULT_LIBERO = PROJECT_ROOT / ".deps/RoboCerebra_zip/LIBERO"
DEFAULT_OPENPI_ROOT = Path("/mnt/raid1/mjh/RoboTwin/policy/pi05")
DEFAULT_PI05_BASE = Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_base/params"
DEFAULT_PI05_LORA = PROJECT_ROOT / "outputs/pi05_robocerebra_lora_random_200ep_1kstep/lora_params.pkl"
DEFAULT_PI05_NORM = PROJECT_ROOT / "outputs/openpi_assets/robocerebra_unified_full/norm_stats.json"
DEFAULT_PI05_SERVICE_URL = "http://127.0.0.1:8766"


OBJECT_NAMES = ["cream_cheese_1", "popcorn_1", "butter_1"]
FULL_TASK_FALLBACK = "Organize all the food boxes into the white storage box."


def add_eval_paths(openvla_oft: Path, robocerebra_eval: Path, libero_root: Path) -> None:
    for path in [openvla_oft, libero_root, robocerebra_eval, PROJECT_ROOT]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    patch_robosuite_numba_cache()


def patch_robosuite_numba_cache() -> None:
    """Disable robosuite's numba disk cache before transform_utils imports.

    The RoboCerebra eval env can import robosuite from site-packages in a way
    that gives numba no cache locator for transform_utils.py. Disabling the
    cache keeps JIT enabled but avoids the import-time failure.
    """
    import importlib.machinery
    import importlib.util

    package_spec = importlib.machinery.PathFinder.find_spec("robosuite", sys.path)
    if package_spec is None or not package_spec.submodule_search_locations:
        return
    base = Path(list(package_spec.submodule_search_locations)[0])

    def load_macro_module(module_name: str, path: Path) -> None:
        if not path.exists():
            return
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.CACHE_NUMBA = False
        sys.modules[module_name] = module

    load_macro_module("robosuite.macros", base / "macros.py")
    load_macro_module("robosuite.utils.macros", base / "utils" / "macros.py")


def add_openpi_paths(openpi_root: Path) -> None:
    for path in [
        PROJECT_ROOT,
        openpi_root / "packages/openpi-client/src",
        openpi_root / "src",
    ]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def as_list(value: Any) -> list[float]:
    return np.asarray(value, dtype=np.float64).astype(float).tolist()


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {"shape": list(arr.shape), "nan": False, "inf": False}
    return {
        "shape": list(arr.shape),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "nan": bool(np.isnan(arr).any()),
        "inf": bool(np.isinf(arr).any()),
    }


def read_full_task_instruction(task_dir: Path) -> str:
    txt = task_dir / "task_description.txt"
    if not txt.exists():
        return FULL_TASK_FALLBACK
    for line in txt.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("Task:"):
            return stripped.split(":", 1)[1].strip()
    return FULL_TASK_FALLBACK


def parse_task_description(task_dir: Path) -> tuple[list[dict[str, Any]], int]:
    """Return original task steps and final demo boundary."""
    txt = task_dir / "task_description.txt"
    steps: list[dict[str, Any]] = []
    final_end = 0
    lines = [line.strip() for line in txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith("Step:"):
            instruction = lines[i].split(":", 1)[1].strip()
            frame_start = None
            frame_end = None
            if i + 1 < len(lines) and lines[i + 1].startswith("["):
                raw = lines[i + 1].strip("[]")
                parts = [part.strip() for part in raw.split(",")]
                if len(parts) == 2:
                    frame_start = int(parts[0])
                    frame_end = int(parts[1])
                    final_end = max(final_end, frame_end)
                i += 1
            steps.append(
                {
                    "subgoal_index": len(steps),
                    "instruction": instruction,
                    "demo_frame_start": frame_start,
                    "demo_frame_end": frame_end,
                }
            )
        i += 1
    return steps, final_end


def load_goal_subgoals(task_dir: Path) -> tuple[dict[str, list[list[str]]], list[dict[str, Any]]]:
    goal_json = task_dir / "goal.json"
    raw = json.loads(goal_json.read_text(encoding="utf-8"))
    goal: dict[str, list[list[str]]] = {}
    subgoals: list[dict[str, Any]] = []
    for object_name, entries in raw.items():
        goal[object_name] = []
        for entry in entries:
            state_pair = entry["state_pair"] if isinstance(entry, dict) else entry
            task_step = int(entry.get("task_step", len(subgoals))) if isinstance(entry, dict) else len(subgoals)
            predicate = [str(item).lower() for item in state_pair]
            goal[object_name].append(predicate)
            subgoals.append(
                {
                    "subgoal_index": task_step,
                    "object": object_name,
                    "predicate": predicate,
                    "completed": False,
                    "completion_step": None,
                }
            )
    subgoals.sort(key=lambda item: item["subgoal_index"])
    return goal, subgoals


def get_object_pose(env, object_name: str) -> dict[str, list[float] | str]:
    try:
        state = env.object_states_dict[object_name].get_geom_state()
        return {"pos": as_list(state["pos"]), "quat": as_list(state["quat"])}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def current_regions(env, object_name: str) -> list[str]:
    regions = []
    for region in getattr(env, "regions", []):
        try:
            if env._eval_predicate(["in", object_name, region]):
                regions.append(region)
        except Exception as exc:
            regions.append(f"ERROR:{region}:{type(exc).__name__}:{exc}")
    return regions or ["None"]


def get_object_pos(env, object_name: str) -> np.ndarray | None:
    pose = get_object_pose(env, object_name)
    if "pos" not in pose:
        return None
    return np.asarray(pose["pos"], dtype=np.float32)


def iter_contact_geom_name_pairs(env) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    try:
        sim = env.sim
        model = sim.model
        data = sim.data
        for i in range(int(data.ncon)):
            contact = data.contact[i]
            name1 = model.geom_id2name(int(contact.geom1)) or ""
            name2 = model.geom_id2name(int(contact.geom2)) or ""
            pairs.append((name1, name2))
    except Exception:
        return pairs
    return pairs


def contact_diagnostics(env, object_names: list[str]) -> dict[str, Any]:
    pairs = iter_contact_geom_name_pairs(env)
    gripper_hints = ("gripper", "finger", "leftpad", "rightpad", "hand", "panda0")
    by_object: dict[str, dict[str, Any]] = {}
    for object_name in object_names:
        object_key = object_name.lower()
        matches = []
        gripper_contact = False
        any_contact = False
        for name1, name2 in pairs:
            joined = f"{name1}|{name2}".lower()
            if object_key not in joined:
                continue
            any_contact = True
            is_gripper = any(hint in joined for hint in gripper_hints)
            gripper_contact = gripper_contact or is_gripper
            if len(matches) < 8:
                matches.append({"geom1": name1, "geom2": name2, "gripper": is_gripper})
        by_object[object_name] = {
            "any_contact": any_contact,
            "gripper_contact": gripper_contact,
            "contact_pairs_sample": matches,
        }
    return {
        "num_contacts": len(pairs),
        "object_contacts": by_object,
    }


def gripper_command_close(value: float, *, threshold: float, direction: str) -> bool:
    if direction == "positive":
        return value >= threshold
    if direction == "negative":
        return value <= -threshold
    return abs(value) >= threshold


def checker_state(env, goal: dict[str, list[list[str]]], subgoal_template: list[dict[str, Any]]) -> dict[str, Any]:
    completion, total_completed, all_done = env._check_success(goal)
    progress = getattr(env, "_state_progress", {}).copy()
    subgoals = []
    for item in subgoal_template:
        obj = item["object"]
        obj_goal_index = sum(
            1
            for prior in subgoal_template
            if prior["object"] == obj and prior["subgoal_index"] < item["subgoal_index"]
        )
        completed = int(progress.get(obj, 0)) >= obj_goal_index + 1
        subgoals.append(
            {
                **item,
                "completed": bool(completed),
            }
        )
    return {
        "completion": {key: float(value) for key, value in completion.items()},
        "total_completed": int(total_completed),
        "all_done": bool(all_done),
        "state_progress": {key: int(value) for key, value in progress.items()},
        "subgoals": subgoals,
    }


def first_incomplete_subgoal(checker: dict[str, Any]) -> dict[str, Any] | None:
    for subgoal in checker["subgoals"]:
        if not subgoal["completed"]:
            return subgoal
    return None


class OpenVLABackend:
    def __init__(self, args: argparse.Namespace):
        from config import GenerateConfig
        from eval_openvla import initialize_model

        self.args = args
        self.cfg = GenerateConfig(
            pretrained_checkpoint=str(args.checkpoint),
            robocerebra_root=str(args.bench_root),
            init_files_root=str(args.bench_root / "init_files"),
            task_types=["Ideal"],
            num_trials_per_task=1,
            task_description_suffix="",
            switch_steps=args.max_steps,
            num_steps_wait=args.wait_steps,
            use_init_files=True,
            use_proprio=args.use_proprio,
            load_in_8bit=False,
            load_in_4bit=False,
            local_log_dir=str(args.output_dir),
            run_id_note=f"{args.model_name}_full_task",
            use_wandb=False,
        )
        (
            self.model,
            self.action_head,
            self.proprio_projector,
            self.noisy_action_projector,
            self.processor,
        ) = initialize_model(self.cfg)
        self.action_queue: deque[np.ndarray] = deque()
        self.chunk_id = -1

    @property
    def metadata(self) -> dict[str, Any]:
        stats = self.model.norm_stats[self.cfg.unnorm_key]
        return {
            "backend": "openvla",
            "checkpoint": str(self.args.checkpoint),
            "use_proprio": self.args.use_proprio,
            "unnorm_key": self.cfg.unnorm_key,
            "action_chunk_size": self.cfg.num_open_loop_steps,
            "action_stats": stats.get("action"),
            "proprio_stats": stats.get("proprio"),
        }

    def reset_episode(self) -> None:
        self.action_queue.clear()
        self.chunk_id = -1

    def next_action(self, observation: dict[str, Any], prompt: str) -> tuple[np.ndarray, dict[str, Any]]:
        from experiments.robot.robot_utils import get_action, invert_gripper_action, normalize_gripper_action

        chunk_record: dict[str, Any] | None = None
        if not self.action_queue:
            infer_start = time.perf_counter()
            actions = get_action(
                self.cfg,
                self.model,
                observation,
                prompt,
                self.processor,
                self.action_head,
                self.proprio_projector,
                self.noisy_action_projector,
                use_film=self.cfg.use_film,
            )
            infer_time = time.perf_counter() - infer_start
            actions_np = np.asarray(actions, dtype=np.float32)
            self.chunk_id += 1
            self.action_queue.extend([actions_np[i].copy() for i in range(len(actions_np))])
            chunk_record = {
                "chunk_id": self.chunk_id,
                "prompt": prompt,
                "inference_time_sec": infer_time,
                "raw_action_stats": finite_stats(actions_np),
                "raw_action_dim_min": as_list(np.nanmin(actions_np, axis=0)),
                "raw_action_dim_max": as_list(np.nanmax(actions_np, axis=0)),
            }

        raw_action = self.action_queue.popleft()
        normalized_gripper = normalize_gripper_action(raw_action, binarize=True)
        env_action = invert_gripper_action(normalized_gripper)
        info = {
            "backend": "openvla",
            "chunk_id": self.chunk_id,
            "queue_remaining": len(self.action_queue),
            "raw_action": as_list(raw_action),
            "env_action": as_list(env_action),
            "action_nan": bool(np.isnan(raw_action).any() or np.isnan(env_action).any()),
            "action_inf": bool(np.isinf(raw_action).any() or np.isinf(env_action).any()),
            "chunk_record": chunk_record,
            "gripper_postprocess": "normalize_gripper_action([0,1]->[-1,1], binarize=True) then invert for env",
        }
        return env_action.astype(np.float32), info


class Pi05Backend:
    """Best-effort pi0.5 backend.

    This backend is fully implemented for a Python environment that has both
    RoboCerebra/LIBERO/robosuite and OpenPI/JAX installed. On the current
    machine those dependencies are split across envs, so construction may fail
    cleanly and the comparison runner records that failure per seed.
    """

    def __init__(self, args: argparse.Namespace):
        # Fail fast before loading the large pi0.5 checkpoint if the active
        # environment cannot create the RoboCerebra simulation.
        import robosuite  # noqa: F401

        add_openpi_paths(args.openpi_root)
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from openpi.models import model as openpi_model
        from openpi.models.tokenizer import PaligemmaTokenizer
        from scripts.openpi_robocerebra_config import make_pi05_robocerebra_lora_config

        self.args = args
        self.jax = jax
        self.jnp = jnp
        self.nnx = nnx
        self.openpi_model = openpi_model
        self.norm_stats = json.loads(Path(args.norm_stats).read_text())["norm_stats"]
        self.config = make_pi05_robocerebra_lora_config(
            assets_base_dir=str(Path(args.norm_stats).parent.parent),
            checkpoint_base_dir=str(args.output_dir / "openpi_checkpoints_unused"),
            batch_size=1,
            num_train_steps=1,
        )
        self.tokenizer = PaligemmaTokenizer(self.config.model.max_token_len)
        model = self.config.model.create(jax.random.key(args.seed))
        graphdef, state = nnx.split(model)
        merged_params = self.config.weight_loader.load(state.to_pure_dict())
        state.replace_by_pure_dict(merged_params)
        lora_params = pickle.loads(Path(args.lora_path).read_bytes())
        state.replace_by_pure_dict(lora_params)
        self.model = nnx.merge(graphdef, state)
        self.sample_actions = self.model.sample_actions
        self.action_queue: deque[np.ndarray] = deque()
        self.chunk_id = -1
        self.rng = jax.random.key(args.seed)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "pi05_openpi",
            "base_ckpt": str(self.args.pi05_base),
            "lora_path": str(self.args.lora_path),
            "norm_stats": str(self.args.norm_stats),
            "action_chunk_size": self.config.model.action_horizon,
            "raw_action_dim": 7,
            "model_action_dim": self.config.model.action_dim,
            "postprocess_assumption": "unnormalize first 7 dims with RoboCerebra full norm stats; normalize+invert gripper for LIBERO env",
        }

    def reset_episode(self) -> None:
        self.action_queue.clear()
        self.chunk_id = -1

    def _normalize(self, x: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        return (x - mean[: x.shape[-1]]) / (std[: x.shape[-1]] + 1e-6)

    def _unnormalize(self, x: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        return x * (std[: x.shape[-1]] + 1e-6) + mean[: x.shape[-1]]

    def _pad_dim(self, x: np.ndarray, dim: int) -> np.ndarray:
        if x.shape[-1] >= dim:
            return x
        pad_width = [(0, 0)] * x.ndim
        pad_width[-1] = (0, dim - x.shape[-1])
        return np.pad(x, pad_width, constant_values=0.0)

    def _observation_to_openpi(self, observation: dict[str, Any], prompt: str):
        state = observation["state"].astype(np.float32)
        norm_state = self._normalize(state, self.norm_stats["state"]).astype(np.float32)
        padded_state = self._pad_dim(norm_state[None, :], self.config.model.action_dim).astype(np.float32)
        tokens, masks = self.tokenizer.tokenize(prompt, norm_state)
        batch = {
            "image": {
                "base_0_rgb": observation["full_image"][None, ...].astype(np.uint8),
                "left_wrist_0_rgb": observation["wrist_image"][None, ...].astype(np.uint8),
                "right_wrist_0_rgb": np.zeros_like(observation["full_image"][None, ...], dtype=np.uint8),
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

    def next_action(self, observation: dict[str, Any], prompt: str) -> tuple[np.ndarray, dict[str, Any]]:
        from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action

        chunk_record: dict[str, Any] | None = None
        if not self.action_queue:
            self.rng, sample_rng = self.jax.random.split(self.rng)
            obs = self._observation_to_openpi(observation, prompt)
            infer_start = time.perf_counter()
            pred_norm = self.sample_actions(sample_rng, obs, num_steps=self.args.pi05_sample_steps)
            pred_norm_np = np.asarray(self.jax.device_get(pred_norm))[0, :, :7].astype(np.float32)
            pred_raw = self._unnormalize(pred_norm_np, self.norm_stats["actions"]).astype(np.float32)
            infer_time = time.perf_counter() - infer_start
            self.chunk_id += 1
            self.action_queue.extend([pred_raw[i].copy() for i in range(len(pred_raw))])
            chunk_record = {
                "chunk_id": self.chunk_id,
                "prompt": prompt,
                "inference_time_sec": infer_time,
                "normalized_action_stats": finite_stats(pred_norm_np),
                "raw_action_stats": finite_stats(pred_raw),
                "raw_action_dim_min": as_list(np.nanmin(pred_raw, axis=0)),
                "raw_action_dim_max": as_list(np.nanmax(pred_raw, axis=0)),
            }

        raw_action = self.action_queue.popleft()
        normalized_gripper = normalize_gripper_action(raw_action, binarize=True)
        env_action = invert_gripper_action(normalized_gripper)
        info = {
            "backend": "pi05_openpi",
            "chunk_id": self.chunk_id,
            "queue_remaining": len(self.action_queue),
            "raw_action": as_list(raw_action),
            "env_action": as_list(env_action),
            "action_nan": bool(np.isnan(raw_action).any() or np.isnan(env_action).any()),
            "action_inf": bool(np.isinf(raw_action).any() or np.isinf(env_action).any()),
            "chunk_record": chunk_record,
            "gripper_postprocess": "normalize_gripper_action([0,1]->[-1,1], binarize=True) then invert for env",
        }
        return env_action.astype(np.float32), info


class Pi05ServiceBackend:
    """pi0.5 backend that delegates OpenPI inference to a local HTTP service."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.base_url = str(args.pi05_service_url).rstrip("/")
        self.timeout = float(args.pi05_service_timeout)
        self.action_queue: deque[np.ndarray] = deque()
        self.chunk_id = -1
        self._metadata = self._health().get("metadata", {})

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "pi05_http_client",
            "service_url": self.base_url,
            "service_metadata": self._metadata,
            "raw_action_dim": 7,
            "execute_chunk_len": int(self.args.execute_chunk_len),
            "postprocess_assumption": "server returns unnormalized 7D RoboCerebra action; client applies normalize+invert gripper for LIBERO env",
        }

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        raw = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=raw,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = response.read()
        parsed = json.loads(body.decode("utf-8"))
        if parsed.get("status") not in {"ok", "pi05_robocerebra_server_ready"}:
            raise RuntimeError(f"pi05_service_error: {parsed}")
        return parsed

    def _health(self) -> dict[str, Any]:
        with urllib.request.urlopen(self.base_url + "/health", timeout=self.timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        if parsed.get("status") != "ok":
            raise RuntimeError(f"pi05_service_unhealthy: {parsed}")
        return parsed

    def reset_episode(self) -> None:
        self.action_queue.clear()
        self.chunk_id = -1
        self._post_json("/reset", {"seed": int(getattr(self.args, "current_seed", self.args.seed))})

    def _encode_png(self, image: np.ndarray) -> str:
        from PIL import Image

        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        buffer = BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def next_action(self, observation: dict[str, Any], prompt: str) -> tuple[np.ndarray, dict[str, Any]]:
        from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action

        chunk_record: dict[str, Any] | None = None
        if not self.action_queue:
            payload = {
                "prompt": prompt,
                "front_image_png_b64": self._encode_png(observation["full_image"]),
                "wrist_image_png_b64": self._encode_png(observation["wrist_image"]),
                "state": np.asarray(observation["state"], dtype=np.float32).tolist(),
                "image_size": int(self.args.image_size),
            }
            result = self._post_json("/predict", payload)
            actions_np = np.asarray(result["action_chunk"], dtype=np.float32)
            if actions_np.shape != (32, 7):
                raise ValueError(f"pi05 service returned action_chunk shape {actions_np.shape}, expected (32, 7)")
            execute_chunk_len = max(1, min(int(self.args.execute_chunk_len), actions_np.shape[0]))
            self.chunk_id += 1
            self.action_queue.extend([actions_np[i].copy() for i in range(execute_chunk_len)])
            chunk_record = {
                "chunk_id": self.chunk_id,
                "prompt": prompt,
                "execute_chunk_len": execute_chunk_len,
                "inference_time_sec": float(result.get("inference_time_sec", 0.0)),
                "raw_action_stats": finite_stats(actions_np),
                "executed_raw_action_stats": finite_stats(actions_np[:execute_chunk_len]),
                "raw_action_dim_min": as_list(np.nanmin(actions_np, axis=0)),
                "raw_action_dim_max": as_list(np.nanmax(actions_np, axis=0)),
                "service_nan": bool(result.get("nan")),
                "service_inf": bool(result.get("inf")),
                "service_request_count": result.get("request_count"),
                "server_metadata": result.get("metadata"),
            }

        raw_action = self.action_queue.popleft()
        normalized_gripper = normalize_gripper_action(raw_action, binarize=True)
        env_action = invert_gripper_action(normalized_gripper)
        info = {
            "backend": "pi05_http_client",
            "chunk_id": self.chunk_id,
            "queue_remaining": len(self.action_queue),
            "raw_action": as_list(raw_action),
            "env_action": as_list(env_action),
            "action_nan": bool(np.isnan(raw_action).any() or np.isnan(env_action).any()),
            "action_inf": bool(np.isinf(raw_action).any() or np.isinf(env_action).any()),
            "chunk_record": chunk_record,
            "gripper_postprocess": "normalize_gripper_action([0,1]->[-1,1], binarize=True) then invert for env",
        }
        return env_action.astype(np.float32), info


def make_backend(args: argparse.Namespace):
    if args.model_kind == "openvla":
        return OpenVLABackend(args)
    if args.model_kind == "pi05":
        return Pi05Backend(args)
    if args.model_kind == "pi05_service":
        return Pi05ServiceBackend(args)
    raise ValueError(f"Unsupported model kind: {args.model_kind}")


def write_failure_seed(args: argparse.Namespace, seed: int, model_status: str, error: BaseException, tb: str) -> dict[str, Any]:
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_name": args.model_name,
        "model_kind": args.model_kind,
        "seed": seed,
        "status": model_status,
        "failure_reason": f"{type(error).__name__}: {error}",
        "traceback": tb,
        "full_task_success": 0,
        "completed_subgoals": 0,
        "subgoal_possible": 6,
        "subgoals": [],
        "first_failed_subgoal": 0,
        "rollout_steps": 0,
        "nan_or_inf_action": None,
        "inference_time_sec_total": None,
        "video_path": None,
        "step_log": str(seed_dir / "step_log.jsonl"),
    }
    (seed_dir / "step_log.jsonl").write_text("", encoding="utf-8")
    (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def run_seed(args: argparse.Namespace, seed: int, backend) -> dict[str, Any]:
    from config import GenerateConfig
    from experiments.robot.libero.libero_utils import get_libero_dummy_action
    from experiments.robot.robot_utils import set_seed_everywhere
    from task_runner import setup_task_environment
    from utils import load_init_state, prepare_observation

    set_seed_everywhere(seed)
    args.current_seed = seed
    backend.reset_episode()

    task_dir = args.bench_root / "Ideal" / "case1"
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    step_log_path = seed_dir / "step_log.jsonl"
    chunk_log_path = seed_dir / "chunk_log.jsonl"
    summary_path = seed_dir / "summary.json"
    video_path = seed_dir / "rollout.mp4"
    for path in [step_log_path, chunk_log_path, summary_path, video_path]:
        if path.exists():
            path.unlink()

    full_task_instruction = read_full_task_instruction(task_dir)
    task_steps, demo_final_end = parse_task_description(task_dir)
    goal, subgoal_template = load_goal_subgoals(task_dir)
    max_steps = args.max_steps or demo_final_end
    if args.prompt_mode == "first_gt_subgoal":
        if not task_steps:
            raise RuntimeError("No GT task steps found for first_gt_subgoal prompt mode")
        policy_prompt = task_steps[0]["instruction"]
        evaluation_scope = "first_gt_subgoal"
        default_success_subgoals = 1
    else:
        policy_prompt = full_task_instruction
        evaluation_scope = "full_task"
        default_success_subgoals = len(subgoal_template)
    success_subgoals = int(args.success_subgoals or default_success_subgoals)

    env, bddl_file_path, error = setup_task_environment(task_dir)
    if error:
        raise RuntimeError(error)

    cfg = GenerateConfig(
        robocerebra_root=str(args.bench_root),
        init_files_root=str(args.bench_root / "init_files"),
        task_types=["Ideal"],
        num_steps_wait=args.wait_steps,
        use_init_files=True,
        local_log_dir=str(args.output_dir),
    )

    obs = env.reset()
    initial_state = load_init_state(cfg, "Ideal", "case1")
    if initial_state is not None:
        env.sim.set_state_from_flattened(initial_state)
        env.sim.forward()
        env._post_process()
        env._update_observables(force=True)
        obs = env._get_observations()

    replay_images: list[np.ndarray] = []
    completion_times: dict[int, int] = {}
    nan_or_inf_action = False
    inference_time_total = 0.0
    chunk_count = 0
    env_exception = None

    initial_poses = {obj: get_object_pose(env, obj) for obj in OBJECT_NAMES}
    min_distances: dict[str, float] = {}
    final_distances: dict[str, float] = {}
    min_distances_at_gripper_command_close: dict[str, float] = {obj: float("inf") for obj in OBJECT_NAMES}
    min_distances_at_gripper_qpos_close: dict[str, float] = {obj: float("inf") for obj in OBJECT_NAMES}
    first_gripper_command_close_step: int | None = None
    first_gripper_qpos_close_step: int | None = None
    contact_ever = {
        obj: {
            "any_contact": False,
            "gripper_contact": False,
            "first_any_contact_step": None,
            "first_gripper_contact_step": None,
            "contact_pairs_sample": [],
        }
        for obj in OBJECT_NAMES
    }
    initial_object_z: dict[str, float | None] = {}
    max_object_z: dict[str, float | None] = {}
    for obj in OBJECT_NAMES:
        pose = initial_poses[obj]
        if "pos" in pose:
            pos = np.asarray(pose["pos"], dtype=np.float32)
            min_distances[obj] = float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - pos))
            initial_object_z[obj] = float(pos[2])
            max_object_z[obj] = float(pos[2])
        else:
            min_distances[obj] = float("inf")
            initial_object_z[obj] = None
            max_object_z[obj] = None

    checker = checker_state(env, goal, subgoal_template)
    baseline_checker = None
    rollout_steps = 0

    with step_log_path.open("w", encoding="utf-8") as step_log, chunk_log_path.open("w", encoding="utf-8") as chunk_log:
        try:
            for t in range(max_steps):
                if t < args.wait_steps:
                    obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
                    continue
                if t == args.wait_steps:
                    checker = checker_state(env, goal, subgoal_template)
                    baseline_checker = checker

                observation, img = prepare_observation(obs, args.image_size)
                replay_images.append(img)
                action, action_info = backend.next_action(observation, policy_prompt)
                if action_info.get("chunk_record") is not None:
                    chunk_count += 1
                    inference_time_total += float(action_info["chunk_record"].get("inference_time_sec", 0.0))
                    chunk_log.write(json.dumps(action_info["chunk_record"], ensure_ascii=False) + "\n")
                    chunk_log.flush()
                nan_or_inf_action = nan_or_inf_action or bool(action_info["action_nan"] or action_info["action_inf"])
                env_action_array = np.asarray(action, dtype=np.float32)
                raw_action_array = np.asarray(action_info.get("raw_action", []), dtype=np.float32)
                gripper_command = float(env_action_array[-1]) if env_action_array.size else None
                raw_gripper_command = float(raw_action_array[-1]) if raw_action_array.size else None
                gripper_command_is_close = (
                    gripper_command_close(
                        gripper_command,
                        threshold=args.gripper_close_threshold,
                        direction=args.gripper_close_direction,
                    )
                    if gripper_command is not None
                    else False
                )

                obs, _, _, _ = env.step(action.tolist())
                rollout_steps += 1
                checker = checker_state(env, goal, subgoal_template)
                for subgoal in checker["subgoals"]:
                    if subgoal["completed"] and subgoal["subgoal_index"] not in completion_times:
                        completion_times[subgoal["subgoal_index"]] = t

                eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
                object_poses = {}
                object_regions = {}
                for obj in OBJECT_NAMES:
                    pose = get_object_pose(env, obj)
                    object_poses[obj] = pose
                    object_regions[obj] = current_regions(env, obj)
                    if "pos" in pose:
                        pos = np.asarray(pose["pos"], dtype=np.float32)
                        dist = float(np.linalg.norm(eef - pos))
                        min_distances[obj] = min(min_distances[obj], dist)
                        final_distances[obj] = dist
                        if max_object_z[obj] is None:
                            max_object_z[obj] = float(pos[2])
                        else:
                            max_object_z[obj] = max(float(max_object_z[obj]), float(pos[2]))
                        if gripper_command_is_close:
                            min_distances_at_gripper_command_close[obj] = min(
                                min_distances_at_gripper_command_close[obj],
                                dist,
                            )

                gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
                gripper_qpos_mean = float(np.mean(gripper_qpos)) if gripper_qpos.size else None
                gripper_qpos_is_close = (
                    gripper_qpos_mean <= args.gripper_qpos_close_threshold
                    if gripper_qpos_mean is not None
                    else False
                )
                if gripper_command_is_close and first_gripper_command_close_step is None:
                    first_gripper_command_close_step = t
                if gripper_qpos_is_close and first_gripper_qpos_close_step is None:
                    first_gripper_qpos_close_step = t
                if gripper_qpos_is_close:
                    for obj, dist in final_distances.items():
                        min_distances_at_gripper_qpos_close[obj] = min(
                            min_distances_at_gripper_qpos_close[obj],
                            dist,
                        )

                contact_state = contact_diagnostics(env, OBJECT_NAMES)
                for obj, item in contact_state["object_contacts"].items():
                    if item["any_contact"]:
                        contact_ever[obj]["any_contact"] = True
                        if contact_ever[obj]["first_any_contact_step"] is None:
                            contact_ever[obj]["first_any_contact_step"] = t
                    if item["gripper_contact"]:
                        contact_ever[obj]["gripper_contact"] = True
                        if contact_ever[obj]["first_gripper_contact_step"] is None:
                            contact_ever[obj]["first_gripper_contact_step"] = t
                    if item["contact_pairs_sample"] and not contact_ever[obj]["contact_pairs_sample"]:
                        contact_ever[obj]["contact_pairs_sample"] = item["contact_pairs_sample"]

                current_target = first_incomplete_subgoal(checker)
                step_record = {
                    "model_name": args.model_name,
                    "seed": seed,
                    "t": t,
                    "policy_step_index": rollout_steps - 1,
                    "prompt": policy_prompt,
                    "evaluation_scope": evaluation_scope,
                    "success_subgoals": success_subgoals,
                    "current_target_subgoal": current_target,
                    "action": {key: value for key, value in action_info.items() if key != "chunk_record"},
                    "gripper_command": gripper_command,
                    "raw_gripper_command": raw_gripper_command,
                    "gripper_command_is_close": gripper_command_is_close,
                    "gripper_command_close_direction": args.gripper_close_direction,
                    "gripper_qpos_mean": gripper_qpos_mean,
                    "gripper_qpos_is_close": gripper_qpos_is_close,
                    "robot_eef_pos": as_list(obs["robot0_eef_pos"]),
                    "robot_eef_quat": as_list(obs["robot0_eef_quat"]),
                    "robot_gripper_qpos": as_list(obs["robot0_gripper_qpos"]),
                    "object_poses": object_poses,
                    "object_regions": object_regions,
                    "eef_to_object_distance": final_distances,
                    "contact_diagnostics": contact_state,
                    "object_lift_height": {
                        obj: None
                        if initial_object_z[obj] is None or max_object_z[obj] is None
                        else float(max_object_z[obj] - initial_object_z[obj])
                        for obj in OBJECT_NAMES
                    },
                    "checker": checker,
                    "completed_subgoals_so_far": int(checker["total_completed"]),
                }
                step_log.write(json.dumps(step_record, ensure_ascii=False) + "\n")
                step_log.flush()

                if checker["all_done"] or int(checker["total_completed"]) >= success_subgoals:
                    break
        except Exception as exc:
            env_exception = {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }

    final_checker = checker_state(env, goal, subgoal_template)
    final_poses = {obj: get_object_pose(env, obj) for obj in OBJECT_NAMES}
    for item in final_checker["subgoals"]:
        idx = item["subgoal_index"]
        item["completion_step"] = completion_times.get(idx)
        if idx < len(task_steps):
            item["instruction"] = task_steps[idx]["instruction"]
            item["demo_frame_start"] = task_steps[idx]["demo_frame_start"]
            item["demo_frame_end"] = task_steps[idx]["demo_frame_end"]

    first_failed = first_incomplete_subgoal(final_checker)
    closest_object = min(min_distances.items(), key=lambda kv: kv[1])[0] if min_distances else None
    first_target = subgoal_template[0]["object"] if subgoal_template else None
    correct_object_approached = bool(
        first_target in min_distances and np.isfinite(min_distances[first_target]) and min_distances[first_target] < args.correct_object_distance
    )
    object_lift_height = {
        obj: None
        if initial_object_z[obj] is None or max_object_z[obj] is None
        else float(max_object_z[obj] - initial_object_z[obj])
        for obj in OBJECT_NAMES
    }

    def finite_or_none_dict(values: dict[str, float]) -> dict[str, float | None]:
        return {key: (float(value) if np.isfinite(value) else None) for key, value in values.items()}

    if replay_images:
        import imageio

        imageio.mimwrite(video_path, replay_images, fps=args.video_fps)

    summary = {
        "model_name": args.model_name,
        "model_kind": args.model_kind,
        "seed": seed,
        "status": "completed" if env_exception is None else "rollout_exception",
        "failure_reason": None if env_exception is None else env_exception["error"],
        "traceback": None if env_exception is None else env_exception["traceback"],
        "benchmark": "RoboCerebra",
        "task_type": "Ideal",
        "case": "case1",
        "bddl_file": bddl_file_path,
        "full_task_instruction": full_task_instruction,
        "policy_prompt": policy_prompt,
        "prompt_mode": args.prompt_mode,
        "evaluation_scope": evaluation_scope,
        "success_subgoals": success_subgoals,
        "task_steps": task_steps,
        "goal_json": str(task_dir / "goal.json"),
        "original_goal": goal,
        "baseline_checker": baseline_checker,
        "final_checker": final_checker,
        "full_task_success": int(final_checker["all_done"]),
        "scope_success": int(int(final_checker["total_completed"]) >= success_subgoals),
        "completed_subgoals": int(final_checker["total_completed"]),
        "subgoal_possible": len(subgoal_template),
        "subgoals": final_checker["subgoals"],
        "first_failed_subgoal": None if first_failed is None else first_failed["subgoal_index"],
        "first_failed_subgoal_detail": first_failed,
        "rollout_steps": rollout_steps,
        "max_steps": max_steps,
        "wait_steps": args.wait_steps,
        "action_chunks": chunk_count,
        "nan_or_inf_action": bool(nan_or_inf_action),
        "inference_time_sec_total": inference_time_total,
        "inference_time_sec_per_chunk_avg": inference_time_total / chunk_count if chunk_count else None,
        "initial_object_poses": initial_poses,
        "final_object_poses": final_poses,
        "min_eef_to_object_distance": min_distances,
        "final_eef_to_object_distance": final_distances,
        "min_eef_to_object_distance_at_gripper_command_close": finite_or_none_dict(
            min_distances_at_gripper_command_close
        ),
        "min_eef_to_object_distance_at_gripper_qpos_close": finite_or_none_dict(
            min_distances_at_gripper_qpos_close
        ),
        "first_gripper_command_close_step": first_gripper_command_close_step,
        "first_gripper_qpos_close_step": first_gripper_qpos_close_step,
        "gripper_close_threshold": args.gripper_close_threshold,
        "gripper_close_direction": args.gripper_close_direction,
        "gripper_qpos_close_threshold": args.gripper_qpos_close_threshold,
        "contact_ever": contact_ever,
        "object_lift_height": object_lift_height,
        "closest_object_by_min_eef_distance": closest_object,
        "correct_object_approached": correct_object_approached,
        "correct_object_distance_threshold": args.correct_object_distance,
        "video_path": str(video_path) if replay_images else None,
        "step_log": str(step_log_path),
        "chunk_log": str(chunk_log_path),
        "backend_metadata": backend.metadata,
        "schema_notes": {
            "camera_mapping": {
                "front": "agentview_image",
                "wrist": "robot0_eye_in_hand_image",
            },
            "proprio_order": [
                "robot0_eef_pos_xyz",
                "robot0_eef_quat_axis_angle",
                "robot0_gripper_qpos_2",
            ],
            "env_action_dim": 7,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    env.close()
    print(json.dumps({"seed": seed, "summary": str(summary_path), "success": summary["full_task_success"], "completed_subgoals": summary["completed_subgoals"]}, ensure_ascii=False))
    return summary


def run_model(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_name": args.model_name,
        "model_kind": args.model_kind,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "bench_root": str(args.bench_root),
        "task": "Ideal/case1",
        "seeds": args.seeds,
        "max_steps": args.max_steps,
        "wait_steps": args.wait_steps,
        "full_task_instruction": read_full_task_instruction(args.bench_root / "Ideal" / "case1"),
        "lora_path": str(args.lora_path),
        "pi05_base": str(args.pi05_base),
        "norm_stats": str(args.norm_stats),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    try:
        backend = make_backend(args)
        backend_error = None
        backend_tb = None
    except Exception as exc:
        backend = None
        backend_error = exc
        backend_tb = traceback.format_exc()

    summaries = []
    for seed in args.seeds:
        summary_path = args.output_dir / f"seed_{seed}" / "summary.json"
        if args.skip_completed and summary_path.exists():
            try:
                existing = json.loads(summary_path.read_text(encoding="utf-8"))
                if existing.get("status") == "completed":
                    print(
                        json.dumps(
                            {
                                "seed": seed,
                                "summary": str(summary_path),
                                "skipped": True,
                                "reason": "--skip-completed",
                                "success": existing.get("full_task_success"),
                                "completed_subgoals": existing.get("completed_subgoals"),
                            },
                            ensure_ascii=False,
                        )
                    )
                    summaries.append(existing)
                    continue
            except Exception:
                pass
        if backend is None:
            summaries.append(write_failure_seed(args, seed, "backend_load_failed", backend_error, backend_tb or ""))
            continue
        try:
            summaries.append(run_seed(args, seed, backend))
        except Exception as exc:
            summaries.append(write_failure_seed(args, seed, "seed_run_failed", exc, traceback.format_exc()))

    (args.output_dir / "all_seed_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


def load_summaries(out_root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for model_dir in sorted(out_root.iterdir() if out_root.exists() else []):
        if not model_dir.is_dir():
            continue
        summaries = []
        for summary_path in sorted(model_dir.glob("seed_*/summary.json")):
            try:
                summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            except Exception:
                continue
        if summaries:
            result[model_dir.name] = summaries
    return result


def failure_type(summary: dict[str, Any]) -> str:
    if summary.get("status") not in {"completed", None}:
        return str(summary.get("status"))
    if summary.get("nan_or_inf_action"):
        return "nan_or_inf_action"
    if summary.get("full_task_success"):
        return "success"
    if not summary.get("correct_object_approached"):
        return "did_not_approach_first_target"
    if summary.get("completed_subgoals", 0) == 0:
        return "failed_first_pick"
    return f"failed_subgoal_{summary.get('first_failed_subgoal')}"


def nested_get(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_report(out_root: Path, report_path: Path) -> None:
    summaries_by_model = load_summaries(out_root)
    lines = [
        "# RoboCerebra Full-Task Model Comparison",
        "",
        "Task: `Ideal/case1`.",
        "",
        "Instruction used for every policy call: `Organize all the food boxes into the white storage box.`",
        "",
        "Checker: original `goal.json` with 6 ordered subgoals.",
        "",
        "No planner decomposition or paraphrase prompts are used.",
        "",
        "## Aggregate Results",
        "",
        "| Model | Rollouts completed | Full-task success | Avg completed subgoals | Most common failed subgoal | Main failure type |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for model_name, summaries in summaries_by_model.items():
        rollouts_completed = sum(1 for s in summaries if s.get("status") == "completed")
        success_count = sum(int(s.get("full_task_success", 0)) for s in summaries)
        avg_subgoals = float(np.mean([s.get("completed_subgoals", 0) for s in summaries])) if summaries else 0.0
        failed = [s.get("first_failed_subgoal") for s in summaries if s.get("first_failed_subgoal") is not None]
        failed_label = "none" if not failed else str(Counter(failed).most_common(1)[0][0])
        failure_label = Counter(failure_type(s) for s in summaries).most_common(1)[0][0] if summaries else "none"
        lines.append(
            f"| `{model_name}` | {rollouts_completed}/3 | {success_count}/3 | {avg_subgoals:.2f}/6 | {failed_label} | {failure_label} |"
        )

    lines.extend(["", "## Per-Seed Results", ""])
    for model_name, summaries in summaries_by_model.items():
        lines.extend([
            f"### `{model_name}`",
            "",
            "| Seed | Status | Success | Completed | First failed | Closest object | Min EEF cream cheese | Gripper-close EEF cream cheese | Gripper contact cream cheese | Lift cream cheese | Correct first object approached | NaN/Inf action | Steps | Inference time | Failure reason | Video |",
            "| ---: | --- | ---: | ---: | --- | --- | ---: | ---: | --- | ---: | --- | --- | ---: | ---: | --- | --- |",
        ])
        for s in sorted(summaries, key=lambda item: item.get("seed", -1)):
            video = s.get("video_path") or ""
            inf = s.get("inference_time_sec_total")
            inf_text = "" if inf is None else f"{inf:.2f}s"
            failure_reason = s.get("failure_reason") or ""
            first_obj = "cream_cheese_1"
            lines.append(
                "| {seed} | {status} | {succ} | {comp}/6 | {failed} | {closest} | {min_dist} | {close_dist} | {contact} | {lift} | {approach} | {nan} | {steps} | {inf} | {failure} | {video} |".format(
                    seed=s.get("seed"),
                    status=s.get("status"),
                    succ=s.get("full_task_success"),
                    comp=s.get("completed_subgoals"),
                    failed=s.get("first_failed_subgoal"),
                    closest=s.get("closest_object_by_min_eef_distance"),
                    min_dist=fmt_float(nested_get(s, "min_eef_to_object_distance", first_obj)),
                    close_dist=fmt_float(
                        nested_get(s, "min_eef_to_object_distance_at_gripper_command_close", first_obj)
                    ),
                    contact=nested_get(s, "contact_ever", first_obj, "gripper_contact"),
                    lift=fmt_float(nested_get(s, "object_lift_height", first_obj)),
                    approach=s.get("correct_object_approached"),
                    nan=s.get("nan_or_inf_action"),
                    steps=s.get("rollout_steps"),
                    inf=inf_text,
                    failure=failure_reason.replace("|", "\\|"),
                    video=video,
                )
            )
        lines.append("")

    lines.extend(["## Notes", ""])
    if not summaries_by_model:
        lines.append("- No model summaries were found yet.")
    lines.extend(
        [
            "- `correct_object_approached` uses the first target object (`cream_cheese_1`) and a conservative EEF-distance threshold from the run config.",
            "- If a model has `backend_load_failed`, its seed directories still contain `summary.json` and empty `step_log.jsonl` so the aggregate table remains stable.",
            "- The pi0.5 backend requires a Python environment containing both OpenPI/JAX and RoboCerebra/LIBERO/robosuite. On the current machine those dependencies are split across existing envs unless a combined env is created.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run-model", "report"], default="run-model")
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--model-kind", choices=["openvla", "pi05", "pi05_service"], default="openvla")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--use-proprio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_ROOT / "model")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--wait-steps", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--correct-object-distance", type=float, default=0.12)
    parser.add_argument("--gripper-close-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-close-direction", choices=["abs", "positive", "negative"], default="abs")
    parser.add_argument("--gripper-qpos-close-threshold", type=float, default=0.02)
    parser.add_argument("--openpi-root", type=Path, default=DEFAULT_OPENPI_ROOT)
    parser.add_argument("--pi05-base", type=Path, default=DEFAULT_PI05_BASE)
    parser.add_argument("--lora-path", type=Path, default=DEFAULT_PI05_LORA)
    parser.add_argument("--norm-stats", type=Path, default=DEFAULT_PI05_NORM)
    parser.add_argument("--pi05-sample-steps", type=int, default=5)
    parser.add_argument("--pi05-service-url", default=DEFAULT_PI05_SERVICE_URL)
    parser.add_argument("--pi05-service-timeout", type=float, default=600.0)
    parser.add_argument(
        "--execute-chunk-len",
        type=int,
        default=32,
        help="For pi0.5 service rollout, execute only the first N actions from each 32-step predicted chunk.",
    )
    parser.add_argument("--prompt-mode", choices=["full_task", "first_gt_subgoal"], default="full_task")
    parser.add_argument("--success-subgoals", type=int, default=None)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--openvla-oft", type=Path, default=DEFAULT_OPENVLA_OFT)
    parser.add_argument("--robocerebra-eval", type=Path, default=DEFAULT_ROBOCEREBRA_EVAL)
    parser.add_argument("--libero-root", type=Path, default=DEFAULT_LIBERO)
    args = parser.parse_args()
    add_eval_paths(args.openvla_oft, args.robocerebra_eval, args.libero_root)
    return args


def main() -> int:
    args = parse_args()
    if args.mode == "report":
        write_report(args.out_root, args.report_path)
        return 0
    return run_model(args)


if __name__ == "__main__":
    raise SystemExit(main())
