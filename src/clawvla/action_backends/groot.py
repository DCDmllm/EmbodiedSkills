from __future__ import annotations

import json
from importlib.util import find_spec
import os
from pathlib import Path
import socket
import subprocess
from typing import Any

from ..schema import ActionChunk, CameraView, MotionGoal, ObservationBundle, RobotArmState, WorldState
from .base import ActionBackendResult


class GrootActionBackend:
    name = "groot"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self._runtime: dict[str, Any] | None = None

    def build_action_chunk(
        self,
        motion_goal: MotionGoal | None,
        world_state: WorldState | None,
        observation: ObservationBundle | None,
        request: dict[str, Any],
    ) -> ActionBackendResult:
        if not self.config.get("enabled", False):
            return self._unavailable("groot_backend_disabled", motion_goal, request)
        pretrained_path = self.config.get("pretrained_path")
        if not pretrained_path:
            return self._unavailable("groot_pretrained_path_missing", motion_goal, request, {"diagnosis": self.diagnose()})
        if _is_missing_local_path(pretrained_path):
            return self._unavailable("groot_pretrained_path_not_found", motion_goal, request, {"diagnosis": self.diagnose()})
        if self._use_subprocess():
            try:
                chunk = self._build_action_chunk_subprocess(motion_goal, world_state, observation, request)
            except Exception as exc:
                return self._unavailable(
                    "groot_subprocess_inference_failed",
                    motion_goal,
                    request,
                    {"exception": f"{type(exc).__name__}: {exc}"},
                )
            return self._result_from_chunk(chunk)
        diagnosis = self.diagnose(load_policy=False)
        if not diagnosis.get("dependencies", {}).get("ok", False):
            return self._unavailable("groot_dependencies_unavailable", motion_goal, request, {"diagnosis": diagnosis})
        try:
            chunk = self._build_action_chunk_direct(motion_goal, world_state, observation, request)
        except Exception as exc:
            return self._unavailable(
                "groot_inference_failed",
                motion_goal,
                request,
                {"exception": f"{type(exc).__name__}: {exc}", "diagnosis": diagnosis},
            )
        return self._result_from_chunk(chunk)

    def action_spec(self) -> dict[str, Any]:
        summary = self._policy_summary()
        action_shape = summary.get("env_action_shape") or summary.get("action_shape") or self._configured_action_shape()
        env_action_dim = int(action_shape[0]) if isinstance(action_shape, list) and action_shape else None
        return {
            "backend": self.name,
            "checkpoint_format": summary.get("checkpoint_format"),
            "policy_type": summary.get("policy_type", "groot"),
            "model_action_dim": summary.get("model_action_dim"),
            "env_action_dim": env_action_dim,
            "types": {str(self.config.get("action_type") or "robocasa_action"): env_action_dim}
            if env_action_dim is not None
            else {},
            "horizon": summary.get("n_action_steps") or summary.get("chunk_size"),
        }

    def health(self) -> dict[str, Any]:
        if not self.config.get("enabled", False):
            return {"ok": False, "backend": self.name, "reason": "action_backend_disabled"}
        if not self.config.get("pretrained_path"):
            return {"ok": False, "backend": self.name, "reason": "action_backend_pretrained_path_missing"}
        if _is_missing_local_path(self.config.get("pretrained_path")):
            return {
                "ok": False,
                "backend": self.name,
                "reason": "action_backend_pretrained_path_not_found",
                "pretrained_path": str(self.config.get("pretrained_path")),
            }
        runtime_cfg = self._runtime_cfg()
        if runtime_cfg.get("mode") == "worker" and os.environ.get("CLAWVLA_GROOT_DIRECT") != "1":
            worker = _worker_health(runtime_cfg)
            return {
                "ok": bool(worker.get("ok")),
                "backend": self.name,
                "reason": "ok" if worker.get("ok") else f"groot_worker_{worker.get('reason')}",
                "worker": worker,
                "policy_summary": self._policy_summary(),
            }
        deps = _dependency_report()
        return {
            "ok": bool(deps.get("ok")),
            "backend": self.name,
            "reason": "ok" if deps.get("ok") else "groot_dependencies_unavailable",
            "dependencies": deps,
            "policy_summary": self._policy_summary(),
        }

    def diagnose(self, load_policy: bool = False) -> dict[str, Any]:
        report = {
            "backend": self.name,
            "status": "groot_diagnosed",
            "config": self.public_config(),
            "dependencies": _dependency_report(),
            "pretrained_path": self.config.get("pretrained_path"),
            "pretrained_path_is_missing_local_path": _is_missing_local_path(self.config.get("pretrained_path")),
            "policy_summary": self._policy_summary(),
            "policy_load": {"requested": bool(load_policy), "status": "not_requested"},
        }
        if load_policy:
            try:
                runtime = self._load_runtime()
                report["policy_load"] = {
                    "requested": True,
                    "status": "policy_loaded",
                    "policy_class": runtime["policy_class"],
                    "device": runtime["device"],
                    "input_features": runtime["input_features"],
                    "output_features": runtime["output_features"],
                    "pretrained_path": runtime["pretrained_path"],
                }
            except Exception as exc:
                report["policy_load"] = {
                    "requested": True,
                    "status": "policy_load_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return report

    def _use_subprocess(self) -> bool:
        if os.environ.get("CLAWVLA_GROOT_DIRECT") == "1":
            return False
        return self._runtime_cfg().get("mode") in {"subprocess", "worker"}

    def _runtime_cfg(self) -> dict[str, Any]:
        runtime_cfg = self.config.get("runtime", {})
        return dict(runtime_cfg) if isinstance(runtime_cfg, dict) else {}

    def _build_action_chunk_subprocess(
        self,
        motion_goal: MotionGoal | None,
        world_state: WorldState | None,
        observation: ObservationBundle | None,
        request: dict[str, Any],
    ) -> ActionChunk:
        runtime_cfg = self._runtime_cfg()
        artifact_dir = _artifact_dir_from_observation(observation)
        if runtime_cfg.get("mode") == "worker":
            return self._request_worker(motion_goal, world_state, observation, request, runtime_cfg, artifact_dir)

        output_path = artifact_dir / "groot_action_chunk.json"
        command = [
            *_python_command_prefix(runtime_cfg),
            "-m",
            "clawvla.scripts.groot_inference_smoke",
            "--config",
            str(runtime_cfg.get("config_path") or "/mnt/wangwai/vla/clawvla/configs/robocasa_groot_enabled_probe.json"),
            "--artifact-dir",
            str(artifact_dir),
            "--horizon",
            str(request.get("horizon") or self.config.get("policy_kwargs", {}).get("n_action_steps") or 10),
            "--output",
            str(output_path),
        ]
        motion_plan = request.get("motion_plan")
        if isinstance(motion_plan, dict) and motion_plan.get("vla_prompt"):
            command.extend(["--prompt", str(motion_plan["vla_prompt"])])
        env = _subprocess_env(runtime_cfg)
        subprocess.run(command, check=True, env=env)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return _chunk_from_payload(payload["action_chunk"])

    def _request_worker(
        self,
        motion_goal: MotionGoal | None,
        world_state: WorldState | None,
        observation: ObservationBundle | None,
        request: dict[str, Any],
        runtime_cfg: dict[str, Any],
        artifact_dir: Path,
    ) -> ActionChunk:
        _ = (motion_goal, world_state, observation)
        prompt = _resolve_prompt(request)
        payload = {
            "artifact_dir": str(artifact_dir),
            "motion_plan": request.get("motion_plan"),
            "prompt": prompt,
            "horizon": request.get("horizon") or self.config.get("policy_kwargs", {}).get("n_action_steps") or 10,
        }
        host = str(runtime_cfg.get("host") or "127.0.0.1")
        port = int(runtime_cfg.get("port") or 8766)
        with socket.create_connection((host, port), timeout=float(runtime_cfg.get("timeout", 900.0))) as sock:
            sock.sendall((json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8"))
            response = _read_socket_json_line(sock)
        if response.get("success") is False:
            errors = response.get("errors")
            if not isinstance(errors, list):
                errors = []
            raise RuntimeError(f"{response.get('status')}: {';'.join(str(error) for error in errors)}")
        return _chunk_from_payload(response["action_chunk"])

    def _build_action_chunk_direct(
        self,
        motion_goal: MotionGoal | None,
        world_state: WorldState | None,
        observation: ObservationBundle | None,
        request: dict[str, Any],
    ) -> ActionChunk:
        runtime = self._load_runtime()
        inference = self._infer_actions(runtime, motion_goal, world_state, observation, request)
        commands = inference["commands"]
        metadata = {
            "backend": self.name,
            "status": "groot_action_chunk_built",
            "checkpoint_format": runtime["checkpoint_format"],
            "action_type": str(self.config.get("action_type") or "robocasa_action"),
            "prompt": inference["prompt"],
            "state_source": inference["state_source"],
            "image_sources": inference["image_sources"],
            "raw_action_shape": inference["raw_action_shape"],
            "model_action_dim": inference["model_action_dim"],
            "env_action_dim": inference["env_action_dim"],
            "command_shape": [len(commands), len(commands[0]) if commands else 0],
            "policy_class": runtime["policy_class"],
            "pretrained_path": runtime["pretrained_path"],
            "motion_goal": motion_goal.to_dict() if hasattr(motion_goal, "to_dict") else None,
            "request": dict(request),
        }
        return ActionChunk(
            action_type=str(self.config.get("action_type") or "robocasa_action"),
            commands=commands,
            control_horizon=len(commands),
            metadata=metadata,
        )

    def _load_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime

        import torch
        from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.groot.configuration_groot import GrootConfig
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

        pretrained_path = str(self.config.get("pretrained_path") or "")
        local_files_only = bool(self.config.get("local_files_only", False))
        if Path(pretrained_path).exists():
            local_files_only = True

        policy_kwargs = dict(self.config.get("policy_kwargs", {})) if isinstance(self.config.get("policy_kwargs"), dict) else {}
        device = str(self.config.get("device") or policy_kwargs.pop("device", "cuda"))
        try:
            policy_cfg = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_path,
                local_files_only=local_files_only,
                cli_overrides=_policy_cli_overrides(policy_kwargs),
            )
        except Exception:
            policy_cfg = GrootConfig(base_model_path=pretrained_path, **policy_kwargs)
        if getattr(policy_cfg, "type", "groot") != "groot":
            raise ValueError(f"groot_policy_type_mismatch:{getattr(policy_cfg, 'type', None)}")
        policy_cfg.device = device
        policy_cfg.pretrained_path = pretrained_path
        for key, value in policy_kwargs.items():
            if hasattr(policy_cfg, key):
                setattr(policy_cfg, key, value)

        dataset_stats = _load_dataset_stats(self.config.get("dataset_stats_path"))
        state_dim = int(self.config.get("state_dim") or getattr(policy_cfg, "max_state_dim", 64))
        env_action_dim = int(
            self.config.get("env_action_dim")
            or self.config.get("action_dim")
            or _action_dim_from_dataset_stats(dataset_stats)
            or _action_dim_from_config(policy_cfg)
            or getattr(policy_cfg, "max_action_dim", 32)
        )
        policy_kwargs_max_action_dim = (
            policy_kwargs.get("max_action_dim") if isinstance(policy_kwargs, dict) else None
        )
        model_action_dim = int(
            getattr(policy_cfg, "max_action_dim", None)
            or _model_action_dim_from_config(policy_cfg)
            or policy_kwargs_max_action_dim
            or 32
        )
        image_size = tuple(getattr(policy_cfg, "image_size", (224, 224)) or (224, 224))
        if not getattr(policy_cfg, "input_features", None):
            policy_cfg.input_features = {
                f"{OBS_IMAGES}.robot0_agentview_left": PolicyFeature(type=FeatureType.VISUAL, shape=(3, *image_size)),
                f"{OBS_IMAGES}.robot0_agentview_right": PolicyFeature(type=FeatureType.VISUAL, shape=(3, *image_size)),
                f"{OBS_IMAGES}.robot0_eye_in_hand": PolicyFeature(type=FeatureType.VISUAL, shape=(3, *image_size)),
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            }
        if not getattr(policy_cfg, "output_features", None):
            policy_cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(env_action_dim,))}

        policy = GrootPolicy.from_pretrained(
            pretrained_name_or_path=pretrained_path,
            config=policy_cfg,
            local_files_only=local_files_only,
        )
        policy.eval()
        if hasattr(policy, "to"):
            policy.to(torch.device(device))

        try:
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=pretrained_path if Path(pretrained_path).exists() else None,
                dataset_stats=dataset_stats,
            )
        except Exception:
            preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_cfg, dataset_stats=dataset_stats)

        self._runtime = {
            "torch": torch,
            "device": device,
            "policy_cfg": policy_cfg,
            "policy": policy,
            "policy_class": type(policy).__name__,
            "preprocessor": preprocessor,
            "postprocessor": postprocessor,
            "pretrained_path": pretrained_path,
            "checkpoint_format": "lerobot_or_groot",
            "input_features": _feature_dict(getattr(policy_cfg, "input_features", {})),
            "output_features": _feature_dict(getattr(policy_cfg, "output_features", {})),
            "model_action_dim": model_action_dim,
            "env_action_dim": env_action_dim,
        }
        return self._runtime

    def _infer_actions(
        self,
        runtime: dict[str, Any],
        motion_goal: MotionGoal | None,
        world_state: WorldState | None,
        observation: ObservationBundle | None,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        _ = (motion_goal, world_state)
        torch = runtime["torch"]
        policy = runtime["policy"]
        prompt = _resolve_prompt(request)
        frame, frame_metadata = _groot_frame(observation, prompt, self.config)
        batch = runtime["preprocessor"](frame)
        if hasattr(policy, "reset"):
            policy.reset()
        action_dim = self.action_spec().get("types", {}).get(str(self.config.get("action_type") or "robocasa_action"))
        horizon = int(request.get("horizon") or self.config.get("policy_kwargs", {}).get("n_action_steps") or 10)
        commands: list[list[float]] = []
        raw_shape: list[int] | None = None
        with torch.inference_mode():
            for _step in range(horizon):
                raw_action = policy.select_action(batch)
                raw_shape = list(raw_action.shape)
                action = runtime["postprocessor"](raw_action)
                if hasattr(action, "detach"):
                    array = action.detach().float().cpu().numpy()
                else:
                    import numpy as np

                    array = np.asarray(action, dtype=float)
                vector = array.reshape(-1)
                if action_dim is not None and int(action_dim) > 0 and vector.shape[0] != int(action_dim):
                    raise ValueError(f"groot_action_dim_mismatch:{vector.shape[0]}!={action_dim}")
                commands.append([float(item) for item in vector.tolist()])
        return {
            "commands": commands,
            "prompt": prompt,
            "state_source": frame_metadata["state_source"],
            "image_sources": frame_metadata["image_sources"],
            "raw_action_shape": raw_shape,
            "model_action_dim": runtime.get("model_action_dim"),
            "env_action_dim": runtime.get("env_action_dim"),
        }

    def _policy_summary(self) -> dict[str, Any]:
        pretrained_path = self.config.get("pretrained_path")
        if not pretrained_path:
            return {"status": "pretrained_path_missing"}
        root = Path(str(pretrained_path))
        if root.exists():
            config_path = root / "config.json"
            if config_path.exists():
                try:
                    payload = json.loads(config_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    return {"status": "config_json_unreadable", "config_path": str(config_path), "error": str(exc)}
                if payload.get("model_type") == "gr00t_n1_5":
                    policy_kwargs = self.config.get("policy_kwargs", {})
                    configured_steps = (
                        policy_kwargs.get("n_action_steps", payload.get("action_horizon", 16))
                        if isinstance(policy_kwargs, dict)
                        else payload.get("action_horizon", 16)
                    )
                    return {
                        "status": "raw_groot_config_loaded",
                        "checkpoint_format": "raw_groot",
                        "policy_type": "groot",
                        "state_shape": self._configured_state_shape(),
                        "env_action_shape": self._configured_action_shape(),
                        "action_shape": self._configured_action_shape(),
                        "chunk_size": payload.get("action_horizon"),
                        "n_action_steps": min(int(configured_steps), 16),
                        "model_type": payload.get("model_type"),
                        "action_horizon": payload.get("action_horizon"),
                        "model_action_dim": payload.get("action_dim"),
                        "compute_dtype": payload.get("compute_dtype"),
                    }
                return {
                    "status": "config_json_loaded",
                    "checkpoint_format": "lerobot",
                    "policy_type": payload.get("type", "groot"),
                    "input_features": payload.get("input_features", {}),
                    "output_features": payload.get("output_features", {}),
                    "state_shape": _feature_shape(payload.get("input_features", {}).get("observation.state"))
                    if isinstance(payload.get("input_features"), dict)
                    else None,
                    "action_shape": _feature_shape(payload.get("output_features", {}).get("action"))
                    if isinstance(payload.get("output_features"), dict)
                    else None,
                    "chunk_size": payload.get("chunk_size"),
                    "n_action_steps": payload.get("n_action_steps"),
                    "base_model_path": payload.get("base_model_path"),
                    "embodiment_tag": payload.get("embodiment_tag"),
                }
            return {"status": "config_json_missing", "checkpoint_format": "local_path", "path": str(root)}
        return {
            "status": "remote_or_base_model",
            "checkpoint_format": "remote_or_base_groot",
            "policy_type": "groot",
            "repo_id": str(pretrained_path),
            "action_shape": self._configured_action_shape(),
            "state_shape": self._configured_state_shape(),
        }

    def _configured_action_shape(self) -> list[int] | None:
        dim = self.config.get("env_action_dim") or self.config.get("action_dim")
        if dim is None and isinstance(self.config.get("policy_kwargs"), dict):
            dim = self.config["policy_kwargs"].get("max_action_dim")
        return [int(dim)] if dim is not None else None

    def _configured_state_shape(self) -> list[int] | None:
        dim = self.config.get("state_dim")
        if dim is None and isinstance(self.config.get("policy_kwargs"), dict):
            dim = self.config["policy_kwargs"].get("max_state_dim")
        return [int(dim)] if dim is not None else None

    def _unavailable(
        self,
        reason: str,
        motion_goal: MotionGoal | None,
        request: dict[str, Any],
        extra_metadata: dict[str, Any] | None = None,
    ) -> ActionBackendResult:
        metadata = {
            "backend": self.name,
            "status": "groot_unavailable",
            "reason": reason,
            "retryable": False,
            "motion_goal": motion_goal.to_dict() if hasattr(motion_goal, "to_dict") else None,
            "request": dict(request),
            "config": self.public_config(),
        }
        metadata.update(dict(extra_metadata or {}))
        chunk = ActionChunk(action_type="unavailable", commands=[], control_horizon=0, metadata=metadata)
        return ActionBackendResult(
            success=False,
            status="groot_unavailable",
            action_chunk=chunk,
            metadata=chunk.metadata,
            errors=[reason],
        )

    def _result_from_chunk(self, chunk: ActionChunk) -> ActionBackendResult:
        metadata = dict(chunk.metadata or {})
        if chunk.action_type == "unavailable" or not chunk.commands:
            reason = str(metadata.get("reason") or metadata.get("status") or "empty_action_chunk")
            return ActionBackendResult(
                success=False,
                status=str(metadata.get("status") or "groot_action_chunk_unavailable"),
                action_chunk=chunk,
                metadata=metadata,
                errors=[reason],
            )
        return ActionBackendResult(
            success=True,
            status=str(metadata.get("status") or "groot_action_chunk_built"),
            action_chunk=chunk,
            metadata=metadata,
            errors=[],
        )

    def public_config(self) -> dict[str, Any]:
        return {
            "type": self.config.get("type", "groot"),
            "enabled": bool(self.config.get("enabled", False)),
            "policy_type": self.config.get("policy_type", "groot"),
            "pretrained_path": self.config.get("pretrained_path"),
            "device": self.config.get("device"),
            "action_type": self.config.get("action_type", "robocasa_action"),
            "state_dim": self.config.get("state_dim"),
            "env_action_dim": self.config.get("env_action_dim") or self.config.get("action_dim"),
            "legacy_action_dim": self.config.get("action_dim"),
            "policy_kwargs": dict(self.config.get("policy_kwargs", {}))
            if isinstance(self.config.get("policy_kwargs"), dict)
            else {},
            "environment_adapter": dict(self.config.get("environment_adapter", {}))
            if isinstance(self.config.get("environment_adapter"), dict)
            else {},
            "image_mapping": dict(self.config.get("image_mapping", {}))
            if isinstance(self.config.get("image_mapping"), dict)
            else {},
            "runtime": dict(self.config.get("runtime", {})) if isinstance(self.config.get("runtime"), dict) else {},
        }


def _resolve_prompt(request: dict[str, Any]) -> str:
    motion_plan = request.get("motion_plan")
    if isinstance(motion_plan, dict) and motion_plan.get("vla_prompt"):
        return str(motion_plan["vla_prompt"])
    raise ValueError(
        "motion_plan.vla_prompt is required for GR00T execution; run motion.plan_motion first so the VLA "
        "receives only the current subgoal instruction."
    )


def _groot_frame(
    observation: ObservationBundle | None,
    prompt: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    import torch
    from PIL import Image

    if observation is None:
        raise ValueError("Observation is required for GR00T inference.")
    state, state_source = _state_from_observation(observation, config)
    image_mapping = _image_mapping(config)
    image_sources: dict[str, str] = {}
    frame: dict[str, Any] = {
        "observation.state": torch.as_tensor(state, dtype=torch.float32),
        "task": prompt,
    }
    for policy_key, camera_name in image_mapping.items():
        view = observation.camera_views.get(camera_name)
        if view is None or not view.rgb_path:
            raise ValueError(f"Missing GR00T RGB artifact for camera {camera_name}.")
        image = Image.open(view.rgb_path).convert("RGB")
        array = np.asarray(image, dtype=np.uint8).copy()
        tensor = torch.as_tensor(array, dtype=torch.float32).permute(2, 0, 1) / 255.0
        frame[policy_key] = tensor
        image_sources[policy_key] = str(view.rgb_path)
    return frame, {"state_source": state_source, "image_sources": image_sources}


def _image_mapping(config: dict[str, Any]) -> dict[str, str]:
    payload = config.get("image_mapping")
    if isinstance(payload, dict) and payload:
        return {str(key): str(value) for key, value in payload.items()}
    return {
        "observation.images.robot0_agentview_left": "robot0_agentview_left",
        "observation.images.robot0_agentview_right": "robot0_agentview_right",
        "observation.images.robot0_eye_in_hand": "robot0_eye_in_hand",
    }


def _state_from_observation(observation: ObservationBundle, config: dict[str, Any]) -> tuple[list[float], str]:
    configured_keys = config.get("state_keys")
    keys = [str(item) for item in configured_keys] if isinstance(configured_keys, list) else []
    keys.extend(["groot_state", "robocasa_state", "observation.state", "state"])
    for key in keys:
        value = observation.raw.get(key) if isinstance(observation.raw, dict) else None
        vector = _float_vector(value)
        if vector is not None:
            return vector, f"observation.raw.{key}"

    raw_ref = observation.raw.get("summary_ref") if isinstance(observation.raw, dict) else None
    if raw_ref:
        payload = json.loads(Path(str(raw_ref)).read_text(encoding="utf-8"))
        for key in keys:
            vector = _float_vector(payload.get(key))
            if vector is not None:
                return vector, f"observation.raw.summary_ref.{key}"

    for name, arm in (observation.robot_arms or {}).items():
        metadata = getattr(arm, "metadata", {}) if isinstance(arm, RobotArmState) else {}
        if isinstance(metadata, dict):
            for key in keys:
                vector = _float_vector(metadata.get(key))
                if vector is not None:
                    return vector, f"observation.robot_arms.{name}.metadata.{key}"
    raise ValueError("GR00T state vector is missing from observation.")


def _float_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return None
    try:
        return [float(item) for item in _flatten(value)]
    except (TypeError, ValueError):
        return None


def _flatten(value: list[Any]) -> list[Any]:
    result: list[Any] = []
    for item in value:
        if isinstance(item, list):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result


def _feature_shape(payload: Any) -> list[int] | None:
    if isinstance(payload, dict) and isinstance(payload.get("shape"), list):
        return [int(item) for item in payload["shape"]]
    shape = getattr(payload, "shape", None)
    if shape is None:
        return None
    return [int(item) for item in shape]


def _feature_dict(features: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(features, dict):
        return {}
    return {
        str(key): {
            "type": str(getattr(value, "type", value.get("type") if isinstance(value, dict) else None)),
            "shape": _feature_shape(value),
        }
        for key, value in features.items()
    }


def _action_dim_from_config(policy_cfg: Any) -> int | None:
    output_features = getattr(policy_cfg, "output_features", {})
    if isinstance(output_features, dict):
        shape = _feature_shape(output_features.get("action"))
        if shape:
            return int(shape[0])
    return None


def _model_action_dim_from_config(policy_cfg: Any) -> int | None:
    action_dim = getattr(policy_cfg, "action_dim", None)
    if action_dim is not None:
        return int(action_dim)
    action_head_cfg = getattr(policy_cfg, "action_head_cfg", None)
    if isinstance(action_head_cfg, dict) and action_head_cfg.get("action_dim") is not None:
        return int(action_head_cfg["action_dim"])
    return None


def _action_dim_from_dataset_stats(dataset_stats: dict[str, dict[str, Any]] | None) -> int | None:
    if not isinstance(dataset_stats, dict):
        return None
    action_stats = dataset_stats.get("action")
    if not isinstance(action_stats, dict):
        return None
    for key in ("mean", "min", "max", "std"):
        value = action_stats.get(key)
        if isinstance(value, list) and value:
            return len(value)
    return None


def _policy_cli_overrides(kwargs: dict[str, Any]) -> list[str]:
    overrides = []
    for key, value in kwargs.items():
        if value is None:
            continue
        rendered = "true" if isinstance(value, bool) and value else "false" if isinstance(value, bool) else str(value)
        overrides.append(f"--{key}={rendered}")
    return overrides


def _load_dataset_stats(path_value: Any) -> dict[str, dict[str, Any]] | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "new_embodiment" in payload:
        stats = payload["new_embodiment"].get("statistics", {})
        state_blocks = stats.get("state", {}) if isinstance(stats, dict) else {}
        action_blocks = stats.get("action", {}) if isinstance(stats, dict) else {}
        return {
            "observation.state": _flatten_stats_blocks(
                state_blocks,
                [
                    "base_position",
                    "base_rotation",
                    "end_effector_position_relative",
                    "end_effector_rotation_relative",
                    "gripper_qpos",
                ],
            ),
            "action": _flatten_stats_blocks(
                action_blocks,
                [
                    "base_motion",
                    "control_mode",
                    "end_effector_position",
                    "end_effector_rotation",
                    "gripper_close",
                ],
            ),
        }
    stats = payload.get("stats", payload)
    return stats if isinstance(stats, dict) else None


def _flatten_stats_blocks(blocks: dict[str, Any], order: list[str]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {key: [] for key in ("min", "max", "mean", "std", "q01", "q99")}
    for name in order:
        block = blocks.get(name, {}) if isinstance(blocks, dict) else {}
        for key in result:
            values = block.get(key, []) if isinstance(block, dict) else []
            result[key].extend(float(item) for item in values)
    return result


def _dependency_report() -> dict[str, Any]:
    report = {
        "lerobot": find_spec("lerobot") is not None,
        "torch": find_spec("torch") is not None,
        "flash_attn": find_spec("flash_attn") is not None,
        "transformers": find_spec("transformers") is not None,
    }
    report["ok"] = all(bool(value) for value in report.values())
    if report["torch"]:
        try:
            import torch

            report["torch_version"] = getattr(torch, "__version__", None)
            report["torch_cuda"] = getattr(torch.version, "cuda", None)
            report["cuda_available"] = bool(torch.cuda.is_available())
        except Exception as exc:
            report["torch_error"] = f"{type(exc).__name__}: {exc}"
            report["ok"] = False
    if report["flash_attn"]:
        try:
            import flash_attn

            report["flash_attn_version"] = getattr(flash_attn, "__version__", None)
        except Exception as exc:
            report["flash_attn_error"] = f"{type(exc).__name__}: {exc}"
            report["ok"] = False
    return report


def _is_missing_local_path(value: Any) -> bool:
    if not value:
        return False
    text = str(value)
    path = Path(text)
    return (path.is_absolute() or text.startswith(".")) and not path.exists()


def _artifact_dir_from_observation(observation: ObservationBundle | None) -> Path:
    if observation is None or not isinstance(observation.raw, dict) or not observation.raw.get("summary_ref"):
        raise ValueError("Observation raw.summary_ref is required for subprocess GR00T inference.")
    return Path(str(observation.raw["summary_ref"])).parent


def _chunk_from_payload(chunk_payload: dict[str, Any]) -> ActionChunk:
    return ActionChunk(
        action_type=str(chunk_payload["action_type"]),
        commands=[[float(item) for item in command] for command in chunk_payload["commands"]],
        control_horizon=int(chunk_payload.get("control_horizon") or len(chunk_payload["commands"])),
        metadata=dict(chunk_payload.get("metadata") or {}),
    )


def _read_socket_json_line(sock: socket.socket) -> dict[str, Any]:
    chunks: list[bytes] = []
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    data = b"".join(chunks).split(b"\n", 1)[0]
    if not data:
        raise RuntimeError("groot_worker_empty_response")
    return json.loads(data.decode("utf-8"))


def _worker_health(runtime_cfg: dict[str, Any]) -> dict[str, Any]:
    host = str(runtime_cfg.get("host") or "127.0.0.1")
    port = int(runtime_cfg.get("port") or 8766)
    try:
        with socket.create_connection((host, port), timeout=2.0) as sock:
            sock.sendall(b'{"op":"health"}\n')
            payload = sock.recv(4096).split(b"\n", 1)[0]
        response = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        return {"ok": False, "mode": "worker", "host": host, "port": port, "reason": "unreachable", "error": str(exc)}
    return {
        "ok": response.get("status") == "ok",
        "mode": "worker",
        "host": host,
        "port": port,
        "reason": response.get("status"),
        "response": response,
    }


def _python_command_prefix(runtime_cfg: dict[str, Any]) -> list[str]:
    conda_bin = Path(str(runtime_cfg.get("conda_bin") or "/mnt/wangwai/miniconda3/bin/conda"))
    env_name = str(runtime_cfg.get("conda_env") or "groot-py312")
    python_path = conda_bin.parent.parent / "envs" / env_name / "bin" / "python"
    if python_path.exists():
        return [str(python_path)]
    return [str(conda_bin), "run", "--no-capture-output", "-n", env_name, "python"]


def _subprocess_env(runtime_cfg: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = str(runtime_cfg.get("pythonpath") or "/mnt/wangwai/vla/clawvla/src:/mnt/wangwai/lerobot/src")
    env["CLAWVLA_GROOT_DIRECT"] = "1"
    if runtime_cfg.get("cuda_visible_devices"):
        env["CUDA_VISIBLE_DEVICES"] = str(runtime_cfg["cuda_visible_devices"])
    return env


def observation_from_artifact(artifact_dir: Path, prompt: str) -> ObservationBundle:
    image_dir = artifact_dir / "images"
    summary_path = artifact_dir / "raw_observation_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    camera_views: dict[str, CameraView] = {}
    for name in ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"):
        path = image_dir / f"{name}_rgb.png"
        camera_views[name] = CameraView(name=name, rgb_path=str(path))
    robot_arms = {
        "robot0": RobotArmState(
            arm_name="robot0",
            joint_positions=_float_vector(payload.get("robot0_joint_pos")),
            gripper_value=_mean_float(payload.get("robot0_gripper_qpos")),
            metadata={
                "groot_state": _float_vector(payload.get("groot_state") or payload.get("robocasa_state")),
                "state_source": "artifact_summary",
            },
        )
    }
    raw = dict(payload)
    raw["summary_ref"] = str(summary_path)
    return ObservationBundle(task_instruction=prompt, camera_views=camera_views, robot_arms=robot_arms, raw=raw)


def _mean_float(value: Any) -> float | None:
    vector = _float_vector(value)
    if not vector:
        return None
    return float(sum(vector) / len(vector))
