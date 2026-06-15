from __future__ import annotations

from contextlib import contextmanager
import json
from importlib.util import find_spec
import os
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace
from typing import Any

from ..notices import emit_status_notice
from ..schema import ActionChunk, MotionGoal, ObservationBundle, WorldState
from .base import ActionBackendResult


class Pi05ActionBackend:
    name = "pi05"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self._openpi_runtime: dict[str, Any] | None = None

    def build_action_chunk(
        self,
        motion_goal: MotionGoal | None,
        world_state: WorldState | None,
        observation: ObservationBundle | None,
        request: dict[str, Any],
    ) -> ActionBackendResult:
        if not self.config.get("enabled", False):
            return self._unavailable("pi05_backend_disabled", motion_goal, request)
        pretrained_path = self.config.get("pretrained_path")
        if not pretrained_path:
            return self._unavailable("pi05_pretrained_path_missing", motion_goal, request, {"diagnosis": self.diagnose()})
        if not Path(str(pretrained_path)).exists():
            return self._unavailable("pi05_pretrained_path_not_found", motion_goal, request, {"diagnosis": self.diagnose()})

        diagnosis = self.diagnose(load_policy=bool(request.get("load_policy_for_diagnosis", False)))
        summary = diagnosis.get("policy_summary", {})
        if isinstance(summary, dict) and summary.get("checkpoint_format") == "openpi":
            if not summary.get("openpi_src_available", False):
                return self._unavailable(
                    "pi05_openpi_source_unavailable",
                    motion_goal,
                    request,
                    {"diagnosis": diagnosis},
                )
            if self._use_openpi_subprocess():
                chunk = self._build_openpi_action_chunk_subprocess(motion_goal, world_state, observation, request)
                return self._result_from_chunk(chunk)
            chunk = self._build_openpi_action_chunk(motion_goal, world_state, observation, request, summary)
            return self._result_from_chunk(chunk)
        if find_spec("lerobot") is None:
            return self._unavailable("lerobot_not_installed_in_current_environment", motion_goal, request)
        adapter = diagnosis.get("robotwin_adapter", {})
        if not adapter.get("compatible_for_execution", False):
            return self._unavailable(
                "pi05_robotwin_action_adapter_not_configured",
                motion_goal,
                request,
                {"diagnosis": diagnosis},
            )
        return self._unavailable("pi05_robotwin_observation_adapter_not_implemented", motion_goal, request, {"diagnosis": diagnosis})

    def _use_openpi_subprocess(self) -> bool:
        runtime_cfg = self.config.get("openpi_runtime", {})
        if os.environ.get("CLAWVLA_PI05_DIRECT") == "1":
            return False
        return isinstance(runtime_cfg, dict) and runtime_cfg.get("mode") in {"subprocess", "worker"}

    def _build_openpi_action_chunk_subprocess(
        self,
        motion_goal: MotionGoal | None,
        world_state: WorldState | None,
        observation: ObservationBundle | None,
        request: dict[str, Any],
    ) -> ActionChunk:
        runtime_cfg = self.config.get("openpi_runtime", {})
        runtime_cfg = runtime_cfg if isinstance(runtime_cfg, dict) else {}
        artifact_dir = _artifact_dir_from_observation(observation)
        if runtime_cfg.get("mode") == "worker":
            return self._request_openpi_worker(motion_goal, world_state, observation, request, runtime_cfg, artifact_dir)
        output_path = artifact_dir / "pi05_action_chunk.json"
        prompt = _resolve_prompt(self, motion_goal, world_state, observation, request)
        command = [
            str(runtime_cfg.get("conda_bin") or "/mnt/wangwai/miniconda3/bin/conda"),
            "run",
            "--no-capture-output",
            "-n",
            str(runtime_cfg.get("conda_env") or "openpi-torch-py312"),
            "python",
            "-m",
            "clawvla.scripts.pi05_inference_smoke",
            "--config",
            str(runtime_cfg.get("config_path") or "/mnt/wangwai/vla/clawvla/configs/robotwin_pi05_enabled_probe.json"),
            "--artifact-dir",
            str(artifact_dir),
            "--prompt",
            prompt,
            "--num-steps",
            str(request.get("num_steps") or self.config.get("policy_kwargs", {}).get("sample_num_steps") or 10),
            "--horizon",
            str(request.get("horizon") or self.config.get("policy_kwargs", {}).get("n_action_steps") or 10),
            "--output",
            str(output_path),
        ]
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["PYTHONPATH"] = str(
            runtime_cfg.get("pythonpath") or "/mnt/wangwai/vla/clawvla/src:/mnt/wangwai/RoboTwin/policy/pi05/src"
        )
        env["CLAWVLA_PI05_DIRECT"] = "1"
        if runtime_cfg.get("cuda_visible_devices"):
            env["CUDA_VISIBLE_DEVICES"] = str(runtime_cfg["cuda_visible_devices"])
        subprocess.run(command, check=True, env=env)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        chunk_payload = payload["action_chunk"]
        return ActionChunk(
            action_type=str(chunk_payload["action_type"]),
            commands=[[float(item) for item in command] for command in chunk_payload["commands"]],
            control_horizon=int(chunk_payload.get("control_horizon") or len(chunk_payload["commands"])),
            metadata=dict(chunk_payload.get("metadata") or {}),
        )

    def _request_openpi_worker(
        self,
        motion_goal: MotionGoal | None,
        world_state: WorldState | None,
        observation: ObservationBundle | None,
        request: dict[str, Any],
        runtime_cfg: dict[str, Any],
        artifact_dir: Path,
    ) -> ActionChunk:
        prompt = _resolve_prompt(self, motion_goal, world_state, observation, request)
        payload = {
            "artifact_dir": str(artifact_dir),
            "prompt": prompt,
            "motion_plan": request.get("motion_plan"),
            "num_steps": request.get("num_steps") or self.config.get("policy_kwargs", {}).get("sample_num_steps") or 10,
            "horizon": request.get("horizon") or self.config.get("policy_kwargs", {}).get("n_action_steps") or 10,
        }
        host = str(runtime_cfg.get("host") or "127.0.0.1")
        port = int(runtime_cfg.get("port") or 8765)
        with socket.create_connection((host, port), timeout=float(runtime_cfg.get("timeout", 600.0))) as sock:
            sock.sendall((json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8"))
            response = _read_socket_json_line(sock)
        if response.get("success") is False:
            errors = response.get("errors")
            if not isinstance(errors, list):
                errors = []
            raise RuntimeError(f"{response.get('status')}: {';'.join(str(error) for error in errors)}")
        chunk_payload = response["action_chunk"]
        if not isinstance(chunk_payload, dict):
            raise RuntimeError(f"pi05_worker_missing_action_chunk: response_keys={sorted(str(key) for key in response.keys())}")
        return ActionChunk(
            action_type=str(chunk_payload["action_type"]),
            commands=[[float(item) for item in command] for command in chunk_payload["commands"]],
            control_horizon=int(chunk_payload.get("control_horizon") or len(chunk_payload["commands"])),
            metadata=dict(chunk_payload.get("metadata") or {}),
        )

    def _build_openpi_action_chunk(
        self,
        motion_goal: MotionGoal | None,
        world_state: WorldState | None,
        observation: ObservationBundle | None,
        request: dict[str, Any],
        summary: dict[str, Any],
    ) -> ActionChunk:
        runtime = self._load_openpi_torch_runtime(summary)
        inference = self._infer_openpi_actions(runtime, motion_goal, world_state, observation, request)
        commands = inference["commands"]
        metadata = {
            "backend": self.name,
            "status": "pi05_action_chunk_built",
            "checkpoint_format": "openpi",
            "action_type": self._robotwin_action_type(),
            "prompt": inference["prompt"],
            "state_source": inference["state_source"],
            "raw_action_shape": inference["raw_action_shape"],
            "command_shape": [len(commands), len(commands[0]) if commands else 0],
            "transforms": [
                "AlohaInputs(adapt_to_pi=True)",
                "Normalize(use_quantiles=True)",
                "TokenizePrompt(discrete_state_input=True)",
                "PadStatesAndActions(32)",
                "sample_actions",
                "Unnormalize(use_quantiles=True)",
                "AbsoluteActions(mask=6,-1,6,-1)",
                "AlohaOutputs(adapt_to_pi=True)",
            ],
            "motion_goal": motion_goal.to_dict() if hasattr(motion_goal, "to_dict") else None,
            "request": dict(request),
        }
        return ActionChunk(
            action_type=self._robotwin_action_type(),
            commands=commands,
            control_horizon=len(commands),
            metadata=metadata,
        )

    def _load_openpi_torch_runtime(self, summary: dict[str, Any]) -> dict[str, Any]:
        if self._openpi_runtime is not None:
            return self._openpi_runtime

        import safetensors.torch
        import sentencepiece
        import torch

        pretrained_path = Path(str(self.config.get("pretrained_path") or ""))
        model_path = pretrained_path / "model.safetensors"
        norm_stats_path = _norm_stats_path(pretrained_path, summary)
        tokenizer_path = _tokenizer_path(self)
        openpi_src = self._openpi_src()

        policy_kwargs = self.config.get("policy_kwargs", {})
        policy_kwargs = policy_kwargs if isinstance(policy_kwargs, dict) else {}
        model_config = SimpleNamespace(
            pi05=True,
            dtype=str(policy_kwargs.get("dtype") or "bfloat16"),
            paligemma_variant=str(policy_kwargs.get("paligemma_variant") or "gemma_2b"),
            action_expert_variant=str(policy_kwargs.get("action_expert_variant") or "gemma_300m"),
            action_horizon=int(summary.get("action_horizon") or policy_kwargs.get("action_horizon") or 32),
            action_dim=int(policy_kwargs.get("internal_action_dim") or 32),
            max_token_len=int(policy_kwargs.get("max_token_len") or 200),
        )

        with _openpi_torch_only_import_hooks(str(openpi_src)):
            from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

            model = PI0Pytorch(model_config)
            missing, unexpected = safetensors.torch.load_model(model, str(model_path), strict=False)
            model.to(torch.device(str(self.config.get("device") or "cuda")))
            model.eval()

        tokenizer = sentencepiece.SentencePieceProcessor()
        tokenizer.Load(str(tokenizer_path))
        self._openpi_runtime = {
            "model": model,
            "model_config": model_config,
            "torch": torch,
            "device": str(self.config.get("device") or "cuda"),
            "norm_stats": _load_norm_stats(norm_stats_path),
            "tokenizer": tokenizer,
            "model_path": str(model_path),
            "norm_stats_path": str(norm_stats_path),
            "tokenizer_path": str(tokenizer_path),
            "missing_weights": sorted(str(item) for item in missing),
            "unexpected_weights": sorted(str(item) for item in unexpected),
        }
        return self._openpi_runtime

    def _infer_openpi_actions(
        self,
        runtime: dict[str, Any],
        motion_goal: MotionGoal | None,
        world_state: WorldState | None,
        observation: ObservationBundle | None,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        torch = runtime["torch"]
        device = runtime["device"]
        model = runtime["model"]
        model_config = runtime["model_config"]
        norm_stats = runtime["norm_stats"]
        prompt = _resolve_prompt(self, motion_goal, world_state, observation, request)
        state14, state_source = _state_from_observation(observation)
        decoded_state14 = _decode_aloha_state(state14, adapt_to_pi=True)
        normalized_state14 = _normalize_quantile(decoded_state14, norm_stats["state"])
        padded_state = _pad_last_dim(normalized_state14, int(model_config.action_dim))
        token_ids, token_mask = _tokenize_prompt(
            tokenizer=runtime["tokenizer"],
            prompt=prompt,
            state=normalized_state14,
            max_len=int(model_config.max_token_len),
        )
        obs = SimpleNamespace(
            images=_image_tensors_from_observation(observation, torch, device),
            image_masks={key: torch.ones((1,), dtype=torch.bool, device=device) for key in _openpi_image_keys()},
            state=torch.as_tensor(padded_state[None, :], dtype=torch.float32, device=device),
            tokenized_prompt=torch.as_tensor(token_ids[None, :], dtype=torch.long, device=device),
            tokenized_prompt_mask=torch.as_tensor(token_mask[None, :], dtype=torch.bool, device=device),
            token_ar_mask=torch.zeros((1, int(model_config.max_token_len)), dtype=torch.bool, device=device),
            token_loss_mask=torch.zeros((1, int(model_config.max_token_len)), dtype=torch.bool, device=device),
        )
        num_steps = int(request.get("num_steps") or self.config.get("policy_kwargs", {}).get("sample_num_steps") or 10)
        horizon = int(request.get("horizon") or self.config.get("policy_kwargs", {}).get("n_action_steps") or 10)
        with torch.inference_mode(), _openpi_torch_only_import_hooks(str(self._openpi_src())):
            raw_actions = model.sample_actions(torch.device(device), obs, num_steps=num_steps)
        raw_np = raw_actions.detach().float().cpu().numpy()[0]
        deltas14 = _unnormalize_quantile(raw_np[:, :14], norm_stats["actions"])
        absolute_pi14 = _absolute_actions(deltas14, decoded_state14)
        commands = _encode_aloha_actions(absolute_pi14[:horizon], adapt_to_pi=True).astype(float).tolist()
        return {
            "commands": commands,
            "prompt": prompt,
            "state_source": state_source,
            "raw_action_shape": list(raw_np.shape),
        }

    def diagnose(self, load_policy: bool = False) -> dict[str, Any]:
        pretrained_path = self.config.get("pretrained_path")
        summary = self._policy_summary(pretrained_path)
        report: dict[str, Any] = {
            "backend": self.name,
            "status": "pi05_diagnosed",
            "config": self.public_config(),
            "lerobot_available": find_spec("lerobot") is not None,
            "openpi_import": self._openpi_import_summary(),
            "pretrained_path": pretrained_path,
            "pretrained_path_exists": bool(pretrained_path and Path(str(pretrained_path)).exists()),
            "tokenizer_name": self._resolve_tokenizer_name(),
            "policy_summary": summary,
            "robotwin_adapter": self._robotwin_adapter_summary(summary),
            "policy_load": {"requested": bool(load_policy), "status": "not_requested"},
        }
        if load_policy:
            report["policy_load"] = self._load_policy_for_diagnosis()
        return report

    def _load_policy_for_diagnosis(self) -> dict[str, Any]:
        summary = self._policy_summary(self.config.get("pretrained_path"))
        if summary.get("checkpoint_format") == "openpi":
            return self._load_openpi_policy_for_diagnosis(summary)
        try:
            stack = self._load_policy_stack()
        except Exception as exc:
            return {"requested": True, "status": "policy_load_failed", "error": f"{type(exc).__name__}: {exc}"}
        policy_cfg = stack.get("policy_cfg")
        env_cfg = stack.get("env_cfg")
        return {
            "requested": True,
            "status": "policy_loaded",
            "policy_type": getattr(policy_cfg, "type", None),
            "device": str(getattr(policy_cfg, "device", None)),
            "pretrained_path": getattr(policy_cfg, "pretrained_path", None),
            "tokenizer_name": self._resolve_tokenizer_name(),
            "env_type": getattr(env_cfg, "type", None),
            "env_task": getattr(env_cfg, "task", None),
            "input_features": _feature_dict(getattr(policy_cfg, "input_features", {})),
            "output_features": _feature_dict(getattr(policy_cfg, "output_features", {})),
        }

    def _load_openpi_policy_for_diagnosis(self, summary: dict[str, Any]) -> dict[str, Any]:
        try:
            stack = self._load_openpi_policy_stack(summary)
        except Exception as exc:
            hook_report = self._load_openpi_torch_hook_for_diagnosis(summary, exc)
            if hook_report.get("status") != "policy_load_failed":
                return hook_report
            return {
                "requested": True,
                "status": "policy_load_failed",
                "checkpoint_format": "openpi",
                "error": f"{type(exc).__name__}: {exc}",
                "hint": "Install the RoboTwin policy/pi05 OpenPI dependencies or run with an environment that provides them.",
                "required_pythonpath": self._openpi_src(),
                "fallback_policy_load": hook_report,
            }
        policy = stack.get("policy")
        train_config = stack.get("train_config")
        return {
            "requested": True,
            "status": "policy_loaded",
            "checkpoint_format": "openpi",
            "train_config_name": getattr(train_config, "name", None),
            "asset_id": stack.get("asset_id"),
            "policy_class": type(policy).__name__,
            "device": self.config.get("device", "cuda"),
            "input_features": summary.get("input_features", {}),
            "output_features": summary.get("output_features", {}),
        }

    def _load_openpi_torch_hook_for_diagnosis(
        self,
        summary: dict[str, Any],
        official_error: Exception | None = None,
    ) -> dict[str, Any]:
        pretrained_path = Path(str(self.config.get("pretrained_path") or ""))
        model_path = pretrained_path / "model.safetensors"
        openpi_src = self._openpi_src()
        if not model_path.exists():
            return {
                "requested": True,
                "status": "policy_load_failed",
                "checkpoint_format": "openpi",
                "strategy": "torch_hook_best_effort",
                "error": f"model.safetensors not found at {model_path}",
            }
        if not openpi_src or not Path(openpi_src).exists():
            return {
                "requested": True,
                "status": "policy_load_failed",
                "checkpoint_format": "openpi",
                "strategy": "torch_hook_best_effort",
                "error": "RoboTwin OpenPI source path is missing.",
                "required_pythonpath": openpi_src,
            }

        try:
            import safetensors.torch
            import torch
            from types import SimpleNamespace
        except Exception as exc:
            return {
                "requested": True,
                "status": "policy_load_failed",
                "checkpoint_format": "openpi",
                "strategy": "torch_hook_best_effort",
                "error": f"{type(exc).__name__}: {exc}",
                "hint": "The torch-only hook still requires torch and safetensors.",
            }

        policy_kwargs = self.config.get("policy_kwargs", {})
        policy_kwargs = policy_kwargs if isinstance(policy_kwargs, dict) else {}
        config = SimpleNamespace(
            pi05=True,
            dtype=str(policy_kwargs.get("dtype") or "bfloat16"),
            paligemma_variant=str(policy_kwargs.get("paligemma_variant") or "gemma_2b"),
            action_expert_variant=str(policy_kwargs.get("action_expert_variant") or "gemma_300m"),
            action_horizon=int(summary.get("action_horizon") or policy_kwargs.get("action_horizon") or 32),
            action_dim=int(policy_kwargs.get("internal_action_dim") or 32),
            max_token_len=int(policy_kwargs.get("max_token_len") or 200),
        )
        strict_load = bool(policy_kwargs.get("torch_hook_strict_load", False))
        try:
            with _openpi_torch_only_import_hooks(openpi_src):
                from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

                model = PI0Pytorch(config)
                missing, unexpected = safetensors.torch.load_model(model, str(model_path), strict=strict_load)
        except Exception as exc:
            return {
                "requested": True,
                "status": "policy_load_failed",
                "checkpoint_format": "openpi",
                "strategy": "torch_hook_best_effort",
                "official_openpi_error": f"{type(official_error).__name__}: {official_error}"
                if official_error
                else None,
                "error": f"{type(exc).__name__}: {exc}",
                "hint": "The version check is bypassed, but the installed Transformers runtime may still be structurally incompatible.",
            }

        missing = sorted(str(item) for item in missing)
        unexpected = sorted(str(item) for item in unexpected)
        status = "policy_loaded" if not missing and not unexpected else "policy_loaded_with_weight_mismatch"
        return {
            "requested": True,
            "status": status,
            "checkpoint_format": "openpi",
            "strategy": "torch_hook_best_effort",
            "official_openpi_error": f"{type(official_error).__name__}: {official_error}" if official_error else None,
            "version_check_bypassed": True,
            "flax_jax_bypassed": True,
            "safe_for_execution": False,
            "execution_block_reason": "torch-only hook has not validated exact OpenPI transforms, normalization, and action semantics.",
            "model_class": type(model).__name__,
            "model_path": str(model_path),
            "torch_version": getattr(torch, "__version__", None),
            "transformers_version": _package_version("transformers"),
            "config": {
                "pi05": config.pi05,
                "dtype": config.dtype,
                "paligemma_variant": config.paligemma_variant,
                "action_expert_variant": config.action_expert_variant,
                "action_horizon": config.action_horizon,
                "internal_action_dim": config.action_dim,
                "max_token_len": config.max_token_len,
            },
            "strict_load": strict_load,
            "missing_weight_count": len(missing),
            "unexpected_weight_count": len(unexpected),
            "missing_weight_sample": missing[:20],
            "unexpected_weight_sample": unexpected[:20],
            "input_features": summary.get("input_features", {}),
            "output_features": summary.get("output_features", {}),
        }

    def _load_openpi_policy_stack(self, summary: dict[str, Any]) -> dict[str, Any]:
        openpi_src = self._openpi_src()
        if openpi_src:
            import sys

            if openpi_src not in sys.path:
                sys.path.insert(0, openpi_src)

        from openpi.models import pi0_config
        from openpi.policies import policy_config as openpi_policy_config
        from openpi.training import config as openpi_config
        from openpi.training import optimizer as openpi_optimizer
        from openpi.training import weight_loaders
        import openpi.transforms as openpi_transforms

        policy_kwargs = self.config.get("policy_kwargs", {})
        policy_kwargs = policy_kwargs if isinstance(policy_kwargs, dict) else {}
        train_config_name = str(
            policy_kwargs.get("train_config_name")
            or "pi05_base_finetune_on_robotwin_clean_randomized_joint_training"
        )
        asset_id = str(policy_kwargs.get("asset_id") or summary.get("asset_id") or "pi0.5_clean_randomize_joint_training")
        train_config = _openpi_train_config(train_config_name, asset_id, Path(str(self.config.get("pretrained_path"))))
        policy = openpi_policy_config.create_trained_policy(
            train_config,
            str(self.config.get("pretrained_path")),
            robotwin_repo_id=asset_id,
            pytorch_device=str(self.config.get("device") or "cuda"),
        )
        return {
            "policy": policy,
            "train_config": train_config,
            "asset_id": asset_id,
            "modules": {
                "pi0_config": pi0_config,
                "optimizer": openpi_optimizer,
                "weight_loaders": weight_loaders,
                "transforms": openpi_transforms,
            },
        }

    def _load_policy_stack(self) -> dict[str, Any]:
        from lerobot.configs import PreTrainedConfig
        from lerobot.envs import make_env_pre_post_processors
        from lerobot.envs.configs import LiberoEnv
        from lerobot.policies import make_policy, make_policy_config, make_pre_post_processors

        policy_type = str(self.config.get("policy_type", "pi05"))
        pretrained_path = str(self.config.get("pretrained_path") or "")
        kwargs = dict(self.config.get("policy_kwargs", {})) if isinstance(self.config.get("policy_kwargs"), dict) else {}
        kwargs["device"] = self.config.get("device", "cuda")
        try:
            policy_cfg = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_path,
                local_files_only=True,
                cli_overrides=_policy_cli_overrides(kwargs),
            )
        except Exception as exc:
            emit_status_notice(
                "lerobot_pretrained_config_unavailable",
                success=True,
                source="pi05._load_policy_stack",
                reason=f"{type(exc).__name__}: {exc}",
                payload={"fallback": "make_policy_config"},
            )
            kwargs["pretrained_path"] = pretrained_path
            policy_cfg = make_policy_config(policy_type, **kwargs)
        policy_cfg.pretrained_path = pretrained_path

        env_cfg_payload = self.config.get("lerobot_env", {})
        env_cfg_payload = env_cfg_payload if isinstance(env_cfg_payload, dict) else {}
        env_cfg = LiberoEnv(
            task=str(env_cfg_payload.get("task", "libero_object")),
            obs_type=str(env_cfg_payload.get("obs_type", "pixels_agent_pos")),
        )
        policy = make_policy(policy_cfg, env_cfg=env_cfg)
        policy.eval()
        preprocessor_overrides = {"device_processor": {"device": str(policy.config.device)}}
        tokenizer_name = self._resolve_tokenizer_name()
        if tokenizer_name:
            preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": tokenizer_name}
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_cfg.pretrained_path,
            preprocessor_overrides=preprocessor_overrides,
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
        return {
            "env_cfg": env_cfg,
            "policy_cfg": policy_cfg,
            "policy": policy,
            "preprocessor": preprocessor,
            "postprocessor": postprocessor,
            "env_preprocessor": env_preprocessor,
            "env_postprocessor": env_postprocessor,
        }

    def _policy_summary(self, pretrained_path: Any) -> dict[str, Any]:
        if not pretrained_path:
            return {"status": "pretrained_path_missing"}
        root = Path(str(pretrained_path))
        config_path = root / "config.json"
        if not config_path.exists():
            openpi_summary = self._openpi_checkpoint_summary(root)
            if openpi_summary["status"] != "openpi_checkpoint_missing":
                return openpi_summary
            return {"status": "config_json_missing", "config_path": str(config_path)}
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"status": "config_json_unreadable", "config_path": str(config_path), "error": str(exc)}
        return {
            "status": "config_json_loaded",
            "checkpoint_format": "lerobot",
            "policy_type": payload.get("type"),
            "input_features": payload.get("input_features", {}),
            "output_features": payload.get("output_features", {}),
            "chunk_size": payload.get("chunk_size"),
            "n_action_steps": payload.get("n_action_steps"),
            "image_resolution": payload.get("image_resolution"),
        }

    def _openpi_checkpoint_summary(self, root: Path) -> dict[str, Any]:
        model_path = root / "model.safetensors"
        metadata_path = root / "metadata.pt"
        norm_stats_candidates = sorted(root.glob("assets/*/norm_stats.json"))
        if not model_path.exists() and not norm_stats_candidates:
            return {"status": "openpi_checkpoint_missing", "checkpoint_format": "unknown"}

        norm_summary: dict[str, Any] = {"available": False}
        action_shape = None
        state_shape = None
        asset_id = None
        norm_stats_path = None
        if norm_stats_candidates:
            norm_stats_path = norm_stats_candidates[0]
            asset_id = norm_stats_path.parent.name
            norm_summary = _norm_stats_summary(norm_stats_path)
            state_shape = norm_summary.get("state_shape")
            action_shape = norm_summary.get("action_shape")

        return {
            "status": "openpi_checkpoint_loaded",
            "checkpoint_format": "openpi",
            "policy_type": self.config.get("policy_type", "pi05"),
            "model_path": str(model_path),
            "model_path_exists": model_path.exists(),
            "metadata_path": str(metadata_path),
            "metadata_path_exists": metadata_path.exists(),
            "norm_stats_path": str(norm_stats_path) if norm_stats_path else None,
            "asset_id": asset_id,
            "norm_stats": norm_summary,
            "input_features": {
                "images.cam_high": {"type": "image", "shape": None},
                "images.cam_left_wrist": {"type": "image", "shape": None},
                "images.cam_right_wrist": {"type": "image", "shape": None},
                "state": {"type": "state", "shape": state_shape},
                "prompt": {"type": "text", "shape": None},
            },
            "output_features": {"action": {"type": "action", "shape": action_shape}},
            "action_horizon": 32,
            "image_resolution": [224, 224],
            "uses_delta_joint_actions": True,
            "action_semantics_after_policy_output": "absolute_qpos_after_openpi_absolute_actions_transform",
            "openpi_src": self._openpi_src(),
            "openpi_src_available": bool(self._openpi_src() and Path(str(self._openpi_src())).exists()),
        }

    def _robotwin_adapter_summary(self, policy_summary: dict[str, Any]) -> dict[str, Any]:
        adapter_cfg = self.config.get("robotwin_adapter", {})
        adapter_cfg = adapter_cfg if isinstance(adapter_cfg, dict) else {}
        mode = str(adapter_cfg.get("mode", "diagnostic_only"))
        output_features = policy_summary.get("output_features", {}) if isinstance(policy_summary, dict) else {}
        action_shape = _feature_shape(output_features.get("action")) if isinstance(output_features, dict) else None
        checkpoint_format = policy_summary.get("checkpoint_format") if isinstance(policy_summary, dict) else None
        expects_openpi = checkpoint_format == "openpi"
        compatible = mode not in {"diagnostic_only", "unconfigured"} and action_shape in ([14], [16])
        reason = None
        if not compatible:
            if mode in {"diagnostic_only", "unconfigured"}:
                reason = "robotwin adapter is diagnostic-only; execution is disabled until policy inference and action semantics are validated"
            elif action_shape in ([7],):
                reason = "current pi05 checkpoint is LIBERO-style 7D single-arm action; RoboTwin take_action expects dual-arm qpos(14D) or ee(16D)"
            else:
                reason = "policy action shape is not compatible with RoboTwin qpos(14D) or ee(16D) command"
        return {
            "mode": mode,
            "compatible_for_execution": compatible,
            "checkpoint_format": checkpoint_format,
            "expects_openpi_policy": expects_openpi,
            "policy_action_shape": action_shape,
            "robotwin_qpos_command_shape": [14],
            "robotwin_ee_command_shape": [16],
            "reason": reason,
            "image_mapping": adapter_cfg.get("image_mapping"),
            "state_mapping": adapter_cfg.get("state_mapping"),
            "action_type": adapter_cfg.get("action_type", "qpos"),
            "uses_delta_joint_actions": policy_summary.get("uses_delta_joint_actions")
            if isinstance(policy_summary, dict)
            else None,
        }

    def _openpi_import_summary(self) -> dict[str, Any]:
        openpi_src = self._openpi_src()
        return {
            "importable": find_spec("openpi") is not None,
            "openpi_src": openpi_src,
            "openpi_src_exists": bool(openpi_src and Path(str(openpi_src)).exists()),
            "required_pythonpath": openpi_src,
        }

    def _openpi_src(self) -> str | None:
        configured = self.config.get("openpi_src")
        if configured:
            return str(configured)
        robotwin_adapter = self.config.get("robotwin_adapter", {})
        if isinstance(robotwin_adapter, dict) and robotwin_adapter.get("openpi_src"):
            return str(robotwin_adapter.get("openpi_src"))
        default = Path("/mnt/wangwai/RoboTwin/policy/pi05/src")
        return str(default) if default.exists() else None

    def _resolve_tokenizer_name(self) -> str | None:
        configured = self.config.get("tokenizer_name")
        if configured:
            return str(configured)
        snapshot = _local_hf_snapshot("models--google--paligemma-3b-pt-224")
        if snapshot is not None and (snapshot / "tokenizer.model").exists():
            return str(snapshot)
        return None

    def _robotwin_action_type(self) -> str:
        adapter_cfg = self.config.get("robotwin_adapter", {})
        if isinstance(adapter_cfg, dict) and adapter_cfg.get("action_type"):
            return str(adapter_cfg["action_type"])
        return "qpos"

    def _unavailable(
        self,
        reason: str,
        motion_goal: MotionGoal | None,
        request: dict[str, Any],
        extra_metadata: dict[str, Any] | None = None,
    ) -> ActionBackendResult:
        metadata = {
            "backend": self.name,
            "status": "pi05_unavailable",
            "reason": reason,
            "retryable": False,
            "motion_goal": motion_goal.to_dict() if hasattr(motion_goal, "to_dict") else None,
            "request": dict(request),
            "config": self.public_config(),
        }
        metadata.update(dict(extra_metadata or {}))
        chunk = ActionChunk(
            action_type="unavailable",
            commands=[],
            control_horizon=0,
            metadata=metadata,
        )
        return ActionBackendResult(
            success=False,
            status="pi05_unavailable",
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
                status=str(metadata.get("status") or "pi05_action_chunk_unavailable"),
                action_chunk=chunk,
                metadata=metadata,
                errors=[reason],
            )
        return ActionBackendResult(
            success=True,
            status=str(metadata.get("status") or "pi05_action_chunk_built"),
            action_chunk=chunk,
            metadata=metadata,
            errors=[],
        )

    def public_config(self) -> dict[str, Any]:
        return {
            "type": self.config.get("type", "pi05"),
            "enabled": bool(self.config.get("enabled", False)),
            "policy_type": self.config.get("policy_type", "pi05"),
            "pretrained_path": self.config.get("pretrained_path"),
            "device": self.config.get("device"),
            "openpi_src": self.config.get("openpi_src"),
            "policy_kwargs": dict(self.config.get("policy_kwargs", {}))
            if isinstance(self.config.get("policy_kwargs"), dict)
            else {},
            "tokenizer_name": self.config.get("tokenizer_name"),
            "lerobot_env": dict(self.config.get("lerobot_env", {}))
            if isinstance(self.config.get("lerobot_env"), dict)
            else {},
            "robotwin_adapter": dict(self.config.get("robotwin_adapter", {}))
            if isinstance(self.config.get("robotwin_adapter"), dict)
            else {},
        }


def _policy_cli_overrides(kwargs: dict[str, Any]) -> list[str]:
    overrides = []
    for key, value in kwargs.items():
        if value is None:
            continue
        rendered = "true" if isinstance(value, bool) and value else "false" if isinstance(value, bool) else str(value)
        overrides.append(f"--{key}={rendered}")
    return overrides


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


def _norm_stats_summary(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"available": False, "path": str(path), "error": str(exc)}
    norm_stats = payload.get("norm_stats", payload)
    state = norm_stats.get("state", {}) if isinstance(norm_stats, dict) else {}
    actions = norm_stats.get("actions", {}) if isinstance(norm_stats, dict) else {}
    state_shape = _stats_shape(state)
    action_shape = _stats_shape(actions)
    return {
        "available": True,
        "path": str(path),
        "state_shape": state_shape,
        "action_shape": action_shape,
        "keys": sorted(str(key) for key in norm_stats.keys()) if isinstance(norm_stats, dict) else [],
        "state_stats": _stats_keys_and_lengths(state),
        "action_stats": _stats_keys_and_lengths(actions),
    }


def _stats_shape(stats: Any) -> list[int] | None:
    if not isinstance(stats, dict):
        return None
    for key in ("mean", "q01", "std", "q99"):
        value = stats.get(key)
        if isinstance(value, list):
            return [len(value)]
    return None


def _stats_keys_and_lengths(stats: Any) -> dict[str, int]:
    if not isinstance(stats, dict):
        return {}
    return {str(key): len(value) for key, value in stats.items() if isinstance(value, list)}


def _norm_stats_path(pretrained_path: Path, summary: dict[str, Any]) -> Path:
    configured = summary.get("norm_stats_path")
    if configured:
        return Path(str(configured))
    candidates = sorted(pretrained_path.glob("assets/*/norm_stats.json"))
    if not candidates:
        raise FileNotFoundError(f"norm_stats.json not found under {pretrained_path / 'assets'}")
    return candidates[0]


def _tokenizer_path(backend: Pi05ActionBackend) -> Path:
    resolved = backend._resolve_tokenizer_name()
    if not resolved:
        raise FileNotFoundError("PaliGemma tokenizer path is not configured and no local snapshot was found.")
    path = Path(resolved)
    return path / "tokenizer.model" if path.is_dir() else path


def _load_norm_stats(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("norm_stats", payload)


def _resolve_prompt(
    backend: Pi05ActionBackend,
    motion_goal: MotionGoal | None,
    world_state: WorldState | None,
    observation: ObservationBundle | None,
    request: dict[str, Any],
) -> str:
    _ = (backend, motion_goal, world_state, observation)
    motion_plan = request.get("motion_plan")
    if isinstance(motion_plan, dict) and motion_plan.get("vla_prompt"):
        return str(motion_plan["vla_prompt"])
    raise ValueError(
        "motion_plan.vla_prompt is required for VLA execution; refusing to fall back to request.prompt, "
        "observation.task_instruction, world_state.task_instruction, motion_goal.motion_hint, default_prompt, "
        "or 'perform the task'. Run motion.plan_motion first so the VLA receives only the current subgoal instruction."
    )


def _state_from_observation(observation: ObservationBundle | None) -> tuple["np.ndarray", str]:
    import numpy as np

    if observation is not None:
        raw_ref = observation.raw.get("summary_ref") if isinstance(observation.raw, dict) else None
        if raw_ref:
            payload = json.loads(Path(str(raw_ref)).read_text(encoding="utf-8"))
            vector = payload.get("joint_action_vector")
            if isinstance(vector, list):
                return np.asarray(vector, dtype=np.float32), "observation_summary.joint_action_vector"
        arms = observation.robot_arms if isinstance(observation.robot_arms, dict) else {}
        left = arms.get("left")
        right = arms.get("right")
        left_joints = getattr(left, "joint_positions", None)
        right_joints = getattr(right, "joint_positions", None)
        left_gripper = getattr(left, "gripper_value", None)
        right_gripper = getattr(right, "gripper_value", None)
        if left_joints is not None and right_joints is not None and left_gripper is not None and right_gripper is not None:
            return (
                np.asarray([*left_joints, left_gripper, *right_joints, right_gripper], dtype=np.float32),
                "observation.robot_arms",
            )
    raise ValueError("14D RoboTwin joint state is missing from observation.")


def _artifact_dir_from_observation(observation: ObservationBundle | None) -> Path:
    if observation is None or not isinstance(observation.raw, dict) or not observation.raw.get("summary_ref"):
        raise ValueError("Observation raw.summary_ref is required for subprocess pi05 inference.")
    return Path(str(observation.raw["summary_ref"])).parent


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
        raise RuntimeError("pi05_worker_empty_response")
    return json.loads(data.decode("utf-8"))


def _openpi_image_keys() -> tuple[str, str, str]:
    return ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _image_tensors_from_observation(observation: ObservationBundle | None, torch: Any, device: str) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    if observation is None:
        raise ValueError("Observation is required for pi05 inference.")
    mapping = {
        "base_0_rgb": "head_camera",
        "left_wrist_0_rgb": "left_camera",
        "right_wrist_0_rgb": "right_camera",
    }
    images = {}
    for output_key, camera_name in mapping.items():
        view = observation.camera_views.get(camera_name)
        if view is None or not view.rgb_path:
            raise ValueError(f"Missing RGB artifact for camera {camera_name}.")
        image = Image.open(view.rgb_path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        images[output_key] = torch.as_tensor(array[None, ...], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    return images


def _tokenize_prompt(tokenizer: Any, prompt: str, state: "np.ndarray", max_len: int) -> tuple["np.ndarray", "np.ndarray"]:
    import numpy as np

    discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized_state))
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    full_prompt = f"Task: {cleaned}, State: {state_str};\nAction: "
    tokens = [tokenizer.bos_id(), *list(tokenizer.EncodeAsIds(full_prompt))]
    if len(tokens) < max_len:
        mask = [True] * len(tokens) + [False] * (max_len - len(tokens))
        tokens = tokens + [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
        mask = [True] * max_len
    return np.asarray(tokens, dtype=np.int64), np.asarray(mask, dtype=np.bool_)


def _normalize_quantile(value: "np.ndarray", stats: dict[str, Any]) -> "np.ndarray":
    import numpy as np

    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    return (value - q01[: value.shape[-1]]) / (q99[: value.shape[-1]] - q01[: value.shape[-1]] + 1e-6) * 2.0 - 1.0


def _unnormalize_quantile(value: "np.ndarray", stats: dict[str, Any]) -> "np.ndarray":
    import numpy as np

    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    return (value + 1.0) / 2.0 * (q99[: value.shape[-1]] - q01[: value.shape[-1]] + 1e-6) + q01[: value.shape[-1]]


def _pad_last_dim(value: "np.ndarray", target_dim: int) -> "np.ndarray":
    import numpy as np

    if value.shape[-1] >= target_dim:
        return value.astype(np.float32)
    pad_width = [(0, 0)] * value.ndim
    pad_width[-1] = (0, target_dim - value.shape[-1])
    return np.pad(value, pad_width, constant_values=0.0).astype(np.float32)


def _absolute_actions(actions: "np.ndarray", state: "np.ndarray") -> "np.ndarray":
    import numpy as np

    mask = np.asarray([True] * 6 + [False] + [True] * 6 + [False])
    result = np.asarray(actions, dtype=np.float32).copy()
    result[:, :14] += np.expand_dims(np.where(mask, state[:14], 0), axis=0)
    return result


def _decode_aloha_state(state: "np.ndarray", *, adapt_to_pi: bool) -> "np.ndarray":
    import numpy as np

    state = np.asarray(state, dtype=np.float32).copy()
    if adapt_to_pi:
        state = _joint_flip_mask() * state
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_aloha_actions(actions: "np.ndarray", *, adapt_to_pi: bool) -> "np.ndarray":
    import numpy as np

    actions = np.asarray(actions, dtype=np.float32).copy()
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _joint_flip_mask() -> "np.ndarray":
    import numpy as np

    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1], dtype=np.float32)


def _normalize(value: "np.ndarray", min_val: float, max_val: float) -> "np.ndarray":
    return (value - min_val) / (max_val - min_val)


def _unnormalize(value: "np.ndarray", min_val: float, max_val: float) -> "np.ndarray":
    return value * (max_val - min_val) + min_val


def _gripper_to_angular(value: "np.ndarray") -> "np.ndarray":
    import numpy as np

    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)
    value = np.arcsin(np.clip((0.022**2 + value**2 - 0.036**2) / (2 * 0.022 * value), -1.0, 1.0))
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value: "np.ndarray") -> "np.ndarray":
    return _normalize(value + 0.5476, min_val=-0.6213, max_val=1.4910)


def _openpi_train_config(train_config_name: str, asset_id: str, checkpoint_dir: Path) -> Any:
    from openpi.models import pi0_config
    from openpi.training import config as openpi_config
    from openpi.training import optimizer as openpi_optimizer
    from openpi.training import weight_loaders
    import openpi.transforms as openpi_transforms

    repack = openpi_transforms.Group(
        inputs=[
            openpi_transforms.RepackTransform(
                {
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "actions",
                    "prompt": "prompt",
                }
            )
        ]
    )
    return openpi_config.TrainConfig(
        name=train_config_name,
        project_name="pi05_finetune",
        exp_name="robotwin_clean_randomized_joint_training",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        weight_loader=weight_loaders.NoOpWeightLoader(),
        pytorch_weight_path=str(checkpoint_dir),
        lr_schedule=openpi_optimizer.ConstantScheduleWithWarmup(),
        data=openpi_config.LeRobotAlohaDataConfig(
            repo_id="clean_randomized_joint_training",
            assets=openpi_config.AssetsConfig(
                assets_dir=str(checkpoint_dir / "assets"),
                asset_id=asset_id,
            ),
            adapt_to_pi=True,
            default_prompt=None,
            use_delta_joint_actions=True,
            repack_transforms=repack,
            base_config=openpi_config.DataConfig(
                prompt_from_task=True,
            ),
            action_sequence_keys=("actions",),
        ),
        seed=42,
        batch_size=128,
        num_workers=16,
        num_train_steps=1_000_000,
        log_interval=100,
        save_interval=5000,
        keep_period=5000,
        resume=True,
        wandb_enabled=False,
    )


@contextmanager
def _openpi_torch_only_import_hooks(openpi_src: str):
    import dataclasses
    import sys
    import types

    import torch
    import torch.nn.functional as torch_f

    if openpi_src not in sys.path:
        sys.path.insert(0, openpi_src)

    import openpi
    import openpi.models
    import openpi.shared

    @dataclasses.dataclass
    class GemmaConfig:
        width: int
        depth: int
        mlp_dim: int
        num_heads: int
        num_kv_heads: int
        head_dim: int
        lora_configs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def get_config(variant: str) -> GemmaConfig:
        if variant == "gemma_300m":
            return GemmaConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)
        if variant == "gemma_2b":
            return GemmaConfig(width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256)
        if variant in {"gemma_300m_lora", "gemma_2b_lora"}:
            raise ValueError(f"torch-only diagnostic hook does not support LoRA variant yet: {variant}")
        raise ValueError(f"unknown Gemma variant for torch-only diagnostic hook: {variant}")

    def resize_with_pad_torch(
        images,
        height: int,
        width: int,
        mode: str = "bilinear",
    ):
        channels_last = images.shape[-1] <= 4
        if channels_last:
            if images.dim() == 3:
                images = images.unsqueeze(0)
            images = images.permute(0, 3, 1, 2)
        elif images.dim() == 3:
            images = images.unsqueeze(0)

        _, _, cur_height, cur_width = images.shape
        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized = torch_f.interpolate(
            images,
            size=(resized_height, resized_width),
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
        )
        if images.dtype == torch.uint8:
            resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
            pad_value = 0
        else:
            resized = resized.clamp(-1.0, 1.0)
            pad_value = -1.0

        pad_h0, remainder_h = divmod(height - resized_height, 2)
        pad_w0, remainder_w = divmod(width - resized_width, 2)
        padded = torch_f.pad(
            resized,
            (pad_w0, pad_w0 + remainder_w, pad_h0, pad_h0 + remainder_h),
            mode="constant",
            value=pad_value,
        )
        if channels_last:
            padded = padded.permute(0, 2, 3, 1)
        return padded

    gemma_module = types.ModuleType("openpi.models.gemma")
    gemma_module.Config = GemmaConfig
    gemma_module.Variant = str
    gemma_module.get_config = get_config

    image_tools_module = types.ModuleType("openpi.shared.image_tools")
    image_tools_module.resize_with_pad_torch = resize_with_pad_torch

    check_module = types.ModuleType("transformers.models.siglip.check")
    check_module.check_whether_transformers_replace_is_installed_correctly = lambda: True

    module_names = [
        "openpi.models.gemma",
        "openpi.shared.image_tools",
        "transformers.models.siglip.check",
    ]
    previous_modules = {name: sys.modules.get(name) for name in module_names}
    previous_attrs = {
        (openpi.models, "gemma"): getattr(openpi.models, "gemma", None),
        (openpi.shared, "image_tools"): getattr(openpi.shared, "image_tools", None),
    }
    attr_missing = {key: not hasattr(key[0], key[1]) for key in previous_attrs}
    previous_compile = getattr(torch, "compile", None)

    sys.modules["openpi.models.gemma"] = gemma_module
    sys.modules["openpi.shared.image_tools"] = image_tools_module
    sys.modules["transformers.models.siglip.check"] = check_module
    setattr(openpi.models, "gemma", gemma_module)
    setattr(openpi.shared, "image_tools", image_tools_module)
    if previous_compile is not None:
        torch.compile = lambda fn=None, *args, **kwargs: fn if fn is not None else (lambda wrapped: wrapped)

    try:
        yield
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        for (obj, attr), previous in previous_attrs.items():
            if attr_missing[(obj, attr)]:
                try:
                    delattr(obj, attr)
                except AttributeError:
                    pass
            else:
                setattr(obj, attr, previous)
        if previous_compile is not None:
            torch.compile = previous_compile


def _package_version(package_name: str) -> str | None:
    try:
        from importlib import metadata

        return metadata.version(package_name)
    except Exception:
        return None


def _local_hf_snapshot(model_dir_name: str) -> Path | None:
    cache_root = Path("/mnt/wangwai/.cache/huggingface/hub") / model_dir_name
    refs_main = cache_root / "refs" / "main"
    if refs_main.exists():
        try:
            snapshot = cache_root / "snapshots" / refs_main.read_text(encoding="utf-8").strip()
            if snapshot.exists():
                return snapshot
        except OSError as exc:
            emit_status_notice(
                "hf_snapshot_ref_unavailable",
                success=True,
                source="pi05._local_hf_snapshot",
                reason=f"{type(exc).__name__}: {exc}",
                payload={"model_dir_name": model_dir_name, "fallback": "latest_snapshot_dir"},
            )
    snapshots_dir = cache_root / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshots:
            return snapshots[-1]
    return None
