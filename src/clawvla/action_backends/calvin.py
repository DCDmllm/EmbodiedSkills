from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import ActionChunk, MotionGoal, ObservationBundle, WorldState
from .base import ActionBackendResult


class CalvinHttpActionBackend:
    name = "calvin_http"
    # X-VLA consumes current images, proprioception, and natural language
    # directly. Object candidate ids are optional diagnostics, not inputs.
    requires_candidate_bindings = False

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})

    def task_plan_contract(self, task_instruction: str) -> dict[str, Any]:
        return {
            "mode": "atomic_instruction_passthrough",
            "instruction": str(task_instruction).strip(),
            "max_subgoals": 1,
            "candidate_bindings_required": False,
            "completion_authority": "environment_oracle",
        }

    def build_action_chunk(
        self,
        motion_goal: MotionGoal | None,
        world_state: WorldState | None,
        observation: ObservationBundle | None,
        request: dict[str, Any],
    ) -> ActionBackendResult:
        _ = (motion_goal, world_state)
        if not self.config.get("enabled", False):
            return self._unavailable("calvin_http_backend_disabled", request)
        url = str(self.config.get("url") or self.config.get("endpoint") or "").strip()
        if not url:
            return self._unavailable("calvin_http_url_missing", request)
        try:
            payload, payload_metadata = self._request_payload(observation, request)
            response_payload = self._post(url, payload)
            response_actions = response_payload.get(
                "action", response_payload.get("actions")
            )
            payload_metadata["response_action_shape"] = (
                list(np.asarray(response_actions).shape)
                if response_actions is not None
                else None
            )
            commands = _commands_from_response(
                response_payload,
                horizon=int(payload_metadata["execution_horizon"]),
            )
        except Exception as exc:
            return self._unavailable("calvin_http_inference_failed", request, {"exception": f"{type(exc).__name__}: {exc}"})
        action_type = str(self.config.get("action_type") or "calvin_ee_pose_10d")
        chunk = ActionChunk(
            action_type=action_type,
            commands=commands,
            control_horizon=len(commands),
            metadata={
                "backend": self.name,
                "status": "calvin_http_action_chunk_built",
                "action_type": action_type,
                "url": _public_url(url),
                "horizon": len(commands),
                "request": dict(request),
                **payload_metadata,
            },
        )
        return ActionBackendResult(
            success=True,
            status="calvin_http_action_chunk_built",
            action_chunk=chunk,
            metadata=chunk.metadata,
            errors=[],
        )

    def action_spec(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "types": {str(self.config.get("action_type") or "calvin_ee_pose_10d"): 10},
            "horizon": int(self.config.get("horizon") or self.config.get("steps") or 10),
            "inference_steps": int(
                self.config.get("inference_steps")
                or self.config.get("diffusion_steps")
                or self.config.get("steps")
                or 10
            ),
            "serialization": str(self.config.get("serialization") or "json_numpy"),
        }

    def health(self) -> dict[str, Any]:
        if not self.config.get("enabled", False):
            return {"ok": False, "backend": self.name, "reason": "action_backend_disabled"}
        url = str(self.config.get("url") or self.config.get("endpoint") or "").strip()
        if not url:
            return {"ok": False, "backend": self.name, "reason": "action_backend_url_missing"}
        return {
            "ok": True,
            "backend": self.name,
            "reason": "configured",
            "url": _public_url(url),
            "checkpoint_id": self.config.get("checkpoint_id"),
            "checkpoint_sha256": self.config.get("checkpoint_sha256"),
        }

    def public_config(self) -> dict[str, Any]:
        return {
            "type": self.config.get("type", "calvin_http"),
            "enabled": bool(self.config.get("enabled", False)),
            "url": _public_url(str(self.config.get("url") or self.config.get("endpoint") or "")),
            "action_type": self.config.get("action_type", "calvin_ee_pose_10d"),
            "horizon": self.config.get("horizon") or self.config.get("steps"),
            "inference_steps": (
                self.config.get("inference_steps")
                or self.config.get("diffusion_steps")
                or self.config.get("steps")
            ),
            "serialization": self.config.get("serialization", "json_numpy"),
            "image_mapping": dict(self.config.get("image_mapping", {}))
            if isinstance(self.config.get("image_mapping"), dict)
            else {},
            "checkpoint_id": self.config.get("checkpoint_id"),
            "checkpoint_sha256": self.config.get("checkpoint_sha256"),
        }

    def _request_payload(self, observation: ObservationBundle | None, request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if observation is None:
            raise ValueError("calvin_http_observation_missing")
        prompt = _resolve_prompt(request, observation)
        proprio = _calvin_proprio_from_observation(observation)
        image_payload, image_metadata = _image_payloads(observation, self.config)
        configured_horizon = int(self.config.get("horizon") or self.config.get("steps") or 10)
        requested_horizon = int(request.get("horizon") or configured_horizon)
        if configured_horizon <= 0:
            raise ValueError(f"calvin_http_invalid_configured_horizon:{configured_horizon}")
        if requested_horizon <= 0:
            raise ValueError(f"calvin_http_invalid_horizon:{requested_horizon}")
        execution_horizon = min(requested_horizon, configured_horizon)
        inference_steps = int(
            request.get("inference_steps")
            or self.config.get("inference_steps")
            or self.config.get("diffusion_steps")
            or self.config.get("steps")
            or 10
        )
        if inference_steps <= 0:
            raise ValueError(f"calvin_http_invalid_inference_steps:{inference_steps}")
        serialization = str(self.config.get("serialization") or "json_numpy")
        payload = {
            "language_instruction": prompt,
            "proprio": _serialize_payload(proprio, serialization),
            "domain_id": int(self.config.get("domain_id", 2)),
            # X-VLA interprets `steps` as the flow-matching sampling count. The
            # number of returned actions is fixed by checkpoint `num_actions`.
            "steps": inference_steps,
            **image_payload,
        }
        return payload, {
            "prompt": prompt,
            "state_source": "observation.raw.calvin_proprio",
            "image_sources": image_metadata,
            "domain_id": payload["domain_id"],
            "serialization": serialization,
            "requested_horizon": requested_horizon,
            "configured_horizon": configured_horizon,
            "execution_horizon": execution_horizon,
            "inference_steps": inference_steps,
        }

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        import requests

        response = requests.post(url, json=payload, timeout=float(self.config.get("timeout", 30.0)))
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError(f"calvin_http_response_must_be_object:{type(data).__name__}")
        return data

    def _unavailable(
        self,
        reason: str,
        request: dict[str, Any],
        extra_metadata: dict[str, Any] | None = None,
    ) -> ActionBackendResult:
        metadata = {
            "backend": self.name,
            "status": "calvin_http_unavailable",
            "reason": reason,
            "retryable": False,
            "request": dict(request),
            "config": self.public_config(),
        }
        metadata.update(dict(extra_metadata or {}))
        chunk = ActionChunk(action_type="unavailable", commands=[], control_horizon=0, metadata=metadata)
        return ActionBackendResult(
            success=False,
            status="calvin_http_unavailable",
            action_chunk=chunk,
            metadata=metadata,
            errors=[reason],
        )


def _resolve_prompt(request: dict[str, Any], observation: ObservationBundle) -> str:
    motion_plan = request.get("motion_plan")
    if isinstance(motion_plan, dict) and motion_plan.get("vla_prompt"):
        return str(motion_plan["vla_prompt"])
    if observation.task_instruction:
        return str(observation.task_instruction)
    raise ValueError("calvin_http_prompt_missing:motion_plan.vla_prompt_or_observation.task_instruction_required")


def _calvin_proprio_from_observation(observation: ObservationBundle) -> list[float]:
    raw = observation.raw if isinstance(observation.raw, dict) else {}
    proprio = _float_vector(raw.get("calvin_proprio"), expected_dim=20)
    if proprio is not None:
        return proprio
    summary_ref = raw.get("summary_ref")
    if summary_ref:
        payload = json.loads(Path(str(summary_ref)).read_text(encoding="utf-8"))
        proprio = _float_vector(payload.get("calvin_proprio"), expected_dim=20)
        if proprio is not None:
            return proprio
    raise ValueError("calvin_http_calvin_proprio_missing")


def _image_payloads(observation: ObservationBundle, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    mapping = config.get("image_mapping")
    if isinstance(mapping, dict) and mapping:
        image_mapping = {str(key): str(value) for key, value in mapping.items()}
    else:
        image_mapping = {"image0": "static", "image1": "gripper"}
    serialization = str(config.get("serialization") or "json_numpy")
    payload: dict[str, Any] = {}
    metadata: dict[str, str] = {}
    for payload_key, camera_name in image_mapping.items():
        view = observation.camera_views.get(camera_name)
        if view is None or not view.rgb_path:
            raise ValueError(f"calvin_http_missing_rgb_artifact:{camera_name}")
        image = _load_rgb_array(view.rgb_path)
        payload[payload_key] = _serialize_payload(image, serialization)
        metadata[payload_key] = str(view.rgb_path)
    return payload, metadata


def _load_rgb_array(path: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()


def _serialize_payload(value: Any, serialization: str) -> Any:
    if serialization == "json_numpy":
        import json_numpy

        return json_numpy.dumps(value)
    if serialization == "list":
        return np.asarray(value).tolist()
    raise ValueError(f"calvin_http_unsupported_serialization:{serialization}")


def _commands_from_response(response: dict[str, Any], horizon: int) -> list[list[float]]:
    actions = response.get("action", response.get("actions"))
    if actions is None:
        raise KeyError("calvin_http_response_missing_action")
    array = np.asarray(actions, dtype=np.float32)
    if array.size == 0:
        raise ValueError("calvin_http_empty_action_response")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] < 10:
        raise ValueError(f"calvin_http_action_shape_invalid:{list(array.shape)}:expected=[N,>=10]")
    # The released CALVIN checkpoint has num_actions=30 and real_action_dim=20.
    # The official client consumes a bounded prefix of rows and executes the
    # first 10 columns of each ee6d action. `steps` does not control row count.
    array = array[:horizon, :10]
    if not np.isfinite(array).all():
        raise ValueError("calvin_http_action_contains_nonfinite")
    return [[float(item) for item in row.tolist()] for row in array]


def _float_vector(value: Any, expected_dim: int | None = None) -> list[float] | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if expected_dim is not None and vector.shape[0] != expected_dim:
        return None
    if not np.isfinite(vector).all():
        return None
    return [float(item) for item in vector.tolist()]


def _public_url(url: str) -> str:
    text = str(url or "")
    if "@" not in text:
        return text
    prefix, _, suffix = text.rpartition("@")
    scheme, _, _rest = prefix.partition("://")
    return f"{scheme}://***@{suffix}" if scheme else f"***@{suffix}"
