from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import ArtifactStore
from ..config import EnvironmentConfig, PROJECT_ROOT
from ..notices import emit_status_notice
from ..schema import ActionChunk, CameraView, ObservationBundle, RobotArmState
from .base import RobotEnvAdapter


ROBOCASA_CAMERA_MAP = {
    "video.robot0_agentview_left": ("robot0_agentview_left", "observation.images.robot0_agentview_left"),
    "video.robot0_agentview_right": ("robot0_agentview_right", "observation.images.robot0_agentview_right"),
    "video.robot0_eye_in_hand": ("robot0_eye_in_hand", "observation.images.robot0_eye_in_hand"),
    "robot0_agentview_left_image": ("robot0_agentview_left", "observation.images.robot0_agentview_left"),
    "robot0_agentview_right_image": ("robot0_agentview_right", "observation.images.robot0_agentview_right"),
    "robot0_eye_in_hand_image": ("robot0_eye_in_hand", "observation.images.robot0_eye_in_hand"),
}

ROBOCASA_GROOT_STATE_BLOCKS = (
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
)

ROBOCASA_GROOT_ACTION_BLOCKS = (
    ("action.base_motion", 4),
    ("action.control_mode", 1),
    ("action.end_effector_position", 3),
    ("action.end_effector_rotation", 3),
    ("action.gripper_close", 1),
)


class RoboCasaAdapter(RobotEnvAdapter):
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        params = dict(config.params)
        self.env_id = str(params.get("env_id") or params.get("task") or config.task_name or "robocasa/PickPlaceCounterToCabinet")
        self.camera_names = _string_list(
            params.get("camera_names")
            or ["video.robot0_agentview_left", "video.robot0_agentview_right", "video.robot0_eye_in_hand"]
        )
        self.state_keys = _string_list(
            params.get("state_keys")
            or ["observation.state", *ROBOCASA_GROOT_STATE_BLOCKS, "robot0_proprio-state", "robot0_robot-state"]
        )
        self.max_episode_steps = int(params.get("max_episode_steps", 500))
        self.make_kwargs = dict(params.get("make_kwargs", {})) if isinstance(params.get("make_kwargs"), dict) else {}
        self.expose_environment_semantics = any(
            _truthy(value)
            for value in (
                params.get("expose_environment_semantics"),
                config.metadata.get("expose_environment_semantics"),
                config.metadata.get("debug_expose_environment_semantics"),
            )
        )
        self.artifacts = ArtifactStore(config.artifact_dir or Path(PROJECT_ROOT) / "tmp_artifacts" / "robocasa")
        self.env: Any | None = None
        self.last_raw_observation: dict[str, Any] | None = None
        self.last_observation: ObservationBundle | None = None
        self.last_reward: float | None = None
        self.last_done: bool | None = None
        self.last_info: dict[str, Any] = {}
        self.step_count = 0

    def capture_views(self, **kwargs: Any) -> ObservationBundle:
        if kwargs.get("setup") or self.env is None:
            raw_observation = self.setup(instruction=kwargs.get("instruction"))
        else:
            raw_observation = kwargs.get("raw_observation") or self.last_raw_observation
        if not isinstance(raw_observation, dict):
            raise RuntimeError("robocasa_observation_unavailable:no_raw_observation")
        observation = normalize_robocasa_observation(
            raw_observation,
            task_instruction=kwargs.get("instruction") or self.task_language(),
            metadata=self._capture_metadata(kwargs),
            artifacts=self.artifacts,
            artifact_prefix=str(kwargs.get("artifact_prefix", "capture")),
            camera_names=self.camera_names,
            state_keys=self.state_keys,
        )
        self.last_raw_observation = raw_observation
        self.last_observation = observation
        return observation

    def setup(self, instruction: str | None = None) -> dict[str, Any]:
        _ = instruction
        if self.env is None:
            self.env = self._make_env()
        reset_result = self.env.reset(seed=int(self.config.seed or 0)) if hasattr(self.env, "reset") else None
        raw_observation, info = _split_reset(reset_result)
        self.last_info = dict(info or {})
        self.step_count = 0
        if not isinstance(raw_observation, dict):
            raise RuntimeError(f"robocasa_reset_returned_non_dict_observation:{type(raw_observation).__name__}")
        self.last_raw_observation = raw_observation
        return raw_observation

    def execute_action(self, action_chunk: ActionChunk | None) -> dict[str, Any]:
        if action_chunk is None:
            return _execution_unavailable("missing_action_chunk", action_chunk, self)
        if action_chunk.action_type != "robocasa_action" or not action_chunk.commands:
            return _execution_unavailable("unsupported_or_empty_action_chunk", action_chunk, self)
        if self.env is None:
            return _execution_unavailable("robocasa_env_not_setup", action_chunk, self)

        before_state = _safe_state_from_robocasa(self.last_raw_observation, self.state_keys)
        executed: list[list[float]] = []
        success = False
        done = False
        info: dict[str, Any] = {}
        raw_observation: dict[str, Any] | None = None
        for command in action_chunk.commands:
            action = _float_vector(command)
            if action is None:
                return _execution_unavailable("invalid_robocasa_action_command", action_chunk, self)
            step_result = self.env.step(_robocasa_env_action(action, self.env))
            raw_observation, reward, done, info = _split_step(step_result)
            self.step_count += 1
            self.last_reward = float(reward)
            self.last_done = bool(done)
            self.last_info = dict(info or {})
            success = _success_from_env(self.env, info)
            executed.append(action)
            if done or success or self.step_count >= self.max_episode_steps:
                break
        if raw_observation is None:
            return _execution_unavailable("robocasa_execute_returned_no_observation", action_chunk, self)

        artifact_prefix = str(action_chunk.metadata.get("artifact_prefix", "execute")) if isinstance(action_chunk.metadata, dict) else "execute"
        self.last_raw_observation = raw_observation
        self.last_observation = normalize_robocasa_observation(
            raw_observation,
            task_instruction=getattr(self.last_observation, "task_instruction", None) or self.task_language(),
            metadata=self._capture_metadata({"artifact_prefix": artifact_prefix, "after_execute": True}),
            artifacts=self.artifacts,
            artifact_prefix=f"{artifact_prefix}/after",
            camera_names=self.camera_names,
            state_keys=self.state_keys,
        )
        return {
            "backend": "robocasa",
            "status": "action_executed",
            "success": success,
            "done": done,
            "executed_steps": len(executed),
            "reward": self.last_reward,
            "info": info,
            "observation": self.last_observation.to_dict(),
            "action_chunk": action_chunk.to_dict(),
            "action_effect": _action_effect_report(before_state, self.last_observation.raw.get("groot_state"), executed),
            "task_env_bound": self.env is not None,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "robocasa",
            "env_id": self.env_id,
            "task_name": self.config.task_name or self.env_id,
            "task_language": self.task_language(),
            "artifact_dir": str(self.artifacts.root_dir),
            "task_env_bound": self.env is not None,
        }

    def status(self) -> dict[str, Any]:
        return {
            "backend": "robocasa",
            "ready": self.env is not None and self.last_observation is not None,
            "needs_setup": self.env is None,
            "last_observation_present": self.last_observation is not None,
            "live_env_bound": self.env is not None,
            "env_id": self.env_id,
            "task_language": self.task_language(),
        }

    def preflight_spec(self) -> dict[str, Any]:
        cameras = [ROBOCASA_CAMERA_MAP.get(name, (str(name), None))[0] for name in self.camera_names]
        action_dim = _action_dim_from_env(self.env)
        state_dim = _state_dim_from_observation(self.last_observation) or 16
        return {
            "backend": "robocasa",
            "required_cameras": cameras,
            "action_cameras": cameras,
            "expected_resolution": None,
            "state": {"required": True, "dim": state_dim, "source": ["groot_state", "robocasa_state", *self.state_keys]},
            "action": {"required": True, "types": {"robocasa_action": action_dim}},
        }

    def task_status(self) -> dict[str, Any]:
        success = _success_from_env(self.env, self.last_info) if self.env is not None else None
        return {
            "backend": "robocasa",
            "env_id": self.env_id,
            "task_name": self.config.task_name or self.env_id,
            "task_language": self.task_language(),
            "success": success,
            "done": bool(self.last_done) if self.last_done is not None else success,
            "step_count": self.step_count,
            "reward": self.last_reward,
            "info": dict(self.last_info),
        }

    def task_language(self) -> str | None:
        if self.env is None:
            return None
        for attr in ("language_instruction", "task_language", "task_description"):
            value = getattr(self.env, attr, None)
            if value:
                return str(value)
        unwrapped = getattr(self.env, "unwrapped", None)
        if unwrapped is not None and unwrapped is not self.env:
            for attr in ("language_instruction", "task_language", "task_description"):
                value = getattr(unwrapped, attr, None)
                if value:
                    return str(value)
        return None

    def close(self) -> None:
        if self.env is not None and hasattr(self.env, "close"):
            self.env.close()
        self.env = None

    def _make_env(self) -> Any:
        try:
            import robocasa  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "RoboCasa is not installed. Install robocasa in the runtime environment before running this adapter."
            ) from exc
        import gymnasium as gym

        return gym.make(self.env_id, **self.make_kwargs)

    def _capture_metadata(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        metadata = {
            "backend": "robocasa",
            "env_id": self.env_id,
            "task_name": self.config.task_name or self.env_id,
            "task_language": self.task_language(),
            "seed": self.config.seed,
            "camera_names": list(self.camera_names),
            "state_keys": list(self.state_keys),
            "artifact_dir": str(self.artifacts.root_dir),
            "task_env_bound": self.env is not None,
            "environment_semantics_enabled": self.expose_environment_semantics,
            "environment_semantics": _robocasa_semantic_hints(self.env) if self.expose_environment_semantics else {},
        }
        metadata.update(kwargs)
        return metadata


def normalize_robocasa_observation(
    raw_observation: dict[str, Any],
    task_instruction: str | None = None,
    metadata: dict[str, Any] | None = None,
    artifacts: ArtifactStore | None = None,
    artifact_prefix: str = "capture",
    camera_names: list[str] | None = None,
    state_keys: list[str] | None = None,
) -> ObservationBundle:
    camera_names = list(camera_names or ["video.robot0_agentview_left", "video.robot0_agentview_right", "video.robot0_eye_in_hand"])
    state_keys = list(state_keys or ["observation.state", *ROBOCASA_GROOT_STATE_BLOCKS, "robot0_proprio-state", "robot0_robot-state"])
    camera_views = _camera_views_from_robocasa(raw_observation, camera_names, artifacts, artifact_prefix)
    state = _state_from_robocasa(raw_observation, state_keys)
    robot_arms = {
        "robot0": RobotArmState(
            arm_name="robot0",
            eef_pose=_float_list(_get_nested(raw_observation, "state.end_effector_position_relative"))
            or _float_list(_get_nested(raw_observation, "robot0_eef_pos")),
            gripper_state=_robocasa_gripper_state(_get_nested(raw_observation, "state.gripper_qpos"))
            or _robocasa_gripper_state(_get_nested(raw_observation, "robot0_gripper_qpos")),
            gripper_value=_mean_float(_get_nested(raw_observation, "state.gripper_qpos"))
            or _mean_float(_get_nested(raw_observation, "robot0_gripper_qpos")),
            joint_positions=_float_list(_get_nested(raw_observation, "robot0_joint_pos")),
            metadata={
                "groot_state": state,
                "robocasa_state": state,
                "state_source": "robocasa_raw_observation",
            },
        )
    }
    summary = _raw_summary(raw_observation, camera_names, state)
    semantics_enabled = bool((metadata or {}).get("environment_semantics_enabled")) if isinstance(metadata, dict) else False
    environment_semantics = (
        dict((metadata or {}).get("environment_semantics", {}))
        if semantics_enabled and isinstance(metadata, dict)
        else {}
    )
    raw_ref = artifacts.write_json(f"{artifact_prefix}/raw_observation_summary.json", summary) if artifacts else None
    return ObservationBundle(
        task_instruction=task_instruction,
        camera_views=camera_views,
        robot_arms=robot_arms,
        pointcloud_ref=None,
        raw={
            "keys": sorted(str(key) for key in raw_observation.keys()),
            "summary_ref": raw_ref,
            "groot_state": state,
            "robocasa_state": state,
            "state_source": "robocasa_state",
            "environment_semantics_enabled": semantics_enabled,
            "environment_semantics": environment_semantics,
        },
        metadata=dict(metadata or {}),
    )


def _camera_views_from_robocasa(
    raw_observation: dict[str, Any],
    camera_names: list[str],
    artifacts: ArtifactStore | None,
    artifact_prefix: str,
) -> dict[str, CameraView]:
    camera_views: dict[str, CameraView] = {}
    for raw_name in camera_names:
        view_name, policy_key = ROBOCASA_CAMERA_MAP.get(raw_name, (str(raw_name).removesuffix("_image"), f"observation.images.{raw_name}"))
        raw_image = _get_nested(raw_observation, raw_name)
        image = _orient_image(raw_image)
        rgb_path = artifacts.write_image(f"{artifact_prefix}/images/{view_name}_rgb.png", image) if artifacts and image is not None else None
        camera_views[view_name] = CameraView(
            name=view_name,
            rgb_path=rgb_path,
            depth_path=None,
            mask_path=None,
            intrinsics=None,
            extrinsics=None,
            metadata={
                "raw_name": raw_name,
                "policy_key": policy_key,
                "has_rgb": raw_image is not None,
                "shape": list(np.asarray(image).shape) if image is not None else None,
                "raw_shape": list(np.asarray(raw_image).shape) if raw_image is not None else None,
            },
        )
    return camera_views


def _orient_image(image: Any) -> Any:
    if image is None:
        return None
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            max_value = float(np.nanmax(array)) if array.size else 1.0
            array = array * 255.0 if max_value <= 1.0 else array
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array.copy()


def _state_from_robocasa(raw_observation: dict[str, Any], state_keys: list[str]) -> list[float]:
    if all(_get_nested(raw_observation, key) is not None for key in ROBOCASA_GROOT_STATE_BLOCKS):
        state: list[float] = []
        for key in ROBOCASA_GROOT_STATE_BLOCKS:
            vector = _float_list(_get_nested(raw_observation, key))
            if vector:
                state.extend(vector)
        if state:
            return state
    for key in state_keys:
        vector = _float_list(_get_nested(raw_observation, key))
        if vector is not None:
            return vector
    block_pieces: list[float] = []
    for key in ROBOCASA_GROOT_STATE_BLOCKS + tuple(key.removeprefix("state.") for key in ROBOCASA_GROOT_STATE_BLOCKS):
        vector = _float_list(_get_nested(raw_observation, key))
        if vector:
            block_pieces.extend(vector)
    if block_pieces:
        return block_pieces
    pieces: list[float] = []
    for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_joint_pos"):
        vector = _float_list(_get_nested(raw_observation, key))
        if vector:
            pieces.extend(vector)
    if pieces:
        return pieces
    raise ValueError(f"robocasa_state_unavailable:checked={state_keys}")


def _safe_state_from_robocasa(raw_observation: dict[str, Any] | None, state_keys: list[str]) -> list[float] | None:
    if not isinstance(raw_observation, dict):
        return None
    try:
        return _state_from_robocasa(raw_observation, state_keys)
    except ValueError:
        return None


def _action_effect_report(
    before_state: list[float] | None,
    after_state: Any,
    executed: list[list[float]],
) -> dict[str, Any]:
    action_values = [abs(float(item)) for command in executed for item in command]
    report: dict[str, Any] = {
        "executed_steps": len(executed),
        "max_abs_action": max(action_values) if action_values else 0.0,
        "mean_abs_action": float(np.mean(np.asarray(action_values, dtype=np.float32))) if action_values else 0.0,
        "state_delta_available": False,
        "state_changed": None,
        "max_abs_state_delta": None,
        "sum_abs_state_delta": None,
    }
    after_vector = _float_vector(after_state)
    if before_state is None or after_vector is None or len(before_state) != len(after_vector):
        return report
    deltas = [abs(float(after) - float(before)) for before, after in zip(before_state, after_vector)]
    max_delta = max(deltas) if deltas else 0.0
    sum_delta = float(sum(deltas))
    report.update(
        {
            "state_delta_available": True,
            "state_changed": bool(max_delta > 1e-6),
            "max_abs_state_delta": max_delta,
            "sum_abs_state_delta": sum_delta,
        }
    )
    return report


def _raw_summary(raw_observation: dict[str, Any], camera_names: list[str], state: list[float]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "keys": sorted(str(key) for key in raw_observation.keys()),
        "cameras": {},
        "groot_state": state,
        "robocasa_state": state,
    }
    for name in camera_names:
        value = _get_nested(raw_observation, name)
        summary["cameras"][name] = {"shape": list(np.asarray(value).shape) if value is not None else None}
    for key in ROBOCASA_GROOT_STATE_BLOCKS + (
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot0_joint_pos",
        "robot0_joint_vel",
    ):
        value = _get_nested(raw_observation, key)
        if value is not None:
            summary[key] = _float_list(value)
    return summary


def _get_nested(payload: dict[str, Any], key: str) -> Any:
    if key in payload:
        return payload[key]
    current: Any = payload
    for part in key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _split_reset(result: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
        return obs, dict(info or {})
    return result, {}


def _split_step(result: Any) -> tuple[Any, float, bool, dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward), bool(terminated or truncated), dict(info or {})
    if isinstance(result, tuple) and len(result) == 4:
        obs, reward, done, info = result
        return obs, float(reward), bool(done), dict(info or {})
    raise RuntimeError(f"robocasa_step_return_shape_unsupported:{type(result).__name__}")


def _success_from_env(env: Any | None, info: dict[str, Any]) -> bool:
    if isinstance(info, dict):
        for key in ("success", "is_success", "task_success"):
            if key in info:
                return bool(info[key])
    if env is not None:
        for name in ("check_success", "_check_success"):
            fn = getattr(env, name, None)
            if callable(fn):
                return bool(fn())
            unwrapped = getattr(env, "unwrapped", None)
            fn = getattr(unwrapped, name, None) if unwrapped is not None else None
            if callable(fn):
                return bool(fn())
    return False


def _action_dim_from_env(env: Any | None) -> int | None:
    space = getattr(env, "action_space", None)
    spaces = getattr(space, "spaces", None)
    if isinstance(spaces, dict):
        total = 0
        for item in spaces.values():
            shape = getattr(item, "shape", None)
            if not shape:
                return None
            total += int(np.prod(shape))
        return total
    shape = getattr(space, "shape", None)
    if shape:
        return int(np.prod(shape))
    return None


def _state_dim_from_observation(observation: ObservationBundle | None) -> int | None:
    if observation is None or not isinstance(observation.raw, dict):
        return None
    for key in ("groot_state", "robocasa_state", "observation.state", "state"):
        vector = _float_list(observation.raw.get(key))
        if vector:
            return len(vector)
    return None


def _robocasa_semantic_hints(env: Any | None) -> dict[str, Any]:
    base_env = _unwrap_robocasa_base_env(env)
    if base_env is None:
        return {}
    env_name = str(getattr(base_env, "__class__", type(base_env)).__name__)
    wrapper_name = str(getattr(getattr(env, "unwrapped", None), "env_name", "") or "")
    fixtures = _fixture_summaries(getattr(base_env, "fixture_refs", {}) or {})
    objects = _object_summaries(base_env)
    source = objects.get("obj") or next((item for key, item in objects.items() if not key.startswith("distr")), None)
    target = _target_fixture_hint(env_name or wrapper_name, fixtures)
    return {
        "backend": "robocasa",
        "env_name": wrapper_name or env_name,
        "task_roles": {
            "source": source,
            "target": target,
        },
        "objects": objects,
        "fixtures": fixtures,
    }


def _unwrap_robocasa_base_env(env: Any | None) -> Any | None:
    current = env
    seen: set[int] = set()
    for _ in range(6):
        if current is None or id(current) in seen:
            return current
        seen.add(id(current))
        if hasattr(current, "objects") and hasattr(current, "fixture_refs"):
            return current
        for attr in ("unwrapped", "env", "_env"):
            try:
                child = getattr(current, attr)
            except Exception:
                child = None
            if child is not None and child is not current:
                current = child
                break
        else:
            return current
    return current


def _object_summaries(base_env: Any) -> dict[str, Any]:
    raw_objects = getattr(base_env, "objects", {}) or {}
    result: dict[str, Any] = {}
    for key, obj in raw_objects.items():
        name = str(key)
        result[name] = {
            "env_key": name,
            "label": _object_language(base_env, name) or _friendly_label(name),
            "role_hint": "source" if name == "obj" else "distractor",
            "root_body": _optional_str(getattr(obj, "root_body", None)),
            "visual_geoms": [str(item) for item in (getattr(obj, "visual_geoms", []) or [])],
            "position": _placement_position(base_env, name),
            "source": "robocasa_env_registry",
        }
    return result


def _fixture_summaries(fixture_refs: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in fixture_refs.items():
        fixture = value[0] if isinstance(value, tuple) and value else value
        name = str(key)
        result[name] = {
            "env_key": name,
            "label": _fixture_label(name, fixture),
            "fixture_class": type(fixture).__name__,
            "root_body": _optional_str(getattr(fixture, "root_body", None)),
            "position": _float_list(getattr(fixture, "pos", None)),
            "size": _float_list(getattr(fixture, "size", None)),
            "source": "robocasa_fixture_registry",
        }
    return result


def _target_fixture_hint(env_name: str, fixtures: dict[str, Any]) -> dict[str, Any] | None:
    lower_name = env_name.lower()
    priorities: list[str] = []
    if "cabinet" in lower_name:
        priorities.extend(["cab", "cabinet"])
    if "counter" in lower_name:
        priorities.extend(["counter"])
    if "sink" in lower_name:
        priorities.extend(["sink"])
    if "drawer" in lower_name:
        priorities.extend(["drawer"])
    for wanted in priorities:
        for key, fixture in fixtures.items():
            haystack = f"{key} {fixture.get('label')} {fixture.get('fixture_class')}".lower()
            if wanted in haystack:
                target = dict(fixture)
                target["role_hint"] = "target"
                return target
    return None


def _object_language(base_env: Any, name: str) -> str | None:
    fn = getattr(base_env, "get_obj_lang", None)
    if not callable(fn):
        return None
    try:
        value = fn(name)
    except Exception:
        return None
    text = str(value or "").strip()
    return text or None


def _fixture_label(name: str, fixture: Any) -> str:
    text = f"{name} {type(fixture).__name__}".lower()
    if "cab" in text or "cabinet" in text:
        return "cabinet"
    if "counter" in text:
        return "counter"
    if "sink" in text:
        return "sink"
    if "drawer" in text:
        return "drawer"
    return _friendly_label(name)


def _friendly_label(name: str) -> str:
    return name.replace("_", " ").strip() or "object"


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _placement_position(base_env: Any, name: str) -> list[float] | None:
    placements = getattr(base_env, "object_placements", {}) or {}
    placement = placements.get(name) if isinstance(placements, dict) else None
    if isinstance(placement, tuple) and placement:
        return _float_list(placement[0])
    return None


def _robocasa_env_action(action: list[float], env: Any) -> Any:
    vector = np.asarray(action, dtype=np.float32)
    space = getattr(env, "action_space", None)
    spaces = getattr(space, "spaces", None)
    if not isinstance(spaces, dict):
        return vector
    expected_dim = sum(size for _, size in ROBOCASA_GROOT_ACTION_BLOCKS)
    if vector.size != expected_dim:
        raise ValueError(f"robocasa_action_dim_mismatch:got={vector.size}:expected={expected_dim}")
    result: dict[str, np.ndarray] = {}
    offset = 0
    for key, size in ROBOCASA_GROOT_ACTION_BLOCKS:
        block = vector[offset : offset + size]
        offset += size
        block_space = spaces.get(key)
        low = getattr(block_space, "low", None)
        high = getattr(block_space, "high", None)
        if low is not None and high is not None:
            block = np.clip(block, np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32))
        result[key] = block.astype(np.float32, copy=False)
    return result


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        result = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        result = [str(item).strip() for item in value if str(item).strip()]
    else:
        result = []
    if not result:
        raise ValueError("robocasa_string_list_empty")
    return result


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return None
    try:
        return [float(item) for item in _flatten(value)]
    except (TypeError, ValueError):
        return None


def _float_vector(value: Any) -> list[float] | None:
    vector = _float_list(value)
    return vector if vector and np.isfinite(np.asarray(vector, dtype=np.float32)).all() else None


def _flatten(value: list[Any]) -> list[Any]:
    result: list[Any] = []
    for item in value:
        if isinstance(item, list):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result


def _mean_float(value: Any) -> float | None:
    values = _float_list(value)
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _robocasa_gripper_state(value: Any) -> str | None:
    scalar = _mean_float(value)
    if scalar is None:
        return None
    return "open" if scalar > 0.0 else "closed"


def _execution_unavailable(reason: str, action_chunk: ActionChunk | None, adapter: RoboCasaAdapter) -> dict[str, Any]:
    emit_status_notice(
        "execution_unavailable",
        success=False,
        source="robocasa.execute_action",
        reason=reason,
        payload=action_chunk.to_dict() if hasattr(action_chunk, "to_dict") else action_chunk,
    )
    return {
        "backend": "robocasa",
        "status": "execution_unavailable",
        "reason": reason,
        "retryable": False,
        "action_chunk": action_chunk.to_dict() if hasattr(action_chunk, "to_dict") else action_chunk,
        "task_env_bound": adapter.env is not None,
    }
