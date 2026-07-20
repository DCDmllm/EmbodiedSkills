from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import ArtifactStore
from ..config import EnvironmentConfig, PROJECT_ROOT
from ..notices import emit_status_notice
from ..schema import ActionChunk, CameraView, ObservationBundle, RobotArmState
from .base import RobotEnvAdapter


LIBERO_CAMERA_MAP = {
    "agentview_image": ("agentview", "observation.images.image"),
    "robot0_eye_in_hand_image": ("wrist", "observation.images.image2"),
}


class LiberoAdapter(RobotEnvAdapter):
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        params = dict(config.params)
        self.suite_name = str(params.get("suite") or config.task_name or "libero_object")
        self.task_id = int(params.get("task_id", 0))
        self.episode_index = int(params.get("episode_index", config.seed or 0))
        self.observation_height = int(params.get("observation_height") or params.get("obs_size") or 256)
        self.observation_width = int(params.get("observation_width") or params.get("obs_size") or 256)
        self.camera_names = _camera_names(params.get("camera_names") or ["agentview_image", "robot0_eye_in_hand_image"])
        self.rotate_images_180 = bool(params.get("rotate_images_180", True))
        self.control_mode = str(params.get("control_mode") or "relative")
        self.init_states = bool(params.get("init_states", True))
        self.num_steps_wait = int(params.get("num_steps_wait", 10))
        self.render_gpu_device_id = int(params.get("render_gpu_device_id", -1))
        self.artifacts = ArtifactStore(config.artifact_dir or Path(PROJECT_ROOT) / "tmp_artifacts" / "libero")
        self.env: Any | None = None
        self.suite: Any | None = None
        self.task: Any | None = None
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
            raise RuntimeError("libero_observation_unavailable:no_raw_observation")
        observation = normalize_libero_observation(
            raw_observation,
            task_instruction=kwargs.get("instruction") or self.task_language(),
            metadata=self._capture_metadata(kwargs),
            artifacts=self.artifacts,
            artifact_prefix=str(kwargs.get("artifact_prefix", "capture")),
            camera_names=self.camera_names,
            rotate_images_180=self.rotate_images_180,
        )
        self.last_raw_observation = raw_observation
        self.last_observation = observation
        return observation

    def setup(self, instruction: str | None = None) -> dict[str, Any]:
        _ = instruction
        self._load_suite()
        if self.env is None:
            self.env = self._make_env()
        if hasattr(self.env, "seed"):
            self.env.seed(int(self.config.seed or 0))
        raw_observation = self.env.reset()
        if self.init_states:
            init_states = self._init_states()
            if init_states is not None and len(init_states) > 0:
                raw_observation = self.env.set_init_state(init_states[self.episode_index % len(init_states)])
        if self.control_mode == "absolute":
            for robot in getattr(self.env, "robots", []) or []:
                robot.controller.use_delta = False
        elif self.control_mode == "relative":
            for robot in getattr(self.env, "robots", []) or []:
                robot.controller.use_delta = True
        else:
            raise ValueError(f"invalid_libero_control_mode:{self.control_mode}")
        for _ in range(self.num_steps_wait):
            raw_observation, self.last_reward, self.last_done, self.last_info = self.env.step(_libero_dummy_action())
        self.step_count = 0
        self.last_raw_observation = raw_observation
        return raw_observation

    def execute_action(self, action_chunk: ActionChunk | None) -> dict[str, Any]:
        if action_chunk is None:
            return _execution_unavailable("missing_action_chunk", action_chunk, self)
        if action_chunk.action_type != "libero_ee_delta" or not action_chunk.commands:
            return _execution_unavailable("unsupported_or_empty_action_chunk", action_chunk, self)
        if self.env is None:
            return _execution_unavailable("libero_env_not_setup", action_chunk, self)

        executed: list[list[float]] = []
        success = False
        done = False
        info: dict[str, Any] = {}
        raw_observation: dict[str, Any] | None = None
        for command in action_chunk.commands:
            action = _float_vector(command, expected_dim=7)
            if action is None:
                return _execution_unavailable("invalid_libero_7d_action_command", action_chunk, self)
            raw_observation, reward, done, info = self.env.step(np.asarray(action, dtype=np.float32))
            self.step_count += 1
            self.last_reward = float(reward)
            self.last_done = bool(done)
            self.last_info = dict(info or {})
            success = bool(self.env.check_success()) if hasattr(self.env, "check_success") else bool(info.get("is_success"))
            executed.append(action)
            if done or success:
                break
        if raw_observation is None:
            return _execution_unavailable("libero_execute_returned_no_observation", action_chunk, self)

        artifact_prefix = str(action_chunk.metadata.get("artifact_prefix", "execute")) if isinstance(action_chunk.metadata, dict) else "execute"
        self.last_raw_observation = raw_observation
        self.last_observation = normalize_libero_observation(
            raw_observation,
            task_instruction=getattr(self.last_observation, "task_instruction", None) or self.task_language(),
            metadata=self._capture_metadata({"artifact_prefix": artifact_prefix, "after_execute": True}),
            artifacts=self.artifacts,
            artifact_prefix=f"{artifact_prefix}/after",
            camera_names=self.camera_names,
            rotate_images_180=self.rotate_images_180,
        )
        return {
            "backend": "libero",
            "status": "action_executed",
            "success": success,
            "done": done,
            "executed_steps": len(executed),
            "reward": self.last_reward,
            "info": info,
            "observation": self.last_observation.to_dict(),
            "action_chunk": action_chunk.to_dict(),
            "task_env_bound": self.env is not None,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "libero",
            "suite": self.suite_name,
            "task_name": self.task_name(),
            "task_id": self.task_id,
            "task_language": self.task_language(),
            "artifact_dir": str(self.artifacts.root_dir),
            "task_env_bound": self.env is not None,
        }

    def status(self) -> dict[str, Any]:
        return {
            "backend": "libero",
            "ready": self.env is not None and self.last_observation is not None,
            "needs_setup": self.env is None,
            "last_observation_present": self.last_observation is not None,
            "live_env_bound": self.env is not None,
            "suite": self.suite_name,
            "task_id": self.task_id,
            "task_name": self.task_name(),
            "task_language": self.task_language(),
        }

    def preflight_spec(self) -> dict[str, Any]:
        cameras = [LIBERO_CAMERA_MAP.get(name, (str(name), None))[0] for name in self.camera_names]
        return {
            "backend": "libero",
            "required_cameras": cameras,
            "action_cameras": cameras,
            "expected_resolution": [self.observation_width, self.observation_height],
            "state": {"required": True, "dim": 8, "source": "libero_state8"},
            "action": {"required": True, "types": {"libero_ee_delta": 7}},
        }

    def task_status(self) -> dict[str, Any]:
        success = bool(self.env.check_success()) if self.env is not None and hasattr(self.env, "check_success") else None
        return {
            "backend": "libero",
            "suite": self.suite_name,
            "task_id": self.task_id,
            "task_name": self.task_name(),
            "task_language": self.task_language(),
            "success": success,
            "done": bool(self.last_done) if self.last_done is not None else success,
            "step_count": self.step_count,
            "reward": self.last_reward,
            "info": dict(self.last_info),
        }

    def task_name(self) -> str | None:
        if self.task is not None:
            return str(getattr(self.task, "name", "") or "")
        return str(self.config.task_name or self.suite_name)

    def task_language(self) -> str | None:
        if self.task is not None:
            return str(getattr(self.task, "language", "") or "") or None
        return None

    def close(self) -> None:
        if self.env is not None and hasattr(self.env, "close"):
            self.env.close()
        self.env = None

    def _load_suite(self) -> None:
        if self.suite is not None and self.task is not None:
            return
        from libero.libero import benchmark

        benchmarks = benchmark.get_benchmark_dict()
        if self.suite_name not in benchmarks:
            raise ValueError(f"unknown_libero_suite:{self.suite_name}:available={sorted(benchmarks)}")
        self.suite = benchmarks[self.suite_name]()
        task_count = len(getattr(self.suite, "tasks", []) or [])
        if self.task_id < 0 or self.task_id >= task_count:
            raise ValueError(f"libero_task_id_out_of_range:{self.task_id}:task_count={task_count}")
        self.task = self.suite.get_task(self.task_id)

    def _make_env(self) -> Any:
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        assert self.task is not None
        bddl_path = Path(get_libero_path("bddl_files")) / self.task.problem_folder / self.task.bddl_file
        if not bddl_path.exists():
            raise FileNotFoundError(f"libero_bddl_file_not_found:{bddl_path}")
        return OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            camera_heights=self.observation_height,
            camera_widths=self.observation_width,
            camera_names=[_render_camera_name(name) for name in self.camera_names],
            render_gpu_device_id=self.render_gpu_device_id,
        )

    def _init_states(self) -> Any | None:
        if self.suite is None:
            return None
        if hasattr(self.suite, "get_task_init_states"):
            return self.suite.get_task_init_states(self.task_id)
        return None

    def _capture_metadata(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        metadata = {
            "backend": "libero",
            "suite": self.suite_name,
            "task_id": self.task_id,
            "task_name": self.task_name(),
            "task_language": self.task_language(),
            "seed": self.config.seed,
            "episode_index": self.episode_index,
            "control_mode": self.control_mode,
            "camera_names": list(self.camera_names),
            "rotate_images_180": self.rotate_images_180,
            "artifact_dir": str(self.artifacts.root_dir),
            "task_env_bound": self.env is not None,
        }
        metadata.update(kwargs)
        return metadata


def normalize_libero_observation(
    raw_observation: dict[str, Any],
    task_instruction: str | None = None,
    metadata: dict[str, Any] | None = None,
    artifacts: ArtifactStore | None = None,
    artifact_prefix: str = "capture",
    camera_names: list[str] | None = None,
    rotate_images_180: bool = True,
) -> ObservationBundle:
    camera_names = list(camera_names or ["agentview_image", "robot0_eye_in_hand_image"])
    camera_views = _camera_views_from_libero(
        raw_observation,
        camera_names,
        artifacts,
        artifact_prefix,
        rotate_images_180=rotate_images_180,
    )
    state8 = _state8_from_libero(raw_observation)
    robot_arms = {
        "panda": RobotArmState(
            arm_name="panda",
            eef_pose=_float_list(raw_observation.get("robot0_eef_pos")),
            gripper_state=_libero_gripper_state(raw_observation.get("robot0_gripper_qpos")),
            gripper_value=_gripper_scalar(raw_observation.get("robot0_gripper_qpos")),
            joint_positions=_float_list(raw_observation.get("robot0_joint_pos")),
            metadata={
                "state8": state8,
                "eef_quat_xyzw": _float_list(raw_observation.get("robot0_eef_quat")),
                "gripper_qpos": _float_list(raw_observation.get("robot0_gripper_qpos")),
                "joint_vel": _float_list(raw_observation.get("robot0_joint_vel")),
                "state_source": "libero_raw_observation",
            },
        )
    }
    summary = _raw_summary(raw_observation, camera_names, state8)
    raw_ref = artifacts.write_json(f"{artifact_prefix}/raw_observation_summary.json", summary) if artifacts else None
    return ObservationBundle(
        task_instruction=task_instruction,
        camera_views=camera_views,
        robot_arms=robot_arms,
        pointcloud_ref=None,
        raw={
            "keys": sorted(str(key) for key in raw_observation.keys()),
            "summary_ref": raw_ref,
            "libero_state8": state8,
            "state_source": "libero_state8",
            "image_orientation": "rotated_180" if rotate_images_180 else "libero_raw",
        },
        metadata=dict(metadata or {}),
    )


def _camera_views_from_libero(
    raw_observation: dict[str, Any],
    camera_names: list[str],
    artifacts: ArtifactStore | None,
    artifact_prefix: str,
    *,
    rotate_images_180: bool,
) -> dict[str, CameraView]:
    camera_views: dict[str, CameraView] = {}
    for raw_name in camera_names:
        view_name, policy_key = LIBERO_CAMERA_MAP.get(raw_name, (str(raw_name), f"observation.images.{raw_name}"))
        raw_image = raw_observation.get(raw_name)
        image = _orient_libero_image(raw_image, rotate_images_180=rotate_images_180)
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
                "source_orientation": "libero_raw",
                "orientation": "rotated_180" if rotate_images_180 else "libero_raw",
                "rotation_applied": "rot180" if rotate_images_180 else None,
            },
        )
    return camera_views


def _orient_libero_image(image: Any, *, rotate_images_180: bool) -> Any:
    if image is None or not rotate_images_180:
        return image
    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError(f"libero_image_invalid_ndim:{array.ndim}")
    return np.flip(array, axis=(0, 1)).copy()


def _state8_from_libero(raw_observation: dict[str, Any]) -> list[float]:
    eef_pos = _float_list(raw_observation.get("robot0_eef_pos"))
    eef_quat = _float_list(raw_observation.get("robot0_eef_quat"))
    gripper_qpos = _float_list(raw_observation.get("robot0_gripper_qpos"))
    if eef_pos is None or len(eef_pos) != 3:
        raise ValueError("libero_state_unavailable:missing_robot0_eef_pos")
    if eef_quat is None or len(eef_quat) != 4:
        raise ValueError("libero_state_unavailable:missing_robot0_eef_quat")
    if gripper_qpos is None or len(gripper_qpos) != 2:
        raise ValueError("libero_state_unavailable:missing_robot0_gripper_qpos")
    axis_angle = _quat_xyzw_to_axis_angle(eef_quat)
    return [*eef_pos, *axis_angle, *gripper_qpos]


def _quat_xyzw_to_axis_angle(quat: list[float]) -> list[float]:
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 0:
        raise ValueError("libero_state_unavailable:zero_norm_eef_quat")
    q = q / norm
    xyz = q[:3]
    w = float(np.clip(q[3], -1.0, 1.0))
    den = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if den <= 1e-10:
        return [0.0, 0.0, 0.0]
    axis = xyz / den
    angle = 2.0 * np.arccos(w)
    return [float(item) for item in (axis * angle).tolist()]


def _raw_summary(raw_observation: dict[str, Any], camera_names: list[str], state8: list[float]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "keys": sorted(str(key) for key in raw_observation.keys()),
        "cameras": {},
        "libero_state8": state8,
    }
    for name in camera_names:
        value = raw_observation.get(name)
        summary["cameras"][name] = {"shape": list(np.asarray(value).shape) if value is not None else None}
    for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_joint_pos", "robot0_joint_vel"):
        value = raw_observation.get(key)
        if value is not None:
            summary[key] = _float_list(value)
    return summary


def _camera_names(value: Any) -> list[str]:
    if isinstance(value, str):
        names = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        names = [str(item).strip() for item in value if str(item).strip()]
    else:
        names = []
    if not names:
        raise ValueError("libero_camera_names_empty")
    return names


def _render_camera_name(raw_observation_key: str) -> str:
    if raw_observation_key.endswith("_image"):
        return raw_observation_key[: -len("_image")]
    return raw_observation_key


def _float_list(value: Any) -> list[float] | None:
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


def _gripper_scalar(value: Any) -> float | None:
    values = _float_list(value)
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _libero_gripper_state(value: Any) -> str | None:
    scalar = _gripper_scalar(value)
    if scalar is None:
        return None
    return "open" if scalar > 0.0 else "closed"


def _float_vector(value: Any, expected_dim: int) -> list[float] | None:
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.shape[0] != expected_dim or not np.isfinite(vector).all():
        return None
    return [float(item) for item in vector.tolist()]


def _libero_dummy_action() -> np.ndarray:
    return np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


def _execution_unavailable(reason: str, action_chunk: ActionChunk | None, adapter: LiberoAdapter) -> dict[str, Any]:
    emit_status_notice(
        "execution_unavailable",
        success=False,
        source="libero.execute_action",
        reason=reason,
        payload=action_chunk.to_dict() if hasattr(action_chunk, "to_dict") else action_chunk,
    )
    return {
        "backend": "libero",
        "status": "execution_unavailable",
        "reason": reason,
        "retryable": False,
        "action_chunk": action_chunk.to_dict() if hasattr(action_chunk, "to_dict") else action_chunk,
        "task_env_bound": adapter.env is not None,
    }
