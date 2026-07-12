from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import ArtifactStore
from ..config import EnvironmentConfig
from ..notices import emit_status_notice
from ..schema import ActionChunk, CameraView, ObservationBundle, RobotArmState
from .base import RobotEnvAdapter


CALVIN_CAMERA_MAP = {
    "rgb_static": ("static", "observation.images.rgb_static"),
    "static": ("static", "observation.images.rgb_static"),
    "rgb_gripper": ("gripper", "observation.images.rgb_gripper"),
    "gripper": ("gripper", "observation.images.rgb_gripper"),
}

CALVIN_ACTION_TYPES = {"calvin_ee_pose_10d", "calvin_ee_delta"}


class CalvinAdapter(RobotEnvAdapter):
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        params = dict(config.params)
        self.repo_root = Path(str(params.get("repo_root") or "/mnt/wangwai/vla/CALVIN"))
        self.dataset_path = Path(str(params.get("dataset_path") or self.repo_root / "dataset" / "task_D_D"))
        self.validation_dir = Path(str(params.get("validation_dir") or self.dataset_path / "validation"))
        self.camera_names = _string_list(params.get("camera_names") or ["rgb_static", "rgb_gripper"])
        self.sequence_index = int(params.get("sequence_index", 0))
        self.subtask_index = int(params.get("subtask_index", 0))
        self.max_episode_steps = int(params.get("max_episode_steps", 720))
        self.show_gui = _truthy(params.get("show_gui", False))
        self.use_egl = _truthy(params.get("use_egl", True))
        self.scene = str(params["scene"]) if params.get("scene") else None
        self.gripper_close_threshold = float(params.get("gripper_close_threshold", 0.8))
        self.artifacts = ArtifactStore(config.artifact_dir or "/mnt/wangwai/vla/clawvla/tmp_artifacts/calvin")

        self.env: Any | None = None
        self.task_oracle: Any | None = None
        self.val_annotations: Any | None = None
        self.initial_state: dict[str, Any] | None = dict(params["initial_state"]) if isinstance(params.get("initial_state"), dict) else None
        self.eval_sequence: tuple[str, ...] | None = (
            tuple(str(item) for item in params["eval_sequence"])
            if isinstance(params.get("eval_sequence"), list)
            else None
        )
        self.current_subtask: str | None = str(params["subtask"]) if params.get("subtask") else None
        if self.eval_sequence is None and self.current_subtask is not None:
            self.eval_sequence = (self.current_subtask,)
        self.start_info: dict[str, Any] | None = None
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
            if raw_observation is None and self.env is not None and hasattr(self.env, "get_obs"):
                raw_observation = self.env.get_obs()
        if not isinstance(raw_observation, dict):
            raise RuntimeError("calvin_observation_unavailable:no_raw_observation")
        observation = normalize_calvin_observation(
            raw_observation,
            task_instruction=kwargs.get("instruction") or self.task_language(),
            metadata=self._capture_metadata(kwargs),
            artifacts=self.artifacts,
            artifact_prefix=str(kwargs.get("artifact_prefix", "capture")),
            camera_names=self.camera_names,
        )
        self.last_raw_observation = raw_observation
        self.last_observation = observation
        return observation

    def setup(self, instruction: str | None = None) -> dict[str, Any]:
        _ = instruction
        self._ensure_repo_pythonpath()
        self._check_validation_dir()
        self._load_task_context()
        if self.env is None:
            self.env = self._make_env()
        if self.initial_state is None:
            raise RuntimeError("calvin_initial_state_unavailable")
        robot_obs, scene_obs = _calvin_env_state_for_initial_condition(self.initial_state)
        raw_observation = self.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        self.start_info = dict(self.env.get_info()) if hasattr(self.env, "get_info") else None
        self.last_info = dict(self.start_info or {})
        self.last_done = False
        self.last_reward = 0.0
        self.step_count = 0
        if not isinstance(raw_observation, dict):
            raise RuntimeError(f"calvin_reset_returned_non_dict_observation:{type(raw_observation).__name__}")
        self.last_raw_observation = raw_observation
        return raw_observation

    def execute_action(self, action_chunk: ActionChunk | None) -> dict[str, Any]:
        if action_chunk is None:
            return _execution_unavailable("missing_action_chunk", action_chunk, self)
        if action_chunk.action_type not in CALVIN_ACTION_TYPES or not action_chunk.commands:
            return _execution_unavailable("unsupported_or_empty_action_chunk", action_chunk, self)
        if self.env is None:
            return _execution_unavailable("calvin_env_not_setup", action_chunk, self)

        before_state = _safe_robot_obs(self.last_raw_observation)
        executed: list[list[float]] = []
        success = False
        done = False
        info: dict[str, Any] = {}
        raw_observation: dict[str, Any] | None = None
        for command in action_chunk.commands:
            vector = _float_vector(command, expected_dim=10)
            if vector is None:
                return _execution_unavailable("invalid_calvin_10d_action_command", action_chunk, self)
            env_action = _calvin_env_action(vector, gripper_close_threshold=self.gripper_close_threshold)
            raw_observation, reward, done, info = _split_step(self.env.step(env_action))
            self.step_count += 1
            self.last_reward = float(reward)
            self.last_done = bool(done)
            self.last_info = dict(info or {})
            executed.append(vector)
            success = self._success_from_info(self.last_info)
            if done or success or self.step_count >= self.max_episode_steps:
                break
        if raw_observation is None:
            return _execution_unavailable("calvin_execute_returned_no_observation", action_chunk, self)

        artifact_prefix = str(action_chunk.metadata.get("artifact_prefix", "execute")) if isinstance(action_chunk.metadata, dict) else "execute"
        self.last_raw_observation = raw_observation
        self.last_observation = normalize_calvin_observation(
            raw_observation,
            task_instruction=getattr(self.last_observation, "task_instruction", None) or self.task_language(),
            metadata=self._capture_metadata({"artifact_prefix": artifact_prefix, "after_execute": True}),
            artifacts=self.artifacts,
            artifact_prefix=f"{artifact_prefix}/after",
            camera_names=self.camera_names,
        )
        return {
            "backend": "calvin",
            "status": "action_executed",
            "success": success,
            "done": done or success or self.step_count >= self.max_episode_steps,
            "executed_steps": len(executed),
            "reward": self.last_reward,
            "info": info,
            "observation": self.last_observation.to_dict(),
            "action_chunk": action_chunk.to_dict(),
            "action_effect": _action_effect_report(before_state, _safe_robot_obs(raw_observation), executed),
            "task_env_bound": self.env is not None,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "calvin",
            "task_name": self.config.task_name or "calvin",
            "task_language": self.task_language(),
            "subtask": self.current_subtask,
            "sequence_index": self.sequence_index,
            "subtask_index": self.subtask_index,
            "dataset_path": str(self.dataset_path),
            "validation_dir": str(self.validation_dir),
            "artifact_dir": str(self.artifacts.root_dir),
            "task_env_bound": self.env is not None,
        }

    def status(self) -> dict[str, Any]:
        return {
            "backend": "calvin",
            "ready": self.env is not None and self.last_observation is not None,
            "needs_setup": self.env is None,
            "last_observation_present": self.last_observation is not None,
            "live_env_bound": self.env is not None,
            "task_name": self.config.task_name or "calvin",
            "task_language": self.task_language(),
            "subtask": self.current_subtask,
            "step_count": self.step_count,
        }

    def preflight_spec(self) -> dict[str, Any]:
        cameras = [CALVIN_CAMERA_MAP.get(name, (_remove_prefix(str(name), "rgb_"), None))[0] for name in self.camera_names]
        return {
            "backend": "calvin",
            "required_cameras": cameras,
            "action_cameras": cameras,
            "expected_resolution": None,
            "state": {"required": True, "dim": 20, "source": "calvin_proprio"},
            "action": {"required": True, "types": {"calvin_ee_pose_10d": 10}},
        }

    def task_status(self) -> dict[str, Any]:
        success = self._success_from_info(self.last_info) if self.env is not None else None
        return {
            "backend": "calvin",
            "task_name": self.config.task_name or "calvin",
            "task_language": self.task_language(),
            "subtask": self.current_subtask,
            "sequence_index": self.sequence_index,
            "subtask_index": self.subtask_index,
            "success": success,
            "done": bool(self.last_done) if self.last_done is not None else success,
            "step_count": self.step_count,
            "reward": self.last_reward,
            "info": dict(self.last_info),
        }

    def task_language(self) -> str | None:
        if self.current_subtask and self.val_annotations is not None:
            try:
                annotation = self.val_annotations[self.current_subtask][0]
            except Exception:
                annotation = None
            if annotation:
                return str(annotation).split("\n", 1)[0].replace("\u2019", "'").strip()
        return None

    def close(self) -> None:
        if self.env is not None and hasattr(self.env, "close"):
            self.env.close()
        self.env = None

    def _ensure_repo_pythonpath(self) -> None:
        for path in (self.repo_root / "calvin_env", self.repo_root / "calvin_models"):
            text = str(path)
            if path.exists() and text not in sys.path:
                sys.path.insert(0, text)

    def _check_validation_dir(self) -> None:
        if not self.repo_root.exists():
            raise FileNotFoundError(f"calvin_repo_root_not_found:{self.repo_root}")
        if not self.validation_dir.exists():
            raise FileNotFoundError(f"calvin_dataset_validation_dir_not_found:{self.validation_dir}")
        merged_config = self.validation_dir / ".hydra" / "merged_config.yaml"
        if not merged_config.exists():
            raise FileNotFoundError(f"calvin_dataset_merged_config_not_found:{merged_config}")

    def _load_task_context(self) -> None:
        if self.initial_state is not None and self.eval_sequence is not None and self.task_oracle is not None:
            return
        from omegaconf import OmegaConf
        import hydra

        if self.initial_state is None or self.eval_sequence is None:
            from calvin_agent.evaluation.multistep_sequences import get_sequences

            sequences = list(get_sequences(max(self.sequence_index + 1, 1), num_workers=1))
            if self.sequence_index < 0 or self.sequence_index >= len(sequences):
                raise ValueError(f"calvin_sequence_index_out_of_range:{self.sequence_index}:available={len(sequences)}")
            initial_state, sequence = sequences[self.sequence_index]
            self.initial_state = dict(initial_state)
            self.eval_sequence = tuple(str(item) for item in sequence)
        if self.current_subtask is None:
            if self.subtask_index < 0 or self.subtask_index >= len(self.eval_sequence):
                raise ValueError(
                    f"calvin_subtask_index_out_of_range:{self.subtask_index}:sequence_len={len(self.eval_sequence)}"
                )
            self.current_subtask = self.eval_sequence[self.subtask_index]
        conf_dir = self.repo_root / "calvin_models" / "conf"
        task_cfg = OmegaConf.load(conf_dir / "callbacks" / "rollout" / "tasks" / "new_playtable_tasks.yaml")
        self.task_oracle = hydra.utils.instantiate(task_cfg)
        self.val_annotations = OmegaConf.load(conf_dir / "annotations" / "new_playtable_validation.yaml")
        if self.current_subtask not in self.val_annotations:
            raise KeyError(f"calvin_subtask_annotation_missing:{self.current_subtask}")

    def _make_env(self) -> Any:
        from calvin_env.envs.play_table_env import get_env

        kwargs: dict[str, Any] = {}
        if self.scene:
            kwargs["scene"] = self.scene
        env = get_env(self.validation_dir, obs_space=_calvin_obs_space(self.camera_names), show_gui=self.show_gui, **kwargs)
        if not self.use_egl and getattr(env, "use_egl", None):
            raise RuntimeError(
                "calvin_use_egl_false_not_supported_by_official_get_env_after_init; "
                "set use_egl=true or create a validation merged_config.yaml with env.use_egl=false."
            )
        return env

    def _success_from_info(self, current_info: dict[str, Any] | None) -> bool | None:
        if self.task_oracle is None or self.start_info is None or not self.current_subtask or current_info is None:
            return None
        task_info = self.task_oracle.get_task_info_for_set(self.start_info, current_info, {self.current_subtask})
        return bool(task_info)

    def _capture_metadata(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        metadata = {
            "backend": "calvin",
            "task_name": self.config.task_name or "calvin",
            "task_language": self.task_language(),
            "subtask": self.current_subtask,
            "sequence_index": self.sequence_index,
            "subtask_index": self.subtask_index,
            "seed": self.config.seed,
            "camera_names": list(self.camera_names),
            "dataset_path": str(self.dataset_path),
            "validation_dir": str(self.validation_dir),
            "artifact_dir": str(self.artifacts.root_dir),
            "task_env_bound": self.env is not None,
        }
        metadata.update(kwargs)
        return metadata


def normalize_calvin_observation(
    raw_observation: dict[str, Any],
    task_instruction: str | None = None,
    metadata: dict[str, Any] | None = None,
    artifacts: ArtifactStore | None = None,
    artifact_prefix: str = "capture",
    camera_names: list[str] | None = None,
) -> ObservationBundle:
    camera_names = list(camera_names or ["rgb_static", "rgb_gripper"])
    camera_views = _camera_views_from_calvin(raw_observation, camera_names, artifacts, artifact_prefix)
    robot_obs = _float_vector(raw_observation.get("robot_obs"))
    if robot_obs is None:
        raise ValueError("calvin_state_unavailable:missing_robot_obs")
    calvin_proprio = _calvin_proprio(robot_obs)
    scene_obs = _float_vector(raw_observation.get("scene_obs"))
    robot_arms = {
        "panda": RobotArmState(
            arm_name="panda",
            eef_pose=robot_obs[:6] if len(robot_obs) >= 6 else None,
            gripper_state=_calvin_gripper_state(robot_obs),
            gripper_value=robot_obs[6] if len(robot_obs) > 6 else None,
            joint_positions=robot_obs[7:14] if len(robot_obs) >= 14 else None,
            metadata={
                "calvin_proprio": calvin_proprio,
                "robot_obs": robot_obs,
                "scene_obs": scene_obs,
                "state_source": "calvin_raw_observation",
            },
        )
    }
    summary = _raw_summary(raw_observation, camera_names, robot_obs, scene_obs, calvin_proprio)
    raw_ref = artifacts.write_json(f"{artifact_prefix}/raw_observation_summary.json", summary) if artifacts else None
    return ObservationBundle(
        task_instruction=task_instruction,
        camera_views=camera_views,
        robot_arms=robot_arms,
        pointcloud_ref=None,
        raw={
            "keys": sorted(str(key) for key in raw_observation.keys()),
            "summary_ref": raw_ref,
            "calvin_proprio": calvin_proprio,
            "robot_obs": robot_obs,
            "scene_obs": scene_obs,
            "state_source": "calvin_proprio",
        },
        metadata=dict(metadata or {}),
    )


def _camera_views_from_calvin(
    raw_observation: dict[str, Any],
    camera_names: list[str],
    artifacts: ArtifactStore | None,
    artifact_prefix: str,
) -> dict[str, CameraView]:
    rgb_obs = raw_observation.get("rgb_obs", {})
    depth_obs = raw_observation.get("depth_obs", {})
    if not isinstance(rgb_obs, dict):
        rgb_obs = {}
    if not isinstance(depth_obs, dict):
        depth_obs = {}
    camera_views: dict[str, CameraView] = {}
    for requested_name in camera_names:
        raw_name = _calvin_rgb_key(requested_name)
        view_name, policy_key = CALVIN_CAMERA_MAP.get(
            requested_name,
            CALVIN_CAMERA_MAP.get(raw_name, (_remove_prefix(raw_name, "rgb_"), f"observation.images.{raw_name}")),
        )
        depth_name = f"depth_{view_name}"
        raw_image = rgb_obs.get(raw_name)
        image = _orient_image(raw_image)
        depth = depth_obs.get(depth_name)
        rgb_path = artifacts.write_image(f"{artifact_prefix}/images/{view_name}_rgb.png", image) if artifacts and image is not None else None
        depth_path = artifacts.write_depth(f"{artifact_prefix}/depth/{view_name}_depth.npy", depth) if artifacts and depth is not None else None
        camera_views[view_name] = CameraView(
            name=view_name,
            rgb_path=rgb_path,
            depth_path=depth_path,
            mask_path=None,
            intrinsics=None,
            extrinsics=None,
            metadata={
                "raw_name": raw_name,
                "depth_name": depth_name,
                "policy_key": policy_key,
                "has_rgb": raw_image is not None,
                "has_depth": depth is not None,
                "shape": list(np.asarray(image).shape) if image is not None else None,
                "raw_shape": list(np.asarray(raw_image).shape) if raw_image is not None else None,
                "depth_shape": list(np.asarray(depth).shape) if depth is not None else None,
            },
        )
    return camera_views


def _calvin_rgb_key(name: str) -> str:
    return name if name.startswith("rgb_") else f"rgb_{name}"


def _calvin_depth_key(name: str) -> str:
    camera = _remove_prefix(_calvin_rgb_key(name), "rgb_")
    return f"depth_{camera}"


def _calvin_obs_space(camera_names: list[str]) -> dict[str, list[str]]:
    return {
        "rgb_obs": [_calvin_rgb_key(name) for name in camera_names],
        "depth_obs": [_calvin_depth_key(name) for name in camera_names],
    }


def _remove_prefix(value: str, prefix: str) -> str:
    return value[len(prefix) :] if value.startswith(prefix) else value


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


def _calvin_proprio(robot_obs: list[float]) -> list[float]:
    if len(robot_obs) < 7:
        raise ValueError(f"calvin_state_unavailable:robot_obs_too_short:{len(robot_obs)}")
    base = [*robot_obs[:3], *_euler_xyz_to_rot6d(robot_obs[3:6]), 1.0 if robot_obs[-1] > 0.0 else 0.0]
    return [*base, *([0.0] * len(base))]


def _euler_xyz_to_rot6d(euler: list[float]) -> list[float]:
    try:
        from scipy.spatial.transform import Rotation as R

        matrix = R.from_euler("xyz", np.asarray(euler, dtype=np.float64), degrees=False).as_matrix()
    except Exception:
        matrix = _euler_xyz_matrix(euler)
    rot6d = np.asarray(matrix, dtype=np.float64)[:, :2].reshape(-1)
    return [float(item) for item in rot6d.tolist()]


def _euler_xyz_matrix(euler: list[float]) -> np.ndarray:
    x, y, z = [float(item) for item in euler]
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    rx = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _calvin_env_action(command: list[float], *, gripper_close_threshold: float) -> tuple[np.ndarray, np.ndarray, int]:
    vector = np.asarray(command, dtype=np.float64)
    position = vector[:3].astype(np.float32)
    quaternion = _rot6d_to_quat_xyzw(vector[3:9]).astype(np.float32)
    gripper = 1 if float(vector[9]) < gripper_close_threshold else -1
    return position, quaternion, gripper


def _rot6d_to_quat_xyzw(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(6)
    a1 = vector[0:5:2]
    a2 = vector[1:6:2]
    b1 = _normalize(a1)
    proj = float(np.sum(b1 * a2)) * b1
    b2 = _normalize(a2 - proj)
    b3 = np.cross(b1, b2)
    matrix = np.stack((b1, b2, b3), axis=-1)
    return _matrix_to_quat_xyzw(matrix)


def _normalize(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("calvin_rotation6d_invalid_zero_norm")
    return value / norm


def _matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _raw_summary(
    raw_observation: dict[str, Any],
    camera_names: list[str],
    robot_obs: list[float],
    scene_obs: list[float] | None,
    calvin_proprio: list[float],
) -> dict[str, Any]:
    rgb_obs = raw_observation.get("rgb_obs", {})
    depth_obs = raw_observation.get("depth_obs", {})
    if not isinstance(rgb_obs, dict):
        rgb_obs = {}
    if not isinstance(depth_obs, dict):
        depth_obs = {}
    summary: dict[str, Any] = {
        "keys": sorted(str(key) for key in raw_observation.keys()),
        "cameras": {},
        "robot_obs": robot_obs,
        "scene_obs": scene_obs,
        "calvin_proprio": calvin_proprio,
    }
    for requested_name in camera_names:
        raw_name = _calvin_rgb_key(requested_name)
        depth_name = f"depth_{_remove_prefix(raw_name, 'rgb_')}"
        image = rgb_obs.get(raw_name)
        depth = depth_obs.get(depth_name)
        summary["cameras"][raw_name] = {
            "shape": list(np.asarray(image).shape) if image is not None else None,
            "depth_shape": list(np.asarray(depth).shape) if depth is not None else None,
        }
    return summary


def _safe_robot_obs(raw_observation: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(raw_observation, dict):
        return None
    return _float_vector(raw_observation.get("robot_obs"))


def _action_effect_report(
    before_state: list[float] | None,
    after_state: list[float] | None,
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
    if before_state is None or after_state is None or len(before_state) != len(after_state):
        return report
    deltas = [abs(float(after) - float(before)) for before, after in zip(before_state, after_state)]
    max_delta = max(deltas) if deltas else 0.0
    report.update(
        {
            "state_delta_available": True,
            "state_changed": bool(max_delta > 1e-6),
            "max_abs_state_delta": max_delta,
            "sum_abs_state_delta": float(sum(deltas)),
        }
    )
    return report


def _split_step(result: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 4:
        obs, reward, done, info = result
        if not isinstance(obs, dict):
            raise RuntimeError(f"calvin_step_returned_non_dict_observation:{type(obs).__name__}")
        return obs, float(reward), bool(done), dict(info or {})
    raise RuntimeError(f"calvin_step_return_shape_unsupported:{type(result).__name__}")


def _execution_unavailable(reason: str, action_chunk: ActionChunk | None, adapter: CalvinAdapter) -> dict[str, Any]:
    emit_status_notice(
        "execution_unavailable",
        success=False,
        source="calvin.execute_action",
        reason=reason,
        payload=action_chunk.to_dict() if hasattr(action_chunk, "to_dict") else action_chunk,
    )
    return {
        "backend": "calvin",
        "status": "execution_unavailable",
        "reason": reason,
        "retryable": False,
        "action_chunk": action_chunk.to_dict() if hasattr(action_chunk, "to_dict") else action_chunk,
        "task_env_bound": adapter.env is not None,
    }


def _calvin_gripper_state(robot_obs: list[float]) -> str | None:
    if len(robot_obs) <= 6:
        return None
    return "open" if float(robot_obs[6]) > 0.04 else "closed"


def _calvin_env_state_for_initial_condition(initial_condition: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    required = {"slider", "drawer", "lightbulb", "led", "red_block", "blue_block", "pink_block", "grasped"}
    missing = sorted(required - set(initial_condition))
    if missing:
        raise KeyError(f"calvin_initial_condition_missing_keys:{missing}")
    robot_obs = np.asarray(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ],
        dtype=np.float64,
    )
    block_rot_z_range = (np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
    block_slider_left = np.asarray([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01], dtype=np.float64)
    block_slider_right = np.asarray([7.03416330e-02, 9.24044687e-02, 4.60990009e-01], dtype=np.float64)
    block_table = [
        np.asarray([5.00000896e-02, -1.20000177e-01, 4.59990009e-01], dtype=np.float64),
        np.asarray([2.29995412e-01, -1.19995140e-01, 4.59990010e-01], dtype=np.float64),
    ]
    seed = _fnv1_32(str(initial_condition.values()))
    with _temp_numpy_seed(seed):
        np.random.shuffle(block_table)
        scene_obs = np.zeros(24, dtype=np.float64)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = float(initial_condition["lightbulb"])
        scene_obs[5] = float(initial_condition["led"])
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)
    return robot_obs, scene_obs


def _fnv1_32(value: str) -> int:
    result = 2166136261
    for byte in value.encode("utf-8"):
        result = (result * 16777619) & 0xFFFFFFFF
        result ^= byte
    return result


@contextmanager
def _temp_numpy_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(int(seed) & 0xFFFFFFFF)
    try:
        yield
    finally:
        np.random.set_state(state)


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


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        result = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        result = [str(item).strip() for item in value if str(item).strip()]
    else:
        result = []
    if not result:
        raise ValueError("calvin_string_list_empty")
    return result


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False
