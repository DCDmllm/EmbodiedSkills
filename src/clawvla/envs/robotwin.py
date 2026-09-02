from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import ArtifactStore
from ..config import RobotwinConfig, RuntimeEnvironment
from ..notices import emit_status_notice
from ..schema import ActionChunk, CameraView, ObservationBundle, RobotArmState
from .base import RobotEnvAdapter
from .robotwin_session import RoboTwinSession, robotwin_cwd


def robotwin_runtime_environment(config: RuntimeEnvironment) -> dict[str, Any]:
    return {
        "conda_env": config.conda_env,
        "conda_bin": config.conda_bin,
        "pythonpath_prefix": list(config.pythonpath_prefix),
        "env": dict(config.env),
    }


class RoboTwinAdapter(RobotEnvAdapter):
    """RoboTwin adapter.

    This class keeps the new runtime independent from the old AgentVLA scripts while
    preserving the environment contract. Real reset/get_obs/step wiring should be
    implemented here, not inside the scheduler.
    """

    def __init__(self, config: RobotwinConfig):
        self.config = config
        self.bound_task_env: Any | None = None
        self.session: RoboTwinSession | None = None
        self.last_observation: ObservationBundle | None = None
        self.artifacts = ArtifactStore(config.artifact_dir)

    def bind_task_env(self, task_env: Any) -> None:
        self.bound_task_env = task_env
        self.session = RoboTwinSession(self.config, task_env=task_env)

    def ensure_session(self) -> RoboTwinSession:
        if self.session is None:
            self.session = RoboTwinSession(self.config)
        return self.session

    def capture_views(self, **kwargs) -> ObservationBundle:
        if kwargs.get("setup"):
            self.ensure_session().setup(
                instruction=kwargs.get("instruction"),
                overrides=kwargs.get("setup_overrides") if isinstance(kwargs.get("setup_overrides"), dict) else None,
            )
        raw_observation = kwargs.get("raw_observation")
        if raw_observation is None and self.session is not None:
            raw_observation = self.session.get_obs()
        elif raw_observation is None and self.bound_task_env is not None:
            raw_observation = self.bound_task_env.get_obs()
        if isinstance(raw_observation, dict):
            observation = normalize_robotwin_observation(
                raw_observation,
                task_instruction=kwargs.get("instruction"),
                metadata=self._capture_metadata(kwargs),
                artifacts=self.artifacts,
                artifact_prefix=str(kwargs.get("artifact_prefix", "capture")),
            )
            self.last_observation = observation
            return observation

        raise RuntimeError("robotwin_observation_unavailable:no_raw_observation_or_bound_task_env")

    def _capture_metadata(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        metadata = {
            "backend": "robotwin",
            "repo_root": str(Path(self.config.repo_root)),
            "task_name": self.config.task_name,
            "task_config": self.config.task_config,
            "seed": self.config.seed,
            "now_ep_num": self.config.now_ep_num,
            "enable_depth": self.config.enable_depth,
            "enable_pointcloud": self.config.enable_pointcloud,
            "camera_profile": self.config.camera_profile,
            "planner_image_mode": self.config.planner_image_mode,
            "static_camera_preset": self.config.static_camera_preset,
            "artifact_dir": self.config.artifact_dir,
            "task_env_bound": self.bound_task_env is not None or (self.session is not None and self.session.task_env is not None),
        }
        metadata.update(kwargs)
        return metadata

    def execute_action(self, action_chunk: ActionChunk | None) -> dict[str, Any]:
        if action_chunk is None:
            return _execution_unavailable("missing_action_chunk", action_chunk, self)
        if action_chunk.action_type == "unavailable" or not action_chunk.commands:
            return _execution_unavailable("action_chunk_unavailable", action_chunk, self)
        session = self.ensure_session()
        task_env = session.ensure_task_env()
        executed = []
        with robotwin_cwd(self.config.repo_root):
            for command in action_chunk.commands:
                command_payload = _float_command(command)
                if command_payload is None:
                    return _execution_unavailable("invalid_action_command", action_chunk, self)
                task_env.take_action(command_payload, action_type=action_chunk.action_type)
                executed.append(command_payload)
            raw_observation = task_env.get_obs()
            skip_success_check = bool(
                isinstance(action_chunk.metadata, dict)
                and action_chunk.metadata.get("skip_success_check")
            )
            success = (
                None
                if skip_success_check
                else task_env.check_success() if hasattr(task_env, "check_success") else None
            )
        artifact_prefix = str(action_chunk.metadata.get("artifact_prefix", "execute")) if isinstance(action_chunk.metadata, dict) else "execute"
        if isinstance(raw_observation, dict):
            self.last_observation = normalize_robotwin_observation(
                raw_observation,
                task_instruction=getattr(self.last_observation, "task_instruction", None),
                metadata=self._capture_metadata({"artifact_prefix": artifact_prefix, "after_execute": True}),
                artifacts=self.artifacts,
                artifact_prefix=f"{artifact_prefix}/after",
            )
        return {
            "backend": "robotwin",
            "status": "action_executed",
            "success": success,
            "executed_steps": len(executed),
            "observation": self.last_observation.to_dict() if hasattr(self.last_observation, "to_dict") else None,
            "action_chunk": action_chunk.to_dict() if hasattr(action_chunk, "to_dict") else action_chunk,
            "task_env_bound": task_env is not None,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "robotwin",
            "repo_root": self.config.repo_root,
            "task_name": self.config.task_name,
            "task_config": self.config.task_config,
            "task_env_bound": self.bound_task_env is not None,
        }

    def status(self) -> dict[str, Any]:
        session_task_env = getattr(self.session, "task_env", None) if self.session is not None else None
        live_env_bound = self.bound_task_env is not None or session_task_env is not None
        return {
            "backend": "robotwin",
            "ready": live_env_bound and self.last_observation is not None,
            "needs_setup": not live_env_bound,
            "last_observation_present": self.last_observation is not None,
            "live_env_bound": live_env_bound,
            "task_name": self.config.task_name,
            "task_config": self.config.task_config,
        }

    def preflight_spec(self) -> dict[str, Any]:
        return {
            "backend": "robotwin",
            "required_cameras": ["head_camera", "front_camera", "left_camera", "right_camera"],
            "action_cameras": ["head_camera", "left_camera", "right_camera"],
            "expected_resolution": [960, 540],
            "state": {"required": True, "dim": 14, "source": "joint_action_vector"},
            "action": {"required": True, "types": {"qpos": 14, "ee": 16}},
        }

    def task_status(self) -> dict[str, Any]:
        task_env = None
        if self.session is not None:
            task_env = getattr(self.session, "task_env", None)
        if task_env is None:
            task_env = self.bound_task_env
        success = task_env.check_success() if task_env is not None and hasattr(task_env, "check_success") else None
        return {
            "backend": "robotwin",
            "task_name": self.config.task_name,
            "task_config": self.config.task_config,
            "success": success,
            "done": success if isinstance(success, bool) else None,
            "step_count": None,
        }

    def close(self) -> None:
        task_envs = []
        if self.session is not None and self.session.task_env is not None:
            task_envs.append(self.session.task_env)
        if self.bound_task_env is not None and all(self.bound_task_env is not item for item in task_envs):
            task_envs.append(self.bound_task_env)
        first_error: Exception | None = None
        for task_env in task_envs:
            try:
                close_env = getattr(task_env, "close_env", None)
                if callable(close_env):
                    close_env()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
            finally:
                viewer = getattr(task_env, "viewer", None)
                close_viewer = getattr(viewer, "close", None)
                if callable(close_viewer):
                    try:
                        close_viewer()
                    except Exception as exc:
                        if first_error is None:
                            first_error = exc
        self.session = None
        self.bound_task_env = None
        self.last_observation = None
        if first_error is not None:
            raise first_error


def normalize_robotwin_observation(
    raw_observation: dict[str, Any],
    task_instruction: str | None = None,
    metadata: dict[str, Any] | None = None,
    artifacts: ArtifactStore | None = None,
    artifact_prefix: str = "capture",
) -> ObservationBundle:
    camera_views = _camera_views_from_robotwin(raw_observation, artifacts=artifacts, artifact_prefix=artifact_prefix)
    robot_arms = _robot_arms_from_robotwin(raw_observation)
    raw_ref = artifacts.write_json(f"{artifact_prefix}/raw_observation_summary.json", _raw_summary(raw_observation)) if artifacts else None
    pointcloud_ref = None
    if artifacts is not None and raw_observation.get("pointcloud") is not None:
        pointcloud_ref = artifacts.write_pointcloud(f"{artifact_prefix}/pointcloud/pointcloud.npy", raw_observation["pointcloud"])
    return ObservationBundle(
        task_instruction=task_instruction,
        camera_views=camera_views,
        robot_arms=robot_arms,
        pointcloud_ref=pointcloud_ref,
        raw={"keys": sorted(raw_observation.keys()), "summary_ref": raw_ref},
        metadata=dict(metadata or {}),
    )


def _camera_views_from_robotwin(
    raw_observation: dict[str, Any],
    artifacts: ArtifactStore | None = None,
    artifact_prefix: str = "capture",
) -> dict[str, CameraView]:
    camera_views: dict[str, CameraView] = {}
    cameras = raw_observation.get("observation", {})
    if not isinstance(cameras, dict):
        return camera_views
    for camera_name, payload in cameras.items():
        if not isinstance(payload, dict):
            continue
        intrinsics = _float_list(payload.get("intrinsics", payload.get("intrinsic_cv")))
        extrinsics = _float_list(payload.get("extrinsics", payload.get("extrinsic_cv")))
        rgb_path = None
        depth_path = None
        if artifacts is not None and payload.get("rgb") is not None:
            rgb_path = artifacts.write_image(f"{artifact_prefix}/images/{camera_name}_rgb.png", payload["rgb"])
        if artifacts is not None and payload.get("depth") is not None:
            depth_path = artifacts.write_depth(f"{artifact_prefix}/depth/{camera_name}_depth.npy", payload["depth"])
        camera_views[str(camera_name)] = CameraView(
            name=str(camera_name),
            rgb_path=rgb_path,
            depth_path=depth_path,
            mask_path=None,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            metadata={
                "has_rgb": payload.get("rgb") is not None,
                "has_depth": payload.get("depth") is not None,
                "raw_keys": sorted(str(key) for key in payload.keys()),
                "pointcloud_ref": None,
            },
        )
    return camera_views


def _robot_arms_from_robotwin(raw_observation: dict[str, Any]) -> dict[str, RobotArmState]:
    robot_arms: dict[str, RobotArmState] = {}
    endpose = raw_observation.get("endpose", {})
    joint_action = raw_observation.get("joint_action", {})
    for arm_name in ("left", "right"):
        eef_pose = _float_list(endpose.get(f"{arm_name}_endpose")) if isinstance(endpose, dict) else None
        joint_positions = _float_list(joint_action.get(f"{arm_name}_arm")) if isinstance(joint_action, dict) else None
        gripper_value = None
        if isinstance(endpose, dict):
            gripper_value = _float_scalar(endpose.get(f"{arm_name}_gripper"))
        if gripper_value is None and isinstance(joint_action, dict):
            gripper_value = _float_scalar(joint_action.get(f"{arm_name}_gripper"))
        robot_arms[arm_name] = RobotArmState(
            arm_name=arm_name,
            eef_pose=eef_pose,
            gripper_state=_gripper_state(gripper_value),
            gripper_value=gripper_value,
            joint_positions=joint_positions,
        )
    return robot_arms


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


def _float_scalar(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        flattened = _flatten(value)
        value = flattened[0] if flattened else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gripper_state(value: float | None) -> str | None:
    if value is None:
        return None
    return "open" if value > 0.5 else "closed"


def _float_command(command: Any) -> list[float] | None:
    try:
        values = np.asarray(command, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size == 0:
        return None
    return [float(item) for item in values.tolist()]


def _execution_unavailable(reason: str, action_chunk: ActionChunk | None, adapter: RoboTwinAdapter) -> dict[str, Any]:
    emit_status_notice(
        "execution_unavailable",
        success=False,
        source="robotwin.execute_action",
        reason=reason,
        payload=action_chunk.to_dict() if hasattr(action_chunk, "to_dict") else action_chunk,
    )
    return {
        "backend": "robotwin",
        "status": "execution_unavailable",
        "reason": reason,
        "retryable": False,
        "action_chunk": action_chunk.to_dict() if hasattr(action_chunk, "to_dict") else action_chunk,
        "task_env_bound": adapter.bound_task_env is not None
        or (adapter.session is not None and adapter.session.task_env is not None),
    }


def _raw_summary(raw_observation: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"keys": sorted(str(key) for key in raw_observation.keys())}
    cameras = raw_observation.get("observation")
    if isinstance(cameras, dict):
        summary["cameras"] = {
            str(name): sorted(str(key) for key in payload.keys()) if isinstance(payload, dict) else []
            for name, payload in cameras.items()
        }
    for key in ("endpose", "joint_action"):
        payload = raw_observation.get(key)
        if isinstance(payload, dict):
            summary[key] = sorted(str(item) for item in payload.keys())
            if key == "joint_action":
                vector = _joint_action_vector(payload)
                if vector is not None:
                    summary["joint_action_vector"] = vector
    return summary


def _joint_action_vector(joint_action: dict[str, Any]) -> list[float] | None:
    explicit = _float_list(joint_action.get("vector"))
    if explicit is not None:
        return explicit
    left_arm = _float_list(joint_action.get("left_arm"))
    right_arm = _float_list(joint_action.get("right_arm"))
    if left_arm is None or right_arm is None:
        return None
    left_gripper = _float_scalar(joint_action.get("left_gripper"))
    right_gripper = _float_scalar(joint_action.get("right_gripper"))
    if left_gripper is None or right_gripper is None:
        return None
    return [*left_arm, left_gripper, *right_arm, right_gripper]
