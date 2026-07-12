from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..config import RobotwinConfig


@dataclass
class RoboTwinSession:
    config: RobotwinConfig
    task_env: Any | None = None

    def ensure_task_env(self) -> Any:
        if self.task_env is None:
            self.task_env = instantiate_task(self.config.task_name, self.config.repo_root)
        return self.task_env

    def setup(self, instruction: str | None = None, overrides: dict[str, Any] | None = None) -> None:
        task_env = self.ensure_task_env()
        args = prepare_task_args(self.config, overrides=overrides)
        with robotwin_cwd(self.config.repo_root):
            task_env.setup_demo(**args)
            if instruction is not None and hasattr(task_env, "set_instruction"):
                task_env.set_instruction(instruction=instruction)

    def get_obs(self) -> dict[str, Any]:
        task_env = self.ensure_task_env()
        with robotwin_cwd(self.config.repo_root):
            observation = task_env.get_obs()
        return observation if isinstance(observation, dict) else {"raw_observation": observation}


def instantiate_task(task_name: str, repo_root: str) -> Any:
    repo_root_path = Path(repo_root).resolve()
    ensure_repo_on_path(repo_root_path)
    with robotwin_cwd(str(repo_root_path)):
        module = importlib.import_module(f"envs.{task_name}")
    if not hasattr(module, task_name):
        raise AttributeError(f"RoboTwin env module envs.{task_name} does not define class {task_name}.")
    return getattr(module, task_name)()


def prepare_task_args(config: RobotwinConfig, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    repo_root = Path(config.repo_root).resolve()
    task_config_path = repo_root / "task_config" / f"{config.task_config}.yml"
    embodiment_config_path = repo_root / "task_config" / "_embodiment_config.yml"
    camera_config_path = repo_root / "task_config" / "_camera_config.yml"

    args = load_yaml(task_config_path)
    args["task_name"] = config.task_name
    args["task_config"] = config.task_config
    args["seed"] = config.seed
    args["now_ep_num"] = config.now_ep_num
    args["collect_data"] = False
    args["save_data"] = False
    args["eval_video_log"] = False
    args["is_test"] = bool(config.is_test)
    args["eval_mode"] = bool(config.eval_mode)
    if config.render_freq is not None:
        args["render_freq"] = int(config.render_freq)
    args["need_plan"] = bool(config.need_plan) if config.need_plan is not None else False

    camera_map = load_yaml(camera_config_path)
    apply_embodiment_files(args, repo_root, load_yaml(embodiment_config_path))
    apply_camera_profile(args, camera_map, config.camera_profile)
    apply_camera_defaults(args, camera_map)
    args["save_path"] = resolve_repo_relative(repo_root, str(args.get("save_path", "./data")))

    if overrides:
        args = deep_update(args, overrides)
    return args


def apply_embodiment_files(args: dict[str, Any], repo_root: Path, embodiment_map: dict[str, Any]) -> None:
    embodiment = args.get("embodiment")
    if not isinstance(embodiment, list) or not embodiment:
        raise ValueError(f"Invalid RoboTwin embodiment config: {embodiment!r}")

    def file_for(name: str) -> str:
        payload = embodiment_map.get(name)
        if not isinstance(payload, dict) or not payload.get("file_path"):
            raise ValueError(f"Missing embodiment file path for {name}.")
        return str(Path(resolve_repo_relative(repo_root, str(payload["file_path"])) or "").resolve())

    if len(embodiment) == 1:
        args["left_robot_file"] = file_for(str(embodiment[0]))
        args["right_robot_file"] = file_for(str(embodiment[0]))
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        args["left_robot_file"] = file_for(str(embodiment[0]))
        args["right_robot_file"] = file_for(str(embodiment[1]))
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError(f"RoboTwin embodiment should contain 1 or 3 items, got {embodiment!r}.")

    args["left_embodiment_config"] = load_yaml(Path(args["left_robot_file"]) / "config.yml")
    args["right_embodiment_config"] = load_yaml(Path(args["right_robot_file"]) / "config.yml")


def apply_camera_defaults(args: dict[str, Any], camera_map: dict[str, Any]) -> None:
    camera = args.get("camera")
    if not isinstance(camera, dict):
        return
    head_camera_type = camera.get("head_camera_type")
    if isinstance(head_camera_type, str) and isinstance(camera_map.get(head_camera_type), dict):
        args["head_camera_h"] = camera_map[head_camera_type].get("h")
        args["head_camera_w"] = camera_map[head_camera_type].get("w")


def apply_camera_profile(args: dict[str, Any], camera_map: dict[str, Any], camera_profile: str | None) -> None:
    if not camera_profile:
        return
    if camera_profile not in camera_map:
        raise ValueError(f"RoboTwin camera_profile is not defined in _camera_config.yml: {camera_profile}")
    camera = args.setdefault("camera", {})
    if not isinstance(camera, dict):
        raise ValueError(f"RoboTwin task camera config must be a dict, got {type(camera).__name__}")
    camera["head_camera_type"] = camera_profile
    camera["wrist_camera_type"] = camera_profile
    for config_key in ("left_embodiment_config", "right_embodiment_config"):
        embodiment_config = args.get(config_key)
        if not isinstance(embodiment_config, dict):
            continue
        static_cameras = embodiment_config.get("static_camera_list")
        if not isinstance(static_cameras, list):
            continue
        for camera_info in static_cameras:
            if isinstance(camera_info, dict) and camera_info.get("name") in {"head_camera", "front_camera"}:
                camera_info["type"] = camera_profile


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else {}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_repo_relative(repo_root: Path, value: str | None) -> str | None:
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((repo_root / value).resolve())


def ensure_repo_on_path(repo_root: Path) -> None:
    repo_root_text = str(repo_root.resolve())
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)


@contextmanager
def robotwin_cwd(repo_root: str):
    previous = os.getcwd()
    os.chdir(repo_root)
    try:
        yield
    finally:
        os.chdir(previous)
