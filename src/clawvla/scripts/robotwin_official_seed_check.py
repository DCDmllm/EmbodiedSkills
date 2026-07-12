from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
ROBOTWIN_ROOT = WORKSPACE_ROOT / "RoboTwin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check one RoboTwin official eval seed and emit the episode instruction.")
    parser.add_argument("--repo-root", default=str(ROBOTWIN_ROOT))
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--language-num", type=int, default=100)
    parser.add_argument("--camera-profile", default=None)
    return parser.parse_args()


def main() -> None:
    _ensure_process_env()
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    _ensure_repo(repo_root)
    try:
        payload = _check_seed(args, repo_root)
    except Exception as exc:
        payload = {
            "ok": False,
            "task_name": args.task_name,
            "seed": args.seed,
            "status": "seed_check_exception",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def _ensure_process_env() -> None:
    env_lib = Path(sys.executable).resolve().parent.parent / "lib"
    if env_lib.exists():
        existing = os.environ.get("LD_LIBRARY_PATH")
        prefix = str(env_lib)
        if not existing:
            os.environ["LD_LIBRARY_PATH"] = prefix
        elif prefix not in existing.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{prefix}:{existing}"
    os.environ["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
    os.environ["__EGL_VENDOR_LIBRARY_DIRS"] = "/usr/share/glvnd/egl_vendor.d"


def _check_seed(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    task_env = _instantiate_task(repo_root, args.task_name)
    task_args = _task_args(repo_root, args.task_name, args.task_config, args.camera_profile)
    task_args["eval_mode"] = True
    task_args["render_freq"] = 0
    task_args["eval_video_log"] = False
    task_args["collect_data"] = False
    task_args["save_data"] = False
    task_args["need_plan"] = True
    task_args["language_num"] = int(args.language_num)
    try:
        with _cwd(repo_root):
            task_env.setup_demo(now_ep_num=args.episode_index, seed=args.seed, is_test=True, **task_args)
            episode_info = task_env.play_once()
            plan_success = bool(getattr(task_env, "plan_success", False))
            task_success = bool(task_env.check_success()) if hasattr(task_env, "check_success") else False
            instruction = _episode_instruction(
                repo_root=repo_root,
                task_name=args.task_name,
                episode_info=episode_info,
                instruction_type=args.instruction_type,
                language_num=int(args.language_num),
            )
    finally:
        _close_env(task_env)
    ok = plan_success and task_success
    return {
        "ok": ok,
        "task_name": args.task_name,
        "task_config": args.task_config,
        "seed": args.seed,
        "episode_index": args.episode_index,
        "plan_success": plan_success,
        "task_success": task_success,
        "instruction": instruction if ok else None,
        "status": "valid_seed" if ok else "expert_failed",
    }


def _ensure_repo(repo_root: Path) -> None:
    repo_text = str(repo_root)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    desc_utils = str(repo_root / "description" / "utils")
    if desc_utils not in sys.path:
        sys.path.insert(0, desc_utils)


def _instantiate_task(repo_root: Path, task_name: str) -> Any:
    with _cwd(repo_root):
        module = importlib.import_module(f"envs.{task_name}")
    if not hasattr(module, task_name):
        raise AttributeError(f"RoboTwin env module envs.{task_name} does not define class {task_name}.")
    return getattr(module, task_name)()


def _task_args(repo_root: Path, task_name: str, task_config: str, camera_profile: str | None) -> dict[str, Any]:
    config_path = repo_root / "task_config" / f"{task_config}.yml"
    embodiment_path = repo_root / "task_config" / "_embodiment_config.yml"
    camera_path = repo_root / "task_config" / "_camera_config.yml"
    payload = _load_yaml(config_path)
    payload["task_name"] = task_name
    payload["task_config"] = task_config
    camera_map = _load_yaml(camera_path)
    _apply_embodiment(payload, repo_root, _load_yaml(embodiment_path))
    if camera_profile:
        _apply_camera_profile(payload, camera_map, camera_profile)
    _apply_camera_defaults(payload, camera_map)
    payload["save_path"] = _resolve(repo_root, str(payload.get("save_path", "./data")))
    return payload


def _apply_camera_profile(payload: dict[str, Any], camera_map: dict[str, Any], camera_profile: str) -> None:
    if camera_profile not in camera_map:
        raise ValueError(f"camera profile is not defined in _camera_config.yml: {camera_profile}")
    camera = payload.setdefault("camera", {})
    if isinstance(camera, dict):
        camera["head_camera_type"] = camera_profile
        camera["wrist_camera_type"] = camera_profile
    for config_key in ("left_embodiment_config", "right_embodiment_config"):
        embodiment_config = payload.get(config_key)
        if not isinstance(embodiment_config, dict):
            continue
        static_cameras = embodiment_config.get("static_camera_list")
        if not isinstance(static_cameras, list):
            continue
        for camera_info in static_cameras:
            if isinstance(camera_info, dict) and camera_info.get("name") in {"head_camera", "front_camera"}:
                camera_info["type"] = camera_profile


def _apply_camera_defaults(payload: dict[str, Any], camera_map: dict[str, Any]) -> None:
    camera = payload.get("camera")
    if not isinstance(camera, dict):
        return
    head_type = camera.get("head_camera_type")
    if isinstance(head_type, str) and isinstance(camera_map.get(head_type), dict):
        payload["head_camera_h"] = camera_map[head_type].get("h")
        payload["head_camera_w"] = camera_map[head_type].get("w")


def _apply_embodiment(payload: dict[str, Any], repo_root: Path, embodiment_map: dict[str, Any]) -> None:
    embodiment = payload.get("embodiment")
    if not isinstance(embodiment, list) or not embodiment:
        raise ValueError(f"invalid embodiment config: {embodiment!r}")

    def file_for(name: str) -> str:
        item = embodiment_map.get(name)
        if not isinstance(item, dict) or not item.get("file_path"):
            raise ValueError(f"missing embodiment file path for {name}")
        return _resolve(repo_root, str(item["file_path"]))

    if len(embodiment) == 1:
        payload["left_robot_file"] = file_for(str(embodiment[0]))
        payload["right_robot_file"] = file_for(str(embodiment[0]))
        payload["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        payload["left_robot_file"] = file_for(str(embodiment[0]))
        payload["right_robot_file"] = file_for(str(embodiment[1]))
        payload["embodiment_dis"] = embodiment[2]
        payload["dual_arm_embodied"] = False
    else:
        raise ValueError(f"embodiment should contain 1 or 3 items, got {embodiment!r}")
    payload["left_embodiment_config"] = _load_yaml(Path(payload["left_robot_file"]) / "config.yml")
    payload["right_embodiment_config"] = _load_yaml(Path(payload["right_robot_file"]) / "config.yml")


def _episode_instruction(
    *,
    repo_root: Path,
    task_name: str,
    episode_info: Any,
    instruction_type: str,
    language_num: int,
) -> str | None:
    try:
        from generate_episode_instructions import generate_episode_descriptions
    except Exception:
        return None
    info = episode_info.get("info") if isinstance(episode_info, dict) else None
    if info is None:
        return None
    with _cwd(repo_root / "description"):
        results = generate_episode_descriptions(task_name, [info], language_num)
    choices = results[0].get(instruction_type) if results and isinstance(results[0], dict) else None
    if not choices:
        return None
    return str(np.random.choice(choices))


def _close_env(task_env: Any) -> None:
    try:
        task_env.close_env()
    except Exception:
        pass
    viewer = getattr(task_env, "viewer", None)
    if viewer is not None:
        try:
            viewer.close()
        except Exception:
            pass


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else {}


def _resolve(repo_root: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (repo_root / path).resolve())


class _cwd:
    def __init__(self, path: Path):
        self.path = str(path)
        self.previous = ""

    def __enter__(self) -> None:
        self.previous = os.getcwd()
        os.chdir(self.path)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        os.chdir(self.previous)


if __name__ == "__main__":
    main()
