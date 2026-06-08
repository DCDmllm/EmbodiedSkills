from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a ClawVLA command profile.")
    parser.add_argument("--profile", default="/mnt/wangwai/vla/clawvla/configs/run_profiles/qwen3vl_pi05_vllm.json")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--artifact-prefix", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--run", action="store_true", default=None)
    parser.add_argument("--no-run", action="store_false", dest="run")
    parser.add_argument("--initial-observe", action="store_true", default=None)
    parser.add_argument("--no-initial-observe", action="store_true", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Arguments after -- are appended verbatim.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile_path = Path(args.profile)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    command = _build_command(profile, args)
    env = _build_env(profile)
    cwd = _expand(profile.get("cwd") or str(profile_path.parent.parent.parent))
    print(json.dumps({"event": "clawvla_profile_command", "profile": str(profile_path), "command": command}, ensure_ascii=True), file=sys.stderr)
    if args.dry_run:
        print(json.dumps({"command": command, "cwd": cwd, "env_updates": profile.get("env", {})}, indent=2, ensure_ascii=True))
        return
    subprocess.run(command, check=True, cwd=cwd, env=env)


def _build_command(profile: dict[str, Any], cli_args: argparse.Namespace) -> list[str]:
    module = str(profile.get("module") or "clawvla.scripts.run_loop_with_vllm")
    python = _expand(profile.get("python") or sys.executable)
    args_payload = dict(profile.get("args", {})) if isinstance(profile.get("args"), dict) else {}
    repeat_args = dict(profile.get("repeat_args", {})) if isinstance(profile.get("repeat_args"), dict) else {}

    _override(args_payload, "instruction", cli_args.instruction)
    _override(args_payload, "artifact-prefix", cli_args.artifact_prefix)
    _override(args_payload, "max-steps", cli_args.max_steps)
    _override(args_payload, "gpus", cli_args.gpus)
    if cli_args.run is not None:
        args_payload["run"] = bool(cli_args.run)
    if cli_args.initial_observe and not cli_args.no_initial_observe:
        args_payload["initial-observe"] = True
    if cli_args.no_initial_observe:
        args_payload.pop("initial-observe", None)
        args_payload["no-initial-observe"] = True

    command = [python, "-m", module]
    command.extend(_render_args(args_payload))
    command.extend(_render_repeat_args(repeat_args))
    command.extend(_clean_extra_args(cli_args.extra_args))
    return command


def _build_env(profile: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    for key in profile.get("unset_env", []):
        env.pop(str(key), None)
    updates = profile.get("env", {})
    if isinstance(updates, dict):
        for key, value in updates.items():
            env[str(key)] = _expand(value)
    return env


def _render_args(payload: dict[str, Any]) -> list[str]:
    rendered: list[str] = []
    for key, value in payload.items():
        if value is None or value is False:
            continue
        flag = f"--{key}"
        if value is True:
            rendered.append(flag)
        elif isinstance(value, list):
            for item in value:
                rendered.append(f"{flag}={_expand(item)}")
        else:
            rendered.extend([flag, _expand(value)])
    return rendered


def _render_repeat_args(payload: dict[str, Any]) -> list[str]:
    rendered: list[str] = []
    for key, values in payload.items():
        if not isinstance(values, list):
            values = [values]
        for value in values:
            rendered.append(f"--{key}={_expand(value)}")
    return rendered


def _override(payload: dict[str, Any], key: str, value: Any | None) -> None:
    if value is not None:
        payload[key] = value


def _expand(value: Any) -> str:
    return os.path.expandvars(os.path.expanduser(str(value)))


def _clean_extra_args(extra_args: list[str]) -> list[str]:
    if extra_args and extra_args[0] == "--":
        return extra_args[1:]
    return extra_args


if __name__ == "__main__":
    main()
