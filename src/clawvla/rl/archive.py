from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from .config import RLConfig, dump_resolved_config
from .trajectory import TrajectoryWriter


def create_run_archive(config: RLConfig, *, mode: str) -> Path:
    run_id = config.resolved_run_id()
    config.run_id = run_id
    run_dir = Path(config.logging.run_root) / run_id
    for child in ["logs", "trajectories", "rewards", "checkpoints", "artifacts", "env"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    dump_resolved_config(config, run_dir / "resolved_config.yaml")
    manifest = {
        "run_id": run_id,
        "mode": mode,
        "created_at": time.time(),
        "command": sys.argv,
        "cwd": os.getcwd(),
        "config": asdict(config),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    project_cwd = Path.cwd()
    _write_command(run_dir / "git_status.txt", ["git", "status", "--short"], cwd=project_cwd)
    _write_command(run_dir / "git_diff.patch", ["git", "diff", "--"], cwd=project_cwd)
    _write_env(run_dir / "env" / "process_env.json")
    TrajectoryWriter(run_dir / "events.jsonl").write_event(
        "clawvla_rl_archive_created",
        {"run_id": run_id, "mode": mode, "run_dir": str(run_dir)},
    )
    return run_dir


def write_preflight_report(run_dir: Path, report: dict[str, Any]) -> None:
    (run_dir / "preflight_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    TrajectoryWriter(run_dir / "events.jsonl").write_event("clawvla_rl_preflight", report)


def _write_command(path: Path, command: list[str], *, cwd: Path) -> None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        path.write_text(result.stdout, encoding="utf-8")
    except Exception as exc:
        path.write_text(f"archive_command_failed {command}: {type(exc).__name__}: {exc}\n", encoding="utf-8")


def _write_env(path: Path) -> None:
    path.write_text(json.dumps(dict(os.environ), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
