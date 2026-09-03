#!/usr/bin/env python3
"""Export LeRobot RoboCerebra subgoal episodes for VLA training."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ID = "lerobot/robocerebra_unified"
RAW_REPO_ID = "qiukingballball/RoboCerebra"
DEFAULT_HF_BASE_URL = "https://hf-mirror.com"
STEP_RE = re.compile(
    r"Step:\s*(?P<instruction>.*?)\s*"
    r"\[(?P<start>\d+)\s*,\s*(?P<end>\d+)\]"
    r"(?:\s*Related Objects:\s*(?P<objects>.*?))?"
    r"(?=\s*Step:|\s*$)",
    flags=re.DOTALL,
)


def clean_text(value: Any) -> str:
    return " ".join(str(value).replace("\n", " ").split()).strip(" .")


def download_file(repo_id: str, filename: str, target: Path, base_url: str, timeout: int = 120) -> Path:
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    quoted = urllib.parse.quote(filename, safe="/")
    url = f"{base_url.rstrip('/')}/datasets/{repo_id}/resolve/main/{quoted}"
    with urllib.request.urlopen(url, timeout=timeout) as response, target.open("wb") as f:
        shutil.copyfileobj(response, f)
    return target


def ensure_lerobot_files(root: Path, base_url: str, include_data: bool = True) -> dict[str, Path]:
    files = {
        "info": root / "meta/info.json",
        "tasks": root / "meta/tasks.parquet",
        "episodes": root / "meta/episodes/chunk-000/file-000.parquet",
    }
    if include_data:
        files["data"] = root / "data/chunk-000/file-000.parquet"
    remote = {
        "info": "meta/info.json",
        "tasks": "meta/tasks.parquet",
        "episodes": "meta/episodes/chunk-000/file-000.parquet",
        "data": "data/chunk-000/file-000.parquet",
    }
    for key, path in files.items():
        download_file(REPO_ID, remote[key], path, base_url)
    return files


def task_index_map(tasks_df: pd.DataFrame) -> dict[int, str]:
    return {int(row["task_index"]): str(task_text) for task_text, row in tasks_df.iterrows()}


def parse_episode_task(value: Any) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and value:
        return clean_text(value[0])
    return clean_text(value)


def load_episode_records(files: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    info = json.loads(files["info"].read_text(encoding="utf-8"))
    tasks_df = pd.read_parquet(files["tasks"])
    episodes = pd.read_parquet(files["episodes"]).sort_values("episode_index").reset_index(drop=True)
    data = pd.read_parquet(files["data"])
    idx_to_task = task_index_map(tasks_df)

    first_by_episode = data.groupby("episode_index").agg(
        task_index=("task_index", "first"),
        action_dim=("action", lambda s: int(len(s.iloc[0]))),
        state_dim=("observation.state", lambda s: int(len(s.iloc[0]))),
    )
    records: list[dict[str, Any]] = []
    for _, ep in episodes.iterrows():
        episode_index = int(ep["episode_index"])
        task_index = int(first_by_episode.loc[episode_index, "task_index"])
        task_text = idx_to_task.get(task_index, parse_episode_task(ep["tasks"]))
        image_chunk = int(ep["videos/observation.images.image/chunk_index"])
        image_file = int(ep["videos/observation.images.image/file_index"])
        wrist_chunk = int(ep["videos/observation.images.wrist_image/chunk_index"])
        wrist_file = int(ep["videos/observation.images.wrist_image/file_index"])
        records.append(
            {
                "benchmark": "RoboCerebra-LeRobot",
                "episode_index": episode_index,
                "task_index": task_index,
                "subgoal_instruction": task_text,
                "dataset_from_index": int(ep["dataset_from_index"]),
                "dataset_to_index": int(ep["dataset_to_index"]),
                "num_frames": int(ep["length"]),
                "action_dim": int(first_by_episode.loc[episode_index, "action_dim"]),
                "state_dim": int(first_by_episode.loc[episode_index, "state_dim"]),
                "image_key": "observation.images.image",
                "wrist_image_key": "observation.images.wrist_image",
                "image_video_ref": (
                    f"videos/observation.images.image/chunk-{image_chunk:03d}/file-{image_file:03d}.mp4"
                ),
                "wrist_image_video_ref": (
                    f"videos/observation.images.wrist_image/chunk-{wrist_chunk:03d}/file-{wrist_file:03d}.mp4"
                ),
                "image_timestamp_start": float(ep["videos/observation.images.image/from_timestamp"]),
                "image_timestamp_end": float(ep["videos/observation.images.image/to_timestamp"]),
                "wrist_image_timestamp_start": float(ep["videos/observation.images.wrist_image/from_timestamp"]),
                "wrist_image_timestamp_end": float(ep["videos/observation.images.wrist_image/to_timestamp"]),
                "source": REPO_ID,
                "boundary_status": "already_segmented_episode",
            }
        )
    return records, info, data


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_small_sample(
    out_dir: Path,
    records: list[dict[str, Any]],
    data: pd.DataFrame,
    max_episodes: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for record in records[:max_episodes]:
        episode_index = record["episode_index"]
        ep_dir = out_dir / f"episode_{episode_index:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        ep = data[data["episode_index"] == episode_index].sort_values("frame_index")
        actions = np.stack(ep["action"].to_numpy())
        states = np.stack(ep["observation.state"].to_numpy())
        np.save(ep_dir / "actions.npy", actions)
        np.save(ep_dir / "states.npy", states)
        meta = {
            **record,
            "actions_file": "actions.npy",
            "states_file": "states.npy",
            "frame_index_start": int(ep["frame_index"].min()),
            "frame_index_end": int(ep["frame_index"].max()),
            "image_storage": "video_reference_not_downloaded",
        }
        (ep_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_raw_planning_rows(raw_path: Path) -> list[dict[str, Any]]:
    df = pd.read_parquet(raw_path)
    rows: list[dict[str, Any]] = []
    for raw_row_index, row in df.iterrows():
        if row.get("id") is not None:
            episode_id = str(row.get("id"))
            demo_path = row.get("demo_path")
        else:
            episode_id = f"{row.get('scene')}_{row.get('case')}"
            demo_path = row.get("demo")
        subgoals = []
        for subgoal_index, match in enumerate(STEP_RE.finditer(str(row.get("task_description", "")))):
            subgoals.append(
                {
                    "subgoal_index": subgoal_index,
                    "subgoal_instruction": clean_text(match.group("instruction")),
                    "frame_start": int(match.group("start")),
                    "frame_end": int(match.group("end")),
                    "related_objects": clean_text(match.group("objects")) if match.group("objects") else None,
                }
            )
        rows.append(
            {
                "benchmark": "RoboCerebra",
                "episode_id": episode_id,
                "raw_row_index": int(raw_row_index),
                "full_task_instruction": row.get("high_level_instruction"),
                "subgoals": subgoals,
                "num_subgoals": len(subgoals),
                "raw_demo_path": demo_path,
                "source": str(raw_path),
                "boundary_status": "native_task_description",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/robocerebra_metadata/lerobot"))
    parser.add_argument("--hf-base-url", default=DEFAULT_HF_BASE_URL)
    parser.add_argument("--output", type=Path, default=Path("outputs/robocerebra_lerobot_vla_samples.jsonl"))
    parser.add_argument("--episode-probe", type=Path, default=Path("outputs/robocerebra_lerobot_episode_probe.md"))
    parser.add_argument("--small-sample-dir", type=Path, default=Path("outputs/robocerebra_lerobot_small_sample"))
    parser.add_argument("--small-sample-episodes", type=int, default=3)
    parser.add_argument("--planning-output", type=Path, default=Path("outputs/robocerebra_planning_samples.jsonl"))
    parser.add_argument("--raw-planning-path", type=Path, default=Path("outputs/robocerebra_metadata/raw/test.parquet"))
    parser.add_argument("--max-planning-rows", type=int, default=20)
    args = parser.parse_args()

    files = ensure_lerobot_files(args.cache_root, args.hf_base_url, include_data=True)
    records, info, data = load_episode_records(files)
    write_jsonl(args.output, records)
    export_small_sample(args.small_sample_dir, records, data, args.small_sample_episodes)

    planning_rows = parse_raw_planning_rows(args.raw_planning_path)[: args.max_planning_rows]
    write_jsonl(args.planning_output, planning_rows)

    loader_status = "unavailable: Python package `lerobot` is not installed in this environment"
    try:
        import lerobot  # noqa: F401

        loader_status = "available"
    except Exception:
        pass

    lines = [
        "# RoboCerebra LeRobot Episode Probe",
        "",
        f"- source: `{REPO_ID}`",
        f"- official LeRobot loader: {loader_status}",
        f"- total episodes: {len(records)}",
        f"- total frames from info.json: {info.get('total_frames')}",
        f"- fps: {info.get('fps')}",
        f"- image field metadata readable: {'observation.images.image' in info.get('features', {})}",
        f"- wrist image field metadata readable: {'observation.images.wrist_image' in info.get('features', {})}",
        "- image pixels not downloaded; exported as video references plus timestamp ranges",
        "",
        "## First 20 Episodes",
        "",
    ]
    for record in records[:20]:
        lines.append(
            f"- episode_index={record['episode_index']} task_index={record['task_index']} "
            f"frames={record['num_frames']} range={record['dataset_from_index']}:{record['dataset_to_index']} "
            f"action_shape=({record['num_frames']}, {record['action_dim']}) "
            f"state_shape=({record['num_frames']}, {record['state_dim']}) "
            f"image_ref={record['image_video_ref']} "
            f"wrist_ref={record['wrist_image_video_ref']} "
            f"task=`{record['subgoal_instruction']}`"
        )
    args.episode_probe.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {args.episode_probe}")
    print(f"wrote small sample under {args.small_sample_dir}")
    print(f"wrote {args.planning_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
