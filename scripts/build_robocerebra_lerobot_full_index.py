#!/usr/bin/env python3
"""Build an episode index for the full RoboCerebra LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import av
import pandas as pd


DEFAULT_DATASET_DIR = Path("/mnt/raid1/mjh/datasets/robocerebra_lerobot_unified")
DEFAULT_INDEX = Path("outputs/robocerebra_lerobot_full_index.jsonl")
DEFAULT_SUMMARY = Path("outputs/robocerebra_lerobot_full_index_summary.json")
DEFAULT_PROBE = Path("outputs/robocerebra_lerobot_full_index_probe.md")


def run_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc!r}"


def du_h(path: Path) -> str:
    return run_text(["du", "-sh", str(path)]) if path.exists() else "not found"


def task_index_map(tasks_df: pd.DataFrame) -> dict[int, str]:
    return {int(row["task_index"]): " ".join(str(task_text).split()) for task_text, row in tasks_df.iterrows()}


def task_from_episode(value: Any) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return " ".join(str(value).split())


def video_ref(key: str, row: pd.Series) -> str:
    chunk = int(row[f"videos/{key}/chunk_index"])
    file = int(row[f"videos/{key}/file_index"])
    return f"videos/{key}/chunk-{chunk:03d}/file-{file:03d}.mp4"


def load_data(dataset_dir: Path) -> pd.DataFrame:
    paths = sorted((dataset_dir / "data").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No data parquet files found under {dataset_dir / 'data'}")
    frames = [pd.read_parquet(path) for path in paths]
    data = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return data.sort_values("index").reset_index(drop=True)


def decode_first_episode_frame(video_path: Path, timestamp: float) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
        codec = stream.codec_context.name
        target_frame = max(0, int(round(timestamp * fps))) if fps > 0 else 0

        # Seek close to the requested timestamp, then decode the first frame at
        # or after the target frame. Fall back naturally to the first decoded
        # frame if the container lands slightly before/after a keyframe.
        try:
            seek_pts = int(timestamp / float(stream.time_base))
            container.seek(seek_pts, any_frame=False, backward=True, stream=stream)
        except Exception:
            container.seek(0)

        decoded = 0
        selected = None
        for frame in container.decode(stream):
            decoded += 1
            selected = frame
            if frame.pts is not None:
                frame_time = float(frame.pts * stream.time_base)
                if frame_time + 1e-6 >= timestamp:
                    break
            elif decoded >= target_frame + 1:
                break
            if decoded > 600:
                break
        container.close()
        if selected is None:
            raise RuntimeError("no frame decoded")
        arr = selected.to_ndarray(format="rgb24")
        elapsed = time.perf_counter() - start
        return {
            "ok": True,
            "fps": fps,
            "codec": codec,
            "target_timestamp": timestamp,
            "target_frame": target_frame,
            "decoded_frames_after_seek": decoded,
            "shape": list(arr.shape),
            "elapsed_sec": elapsed,
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "target_timestamp": timestamp}


def build_index(dataset_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    info = json.loads((dataset_dir / "meta/info.json").read_text(encoding="utf-8"))
    tasks = pd.read_parquet(dataset_dir / "meta/tasks.parquet")
    episodes = pd.read_parquet(dataset_dir / "meta/episodes/chunk-000/file-000.parquet").sort_values(
        "episode_index"
    )
    data = load_data(dataset_dir)
    task_map = task_index_map(tasks)

    grouped = data.groupby("episode_index").agg(
        count=("index", "count"),
        task_index=("task_index", "first"),
        first_index=("index", "first"),
        last_index=("index", "last"),
        action_dim=("action", lambda s: int(len(s.iloc[0]))),
        state_dim=("observation.state", lambda s: int(len(s.iloc[0]))),
    )

    records: list[dict[str, Any]] = []
    for _, ep in episodes.iterrows():
        episode_index = int(ep["episode_index"])
        agg = grouped.loc[episode_index]
        task_index = int(agg["task_index"])
        frame_start = int(ep["dataset_from_index"])
        frame_end = int(ep["dataset_to_index"])
        num_frames = int(ep["length"])
        front_ref = video_ref("observation.images.image", ep)
        wrist_ref = video_ref("observation.images.wrist_image", ep)
        action_dim = int(agg["action_dim"])
        state_dim = int(agg["state_dim"])
        records.append(
            {
                "episode_index": episode_index,
                "task_index": task_index,
                "task_text": task_map.get(task_index, task_from_episode(ep["tasks"])),
                "dataset_from_index": frame_start,
                "dataset_to_index": frame_end,
                "num_frames": num_frames,
                "front_video_path": str((dataset_dir / front_ref).resolve()),
                "wrist_video_path": str((dataset_dir / wrist_ref).resolve()),
                "front_video_ref": front_ref,
                "wrist_video_ref": wrist_ref,
                "front_video_timestamp_start": float(ep["videos/observation.images.image/from_timestamp"]),
                "wrist_video_timestamp_start": float(ep["videos/observation.images.wrist_image/from_timestamp"]),
                "action_shape": [num_frames, action_dim],
                "state_shape": [num_frames, state_dim],
                "data_count": int(agg["count"]),
                "data_first_index": int(agg["first_index"]),
                "data_last_index": int(agg["last_index"]),
                "aligned": (
                    int(agg["count"]) == num_frames
                    and int(agg["first_index"]) == frame_start
                    and int(agg["last_index"]) == frame_end - 1
                ),
            }
        )
    return records, info, data, episodes


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--index-output", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--probe-output", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--probe-episodes", type=int, default=20)
    parser.add_argument("--probe-seed", type=int, default=42)
    args = parser.parse_args()

    records, info, data, _ = build_index(args.dataset_dir)
    write_jsonl(args.index_output, records)

    rng = random.Random(args.probe_seed)
    sample = rng.sample(records, min(args.probe_episodes, len(records)))
    probe_rows: list[dict[str, Any]] = []
    for record in sample:
        front = decode_first_episode_frame(Path(record["front_video_path"]), record["front_video_timestamp_start"])
        wrist = decode_first_episode_frame(Path(record["wrist_video_path"]), record["wrist_video_timestamp_start"])
        ok = bool(record["aligned"] and front.get("ok") and wrist.get("ok"))
        probe_rows.append({"episode_index": record["episode_index"], "task_text": record["task_text"], "ok": ok, "front": front, "wrist": wrist})

    failed_records = [r for r in records if not r["aligned"]]
    failed_probes = [r for r in probe_rows if not r["ok"]]
    video_refs = sorted({r["front_video_ref"] for r in records} | {r["wrist_video_ref"] for r in records})
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "total_episodes": len(records),
        "total_frames": int(sum(r["num_frames"] for r in records)),
        "info_total_episodes": info.get("total_episodes"),
        "info_total_frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "num_data_rows": int(len(data)),
        "num_video_shards": len(video_refs),
        "num_front_video_shards": len({r["front_video_ref"] for r in records}),
        "num_wrist_video_shards": len({r["wrist_video_ref"] for r in records}),
        "action_shape": info.get("features", {}).get("action", {}).get("shape"),
        "state_shape": info.get("features", {}).get("observation.state", {}).get("shape"),
        "all_episode_ranges_aligned": len(failed_records) == 0,
        "probe_episodes": len(probe_rows),
        "probe_failures": len(failed_probes),
        "index_output": str(args.index_output),
        "probe_output": str(args.probe_output),
        "dataset_disk_usage": du_h(args.dataset_dir),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# RoboCerebra LeRobot Full Index Probe",
        "",
        f"- dataset_dir: `{args.dataset_dir}`",
        f"- total episodes: {summary['total_episodes']}",
        f"- total frames: {summary['total_frames']}",
        f"- data rows: {summary['num_data_rows']}",
        f"- video shards: {summary['num_video_shards']}",
        f"- all episode ranges aligned: {summary['all_episode_ranges_aligned']}",
        f"- probe episodes: {summary['probe_episodes']}",
        f"- probe failures: {summary['probe_failures']}",
        f"- dataset disk usage: `{summary['dataset_disk_usage']}`",
        "",
        "## Decode Probe",
        "",
        "| episode | aligned | front_ok | wrist_ok | front_fps | wrist_fps | front_shape | wrist_shape | task |",
        "|---:|---|---|---|---:|---:|---|---|---|",
    ]
    for row in probe_rows:
        lines.append(
            "| {episode} | {aligned} | {front_ok} | {wrist_ok} | {front_fps} | {wrist_fps} | {front_shape} | {wrist_shape} | {task} |".format(
                episode=row["episode_index"],
                aligned=next(r["aligned"] for r in records if r["episode_index"] == row["episode_index"]),
                front_ok=row["front"].get("ok"),
                wrist_ok=row["wrist"].get("ok"),
                front_fps=row["front"].get("fps", ""),
                wrist_fps=row["wrist"].get("fps", ""),
                front_shape=row["front"].get("shape", row["front"].get("error", "")),
                wrist_shape=row["wrist"].get("shape", row["wrist"].get("error", "")),
                task=row["task_text"].replace("|", "/"),
            )
        )
    lines.extend(["", "## Range Failures", ""])
    if failed_records:
        for record in failed_records[:50]:
            lines.append(
                f"- episode {record['episode_index']}: count={record['data_count']} "
                f"range={record['dataset_from_index']}:{record['dataset_to_index']} "
                f"data_first_last={record['data_first_index']}:{record['data_last_index']}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Decode Failures", ""])
    if failed_probes:
        for row in failed_probes:
            lines.append(f"- episode {row['episode_index']}: front={row['front']} wrist={row['wrist']}")
    else:
        lines.append("- none")
    args.probe_output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {args.index_output}")
    print(f"wrote {args.summary_output}")
    print(f"wrote {args.probe_output}")
    if failed_records or failed_probes:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
