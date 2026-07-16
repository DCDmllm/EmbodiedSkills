#!/usr/bin/env python3
"""Export a tiny visual VLA sample from RoboCerebra LeRobot shards."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
from PIL import Image


REPO_ID = "lerobot/robocerebra_unified"
DEFAULT_HF_BASE_URL = "https://hf-mirror.com"


def download_file(repo_id: str, filename: str, target: Path, base_url: str, timeout: int = 300) -> Path:
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    quoted = urllib.parse.quote(filename, safe="/")
    url = f"{base_url.rstrip('/')}/datasets/{repo_id}/resolve/main/{quoted}"
    with urllib.request.urlopen(url, timeout=timeout) as response, target.open("wb") as f:
        shutil.copyfileobj(response, f)
    return target


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def decode_frame_range(video_path: Path, start: int, end: int) -> tuple[list[np.ndarray], float]:
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
    frames: list[np.ndarray] = []
    for frame_index, frame in enumerate(container.decode(video=0)):
        if frame_index < start:
            continue
        if frame_index >= end:
            break
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return frames, fps


def save_frames(frames: list[np.ndarray], out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame_rgb in enumerate(frames):
        Image.fromarray(frame_rgb).save(out_dir / f"{i:06d}.png")


def disk_usage(path: Path) -> str:
    try:
        return subprocess.check_output(["du", "-sh", str(path)], text=True).splitlines()[0]
    except Exception as exc:
        return f"unavailable: {exc!r}"


def df_h(path: Path) -> str:
    try:
        return subprocess.check_output(["df", "-h", str(path)], text=True)
    except Exception as exc:
        return f"unavailable: {exc!r}"


def save_video_segments(
    video_path: Path,
    segments: list[dict[str, Any]],
) -> tuple[float, int, str]:
    """Decode one video once and save requested inclusive/exclusive segments.

    The LeRobot metadata used here stores frame ranges as global frame indices
    inside each video shard for the first shard. This function keeps that
    existing indexing behavior but avoids repeatedly decoding from frame 0 for
    every episode.
    """

    if not segments:
        return 0.0, 0, "unknown"

    frame_targets: dict[int, list[tuple[Path, int]]] = defaultdict(list)
    max_end = 0
    for segment in segments:
        start = int(segment["start"])
        end = int(segment["end"])
        out_dir = Path(segment["out_dir"])
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        max_end = max(max_end, end)
        for global_i in range(start, end):
            frame_targets[global_i].append((out_dir, global_i - start))

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
    codec = stream.codec_context.name
    saved = 0
    for frame_index, frame in enumerate(container.decode(video=0)):
        if frame_index >= max_end:
            break
        targets = frame_targets.get(frame_index)
        if not targets:
            continue
        frame_rgb = frame.to_ndarray(format="rgb24")
        image = Image.fromarray(frame_rgb)
        for out_dir, local_i in targets:
            image.save(out_dir / f"{local_i:06d}.png")
            saved += 1
    container.close()
    return fps, saved, codec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path("outputs/robocerebra_lerobot_vla_samples.jsonl"))
    parser.add_argument("--data-parquet", type=Path, default=Path("outputs/robocerebra_metadata/lerobot/data/chunk-000/file-000.parquet"))
    parser.add_argument("--video-cache", type=Path, default=Path("outputs/robocerebra_metadata/lerobot/videos"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/robocerebra_lerobot_small_visual_sample"))
    parser.add_argument("--probe", type=Path, default=Path("outputs/robocerebra_visual_alignment_probe.md"))
    parser.add_argument("--hf-base-url", default=DEFAULT_HF_BASE_URL)
    parser.add_argument("--num-episodes", type=int, default=5)
    args = parser.parse_args()

    samples = load_jsonl(args.samples)[: args.num_episodes]
    data = pd.read_parquet(args.data_parquet)

    needed_videos = sorted({r["image_video_ref"] for r in samples} | {r["wrist_image_video_ref"] for r in samples})
    local_video_paths: dict[str, Path] = {}
    for video_ref in needed_videos:
        local_path = args.video_cache / video_ref
        download_file(REPO_ID, video_ref, local_path, args.hf_base_url)
        local_video_paths[video_ref] = local_path

    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    video_segments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in samples:
        episode_index = int(record["episode_index"])
        start = int(record["dataset_from_index"])
        end = int(record["dataset_to_index"])
        ep_dir = args.output_dir / f"episode_{episode_index:06d}"
        image_dir = ep_dir / "images"
        wrist_dir = ep_dir / "wrist_images"
        ep_dir.mkdir(parents=True, exist_ok=True)

        try:
            ep = data.iloc[start:end].copy()
            ep = ep.sort_values("frame_index")
            actions = np.stack(ep["action"].to_numpy())
            states = np.stack(ep["observation.state"].to_numpy())
            np.save(ep_dir / "actions.npy", actions)
            np.save(ep_dir / "states.npy", states)
            video_segments[record["image_video_ref"]].append({"start": start, "end": end, "out_dir": image_dir})
            video_segments[record["wrist_image_video_ref"]].append({"start": start, "end": end, "out_dir": wrist_dir})
        except Exception as exc:
            failures.append({"episode_index": episode_index, "reason": repr(exc)})
            continue

    video_decode_info: dict[str, dict[str, Any]] = {}
    for video_ref, segments in video_segments.items():
        fps, saved, codec = save_video_segments(local_video_paths[video_ref], segments)
        video_decode_info[video_ref] = {"fps": fps, "saved_frames": saved, "codec": codec}

    for record in samples:
        episode_index = int(record["episode_index"])
        if any(f["episode_index"] == episode_index for f in failures):
            continue
        ep_dir = args.output_dir / f"episode_{episode_index:06d}"
        image_dir = ep_dir / "images"
        wrist_dir = ep_dir / "wrist_images"
        actions = np.load(ep_dir / "actions.npy")
        states = np.load(ep_dir / "states.npy")
        image_frames = sorted(image_dir.glob("*.png"))
        wrist_frames = sorted(wrist_dir.glob("*.png"))
        image_fps = float(video_decode_info[record["image_video_ref"]]["fps"])
        wrist_fps = float(video_decode_info[record["wrist_image_video_ref"]]["fps"])

        aligned = (
            len(image_frames)
            == len(wrist_frames)
            == len(actions)
            == len(states)
            == int(record["num_frames"])
        )
        fps_ok = round(image_fps) == 20 and round(wrist_fps) == 20
        meta = {
            **record,
            "actions_file": "actions.npy",
            "states_file": "states.npy",
            "images_dir": "images",
            "wrist_images_dir": "wrist_images",
            "decoded_image_count": len(image_frames),
            "decoded_wrist_image_count": len(wrist_frames),
            "image_fps": image_fps,
            "wrist_image_fps": wrist_fps,
            "image_codec": video_decode_info[record["image_video_ref"]]["codec"],
            "wrist_image_codec": video_decode_info[record["wrist_image_video_ref"]]["codec"],
            "aligned": aligned,
            "fps_ok": fps_ok,
        }
        (ep_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        probe_rows.append(meta)

    lines = [
        "# RoboCerebra Visual Alignment Probe",
        "",
        f"- source: `{REPO_ID}`",
        f"- episodes exported: {len(probe_rows)}",
        f"- requested episodes: {len(samples)}",
        f"- total frames exported: {sum(int(r['num_frames']) for r in probe_rows)}",
        f"- video shards downloaded: {len(needed_videos)}",
        f"- downloaded/used video shards: {', '.join(f'`{v}`' for v in needed_videos)}",
        f"- output_dir: `{args.output_dir}`",
        f"- output disk usage: `{disk_usage(args.output_dir)}`",
        f"- df -h output_dir filesystem:",
        "",
        "```",
        df_h(args.output_dir).strip(),
        "```",
        "",
        "## Video Decode",
        "",
        "| video_ref | codec | fps | saved_frames | local_path |",
        "|---|---|---:|---:|---|",
    ]
    for video_ref in needed_videos:
        info = video_decode_info.get(video_ref, {})
        lines.append(
            f"| `{video_ref}` | {info.get('codec', 'unknown')} | {float(info.get('fps', 0.0)):.3f} | "
            f"{int(info.get('saved_frames', 0))} | `{local_video_paths[video_ref]}` |"
        )
    lines.extend(
        [
            "",
            "## Episode Checks",
            "",
            "| episode | task | num_frames | action_shape | state_shape | images | wrist_images | fps | aligned | notes |",
            "|---:|---|---:|---|---|---:|---:|---|---|---|",
        ]
    )
    for row in probe_rows:
        notes = []
        if not row["aligned"]:
            notes.append("count mismatch")
        if not row["fps_ok"]:
            notes.append("fps mismatch")
        notes_text = "; ".join(notes) if notes else "ok"
        lines.append(
            "| {episode_index} | {task} | {num_frames} | ({num_frames}, {action_dim}) | "
            "({num_frames}, {state_dim}) | {images} | {wrist} | {fps:.3g}/{wfps:.3g} | {aligned} | {notes} |".format(
                episode_index=row["episode_index"],
                task=row["subgoal_instruction"].replace("|", "/"),
                num_frames=row["num_frames"],
                action_dim=row["action_dim"],
                state_dim=row["state_dim"],
                images=row["decoded_image_count"],
                wrist=row["decoded_wrist_image_count"],
                fps=row["image_fps"],
                wfps=row["wrist_image_fps"],
                aligned=row["aligned"] and row["fps_ok"],
                notes=notes_text,
            )
        )
    lines.extend(["", "## Failures / Skips", ""])
    if failures:
        for failure in failures:
            lines.append(f"- episode {failure['episode_index']}: {failure['reason']}")
    else:
        lines.append("- none")
    args.probe.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote visual sample: {args.output_dir}")
    print(f"wrote probe: {args.probe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
