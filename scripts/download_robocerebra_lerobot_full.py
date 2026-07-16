#!/usr/bin/env python3
"""Download the full LeRobot RoboCerebra VLA dataset.

This downloads only the LeRobot-format files needed for VLA training:
metadata, data parquet shards, and front/wrist mp4 video shards. It does not
download the raw RoboCerebra HDF5 archive and does not decode videos to PNG.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ID = "lerobot/robocerebra_unified"
DEFAULT_HF_BASE_URL = "https://hf-mirror.com"
DEFAULT_OUTPUT_DIR = Path("/mnt/raid1/mjh/datasets/robocerebra_lerobot_unified")
DEFAULT_PROBE = Path("outputs/robocerebra_full_download_probe.md")
KNOWN_METADATA = [
    "meta/info.json",
    "meta/tasks.parquet",
    "meta/episodes/chunk-000/file-000.parquet",
]


def run_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc!r}"


def df_h(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    return run_text(["df", "-h", str(path)])


def du_h(path: Path) -> str:
    if not path.exists():
        return "not found"
    return run_text(["du", "-sh", str(path)])


def hf_url(base_url: str, repo_id: str, filename: str) -> str:
    quoted = urllib.parse.quote(filename, safe="/")
    return f"{base_url.rstrip('/')}/datasets/{repo_id}/resolve/main/{quoted}"


def download_file(
    filename: str,
    target_root: Path,
    *,
    base_url: str,
    timeout: int,
    force: bool = False,
) -> dict[str, Any]:
    target = target_root / filename
    partial = target.with_suffix(target.suffix + ".part")
    url = hf_url(base_url, REPO_ID, filename)

    if target.exists() and target.stat().st_size > 0 and not force:
        return {"file": filename, "status": "skipped_exists", "path": str(target), "bytes": target.stat().st_size}

    target.parent.mkdir(parents=True, exist_ok=True)
    resume_at = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if resume_at > 0:
        request.add_header("Range", f"bytes={resume_at}-")

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            mode = "ab" if resume_at > 0 and response.status == 206 else "wb"
            if mode == "wb":
                resume_at = 0
            with partial.open(mode) as f:
                shutil.copyfileobj(response, f)
        partial.replace(target)
        elapsed = time.perf_counter() - start
        return {
            "file": filename,
            "status": "downloaded",
            "path": str(target),
            "bytes": target.stat().st_size,
            "elapsed_sec": elapsed,
            "resumed_from_bytes": resume_at,
        }
    except Exception as exc:
        return {
            "file": filename,
            "status": "failed",
            "path": str(target),
            "partial_path": str(partial),
            "partial_bytes": partial.stat().st_size if partial.exists() else 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def task_text(value: Any) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return " ".join(str(value).split())


def derive_manifest(root: Path) -> list[str]:
    episodes_path = root / "meta/episodes/chunk-000/file-000.parquet"
    if not episodes_path.exists():
        return KNOWN_METADATA.copy()

    episodes = pd.read_parquet(episodes_path)
    files = set(KNOWN_METADATA)

    for _, row in episodes[["data/chunk_index", "data/file_index"]].drop_duplicates().iterrows():
        files.add(f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet")

    for key in ["observation.images.image", "observation.images.wrist_image"]:
        chunk_col = f"videos/{key}/chunk_index"
        file_col = f"videos/{key}/file_index"
        for _, row in episodes[[chunk_col, file_col]].drop_duplicates().iterrows():
            files.add(f"videos/{key}/chunk-{int(row[chunk_col]):03d}/file-{int(row[file_col]):03d}.mp4")

    return sorted(files)


def write_probe(
    path: Path,
    *,
    target_dir: Path,
    hf_base_url: str,
    before_df: str,
    after_df: str,
    results: list[dict[str, Any]],
    manifest: list[str],
    stopped_reason: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = [r for r in results if r["status"] == "downloaded"]
    skipped = [r for r in results if r["status"] == "skipped_exists"]
    failed = [r for r in results if r["status"] == "failed"]
    video_files = [f for f in manifest if f.startswith("videos/")]
    data_files = [f for f in manifest if f.startswith("data/")]
    meta_files = [f for f in manifest if f.startswith("meta/")]

    lines = [
        "# RoboCerebra LeRobot Full Download Probe",
        "",
        f"- source: `{REPO_ID}`",
        f"- target_dir: `{target_dir}`",
        f"- hf_base_url: `{hf_base_url}`",
        f"- manifest_files: {len(manifest)}",
        f"- metadata_files: {len(meta_files)}",
        f"- data_parquet_files: {len(data_files)}",
        f"- video_files: {len(video_files)}",
        f"- downloaded: {len(downloaded)}",
        f"- skipped_existing: {len(skipped)}",
        f"- failed: {len(failed)}",
        f"- stopped_reason: {stopped_reason or 'none'}",
        f"- target disk usage: `{du_h(target_dir)}`",
        "",
        "## df -h Before",
        "",
        "```",
        before_df,
        "```",
        "",
        "## df -h After",
        "",
        "```",
        after_df,
        "```",
        "",
        "## Video Shards",
        "",
    ]
    for file in video_files:
        result = next((r for r in results if r["file"] == file), None)
        status = result["status"] if result else "not_attempted"
        size = result.get("bytes", 0) if result else 0
        lines.append(f"- `{file}`: {status}, {size} bytes")

    lines.extend(["", "## Failures", ""])
    if failed:
        for result in failed:
            lines.append(f"- `{result['file']}`: {result.get('error')} partial_bytes={result.get('partial_bytes')}")
    else:
        lines.append("- none")

    lines.extend(["", "## Manifest", ""])
    for file in manifest:
        lines.append(f"- `{file}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--hf-base-url", default=DEFAULT_HF_BASE_URL)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.target_dir.mkdir(parents=True, exist_ok=True)
    before_df = df_h(args.target_dir)
    results: list[dict[str, Any]] = []
    stopped_reason: str | None = None

    # Stage 1: metadata required to derive the exact LeRobot shard list.
    for filename in KNOWN_METADATA:
        result = download_file(filename, args.target_dir, base_url=args.hf_base_url, timeout=args.timeout, force=args.force)
        results.append(result)
        if result["status"] == "failed":
            stopped_reason = f"metadata download failed: {filename}"
            manifest = KNOWN_METADATA.copy()
            after_df = df_h(args.target_dir)
            write_probe(
                args.probe,
                target_dir=args.target_dir,
                hf_base_url=args.hf_base_url,
                before_df=before_df,
                after_df=after_df,
                results=results,
                manifest=manifest,
                stopped_reason=stopped_reason,
            )
            print(stopped_reason)
            return 1

    manifest = derive_manifest(args.target_dir)
    attempted = {r["file"] for r in results}
    for filename in manifest:
        if filename in attempted:
            continue
        result = download_file(filename, args.target_dir, base_url=args.hf_base_url, timeout=args.timeout, force=args.force)
        results.append(result)
        if result["status"] == "failed":
            stopped_reason = f"download failed: {filename}"
            break

    after_df = df_h(args.target_dir)
    write_probe(
        args.probe,
        target_dir=args.target_dir,
        hf_base_url=args.hf_base_url,
        before_df=before_df,
        after_df=after_df,
        results=results,
        manifest=manifest,
        stopped_reason=stopped_reason,
    )
    print(f"wrote {args.probe}")
    if stopped_reason:
        print(stopped_reason)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
