#!/usr/bin/env python3
"""Inspect RoboCerebra raw and LeRobot metadata without downloading full data."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


RAW_REPO = "qiukingballball/RoboCerebra"
LEROBOT_REPO = "lerobot/robocerebra_unified"
DEFAULT_HF_BASE_URL = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")

RAW_MANIFEST_CANDIDATES = {
    "train": ["train.parquet", "RoboCerebra_trainset/trainingset.parquet"],
    "test": ["test.parquet"],
}

LEROBOT_FILES = [
    "meta/info.json",
    "meta/tasks.parquet",
    "meta/episodes/chunk-000/file-000.parquet",
    "data/chunk-000/file-000.parquet",
]
DOWNLOAD_TIMEOUT_SECONDS = float(os.environ.get("ROBOCEREBRA_DOWNLOAD_TIMEOUT", "8"))


def download_hf_file(repo_id: str, filename: str, cache_dir: Path, base_url: str) -> Path | None:
    """Download one known HF dataset file, returning None on network/path failure."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    quoted = urllib.parse.quote(filename, safe="/")
    url = f"{base_url}/datasets/{repo_id}/resolve/main/{quoted}"
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, target.open("wb") as f:
            shutil.copyfileobj(response, f)
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        if target.exists() and target.stat().st_size == 0:
            target.unlink()
        print(f"DOWNLOAD_FAILED repo={repo_id} file={filename}: {type(exc).__name__}: {exc}", flush=True)
        return None
    return target


def read_parquet(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        print(f"READ_PARQUET_FAILED path={path}: {type(exc).__name__}: {exc}")
        return None


def summarize_value(value: Any, max_chars: int = 260) -> Any:
    if hasattr(value, "tolist"):
        return summarize_value(value.tolist(), max_chars=max_chars)
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "..."
    if isinstance(value, (list, tuple)):
        return [summarize_value(v, max_chars=120) for v in list(value)[:4]]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def inspect_frame(name: str, df: pd.DataFrame, sample_rows: int) -> None:
    print(f"\n## {name}")
    print("rows:", len(df))
    print("columns:", list(df.columns))
    print("index_name:", df.index.name)
    print("index_sample:", [summarize_value(v, max_chars=160) for v in list(df.index[:sample_rows])])
    print("dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"  - {col}: {dtype}")
    print("samples:")
    for idx, row in df.head(sample_rows).iterrows():
        payload = {col: summarize_value(row[col]) for col in df.columns}
        print(f"  [{idx}] {json.dumps(payload, ensure_ascii=False)}")


def inspect_json(name: str, path: Path) -> None:
    print(f"\n## {name}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for key in [
        "codebase_version",
        "robot_type",
        "total_episodes",
        "total_frames",
        "total_tasks",
        "fps",
        "splits",
        "data_path",
        "video_path",
    ]:
        if key in data:
            print(f"{key}: {data[key]}")
    features = data.get("features", {})
    print("features:")
    for key, spec in features.items():
        print(f"  - {key}: dtype={spec.get('dtype')} shape={spec.get('shape')}")


def find_local_or_download(
    repo: str,
    filename: str,
    cache_dir: Path,
    skip_download: bool,
    base_url: str,
) -> Path | None:
    local = cache_dir / filename
    if local.exists() and local.stat().st_size > 0:
        return local
    if skip_download:
        return None
    return download_hf_file(repo, filename, cache_dir, base_url)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/robocerebra_metadata"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--sample-rows", type=int, default=3)
    parser.add_argument("--hf-base-url", default=DEFAULT_HF_BASE_URL)
    parser.add_argument(
        "--include-data-parquet",
        action="store_true",
        help="Also download/read the first LeRobot data parquet. This may be larger than metadata.",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print("# RoboCerebra metadata inspection")
    print("cache_dir:", args.cache_dir)
    print("hf_base_url:", args.hf_base_url)

    for split, candidates in RAW_MANIFEST_CANDIDATES.items():
        print(f"\n# Raw manifest split={split}")
        found = False
        for filename in candidates:
            path = find_local_or_download(
                RAW_REPO,
                filename,
                args.cache_dir / "raw",
                args.skip_download,
                args.hf_base_url.rstrip("/"),
            )
            if path is None:
                continue
            df = read_parquet(path)
            if df is None:
                continue
            inspect_frame(f"{RAW_REPO}:{filename}", df, args.sample_rows)
            found = True
            break
        if not found:
            print(f"No readable local/downloaded manifest for split={split}; tried {candidates}")

    print("\n# LeRobot metadata")
    lerobot_files = LEROBOT_FILES if args.include_data_parquet else LEROBOT_FILES[:-1]
    for filename in lerobot_files:
        path = find_local_or_download(
            LEROBOT_REPO,
            filename,
            args.cache_dir / "lerobot",
            args.skip_download,
            args.hf_base_url.rstrip("/"),
        )
        if path is None:
            print(f"No readable local/downloaded LeRobot file: {filename}")
            continue
        if filename.endswith(".json"):
            inspect_json(f"{LEROBOT_REPO}:{filename}", path)
        elif filename.endswith(".parquet"):
            df = read_parquet(path)
            if df is not None:
                inspect_frame(f"{LEROBOT_REPO}:{filename}", df, args.sample_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
