#!/usr/bin/env python3
"""Export RoboCerebra long-horizon tasks into subgoal-level JSONL."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


RAW_REPO = "qiukingballball/RoboCerebra"
DEFAULT_OUTPUT = Path("outputs/robocerebra_subgoals_sample.jsonl")
DOWNLOAD_TIMEOUT_SECONDS = float(os.environ.get("ROBOCEREBRA_DOWNLOAD_TIMEOUT", "8"))
DEFAULT_HF_BASE_URL = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")

STEP_RE = re.compile(
    r"Step:\s*(?P<instruction>.*?)\s*"
    r"\[(?P<start>\d+)\s*,\s*(?P<end>\d+)\]"
    r"(?:\s*Related Objects:\s*(?P<objects>.*?))?"
    r"(?=\s*Step:|\s*$)",
    flags=re.DOTALL,
)


# These are real HF Dataset Viewer excerpts used only when parquet download is
# unavailable in the current environment. They intentionally include only the
# visible step text and boundaries, not inferred hidden steps.
VIEWER_SAMPLE_ROWS = [
    {
        "id": "case1",
        "task_type": "Ideal",
        "demo_path": "Ideal/case1/demo.hdf5",
        "high_level_instruction": "Organize all the food boxes into the white storage box.",
        "task_description": (
            "Task: Organize all the food boxes into the white storage box. "
            "Step: Pick up cream cheese from coffee table [0, 242] Related Objects: cream cheese, coffee table "
            "Step: Place down cream cheese into white storage box [242, 697] Related Objects: cream cheese, white storage box"
        ),
        "num_steps": 6,
    },
    {
        "id": "case2",
        "task_type": "Ideal",
        "demo_path": "Ideal/case2/demo.hdf5",
        "high_level_instruction": "Place the cream cheese into the white storage box and place the frypan onto the rack",
        "task_description": (
            "Task: Place the cream cheese into the white storage box and place the frypan onto the rack "
            "Step: Pick up the cream cheese from the coffee table. [0, 256] Related Objects: cream cheese, coffee table "
            "Step: Place the cream cheese into the white storage box at top side [257, 880] Related Objects: cream cheese, white storage box"
        ),
        "num_steps": 4,
    },
    {
        "id": "case3",
        "task_type": "Ideal",
        "demo_path": "Ideal/case3/demo.hdf5",
        "high_level_instruction": "Place the plates neatly between the knife and fork, and pour all the snacks and drinks into the plates.",
        "task_description": (
            "Task: Place the plates neatly between the knife and fork, and pour all the snacks and drinks into the plates. "
            "Step: Pick up the plate from coffee table [0, 261] Related Objects: plate, dining set group "
            "Step: Place the plate on dining set group [261, 417] Related Objects: plate, dining set group"
        ),
        "num_steps": 10,
    },
    {
        "id": "case4",
        "task_type": "Ideal",
        "demo_path": "Ideal/case4/demo.hdf5",
        "high_level_instruction": "Clean up the desk by putting the two boxes in the bowl and place the frying pan onto the cabinet.",
        "task_description": (
            "Task: Clean up the desk by putting the two boxes in the bowl and place the frying pan onto the cabinet. "
            "Step: Pick up cream cheese from coffee table [0, 157] Related Objects: cream cheese, coffee table "
            "Step: Place cream cheese in bowl [158, 454] Related Objects: cream cheese, bowl"
        ),
        "num_steps": 6,
    },
    {
        "id": "case5",
        "task_type": "Ideal",
        "demo_path": "Ideal/case5/demo.hdf5",
        "high_level_instruction": "Put the chocolate pudding on the plate and pour the chips into the plate.",
        "task_description": (
            "Task: Put the chocolate pudding on the plate and pour the chips into the plate. "
            "Step: Pick up the chocolate pudding from the coffee table [0, 633] Related Objects: chocolate pudding, coffee table "
            "Step: Place the chocolate pudding on the plate [633, 946] Related Objects: chocolate pudding, plate"
        ),
        "num_steps": 5,
    },
]


def download_raw_manifest(cache_dir: Path, split: str, base_url: str) -> Path | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    candidates = {
        "train": ["train.parquet", "RoboCerebra_trainset/trainingset.parquet"],
        "test": ["test.parquet"],
    }.get(split, [f"{split}.parquet"])
    for filename in candidates:
        target = cache_dir / filename
        if target.exists() and target.stat().st_size > 0:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        quoted = urllib.parse.quote(filename, safe="/")
        url = f"{base_url}/datasets/{RAW_REPO}/resolve/main/{quoted}"
        try:
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, target.open("wb") as f:
                shutil.copyfileobj(response, f)
            return target
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            if target.exists() and target.stat().st_size == 0:
                target.unlink()
            print(f"DOWNLOAD_FAILED {filename}: {type(exc).__name__}: {exc}")
    return None


def normalize_steps(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value
    return value


def clean_text(text: str) -> str:
    return " ".join(str(text).split())


def parse_task_description(task_description: str) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for match in STEP_RE.finditer(task_description or ""):
        instruction = clean_text(match.group("instruction").strip(" ."))
        objects = match.group("objects")
        if objects is not None:
            objects = clean_text(objects.strip(" ."))
        parsed.append(
            {
                "instruction": instruction,
                "frame_start": int(match.group("start")),
                "frame_end": int(match.group("end")),
                "related_objects": objects,
            }
        )
    return parsed


def row_to_subgoals(row: dict[str, Any]) -> Iterable[dict[str, Any]]:
    full_task = row.get("high_level_instruction")
    if not full_task:
        task_match = re.search(r"Task:\s*(.*?)(?=\s*Step:|$)", str(row.get("task_description", "")), re.DOTALL)
        full_task = clean_text(task_match.group(1)) if task_match else None

    raw_id = row.get("id") or row.get("case") or row.get("episode_id") or row.get("demo_path")
    task_type = row.get("task_type")
    if task_type and raw_id and not str(raw_id).startswith(str(task_type).lower()):
        episode_id = f"{str(task_type).lower()}_{raw_id}"
    elif row.get("scene") and row.get("case"):
        episode_id = f"{row.get('scene')}_{row.get('case')}"
    else:
        episode_id = str(raw_id)

    demo_path = row.get("demo_path") or row.get("demo")
    steps = normalize_steps(row.get("steps"))
    parsed_steps = parse_task_description(str(row.get("task_description", "")))
    for index, step in enumerate(parsed_steps):
        related = step.get("related_objects")
        criteria = "Subgoal instruction completed"
        if related:
            criteria += f"; related objects: {related}"
        notes = []
        if demo_path:
            notes.append(f"raw_demo_path={demo_path}")
        if steps is not None and not (isinstance(steps, float) and pd.isna(steps)):
            notes.append("structured steps field present; frame boundaries parsed from task_description")
        yield {
            "benchmark": "RoboCerebra",
            "episode_id": episode_id,
            "full_task_instruction": full_task,
            "subgoal_index": index,
            "frame_start": step["frame_start"],
            "frame_end": step["frame_end"],
            "subgoal_instruction": step["instruction"],
            "completion_criteria": criteria,
            "source": "native_task_description",
            "boundary_status": "native",
            "notes": "; ".join(notes),
        }


def iter_rows_from_manifest(path: Path) -> Iterable[dict[str, Any]]:
    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        yield row.to_dict()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/robocerebra_metadata/raw"))
    parser.add_argument("--hf-base-url", default=DEFAULT_HF_BASE_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-subgoals", type=int, default=10)
    parser.add_argument(
        "--use-viewer-sample",
        action="store_true",
        help="Use real HF Dataset Viewer excerpts embedded in this script when parquet is unavailable.",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    rows: Iterable[dict[str, Any]]
    if args.manifest_path is not None:
        rows = iter_rows_from_manifest(args.manifest_path)
    elif args.use_viewer_sample:
        rows = VIEWER_SAMPLE_ROWS
    else:
        manifest = download_raw_manifest(args.cache_dir, args.split, args.hf_base_url.rstrip("/"))
        if manifest is None:
            raise SystemExit(
                "Could not download raw manifest. Re-run with --manifest-path or --use-viewer-sample."
            )
        rows = iter_rows_from_manifest(manifest)

    count = 0
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            for subgoal in row_to_subgoals(row):
                f.write(json.dumps(subgoal, ensure_ascii=False) + "\n")
                count += 1
                if count >= args.max_subgoals:
                    print(f"wrote {count} subgoals to {args.output}")
                    return 0
    print(f"wrote {count} subgoals to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
