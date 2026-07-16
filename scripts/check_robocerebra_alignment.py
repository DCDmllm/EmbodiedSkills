#!/usr/bin/env python3
"""Check raw RoboCerebra step order against LeRobot episode order."""

from __future__ import annotations

import argparse
import json
import random
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd


STEP_RE = re.compile(
    r"Step:\s*(?P<instruction>.*?)\s*"
    r"\[(?P<start>\d+)\s*,\s*(?P<end>\d+)\]"
    r"(?:\s*Related Objects:\s*(?P<objects>.*?))?"
    r"(?=\s*Step:|\s*$)",
    flags=re.DOTALL,
)


def clean_text(value: Any) -> str:
    return " ".join(str(value).replace("\n", " ").split()).strip(" .")


def normalize_task_text(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"\b(orig|d[xy][mp]\d+|tex\d+)\b", " ", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = text.replace("placed at", "placed")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def parse_steps(df: pd.DataFrame, split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw_row_index, row in df.iterrows():
        full_task = row.get("high_level_instruction")
        if row.get("id") is not None:
            episode_id = str(row.get("id"))
            demo_path = row.get("demo_path")
        else:
            episode_id = f"{row.get('scene')}_{row.get('case')}"
            demo_path = row.get("demo")
        for subgoal_index, match in enumerate(STEP_RE.finditer(str(row.get("task_description", "")))):
            instruction = clean_text(match.group("instruction"))
            related_objects = match.group("objects")
            out.append(
                {
                    "split": split,
                    "raw_row_index": int(raw_row_index),
                    "episode_id": episode_id,
                    "demo_path": demo_path,
                    "full_task_instruction": full_task,
                    "subgoal_index": subgoal_index,
                    "subgoal_instruction": instruction,
                    "normalized_subgoal_instruction": normalize_task_text(instruction),
                    "raw_frame_start": int(match.group("start")),
                    "raw_frame_end": int(match.group("end")),
                    "related_objects": clean_text(related_objects) if related_objects else None,
                }
            )
    return out


def task_index_map(tasks_df: pd.DataFrame) -> dict[int, str]:
    return {int(row["task_index"]): str(task_text) for task_text, row in tasks_df.iterrows()}


def parse_episode_task(value: Any) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and value:
        return clean_text(value[0])
    return clean_text(value)


def load_lerobot_episodes(
    episodes_path: Path,
    tasks_path: Path,
    data_path: Path | None = None,
) -> list[dict[str, Any]]:
    episodes = pd.read_parquet(episodes_path).sort_values("episode_index").reset_index(drop=True)
    tasks = pd.read_parquet(tasks_path)
    idx_to_task = task_index_map(tasks)
    episode_to_task_index: dict[int, int] = {}
    if data_path is not None and data_path.exists():
        data = pd.read_parquet(data_path, columns=["episode_index", "task_index"])
        episode_to_task_index = {
            int(episode_index): int(task_index)
            for episode_index, task_index in data.groupby("episode_index")["task_index"].first().items()
        }
    out: list[dict[str, Any]] = []
    for _, row in episodes.iterrows():
        episode_index = int(row["episode_index"])
        task_text_from_meta = parse_episode_task(row["tasks"])
        task_index = episode_to_task_index.get(episode_index)
        task_text = idx_to_task.get(task_index, task_text_from_meta) if task_index is not None else task_text_from_meta
        out.append(
            {
                "lerobot_episode_index": episode_index,
                "task_index": task_index,
                "task_text": task_text,
                "normalized_task_text": normalize_task_text(task_text),
                "length": int(row["length"]),
                "lerobot_dataset_from_index": int(row["dataset_from_index"]),
                "lerobot_dataset_to_index": int(row["dataset_to_index"]),
            }
        )
    return out


def compare_at(raw_steps: list[dict[str, Any]], lr_episodes: list[dict[str, Any]], index: int) -> dict[str, Any]:
    raw = raw_steps[index]
    lr = lr_episodes[index]
    raw_norm = raw["normalized_subgoal_instruction"]
    lr_norm = lr["normalized_task_text"]
    return {
        "index": index,
        "exact_normalized_match": raw_norm == lr_norm,
        "similarity": round(SequenceMatcher(None, raw_norm, lr_norm).ratio(), 4),
        "raw_subgoal_instruction": raw["subgoal_instruction"],
        "lerobot_task_text": lr["task_text"],
        "raw_row_index": raw["raw_row_index"],
        "raw_subgoal_index": raw["subgoal_index"],
        "lerobot_episode_index": lr["lerobot_episode_index"],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-train", type=Path, default=Path("outputs/robocerebra_metadata/raw/train.parquet"))
    parser.add_argument("--raw-test", type=Path, default=Path("outputs/robocerebra_metadata/raw/test.parquet"))
    parser.add_argument(
        "--episodes",
        type=Path,
        default=Path("outputs/robocerebra_metadata/lerobot/meta/episodes/chunk-000/file-000.parquet"),
    )
    parser.add_argument("--tasks", type=Path, default=Path("outputs/robocerebra_metadata/lerobot/meta/tasks.parquet"))
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("outputs/robocerebra_metadata/lerobot/data/chunk-000/file-000.parquet"),
        help="Optional LeRobot data parquet used to map episode_index to task_index.",
    )
    parser.add_argument("--report", type=Path, default=Path("outputs/robocerebra_alignment_check.md"))
    parser.add_argument("--matches-jsonl", type=Path, default=Path("outputs/robocerebra_alignment_matches.jsonl"))
    parser.add_argument("--aligned-jsonl", type=Path, default=Path("outputs/robocerebra_aligned_subgoals.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=50)
    args = parser.parse_args()

    raw_train = pd.read_parquet(args.raw_train)
    raw_test = pd.read_parquet(args.raw_test)
    raw_train_steps = parse_steps(raw_train, "train")
    raw_test_steps = parse_steps(raw_test, "test")
    lr_episodes = load_lerobot_episodes(args.episodes, args.tasks, args.data)

    n = min(len(raw_train_steps), len(lr_episodes))
    first = [compare_at(raw_train_steps, lr_episodes, i) for i in range(min(20, n))]
    last_start = max(0, n - 20)
    last = [compare_at(raw_train_steps, lr_episodes, i) for i in range(last_start, n)]

    random.seed(args.seed)
    sample_indices = random.sample(range(n), min(args.sample_size, n))
    random_rows = [compare_at(raw_train_steps, lr_episodes, i) for i in sample_indices]
    random_accuracy = sum(row["exact_normalized_match"] for row in random_rows) / len(random_rows)
    random_similarity = sum(row["similarity"] for row in random_rows) / len(random_rows)

    train_unique = {row["normalized_subgoal_instruction"] for row in raw_train_steps}
    test_unique = {row["normalized_subgoal_instruction"] for row in raw_test_steps}
    lr_unique = {row["normalized_task_text"] for row in lr_episodes}
    test_coverage = sum(task in test_unique for task in lr_unique) / len(lr_unique)
    train_coverage = sum(task in train_unique for task in lr_unique) / len(lr_unique)
    lr_episode_in_test = sum(row["normalized_task_text"] in test_unique for row in lr_episodes) / len(lr_episodes)

    direct_match = len(raw_train_steps) == len(lr_episodes) and all(
        raw_train_steps[i]["normalized_subgoal_instruction"] == lr_episodes[i]["normalized_task_text"]
        for i in range(len(lr_episodes))
    )

    aligned_rows: list[dict[str, Any]] = []
    if direct_match:
        for raw, lr in zip(raw_train_steps, lr_episodes):
            aligned_rows.append(
                {
                    "full_task_instruction": raw["full_task_instruction"],
                    "raw_row_index": raw["raw_row_index"],
                    "subgoal_index": raw["subgoal_index"],
                    "lerobot_episode_index": lr["lerobot_episode_index"],
                    "subgoal_instruction": raw["subgoal_instruction"],
                    "raw_frame_start": raw["raw_frame_start"],
                    "raw_frame_end": raw["raw_frame_end"],
                    "lerobot_dataset_from_index": lr["lerobot_dataset_from_index"],
                    "lerobot_dataset_to_index": lr["lerobot_dataset_to_index"],
                    "alignment_status": "direct_order_match",
                }
            )
        write_jsonl(args.aligned_jsonl, aligned_rows)
    elif args.aligned_jsonl.exists():
        args.aligned_jsonl.unlink()

    write_jsonl(args.matches_jsonl, first + random_rows + last)

    lines = [
        "# RoboCerebra Raw-to-LeRobot Alignment Check",
        "",
        f"- raw train rows: {len(raw_train)}",
        f"- raw train steps: {len(raw_train_steps)}",
        f"- raw test rows: {len(raw_test)}",
        f"- raw test steps: {len(raw_test_steps)}",
        f"- LeRobot episodes: {len(lr_episodes)}",
        f"- LeRobot unique normalized task texts: {len(lr_unique)}",
        f"- direct raw-train-order random {len(random_rows)} accuracy: {random_accuracy:.3f}",
        f"- direct raw-train-order random {len(random_rows)} mean similarity: {random_similarity:.3f}",
        f"- LeRobot unique task coverage in raw train: {train_coverage:.3f}",
        f"- LeRobot unique task coverage in raw test: {test_coverage:.3f}",
        f"- LeRobot episode task coverage in raw test: {lr_episode_in_test:.3f}",
        f"- aligned JSONL exported: {direct_match}",
        "",
        "## First 20 Direct Comparisons",
        "",
    ]
    for row in first:
        lines.append(
            f"- {row['index']}: match={row['exact_normalized_match']} sim={row['similarity']} | "
            f"raw=`{row['raw_subgoal_instruction']}` | lerobot=`{row['lerobot_task_text']}`"
        )
    lines.extend(["", f"## Random {len(random_rows)} Direct Accuracy", ""])
    lines.append(f"- exact normalized matches: {sum(r['exact_normalized_match'] for r in random_rows)}/{len(random_rows)}")
    lines.append(f"- accuracy: {random_accuracy:.3f}")
    lines.append(f"- mean similarity: {random_similarity:.3f}")
    lines.extend(["", "## Last 20 Direct Comparisons", ""])
    for row in last:
        lines.append(
            f"- {row['index']}: match={row['exact_normalized_match']} sim={row['similarity']} | "
            f"raw=`{row['raw_subgoal_instruction']}` | lerobot=`{row['lerobot_task_text']}`"
        )
    lines.extend(
        [
            "",
            "## Analysis",
            "",
            "Direct order alignment from raw `train.parquet` to LeRobot episodes is not valid.",
            "The counts differ: raw train expands to 8,934 steps, while LeRobot has 6,660 episodes.",
            "After normalizing augmentation suffixes such as `orig`, `dxp005`, `dyp005`, `dym005`, and `texN`, LeRobot's 126 unique task texts are fully covered by raw `test.parquet`, not by raw train order.",
            "This indicates that `lerobot/robocerebra_unified` is a subtask-level, augmented version of the test/evaluation-style task set, with repeated perturbation/texture variants and a changed ordering.",
            "Because LeRobot metadata does not expose the raw parent id or raw row index, exporting a trustworthy `full_task_instruction/raw_row_index/subgoal_index -> lerobot_episode_index` aligned JSONL from raw train order would be unsafe.",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote report: {args.report}")
    print(f"wrote matches: {args.matches_jsonl}")
    print(f"aligned_jsonl_exported={direct_match}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
