from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

from clawvla.rl.config import PlannerAuxConfig
from clawvla.rl.planner_similarity import PlannerReference, load_planner_reference_index


def parse_args() -> argparse.Namespace:
    defaults = PlannerAuxConfig()
    parser = argparse.ArgumentParser(
        description="Build full-plan RoboTwin SFT rows and optionally mix ordinary grounding rows."
    )
    parser.add_argument("--dataset-root", default=defaults.dataset_root)
    parser.add_argument("--repair-ledger", default=defaults.repair_ledger)
    parser.add_argument("--split-manifest", default=defaults.split_manifest)
    parser.add_argument("--grounding-train", action="append", default=[])
    parser.add_argument("--grounding-val", action="append", default=[])
    parser.add_argument("--planner-ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="runs/data/robotwin_planner_grounding_sft",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.planner_ratio <= 1:
        raise ValueError("--planner-ratio must be in (0, 1]")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split_name, grounding_paths in (("train", args.grounding_train), ("val", args.grounding_val)):
        planner_rows = _planner_rows(
            args.dataset_root,
            args.repair_ledger,
            args.split_manifest,
            split_name,
        )
        grounding_rows = [row for path in grounding_paths for row in _load_rows(Path(path))]
        mixed = _mix_rows(planner_rows, grounding_rows, args.planner_ratio, args.seed)
        path = output_dir / f"{split_name}.jsonl"
        _write_jsonl(path, mixed)
        summaries[split_name] = {
            "path": str(path),
            "planner_rows": len(planner_rows),
            "grounding_source_rows": len(grounding_rows),
            "mixed_rows": len(mixed),
        }
    summary = {
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "repair_ledger": str(Path(args.repair_ledger).expanduser().resolve()),
        "split_manifest": str(Path(args.split_manifest).expanduser().resolve()),
        "planner_ratio": args.planner_ratio,
        "seed": args.seed,
        "splits": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    dataset_info = {
        "robotwin_planner_grounding_train": {
            "file_name": "train.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
        },
        "robotwin_planner_grounding_val": {
            "file_name": "val.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
        },
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def _planner_rows(dataset_root: str, repair_ledger: str, split_manifest: str, split_name: str) -> list[dict[str, Any]]:
    index = load_planner_reference_index(
        str(Path(dataset_root).expanduser().resolve()),
        str(Path(repair_ledger).expanduser().resolve()),
        str(Path(split_manifest).expanduser().resolve()),
        split_name,
        10_000,
    )
    return [_planner_row(reference) for references in index.values() for reference in references]


def _planner_row(reference: PlannerReference) -> dict[str, Any]:
    subgoals = []
    for index, variants in enumerate(reference.subgoals):
        completion = reference.completion_criteria[index] if index < len(reference.completion_criteria) else ""
        subgoals.append(
            {
                "subgoal_id": f"S{index + 1}",
                "type": reference.subgoal_types[index] if index < len(reference.subgoal_types) else "act",
                "instruction": variants[0],
                "status": "pending",
                "completion_criteria": {"natural_language": completion},
            }
        )
    target = {
        "task": reference.task_instruction,
        "subgoals": subgoals,
        "current_subgoal_id": "S1",
        "status": "pending",
    }
    prompt = (
        "Build the complete ordered manipulation subgoal plan for the task below. "
        "Return exactly one JSON object. Each subgoal instruction is sent verbatim to a short-horizon VLA. "
        "Do not add confirmation-only steps.\n\n"
        f"Task: {reference.task_instruction}"
    )
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
        "metadata": {
            "sample_type": "planner_subgoals",
            "subgoal_mask": 1,
            "task_name": reference.task_name,
            "episode_index": reference.episode_index,
        },
    }


def _load_rows(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text) if path.suffix.lower() == ".json" else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    rows = payload if isinstance(payload, list) else [payload]
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        copied = _to_messages_row(row)
        if copied is None:
            continue
        metadata = dict(copied.get("metadata") or {})
        metadata.update({"sample_type": "grounding", "subgoal_mask": 0})
        copied["metadata"] = metadata
        normalized.append(copied)
    return normalized


def _to_messages_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(row.get("messages"), list):
        return dict(row)
    if isinstance(row.get("conversations"), list):
        role_map = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}
        messages = []
        for item in row["conversations"]:
            if not isinstance(item, dict):
                continue
            role = role_map.get(str(item.get("from") or item.get("role") or "").lower())
            content = item.get("value") if item.get("value") is not None else item.get("content")
            if role and content is not None:
                messages.append({"role": role, "content": content})
        if not messages:
            return None
        return {**{key: value for key, value in row.items() if key != "conversations"}, "messages": messages}
    instruction = str(row.get("instruction") or row.get("prompt") or "").strip()
    output = row.get("output") if row.get("output") is not None else row.get("response")
    if instruction and output is not None:
        return {
            **{key: value for key, value in row.items() if key not in {"instruction", "prompt", "output", "response"}},
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": str(output)},
            ],
        }
    return None


def _mix_rows(
    planner_rows: list[dict[str, Any]],
    grounding_rows: list[dict[str, Any]],
    planner_ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    if not grounding_rows or planner_ratio >= 1:
        mixed = list(planner_rows)
    else:
        grounding_count = round(len(planner_rows) * (1 - planner_ratio) / planner_ratio)
        selected = [grounding_rows[index % len(grounding_rows)] for index in range(grounding_count)]
        mixed = [*planner_rows, *selected]
    random.Random(seed).shuffle(mixed)
    return mixed


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
