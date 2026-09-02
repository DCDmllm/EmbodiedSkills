from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable

from clawvla.scripts.collect_agent_skill_sft_pilot import (
    OPENAI_STYLE_SHAREGPT_TAGS,
)


SUBSET_SCHEMA = "clawvla-agent-skill-sft-subset-v2"
@dataclass(frozen=True)
class SourceRow:
    raw: str
    payload: dict[str, Any]
    source_group: str
    source_path: str
    line_number: int
    identity: str

    @property
    def metadata(self) -> dict[str, Any]:
        value = self.payload.get("metadata")
        return value if isinstance(value, dict) else {}

    @property
    def target(self) -> dict[str, Any]:
        messages = self.payload.get("messages")
        if not isinstance(messages, list) or not messages:
            return {}
        content = messages[-1].get("content") if isinstance(messages[-1], dict) else None
        if not isinstance(content, str):
            return {}
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an exact-row RoboTwin Agent skill SFT subset. All task-plan "
            "generation rows are preserved; other runtime skills are sampled "
            "deterministically across task, decision family, and production "
            "history depth."
        )
    )
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument(
        "--engineering-dir", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=30_000)
    parser.add_argument("--val-size", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=20_260_717)
    parser.add_argument(
        "--val-only",
        action="store_true",
        help=(
            "Only add a validation view to an existing subset directory; do "
            "not read or rewrite train.jsonl."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.val_only:
        summary = build_validation_view(
            corpus_dir=args.corpus_dir.expanduser().resolve(),
            output_dir=args.output_dir.expanduser().resolve(),
            val_size=int(args.val_size),
            seed=int(args.seed),
        )
    else:
        summary = build_subset(
            corpus_dir=args.corpus_dir.expanduser().resolve(),
            engineering_dir=args.engineering_dir.expanduser().resolve(),
            output_dir=args.output_dir.expanduser().resolve(),
            train_size=int(args.train_size),
            val_size=int(args.val_size),
            seed=int(args.seed),
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_validation_view(
    *, corpus_dir: Path, output_dir: Path, val_size: int, seed: int
) -> dict[str, Any]:
    if val_size <= 0:
        raise ValueError("val_size_must_be_positive")
    if not corpus_dir.is_dir():
        raise FileNotFoundError(corpus_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)

    normal_val = _load_rows(
        [
            *sorted((corpus_dir / "shards/val/scheduler").glob("*.jsonl")),
            *sorted((corpus_dir / "shards/val/component").glob("*.jsonl")),
        ],
        source_group="normal_val",
        root=corpus_dir,
    )
    mandatory_val = [row for row in normal_val if plan_generation_kind(row)]
    mandatory_ids = {row.identity for row in mandatory_val}
    candidates = [row for row in normal_val if row.identity not in mandatory_ids]
    remaining = val_size - len(mandatory_val)
    if remaining < 0:
        raise ValueError(
            f"val_size_below_mandatory_rows:{val_size}<{len(mandatory_val)}"
        )
    selected = _stable_output_order(
        [
            *mandatory_val,
            *stratified_sample(candidates, remaining, seed=seed),
        ],
        seed,
    )
    audit = _audit_selection(
        source_rows=normal_val,
        selected_rows=selected,
        expected_size=val_size,
        required_plan_rows=mandatory_val,
        required_rows=[],
    )
    if audit["status"] != "PASS":
        raise ValueError("validation_view_audit_failed:" + json.dumps(audit))

    size_label = _size_label(val_size)
    file_name = f"val_small_{size_label}.jsonl"
    dataset_name = f"robotwin_agent_skill_subset_{size_label}_val"
    _write_exact_rows(output_dir / file_name, selected)
    dataset_info_path = output_dir / "dataset_info.json"
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    dataset_info[dataset_name] = _dataset_entry(file_name)
    dataset_info_path.write_text(
        json.dumps(dataset_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": SUBSET_SCHEMA,
        "status": "PASS",
        "seed": seed,
        "source_rows": len(normal_val),
        "selected_rows": len(selected),
        "mandatory_plan_rows": len(mandatory_val),
        "sampled_other_rows": remaining,
        "dataset_name": dataset_name,
        "file": str(output_dir / file_name),
        "row_content_rewritten": False,
        "audit": audit,
        "distribution": _distribution(selected),
    }
    (output_dir / f"val_small_{size_label}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def build_subset(
    *,
    corpus_dir: Path,
    engineering_dir: Path,
    output_dir: Path,
    train_size: int,
    val_size: int,
    seed: int,
) -> dict[str, Any]:
    if train_size <= 0 or val_size <= 0:
        raise ValueError("subset_sizes_must_be_positive")
    if not corpus_dir.is_dir():
        raise FileNotFoundError(corpus_dir)
    if not engineering_dir.is_dir():
        raise FileNotFoundError(engineering_dir)

    normal_train = _load_rows(
        [
            *sorted((corpus_dir / "shards/train/scheduler").glob("*.jsonl")),
            *sorted((corpus_dir / "shards/train/component").glob("*.jsonl")),
        ],
        source_group="normal_train",
        root=corpus_dir,
    )
    engineering_train = _load_rows(
        [
            engineering_dir / "scheduler_train.jsonl",
            engineering_dir / "component_train.jsonl",
        ],
        source_group="engineering_train",
        root=engineering_dir,
    )
    normal_val = _load_rows(
        [
            *sorted((corpus_dir / "shards/val/scheduler").glob("*.jsonl")),
            *sorted((corpus_dir / "shards/val/component").glob("*.jsonl")),
        ],
        source_group="normal_val",
        root=corpus_dir,
    )

    train_plan_rows = [row for row in normal_train if plan_generation_kind(row)]
    engineering_plan_rows = [
        row for row in engineering_train if plan_generation_kind(row)
    ]
    mandatory_train = _unique_rows([*train_plan_rows, *engineering_train])
    mandatory_train_ids = {row.identity for row in mandatory_train}
    normal_train_candidates = [
        row for row in normal_train if row.identity not in mandatory_train_ids
    ]
    remaining_train = train_size - len(mandatory_train)
    if remaining_train < 0:
        raise ValueError(
            f"train_size_below_mandatory_rows:{train_size}<{len(mandatory_train)}"
        )
    sampled_train = stratified_sample(
        normal_train_candidates,
        remaining_train,
        seed=seed,
    )
    selected_train = _stable_output_order([*mandatory_train, *sampled_train], seed)

    mandatory_val = [row for row in normal_val if plan_generation_kind(row)]
    mandatory_val_ids = {row.identity for row in mandatory_val}
    normal_val_candidates = [
        row for row in normal_val if row.identity not in mandatory_val_ids
    ]
    remaining_val = val_size - len(mandatory_val)
    if remaining_val < 0:
        raise ValueError(
            f"val_size_below_mandatory_rows:{val_size}<{len(mandatory_val)}"
        )
    sampled_val = stratified_sample(
        normal_val_candidates,
        remaining_val,
        seed=seed + 1,
    )
    selected_val = _stable_output_order([*mandatory_val, *sampled_val], seed + 1)

    train_audit = _audit_selection(
        source_rows=[*normal_train, *engineering_train],
        selected_rows=selected_train,
        expected_size=train_size,
        required_plan_rows=[*train_plan_rows, *engineering_plan_rows],
        required_rows=engineering_train,
    )
    val_audit = _audit_selection(
        source_rows=normal_val,
        selected_rows=selected_val,
        expected_size=val_size,
        required_plan_rows=mandatory_val,
        required_rows=[],
    )
    status = "PASS" if train_audit["status"] == val_audit["status"] == "PASS" else "FAIL"
    if status != "PASS":
        raise ValueError(
            "subset_audit_failed:"
            + json.dumps(
                {"train": train_audit, "val": val_audit}, ensure_ascii=False
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_exact_rows(output_dir / "train.jsonl", selected_train)
    _write_exact_rows(output_dir / "val.jsonl", selected_val)
    dataset_info = {
        "robotwin_agent_skill_subset_30k_train": _dataset_entry("train.jsonl"),
        "robotwin_agent_skill_subset_3k_val": _dataset_entry("val.jsonl"),
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": SUBSET_SCHEMA,
        "status": status,
        "seed": seed,
        "source": {
            "corpus_dir": str(corpus_dir),
            "engineering_dir": str(engineering_dir),
            "normal_train_rows": len(normal_train),
            "engineering_train_rows": len(engineering_train),
            "normal_val_rows": len(normal_val),
        },
        "selection_policy": {
            "all_normal_plan_generation_rows_preserved": True,
            "all_engineering_gold_rows_preserved": True,
            "other_rows": (
                "proportional by scheduler/component decision family, then "
                "round-robin across task and production history-depth bucket"
            ),
            "row_content_rewritten": False,
            "packing_required": False,
        },
        "train": {
            "selected_rows": len(selected_train),
            "mandatory_normal_plan_rows": len(train_plan_rows),
            "mandatory_engineering_rows": len(engineering_train),
            "engineering_plan_rows": len(engineering_plan_rows),
            "sampled_other_rows": len(sampled_train),
            "audit": train_audit,
            "distribution": _distribution(selected_train),
        },
        "val": {
            "selected_rows": len(selected_val),
            "mandatory_plan_rows": len(mandatory_val),
            "sampled_other_rows": len(sampled_val),
            "audit": val_audit,
            "distribution": _distribution(selected_val),
        },
        "files": {
            "train": str(output_dir / "train.jsonl"),
            "val": str(output_dir / "val.jsonl"),
            "dataset_info": str(output_dir / "dataset_info.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        _readme(summary), encoding="utf-8"
    )
    return summary


def _load_rows(
    paths: Iterable[Path], *, source_group: str, root: Path
) -> list[SourceRow]:
    rows: list[SourceRow] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = str(path.relative_to(root))
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = line.rstrip("\n")
                if not raw.strip():
                    continue
                payload = json.loads(raw)
                identity = sha256(
                    f"{source_group}\0{relative}\0{line_number}\0{raw}".encode()
                ).hexdigest()
                rows.append(
                    SourceRow(
                        raw=raw,
                        payload=payload,
                        source_group=source_group,
                        source_path=relative,
                        line_number=line_number,
                        identity=identity,
                    )
                )
    return rows


def plan_generation_kind(row: SourceRow) -> str | None:
    teacher_rule = str(row.metadata.get("teacher_rule") or "")
    target = row.target
    if teacher_rule == "expert_subtask_segments:oracle_task_plan":
        return "oracle_task_plan"
    if (
        target.get("control") == "run_skill"
        and target.get("next_component") == "scheduler"
        and target.get("next_skill") == "build_task_plan"
    ):
        return "build_task_plan"
    if (
        target.get("control") == "advance_stage"
        and target.get("reason") == "plan_ready"
    ):
        return "plan_ready"
    return None


def semantic_family(row: SourceRow) -> str:
    component = str(row.metadata.get("component") or "unknown")
    teacher_rule = _normalize_reason(str(row.metadata.get("teacher_rule") or ""))
    target = row.target
    if component != "scheduler":
        next_action = str(target.get("next_action") or target.get("action") or "")
        failure_type = str(target.get("failure_type") or "")
        return f"component:{component}:{teacher_rule}:{next_action}:{failure_type}"
    control = str(target.get("control") or "payload")
    reason = _normalize_reason(str(target.get("reason") or ""))
    if control == "run_skill":
        skill = f"{target.get('next_component')}.{target.get('next_skill')}"
        return f"scheduler:{control}:{skill}:{reason}"
    return f"scheduler:{control}:{reason}"


def history_bucket(row: SourceRow) -> str:
    history = row.metadata.get("history_compaction")
    count = int(history.get("effective_recent_loop_steps") or 0) if isinstance(history, dict) else 0
    if count == 0:
        return "0"
    if count <= 4:
        return "1-4"
    if count <= 9:
        return "5-9"
    if count <= 14:
        return "10-14"
    if count <= 19:
        return "15-19"
    return "20"


def stratified_sample(
    rows: list[SourceRow], target_size: int, *, seed: int
) -> list[SourceRow]:
    if target_size < 0:
        raise ValueError("negative_target_size")
    if target_size >= len(rows):
        return list(rows)
    if target_size == 0:
        return []

    by_family: dict[str, list[SourceRow]] = defaultdict(list)
    for row in rows:
        by_family[semantic_family(row)].append(row)
    quotas = _proportional_quotas(
        {family: len(items) for family, items in by_family.items()}, target_size
    )
    selected: list[SourceRow] = []
    for family in sorted(by_family):
        selected.extend(
            _balanced_take(by_family[family], quotas[family], seed=seed)
        )
    if len(selected) != target_size:
        raise AssertionError(f"sample_size_mismatch:{len(selected)}!={target_size}")
    return selected


def _proportional_quotas(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    if target > total:
        raise ValueError("quota_target_exceeds_source")
    if target < len(counts):
        ordered = sorted(counts, key=lambda key: (-counts[key], key))
        return {key: int(key in set(ordered[:target])) for key in counts}

    exact = {key: counts[key] * target / total for key in counts}
    quotas = {key: min(counts[key], max(1, math.floor(exact[key]))) for key in counts}
    while sum(quotas.values()) < target:
        candidates = [key for key in counts if quotas[key] < counts[key]]
        key = max(candidates, key=lambda item: (exact[item] - quotas[item], counts[item], item))
        quotas[key] += 1
    while sum(quotas.values()) > target:
        candidates = [key for key in counts if quotas[key] > 1]
        key = min(candidates, key=lambda item: (exact[item] - quotas[item], -counts[item], item))
        quotas[key] -= 1
    return quotas


def _balanced_take(
    rows: list[SourceRow], target_size: int, *, seed: int
) -> list[SourceRow]:
    if target_size >= len(rows):
        return list(rows)
    groups: dict[tuple[str, str], list[SourceRow]] = defaultdict(list)
    for row in rows:
        task = str(row.metadata.get("task_name") or "unknown")
        groups[(task, history_bucket(row))].append(row)
    for key, items in groups.items():
        items.sort(key=lambda row: _rank(seed, f"{key}:{row.identity}"))
    ordered_keys = sorted(groups, key=lambda key: _rank(seed, repr(key)))
    selected: list[SourceRow] = []
    cursor = 0
    while len(selected) < target_size:
        key = ordered_keys[cursor % len(ordered_keys)]
        if groups[key]:
            selected.append(groups[key].pop())
        cursor += 1
        if cursor % len(ordered_keys) == 0:
            ordered_keys = [key for key in ordered_keys if groups[key]]
            if not ordered_keys and len(selected) < target_size:
                raise AssertionError("stratified_groups_exhausted_early")
            cursor = 0
    return selected


def _normalize_reason(value: str) -> str:
    for prefix in (
        "runtime_state.next_required_decision:stale_preflight_report",
        "runtime_state.next_required_decision:missing_or_stale_action_chunk",
        "runtime_state.next_required_decision:verification_next_action",
        "stale_preflight_report",
        "verification_next_action",
    ):
        if value.startswith(prefix):
            suffix = value[len(prefix) :].lstrip(":")
            if prefix.endswith("verification_next_action") and suffix:
                return f"{prefix}:{suffix}"
            return prefix
    return value


def _rank(seed: int, value: str) -> str:
    return sha256(f"{seed}\0{value}".encode()).hexdigest()


def _stable_output_order(rows: list[SourceRow], seed: int) -> list[SourceRow]:
    return sorted(_unique_rows(rows), key=lambda row: _rank(seed, row.identity))


def _unique_rows(rows: Iterable[SourceRow]) -> list[SourceRow]:
    result: list[SourceRow] = []
    seen: set[str] = set()
    for row in rows:
        if row.identity not in seen:
            seen.add(row.identity)
            result.append(row)
    return result


def _audit_selection(
    *,
    source_rows: list[SourceRow],
    selected_rows: list[SourceRow],
    expected_size: int,
    required_plan_rows: list[SourceRow],
    required_rows: list[SourceRow],
) -> dict[str, Any]:
    source_ids = {row.identity for row in source_rows}
    selected_ids = {row.identity for row in selected_rows}
    plan_ids = {row.identity for row in required_plan_rows}
    required_ids = {row.identity for row in required_rows}
    source_families = {semantic_family(row) for row in source_rows}
    selected_families = {semantic_family(row) for row in selected_rows}
    source_tasks = {str(row.metadata.get("task_name") or "") for row in source_rows}
    selected_tasks = {str(row.metadata.get("task_name") or "") for row in selected_rows}
    image_errors: list[str] = []
    for row in selected_rows:
        messages = row.payload.get("messages") or []
        images = row.payload.get("images") or []
        placeholders = sum(
            str(message.get("content") or "").count("<image>")
            for message in messages
            if isinstance(message, dict)
        )
        if placeholders != len(images):
            image_errors.append(
                f"{row.source_group}:{row.source_path}:{row.line_number}:"
                f"{placeholders}!={len(images)}"
            )
    errors: list[str] = []
    if len(selected_rows) != expected_size:
        errors.append(f"size:{len(selected_rows)}!={expected_size}")
    if len(selected_ids) != len(selected_rows):
        errors.append("duplicate_selected_identity")
    if not selected_ids <= source_ids:
        errors.append("selected_row_not_in_source")
    if not plan_ids <= selected_ids:
        errors.append(f"missing_plan_rows:{len(plan_ids - selected_ids)}")
    if not required_ids <= selected_ids:
        errors.append(f"missing_required_rows:{len(required_ids - selected_ids)}")
    if source_families - selected_families:
        errors.append(
            f"missing_semantic_families:{len(source_families - selected_families)}"
        )
    if source_tasks - selected_tasks:
        errors.append(f"missing_tasks:{len(source_tasks - selected_tasks)}")
    if image_errors:
        errors.append(f"image_placeholder_errors:{len(image_errors)}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "source_rows": len(source_rows),
        "selected_rows": len(selected_rows),
        "required_plan_rows": len(plan_ids),
        "required_rows": len(required_ids),
        "semantic_families": len(selected_families),
        "task_count": len(selected_tasks),
        "image_placeholder_errors": image_errors[:20],
    }


def _distribution(rows: list[SourceRow]) -> dict[str, Any]:
    return {
        "source_groups": dict(sorted(Counter(row.source_group for row in rows).items())),
        "plan_generation": dict(
            sorted(
                Counter(
                    kind
                    for row in rows
                    if (kind := plan_generation_kind(row)) is not None
                ).items()
            )
        ),
        "components": dict(
            sorted(Counter(str(row.metadata.get("component") or "unknown") for row in rows).items())
        ),
        "tasks": dict(
            sorted(Counter(str(row.metadata.get("task_name") or "unknown") for row in rows).items())
        ),
        "history_buckets": dict(sorted(Counter(history_bucket(row) for row in rows).items())),
        "semantic_families": dict(
            sorted(Counter(semantic_family(row) for row in rows).items())
        ),
    }


def _write_exact_rows(path: Path, rows: list[SourceRow]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.raw)
            handle.write("\n")


def _dataset_entry(file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": dict(OPENAI_STYLE_SHAREGPT_TAGS),
    }


def _size_label(size: int) -> str:
    return f"{size // 1000}k" if size % 1000 == 0 else str(size)


def _readme(summary: dict[str, Any]) -> str:
    train = summary["train"]
    val = summary["val"]
    return f"""# RoboTwin Agent skill SFT 30k subset

Status: `{summary['status']}`

This is a deterministic, exact-row view of the production-context r7 corpus and
the r8 engineering set. Source messages, assistant targets, metadata, image
paths, production history compaction, and prompt hashes are not rewritten.

## Selection

- Train rows: {train['selected_rows']}
- Full normal plan-generation rows retained: {train['mandatory_normal_plan_rows']}
- Full engineering rows retained: {train['mandatory_engineering_rows']}
- Other skill/runtime rows sampled: {train['sampled_other_rows']}
- Validation rows: {val['selected_rows']}
- Validation plan-generation rows retained: {val['mandatory_plan_rows']}

Plan generation means the complete production sequence
`build_task_plan -> oracle task-plan payload -> plan_ready`. Other rows are
sampled proportionally by semantic decision family and balanced across task and
the production `recent_loop_history` depth buckets.

Use `packing: false` in LLaMA-Factory so one exact Runtime call remains one
training example.
"""


if __name__ == "__main__":
    main()
