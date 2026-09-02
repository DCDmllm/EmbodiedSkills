from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from clawvla.scripts.collect_agent_skill_sft_pilot import (
    EXECUTION_FAILURE_RECOVERY_SCENARIOS,
    OPENAI_STYLE_SHAREGPT_TAGS,
)


SCHEMA = "clawvla-agent-skill-sft-engineering-multitask-v2"
JSONL_FILES = (
    "scheduler_train.jsonl",
    "component_train.jsonl",
    "runtime_calls.jsonl",
    "rejected_decisions.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate passed production-AgentLoop engineering pilots."
    )
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [path.expanduser().resolve() for path in args.input_dir]
    summary = aggregate_engineering_sets(
        input_dirs=inputs,
        output_dir=args.output_dir.expanduser().resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False))
    if summary.get("status") != "PASS":
        raise SystemExit(1)


def aggregate_engineering_sets(
    *, input_dirs: list[Path], output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing_to_reuse_nonempty_output_dir:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_dirs:
        raise ValueError("missing_engineering_inputs")

    sources: list[dict[str, Any]] = []
    tasks: set[str] = set()
    episodes: set[tuple[str, int]] = set()
    aggregate_counts: Counter[str] = Counter()
    scenario_source_counts: Counter[str] = Counter()
    max_total_tokens = 0
    for input_dir in input_dirs:
        input_dir = input_dir.expanduser().resolve()
        summary_path = input_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if source_summary.get("status") != "PASS":
            raise ValueError(f"engineering_input_not_passed:{input_dir}")
        token_audit = source_summary.get("qwen_token_parity")
        token_audit = token_audit if isinstance(token_audit, dict) else {}
        if token_audit.get("status") != "PASS":
            raise ValueError(f"engineering_input_token_gate_failed:{input_dir}")
        source = source_summary.get("source")
        source = source if isinstance(source, dict) else {}
        task = str(source.get("task_name") or "")
        episode = int(source.get("episode_index", -1))
        if not task or episode < 0:
            raise ValueError(f"engineering_input_missing_source_identity:{input_dir}")
        if (task, episode) in episodes:
            raise ValueError(f"duplicate_engineering_source:{task}:episode{episode}")
        tasks.add(task)
        episodes.add((task, episode))
        counts = source_summary.get("counts")
        counts = counts if isinstance(counts, dict) else {}
        for key, value in counts.items():
            aggregate_counts[str(key)] += int(value or 0)
        for scenario in source_summary.get("scenarios") or []:
            if isinstance(scenario, dict):
                scenario_source_counts[str(scenario.get("scenario") or "")] += 1
        max_total_tokens = max(
            max_total_tokens, int(token_audit.get("max_total_tokens") or 0)
        )
        sources.append(
            {
                "directory": str(input_dir),
                "summary_sha256": _file_sha(summary_path),
                "task_name": task,
                "episode_index": episode,
                "seed": source.get("seed"),
                "counts": counts,
                "qwen_token_parity": {
                    "status": token_audit.get("status"),
                    "gold_rows": token_audit.get("gold_rows"),
                    "max_total_tokens": token_audit.get("max_total_tokens"),
                },
            }
        )

    observed_line_counts: Counter[str] = Counter()
    image_reference_count = 0
    for filename in JSONL_FILES:
        with (output_dir / filename).open("w", encoding="utf-8") as output:
            for source in sources:
                source_path = Path(source["directory"]) / filename
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                with source_path.open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if filename in {"scheduler_train.jsonl", "component_train.jsonl"}:
                            images = list(row.get("images") or [])
                            content = str((row.get("messages") or [{}])[0].get("content") or "")
                            if content.count("<image>") != len(images):
                                raise ValueError(
                                    f"image_placeholder_mismatch:{source_path}:{line_number}"
                                )
                            if not all(Path(path).is_file() for path in images):
                                raise FileNotFoundError(
                                    f"missing_training_image:{source_path}:{line_number}"
                                )
                            image_reference_count += len(images)
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        observed_line_counts[filename] += 1

    expected_lines = {
        "scheduler_train.jsonl": aggregate_counts["scheduler_gold_rows"],
        "component_train.jsonl": aggregate_counts["component_gold_rows"],
        "runtime_calls.jsonl": aggregate_counts["runtime_calls"],
        "rejected_decisions.jsonl": aggregate_counts["rejected_decisions"],
    }
    errors = [
        f"line_count_mismatch:{name}:actual={observed_line_counts[name]}:expected={expected}"
        for name, expected in expected_lines.items()
        if observed_line_counts[name] != expected
    ]
    required_recovery_scenarios = (
        "grounded_error_then_correction",
        "direct_vla_verify_occluded_reobserve",
        *EXECUTION_FAILURE_RECOVERY_SCENARIOS,
        "grounded_preflight_stale_visual_refresh",
        "direct_vla_preflight_invalid_plan_replan",
    )
    for scenario in required_recovery_scenarios:
        if scenario_source_counts[scenario] != len(sources):
            errors.append(
                f"recovery_scenario_source_count:{scenario}:"
                f"actual={scenario_source_counts[scenario]}:expected={len(sources)}"
            )

    dataset_info = {
        "robotwin_agent_skill_engineering_scheduler": {
            "file_name": "scheduler_train.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": dict(OPENAI_STYLE_SHAREGPT_TAGS),
        },
        "robotwin_agent_skill_engineering_components": {
            "file_name": "component_train.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": dict(OPENAI_STYLE_SHAREGPT_TAGS),
        },
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "sources.json").write_text(
        json.dumps(sources, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": SCHEMA,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "source_set_count": len(sources),
        "source_task_count": len(tasks),
        "source_episode_count": len(episodes),
        "source_tasks": sorted(tasks),
        "gold_row_count": (
            observed_line_counts["scheduler_train.jsonl"]
            + observed_line_counts["component_train.jsonl"]
        ),
        "counts": {
            "scheduler_gold_rows": observed_line_counts["scheduler_train.jsonl"],
            "component_gold_rows": observed_line_counts["component_train.jsonl"],
            "runtime_calls": observed_line_counts["runtime_calls.jsonl"],
            "rejected_decisions": observed_line_counts["rejected_decisions.jsonl"],
            "scenarios": aggregate_counts["scenarios"],
            "images": aggregate_counts["images"],
            "error_conditioned_correction_rows": aggregate_counts[
                "error_conditioned_correction_rows"
            ],
        },
        "line_counts": dict(sorted(observed_line_counts.items())),
        "aggregate_source_counts": dict(sorted(aggregate_counts.items())),
        "scenario_source_counts": dict(sorted(scenario_source_counts.items())),
        "required_recovery_scenarios": list(required_recovery_scenarios),
        "training_image_reference_count": image_reference_count,
        "qwen_token_parity": {
            "status": "PASS",
            "source_audits": len(sources),
            "max_total_tokens": max_total_tokens,
            "cutoff_len": 65536,
        },
        "sources": sources,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        _readme(summary), encoding="utf-8"
    )
    return summary


def _readme(summary: dict[str, Any]) -> str:
    return f"""# RoboTwin Agent skill multi-task engineering SFT set

This aggregate contains {summary.get('gold_row_count')} production-context gold rows from {summary.get('source_episode_count')} expert episodes across {summary.get('source_task_count')} tasks. Every source pilot independently passed exact production prompt/history, contiguous replay, image ordering, correction, recovery, and Qwen token gates before aggregation.

Each source contributes complete engineering scenarios for wrong-decision correction, occluded-view reobserve, explicit action-backend failure recovery/retry, stale-preflight visual refresh, and invalid-plan replan. Physical disturbances without real image sources remain excluded.

- `scheduler_train.jsonl`: authoritative scheduler and task-plan gold rows.
- `component_train.jsonl`: vision, verifier, and recovery component gold rows.
- `runtime_calls.jsonl`: complete source teacher calls for auditing.
- `rejected_decisions.jsonl`: diagnostic-only decisions, excluded from SFT gold.
- `sources.json`: source episode, count, hash, and token-gate provenance.
- `summary.json`: aggregate counts and release gate.
"""


def _file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
