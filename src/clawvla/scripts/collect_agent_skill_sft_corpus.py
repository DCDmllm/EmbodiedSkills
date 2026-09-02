from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import BytesIO, StringIO
import json
import math
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from clawvla.blackboard import MAX_CONTEXT_LOOP_HISTORY
from clawvla.scripts.collect_agent_skill_sft_pilot import (
    CAMERA_ROLES,
    OPENAI_STYLE_SHAREGPT_TAGS,
    SCHEMA,
    _executed_action_reports,
    _file_sha,
    _canonical_task_instruction,
    _matching_calls,
    _observation_from_replay,
    _oracle_task_plan,
    _robot_arms_from_hdf5,
    _run_teacher_scenario,
    _runtime_prompt_text,
    _training_row,
    _write_jsonl,
)


CORPUS_SCHEMA = "clawvla-agent-skill-sft-corpus-v2"
ENGINEERING_COVERAGE_DIR_NAME = "agent_skill_engineering"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a resumable 2486-episode production-AgentLoop SFT corpus from "
            "contiguous RoboTwin expert subtask trajectories."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repair-ledger", type=Path, required=True)
    parser.add_argument(
        "--task-instruction-repairs",
        type=Path,
        required=True,
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/runtime/robotwin.json"),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Optional comma-separated task allowlist.",
    )
    parser.add_argument(
        "--identity",
        action="append",
        default=[],
        help="Restrict to an exact task:episode identity; may be repeated.",
    )
    parser.add_argument("--shard-size", type=int, default=50)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose-loop", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    repair_ledger_path = args.repair_ledger.expanduser().resolve()
    task_instruction_repairs_path = args.task_instruction_repairs.expanduser().resolve()
    split_manifest_path = args.split_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if not repair_ledger_path.is_file():
        raise FileNotFoundError(repair_ledger_path)
    if not task_instruction_repairs_path.is_file():
        raise FileNotFoundError(task_instruction_repairs_path)
    if not split_manifest_path.is_file():
        raise FileNotFoundError(split_manifest_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    repair_ledger = _load_repair_ledger(repair_ledger_path)
    repair_ledger_sha = _file_sha(repair_ledger_path)
    task_instruction_repairs = _load_task_instruction_repairs(
        task_instruction_repairs_path
    )
    task_instruction_repairs_sha = _file_sha(task_instruction_repairs_path)
    split_map = _load_split_manifest(split_manifest_path)
    split_manifest_sha = _file_sha(split_manifest_path)
    segment_paths = sorted(
        (dataset_root / "segments").glob("*/episode*.json"),
        key=_segment_sort_key,
    )
    task_allowlist = {
        item.strip() for item in str(args.tasks or "").split(",") if item.strip()
    }
    if task_allowlist:
        segment_paths = [
            path for path in segment_paths if path.parent.name in task_allowlist
        ]
    identity_allowlist = {_parse_identity(value) for value in args.identity}
    if identity_allowlist:
        available = {_segment_identity(path) for path in segment_paths}
        missing = sorted(identity_allowlist - available)
        if missing:
            raise ValueError(f"requested identities are missing from dataset: {missing}")
        segment_paths = [
            path for path in segment_paths if _segment_identity(path) in identity_allowlist
        ]
    start = max(0, int(args.start_index))
    stop = None if args.limit is None else start + max(0, int(args.limit))
    selected_paths = segment_paths[start:stop]
    selection = {
        "all_episode_count": len(segment_paths),
        "selected_episode_count": len(selected_paths),
        "start_index": start,
        "limit": args.limit,
        "task_allowlist": sorted(task_allowlist),
        "identity_allowlist": [
            f"{task}:{episode}" for task, episode in sorted(identity_allowlist)
        ],
    }
    selection_dir = output_dir / "worker_selections"
    selection_dir.mkdir(parents=True, exist_ok=True)
    selection_name = (
        f"selection_{start:04d}_{len(selected_paths):04d}.json"
        if not args.aggregate_only
        else "selection_aggregate_all.json"
    )
    (selection_dir / selection_name).write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    failures: list[dict[str, Any]] = []
    if not args.aggregate_only:
        total = len(selected_paths)
        for offset, segment_path in enumerate(selected_paths, start=1):
            source = json.loads(segment_path.read_text(encoding="utf-8"))
            source["split"] = split_map[
                (str(source.get("task_name") or ""), int(source.get("episode_index", -1)))
            ]
            episode_dir = _episode_dir(output_dir, source)
            summary_path = episode_dir / "episode_summary.json"
            if not args.no_resume and _completed_summary_matches(
                summary_path,
                segment_path=segment_path,
                repair_ledger_sha=repair_ledger_sha,
                task_instruction_repairs_sha=task_instruction_repairs_sha,
                split_manifest_sha=split_manifest_sha,
            ):
                print(
                    json.dumps(
                        {
                            "event": "episode_skip_complete",
                            "index": offset,
                            "total": total,
                            "task": source.get("task_name"),
                            "episode": source.get("episode_index"),
                        },
                        ensure_ascii=False,
                    )
                )
                continue
            try:
                summary = _collect_episode(
                    segment_path=segment_path,
                    source=source,
                    repair_ledger=repair_ledger,
                    repair_ledger_path=repair_ledger_path,
                    repair_ledger_sha=repair_ledger_sha,
                    task_instruction_repairs=task_instruction_repairs,
                    task_instruction_repairs_path=task_instruction_repairs_path,
                    task_instruction_repairs_sha=task_instruction_repairs_sha,
                    split_manifest_path=split_manifest_path,
                    split_manifest_sha=split_manifest_sha,
                    config_path=config_path,
                    output_dir=output_dir,
                    episode_dir=episode_dir,
                    verbose_loop=bool(args.verbose_loop),
                )
                print(
                    json.dumps(
                        {
                            "event": "episode_complete",
                            "index": offset,
                            "total": total,
                            "task": summary["task_name"],
                            "episode": summary["episode_index"],
                            "split": summary["split"],
                            "subgoals": summary["effective_subgoal_count"],
                            "chunks": summary["action_chunk_count"],
                            "scheduler_rows": summary["scheduler_gold_rows"],
                            "component_rows": summary["component_gold_rows"],
                        },
                        ensure_ascii=False,
                    )
                )
            except Exception as exc:
                failure = {
                    "schema": CORPUS_SCHEMA,
                    "status": "FAIL",
                    "task_name": source.get("task_name"),
                    "episode_index": source.get("episode_index"),
                    "segment_path": str(segment_path),
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                }
                episode_dir.mkdir(parents=True, exist_ok=True)
                (episode_dir / "failure.json").write_text(
                    json.dumps(failure, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                failures.append(failure)
                print(json.dumps({"event": "episode_failed", **failure}, ensure_ascii=False))
                if args.fail_fast:
                    raise

    if args.no_aggregate:
        worker_summary = {
            "schema": CORPUS_SCHEMA,
            "status": "PASS" if not failures else "FAIL",
            "selection": selection,
            "failure_count": len(failures),
            "failures": failures,
        }
        worker_path = selection_dir / f"worker_{start:04d}_{len(selected_paths):04d}.summary.json"
        worker_path.write_text(
            json.dumps(worker_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(worker_summary, ensure_ascii=False))
        if failures:
            raise SystemExit(1)
        return

    corpus_summary = _aggregate_corpus(
        output_dir=output_dir,
        expected_paths=selected_paths,
        dataset_root=dataset_root,
        repair_ledger_path=repair_ledger_path,
        repair_ledger_sha=repair_ledger_sha,
        task_instruction_repairs_path=task_instruction_repairs_path,
        task_instruction_repairs_sha=task_instruction_repairs_sha,
        split_manifest_path=split_manifest_path,
        split_manifest_sha=split_manifest_sha,
        split_map=split_map,
        shard_size=max(1, int(args.shard_size)),
    )
    if failures:
        corpus_summary["run_failures"] = failures
        corpus_summary["status"] = "FAIL"
        (output_dir / "corpus_summary.json").write_text(
            json.dumps(corpus_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(corpus_summary, ensure_ascii=False))
    if corpus_summary.get("status") != "PASS":
        raise SystemExit(1)


def _load_repair_ledger(path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    repairs: dict[tuple[str, int, int], dict[str, Any]] = {}
    for line_number, line in enumerate(path.open(encoding="utf-8"), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") != "accepted":
            continue
        validation = record.get("validation")
        if isinstance(validation, dict) and validation.get("valid") is not True:
            continue
        repair = record.get("repair")
        if not isinstance(repair, dict):
            continue
        key = (
            str(record.get("task_name") or ""),
            int(record.get("episode_index", -1)),
            int(record.get("segment_index", -1)),
        )
        if not key[0] or key[1] < 0 or key[2] < 0:
            raise ValueError(f"invalid_repair_ledger_key:line={line_number}:{key}")
        repairs[key] = deepcopy(repair)
    return repairs


def _load_task_instruction_repairs(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    repairs: dict[tuple[str, int], dict[str, Any]] = {}
    for line_number, line in enumerate(path.open(encoding="utf-8"), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        key = (
            str(record.get("task_name") or ""),
            int(record.get("episode_index", -1)),
        )
        instruction = str(record.get("instruction") or "").strip()
        reason = str(record.get("reason") or "").strip()
        if not key[0] or key[1] < 0 or not instruction or not reason:
            raise ValueError(
                f"invalid_task_instruction_repair:line={line_number}:{key}"
            )
        if key in repairs:
            raise ValueError(f"duplicate_task_instruction_repair:{key}")
        repairs[key] = {
            "instruction": instruction,
            "reason": reason,
        }
    return repairs


def _load_split_manifest(path: Path) -> dict[tuple[str, int], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    split_map: dict[tuple[str, int], str] = {}
    for task_name, task_payload in (payload.get("tasks") or {}).items():
        for episode_index in task_payload.get("train_episode_indices") or []:
            split_map[(str(task_name), int(episode_index))] = "train"
        for episode_index in task_payload.get("val_episode_indices") or []:
            key = (str(task_name), int(episode_index))
            if key in split_map:
                raise ValueError(f"split_manifest_overlap:{key}")
            split_map[key] = "val"
    expected = int(payload.get("num_train_episodes") or 0) + int(
        payload.get("num_val_episodes") or 0
    )
    if len(split_map) != expected:
        raise ValueError(
            f"split_manifest_count_mismatch:{len(split_map)}!={expected}"
        )
    return split_map


def _prepare_episode(
    source: dict[str, Any],
    repair_ledger: dict[tuple[str, int, int], dict[str, Any]],
    task_instruction_repairs: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    prepared = deepcopy(source)
    task_name = str(source.get("task_name") or "")
    episode_index = int(source.get("episode_index", -1))
    task_instruction_repair = (task_instruction_repairs or {}).get(
        (task_name, episode_index)
    )
    if task_instruction_repair is not None:
        prepared["original_instruction_before_semantic_repair"] = source.get(
            "instruction"
        )
        prepared["instruction"] = task_instruction_repair["instruction"]
        prepared["task_instruction_semantic_repair"] = deepcopy(
            task_instruction_repair
        )
    effective: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for fallback_index, raw in enumerate(source.get("segments") or []):
        segment = deepcopy(raw)
        original_index = int(segment.get("segment_index", fallback_index))
        start = int(segment.get("frame_start", 0))
        end = int(segment.get("frame_end_exclusive", start))
        if end <= start:
            excluded.append(
                {
                    "original_segment_index": original_index,
                    "frame_start": start,
                    "frame_end_exclusive": end,
                    "reason": "zero_or_negative_frame_segment",
                    "move_return": segment.get("move_return"),
                    "plan_success_after_move": segment.get("plan_success_after_move"),
                }
            )
            continue
        repair = repair_ledger.get((task_name, episode_index, original_index))
        if repair is not None:
            segment["polished_instruction"] = str(repair.get("instruction") or "").strip()
            segment["canonical_instruction"] = segment["polished_instruction"]
            segment["subgoal_type"] = str(repair.get("subgoal_type") or "move").strip()
            segment["completion_criteria"] = str(
                repair.get("completion_criteria") or ""
            ).strip()
            segment["paraphrases"] = [
                str(item).strip()
                for item in repair.get("paraphrases") or []
                if str(item).strip()
            ]
            applied.append(
                {
                    "original_segment_index": original_index,
                    "instruction": segment["polished_instruction"],
                    "subgoal_type": segment["subgoal_type"],
                }
            )
        instruction = str(
            segment.get("polished_instruction")
            or segment.get("canonical_instruction")
            or segment.get("raw_canonical_instruction")
            or ""
        ).strip()
        completion = str(segment.get("completion_criteria") or "").strip()
        subgoal_type = str(segment.get("subgoal_type") or "move").strip().lower()
        if not instruction or not completion or not subgoal_type:
            raise ValueError(
                "effective_segment_missing_teacher_label:"
                f"{task_name}/episode{episode_index}/segment{original_index}:"
                f"instruction={bool(instruction)}:completion={bool(completion)}:"
                f"type={bool(subgoal_type)}"
            )
        segment["segment_index"] = len(effective)
        segment["original_segment_index"] = original_index
        segment["polished_instruction"] = instruction
        segment["canonical_instruction"] = instruction
        segment["subgoal_type"] = subgoal_type
        segment["completion_criteria"] = completion
        effective.append(segment)
    if not effective:
        raise ValueError(f"episode_has_no_positive_frame_subtasks:{task_name}/episode{episode_index}")
    prepared["segments"] = effective
    prepared["segment_count"] = len(effective)
    prepared["effective_annotation_source"] = "segment_json_plus_accepted_repair_ledger"
    return prepared, applied, excluded


def _deterministic_action_budgets(
    *, task_name: str, episode_index: int, segment_lengths: list[int]
) -> list[int]:
    budgets: list[int] = []
    for segment_index, length in enumerate(segment_lengths):
        cursor = 0
        chunk_index = 0
        while cursor < length:
            digest = sha256(
                f"{task_name}:{episode_index}:{segment_index}:{chunk_index}".encode("utf-8")
            ).digest()
            bucket = digest[0] % 100
            if bucket < 72:
                horizon = 32
            elif bucket < 82:
                horizon = 31
            elif bucket < 89:
                horizon = 30
            elif bucket < 94:
                horizon = 29
            elif bucket < 98:
                horizon = 28
            else:
                horizon = 15 + digest[1] % 13
            budgets.append(horizon)
            cursor = min(length, cursor + horizon)
            chunk_index += 1
    return budgets


def _collect_episode(
    *,
    segment_path: Path,
    source: dict[str, Any],
    repair_ledger: dict[tuple[str, int, int], dict[str, Any]],
    repair_ledger_path: Path,
    repair_ledger_sha: str,
    task_instruction_repairs: dict[tuple[str, int], dict[str, Any]],
    task_instruction_repairs_path: Path,
    task_instruction_repairs_sha: str,
    split_manifest_path: Path,
    split_manifest_sha: str,
    config_path: Path,
    output_dir: Path,
    episode_dir: Path,
    verbose_loop: bool,
) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py_required_run_with_robotwin_python") from exc

    prepared, applied_repairs, excluded_segments = _prepare_episode(
        source, repair_ledger, task_instruction_repairs
    )
    task_name = str(prepared.get("task_name") or "")
    episode_index = int(prepared.get("episode_index", -1))
    instruction = _canonical_task_instruction(prepared)
    hdf5_path = _resolve_hdf5_path(prepared)
    source_sha = _file_sha(segment_path)
    effective_segments_sha = sha256(
        json.dumps(
            prepared["segments"],
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    episode_dir.mkdir(parents=True, exist_ok=True)

    commands_by_subgoal: dict[str, list[list[float]]] = {}
    absolute_start_by_subgoal: dict[str, int] = {}
    image_manifest: list[dict[str, Any]] = []
    observation_cache: dict[int, Any] = {}
    with h5py.File(hdf5_path, "r") as handle:
        action_dataset = handle["joint_action/vector"]
        for segment_index, segment in enumerate(prepared["segments"]):
            subgoal_id = f"S{segment_index + 1}"
            start = int(segment["frame_start"])
            end = int(segment["frame_end_exclusive"])
            commands = [
                [float(value) for value in action_dataset[frame].tolist()]
                for frame in range(start, end)
            ]
            if not commands or any(len(command) != 14 for command in commands):
                raise ValueError(
                    f"invalid_contiguous_expert_actions:{task_name}/episode{episode_index}/"
                    f"{subgoal_id}:count={len(commands)}"
                )
            commands_by_subgoal[subgoal_id] = commands
            absolute_start_by_subgoal[subgoal_id] = start

        def observation_for_frame(frame: int) -> Any:
            if frame not in observation_cache:
                paths, manifest, arms = _extract_replay_observation_from_handle(
                    prepared,
                    output_dir,
                    handle,
                    hdf5_path,
                    frame,
                )
                observation_cache[frame] = _observation_from_replay(
                    prepared, paths, arms, frame
                )
                image_manifest.extend(manifest)
            return observation_cache[frame]

        first_start = int(prepared["segments"][0]["frame_start"])
        initial_observation = observation_for_frame(first_start)
        segment_lengths = [
            len(commands_by_subgoal[f"S{index + 1}"])
            for index in range(len(prepared["segments"]))
        ]
        action_budgets = _deterministic_action_budgets(
            task_name=task_name,
            episode_index=episode_index,
            segment_lengths=segment_lengths,
        )
        max_steps = 24 + 12 * len(action_budgets) + 4 * len(segment_lengths)
        scenario = f"corpus_{task_name}_episode{episode_index:04d}"
        run_kwargs = dict(
            config_path=config_path,
            instruction=instruction,
            observation=initial_observation,
            oracle_candidates=[],
            oracle_plan=_oracle_task_plan(prepared, task_instruction=instruction),
            scenario=scenario,
            candidate_bindings_required=False,
            wrong_scheduler_call_indexes=set(),
            max_steps=max_steps,
            action_commands=commands_by_subgoal["S1"],
            post_execution_observation=initial_observation,
            verification_outcome="timeline",
            contiguous_commands_by_subgoal=commands_by_subgoal,
            absolute_start_by_subgoal=absolute_start_by_subgoal,
            observation_for_frame=observation_for_frame,
            action_budgets=action_budgets,
        )
        if verbose_loop:
            scenario_result, runtime_calls = _run_teacher_scenario(**run_kwargs)
        else:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                scenario_result, runtime_calls = _run_teacher_scenario(**run_kwargs)

    errors, action_audit = _validate_corpus_episode(
        scenario_result=scenario_result,
        runtime_calls=runtime_calls,
        segment_lengths=segment_lengths,
        absolute_start_by_subgoal=absolute_start_by_subgoal,
        action_budgets=action_budgets,
    )
    if errors:
        raise RuntimeError(";".join(errors[:20]))
    scheduler_rows = [
        _training_row(call, prepared, segment_path)
        for call in runtime_calls
        if call.get("component") == "scheduler" and call.get("supervision") == "gold"
    ]
    component_rows = [
        _training_row(call, prepared, segment_path)
        for call in runtime_calls
        if call.get("component") != "scheduler" and call.get("supervision") == "gold"
    ]
    row_metadata = {
        "corpus_schema": CORPUS_SCHEMA,
        "split": str(prepared.get("split") or "train"),
        "repair_ledger_path": str(repair_ledger_path),
        "repair_ledger_sha256": repair_ledger_sha,
        "task_instruction_repairs_path": str(task_instruction_repairs_path),
        "task_instruction_repairs_sha256": task_instruction_repairs_sha,
        "task_instruction_repair_applied": bool(
            prepared.get("task_instruction_semantic_repair")
        ),
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_sha256": split_manifest_sha,
        "applied_repair_count": len(applied_repairs),
        "excluded_zero_frame_segment_count": len(excluded_segments),
        "effective_segments_sha256": effective_segments_sha,
    }
    for row in [*scheduler_rows, *component_rows]:
        row["metadata"].update(row_metadata)
    _write_jsonl(episode_dir / "scheduler_train.jsonl", scheduler_rows)
    _write_jsonl(episode_dir / "component_train.jsonl", component_rows)
    _write_jsonl(episode_dir / "runtime_calls.jsonl", runtime_calls)
    (episode_dir / "action_audit.json").write_text(
        json.dumps(action_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (episode_dir / "image_manifest.json").write_text(
        json.dumps(image_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    gold_calls = [call for call in runtime_calls if call.get("supervision") == "gold"]
    max_prompt_call = max(
        gold_calls,
        key=lambda call: len(_runtime_prompt_text(call["runtime_messages"])),
    )
    summary = {
        "schema": CORPUS_SCHEMA,
        "row_schema": SCHEMA,
        "status": "PASS",
        "task_name": task_name,
        "episode_index": episode_index,
        "seed": prepared.get("seed"),
        "split": str(prepared.get("split") or "train"),
        "source_segment_path": str(segment_path),
        "source_segment_sha256": source_sha,
        "source_hdf5_path": str(hdf5_path),
        "repair_ledger_sha256": repair_ledger_sha,
        "task_instruction_repairs_path": str(task_instruction_repairs_path),
        "task_instruction_repairs_sha256": task_instruction_repairs_sha,
        "task_instruction_repair": prepared.get(
            "task_instruction_semantic_repair"
        ),
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_sha256": split_manifest_sha,
        "applied_repair_count": len(applied_repairs),
        "applied_repairs": applied_repairs,
        "excluded_zero_frame_segment_count": len(excluded_segments),
        "excluded_zero_frame_segments": excluded_segments,
        "source_subgoal_count": len(source.get("segments") or []),
        "effective_subgoal_count": len(segment_lengths),
        "segment_lengths": segment_lengths,
        "action_chunk_count": len(action_budgets),
        "action_budgets": action_budgets,
        "action_audit": action_audit,
        "loop_status": scenario_result.get("loop_result", {}).get("status"),
        "loop_steps": len(scenario_result.get("loop_result", {}).get("steps") or []),
        "scheduler_gold_rows": len(scheduler_rows),
        "component_gold_rows": len(component_rows),
        "runtime_calls": len(runtime_calls),
        "images": len(image_manifest),
        "max_prompt_chars": len(_runtime_prompt_text(max_prompt_call["runtime_messages"])),
        "max_prompt_call": {
            "component": max_prompt_call.get("component"),
            "call_index": max_prompt_call.get("call_index"),
            "event_index": max_prompt_call.get("event_index"),
            "scenario": max_prompt_call.get("scenario"),
        },
        "max_recent_loop_history": max(
            (
                int(call.get("context_audit", {}).get("recent_loop_history_count") or 0)
                for call in runtime_calls
            ),
            default=0,
        ),
        "effective_segments_sha256": effective_segments_sha,
    }
    (episode_dir / "episode_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    failure_path = episode_dir / "failure.json"
    if failure_path.exists():
        failure_path.unlink()
    return summary


def _extract_replay_observation_from_handle(
    episode: dict[str, Any],
    output_dir: Path,
    handle: Any,
    hdf5_path: Path,
    frame: int,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    image_root = (
        output_dir
        / "images"
        / str(episode.get("task_name"))
        / f"episode{int(episode.get('episode_index', 0)):04d}"
        / f"frame{frame:04d}"
    )
    image_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    manifest: list[dict[str, Any]] = []
    for camera in CAMERA_ROLES:
        dataset = handle.get(f"observation/{camera}/rgb")
        if dataset is None or frame < 0 or frame >= int(dataset.shape[0]):
            raise ValueError(f"missing_replay_frame:{camera}:{frame}")
        encoded = bytes(dataset[frame]).rstrip(b"\0")
        decoded = Image.open(BytesIO(encoded)).convert("RGB")
        red, green, blue = decoded.split()
        corrected = Image.merge("RGB", (blue, green, red))
        destination = image_root / f"{camera}.jpg"
        corrected.save(destination, format="JPEG", quality=95, subsampling=0)
        resolved = str(destination.resolve())
        paths[camera] = resolved
        manifest.append(
            {
                "camera": camera,
                "frame": frame,
                "source_hdf5": str(hdf5_path),
                "source_encoded_sha256": sha256(encoded).hexdigest(),
                "output_path": resolved,
                "output_sha256": _file_sha(destination),
                "color_repair": "rgb_encoded_as_bgr_v1",
            }
        )
    return paths, manifest, _robot_arms_from_hdf5(handle, frame)


def _validate_corpus_episode(
    *,
    scenario_result: dict[str, Any],
    runtime_calls: list[dict[str, Any]],
    segment_lengths: list[int],
    absolute_start_by_subgoal: dict[str, int],
    action_budgets: list[int],
) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    loop_result = scenario_result.get("loop_result") or {}
    if loop_result.get("status") != "finished":
        errors.append(f"loop_not_finished:{loop_result.get('status')}")
    final_context = scenario_result.get("final_compact_context") or {}
    final_plan = final_context.get("task_plan") or {}
    if final_plan.get("status") != "succeeded":
        errors.append(f"task_plan_not_succeeded:{final_plan.get('status')}")
    if final_plan.get("current_subgoal_id") is not None:
        errors.append(f"current_subgoal_not_cleared:{final_plan.get('current_subgoal_id')}")
    expected_cursors = {
        f"S{index + 1}": length for index, length in enumerate(segment_lengths)
    }
    if scenario_result.get("final_timeline_cursors") != expected_cursors:
        errors.append("final_timeline_cursor_mismatch")

    expected_chunks: list[dict[str, Any]] = []
    budget_index = 0
    for segment_index, length in enumerate(segment_lengths):
        subgoal_id = f"S{segment_index + 1}"
        cursor = 0
        while cursor < length:
            if budget_index >= len(action_budgets):
                errors.append("action_budget_sequence_shorter_than_expected")
                break
            horizon = int(action_budgets[budget_index])
            end = min(length, cursor + horizon)
            valid = end - cursor
            expected_chunks.append(
                {
                    "subgoal_id": subgoal_id,
                    "expert_cursor_start": cursor,
                    "expert_cursor_end_exclusive": end,
                    "expert_segment_length": length,
                    "requested_horizon": horizon,
                    "executed_steps": horizon,
                    "expert_valid_steps": valid,
                    "padding_steps": horizon - valid,
                    "post_frame": absolute_start_by_subgoal[subgoal_id] + end - 1,
                    "complete": end == length,
                }
            )
            cursor = end
            budget_index += 1
    if budget_index != len(action_budgets):
        errors.append(
            f"unused_action_budgets:used={budget_index}:available={len(action_budgets)}"
        )
    reports = _executed_action_reports(scenario_result)
    if len(reports) != len(expected_chunks):
        errors.append(
            f"execution_report_count_mismatch:{len(reports)}!={len(expected_chunks)}"
        )
    scenario_name = str(scenario_result.get("scenario") or "")
    verifier_calls = _matching_calls(
        runtime_calls, scenario=scenario_name, component="verifier"
    )
    emit_calls = _matching_calls(
        runtime_calls,
        scenario=scenario_name,
        component="scheduler",
        next_skill="emit_action_chunk",
    )
    if len(verifier_calls) != len(expected_chunks):
        errors.append("verifier_call_count_mismatch")
    if len(emit_calls) != len(expected_chunks):
        errors.append("emit_call_count_mismatch")
    action_audit: list[dict[str, Any]] = []
    for index, expected in enumerate(expected_chunks):
        report = reports[index] if index < len(reports) else {}
        verifier = verifier_calls[index] if index < len(verifier_calls) else {}
        emit = emit_calls[index] if index < len(emit_calls) else {}
        for key in (
            "subgoal_id",
            "expert_cursor_start",
            "expert_cursor_end_exclusive",
            "expert_segment_length",
            "requested_horizon",
            "executed_steps",
            "expert_valid_steps",
            "padding_steps",
            "post_frame",
        ):
            if report.get(key) != expected[key]:
                errors.append(
                    f"chunk_{index}_{key}_mismatch:{report.get(key)}!={expected[key]}"
                )
        if not 15 <= int(expected["requested_horizon"]) <= 32:
            errors.append(f"chunk_{index}_horizon_out_of_range")
        if report.get("expert_segment_complete_after_chunk") is not expected["complete"]:
            errors.append(f"chunk_{index}_completion_boundary_mismatch")
        observation = report.get("observation")
        observation = observation if isinstance(observation, dict) else {}
        if not str(observation.get("observation_id") or "").endswith(
            f"_frame{int(expected['post_frame']):04d}"
        ):
            errors.append(f"chunk_{index}_post_observation_mismatch")
        target = verifier.get("target") if isinstance(verifier.get("target"), dict) else {}
        if target.get("subgoal_success") is not expected["complete"]:
            errors.append(f"chunk_{index}_verifier_success_mismatch")
        expected_next = "advance_subgoal" if expected["complete"] else "continue_execute"
        if target.get("next_action") != expected_next:
            errors.append(f"chunk_{index}_verifier_next_action_mismatch")
        if not verifier.get("image_paths"):
            errors.append(f"chunk_{index}_verifier_images_missing")
        emit_horizon = emit.get("target", {}).get("payload", {}).get("horizon")
        if emit_horizon != expected["requested_horizon"]:
            errors.append(f"chunk_{index}_emit_horizon_mismatch")
        action_audit.append(
            {
                **expected,
                "verifier_next_action": target.get("next_action"),
                "post_observation_id": observation.get("observation_id"),
                "verify_images": list(verifier.get("image_paths") or []),
            }
        )

    emit_by_subgoal: dict[str, list[dict[str, Any]]] = {}
    for call in emit_calls:
        current = (
            call.get("context_audit", {})
            .get("runtime_state", {})
            .get("current_subgoal")
        )
        current = current if isinstance(current, dict) else {}
        emit_by_subgoal.setdefault(str(current.get("subgoal_id") or ""), []).append(call)
    required_continuation_history = {
        "execute_action",
        "capture_verify_views",
        "verify_progress",
        "repair_stage_transition",
    }
    for subgoal_id, calls in emit_by_subgoal.items():
        for call in calls[1:]:
            history = call.get("context_audit", {}).get("recent_loop_history") or []
            history_skills = {
                str(item.get("next_skill") or "")
                for item in history
                if isinstance(item, dict)
            }
            if not required_continuation_history <= history_skills:
                errors.append(f"continuation_history_missing:{subgoal_id}")

    for call in runtime_calls:
        if call.get("supervision") != "gold":
            errors.append("non_gold_call_in_corpus_episode")
        history_count = int(
            call.get("context_audit", {}).get("recent_loop_history_count") or 0
        )
        if history_count > MAX_CONTEXT_LOOP_HISTORY:
            errors.append("production_history_limit_exceeded")
        if call.get("component") != "scheduler":
            continue
        target = call.get("target") if isinstance(call.get("target"), dict) else {}
        if target.get("control") not in {"run_skill", "advance_stage", "finish_run"}:
            continue
        required = call.get("context_audit", {}).get("runtime_state", {}).get(
            "next_required_decision"
        )
        if not isinstance(required, dict):
            errors.append("scheduler_gold_missing_authoritative_required_decision")
            continue
        for key in ("control", "stage", "next_component", "next_skill"):
            if target.get(key) != required.get(key):
                errors.append(f"scheduler_target_diverges_from_required:{key}")
        if target.get("next_skill") != "emit_action_chunk":
            if dict(target.get("payload") or {}) != dict(required.get("payload") or {}):
                errors.append("scheduler_payload_diverges_from_required")
    return errors, action_audit


def _aggregate_corpus(
    *,
    output_dir: Path,
    expected_paths: list[Path],
    dataset_root: Path,
    repair_ledger_path: Path,
    repair_ledger_sha: str,
    task_instruction_repairs_path: Path,
    task_instruction_repairs_sha: str,
    split_manifest_path: Path,
    split_manifest_sha: str,
    split_map: dict[tuple[str, int], str],
    shard_size: int,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    missing: list[str] = []
    for segment_path in expected_paths:
        source = json.loads(segment_path.read_text(encoding="utf-8"))
        source["split"] = split_map[
            (str(source.get("task_name") or ""), int(source.get("episode_index", -1)))
        ]
        summary_path = _episode_dir(output_dir, source) / "episode_summary.json"
        if not summary_path.is_file():
            missing.append(str(segment_path))
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "PASS":
            missing.append(str(segment_path))
            continue
        summaries.append(summary)

    shard_root = output_dir / "shards"
    for split in ("train", "val"):
        for kind in ("scheduler", "component"):
            directory = shard_root / split / kind
            directory.mkdir(parents=True, exist_ok=True)
            for old in directory.glob("*.jsonl"):
                old.unlink()
    split_kind_counts: Counter[tuple[str, str]] = Counter()
    target_counts: Counter[str] = Counter()
    horizon_counts: Counter[int] = Counter()
    task_counts: Counter[str] = Counter()
    subgoal_type_counts: Counter[str] = Counter()
    episodes_manifest_path = output_dir / "episodes.jsonl"
    with episodes_manifest_path.open("w", encoding="utf-8") as manifest_handle:
        handles: dict[tuple[str, str, int], Any] = {}
        try:
            split_episode_index: Counter[str] = Counter()
            for summary in summaries:
                split = str(summary.get("split") or "train")
                if split not in {"train", "val"}:
                    split = "train"
                episode_shard_index = split_episode_index[split] // shard_size
                split_episode_index[split] += 1
                episode_dir = _episode_dir(output_dir, summary)
                for kind, filename in (
                    ("scheduler", "scheduler_train.jsonl"),
                    ("component", "component_train.jsonl"),
                ):
                    key = (split, kind, episode_shard_index)
                    if key not in handles:
                        path = (
                            shard_root
                            / split
                            / kind
                            / f"part-{episode_shard_index:05d}.jsonl"
                        )
                        handles[key] = path.open("w", encoding="utf-8")
                    with (episode_dir / filename).open(encoding="utf-8") as source_handle:
                        for line in source_handle:
                            handles[key].write(line)
                            split_kind_counts[(split, kind)] += 1
                            if kind == "scheduler":
                                row = json.loads(line)
                                target = json.loads(row["messages"][1]["content"])
                                if target.get("control") == "run_skill":
                                    target_counts[
                                        f"{target.get('next_component')}.{target.get('next_skill')}"
                                    ] += 1
                for horizon in summary.get("action_budgets") or []:
                    horizon_counts[int(horizon)] += 1
                task_counts[str(summary.get("task_name") or "")] += 1
                manifest_handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        finally:
            for handle in handles.values():
                handle.close()

    for segment_path in expected_paths:
        source = json.loads(segment_path.read_text(encoding="utf-8"))
        for segment in source.get("segments") or []:
            start = int(segment.get("frame_start", 0))
            end = int(segment.get("frame_end_exclusive", start))
            if end <= start:
                continue
            subgoal_type_counts[str(segment.get("subgoal_type") or "repair_overlay")] += 1
    dataset_info = {
        "robotwin_agent_skill_scheduler_train": {
            "file_name": "shards/train/scheduler",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": dict(OPENAI_STYLE_SHAREGPT_TAGS),
        },
        "robotwin_agent_skill_components_train": {
            "file_name": "shards/train/component",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": dict(OPENAI_STYLE_SHAREGPT_TAGS),
        },
        "robotwin_agent_skill_scheduler_val": {
            "file_name": "shards/val/scheduler",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": dict(OPENAI_STYLE_SHAREGPT_TAGS),
        },
        "robotwin_agent_skill_components_val": {
            "file_name": "shards/val/component",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": dict(OPENAI_STYLE_SHAREGPT_TAGS),
        },
    }
    engineering_dir = output_dir.parent / ENGINEERING_COVERAGE_DIR_NAME
    engineering_summary_path = engineering_dir / "summary.json"
    engineering_summary: dict[str, Any] | None = None
    if engineering_summary_path.is_file():
        engineering_summary = json.loads(
            engineering_summary_path.read_text(encoding="utf-8")
        )
        if engineering_summary.get("status") != "PASS":
            raise ValueError("engineering_coverage_dataset_not_passed")
        relative_engineering = f"../{ENGINEERING_COVERAGE_DIR_NAME}"
        dataset_info.update(
            {
                "robotwin_agent_skill_engineering_scheduler": {
                    "file_name": f"{relative_engineering}/scheduler_train.jsonl",
                    "formatting": "sharegpt",
                    "columns": {"messages": "messages", "images": "images"},
                    "tags": dict(OPENAI_STYLE_SHAREGPT_TAGS),
                },
                "robotwin_agent_skill_engineering_components": {
                    "file_name": f"{relative_engineering}/component_train.jsonl",
                    "formatting": "sharegpt",
                    "columns": {"messages": "messages", "images": "images"},
                    "tags": dict(OPENAI_STYLE_SHAREGPT_TAGS),
                },
            }
        )
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    source_episode_count = len(expected_paths)
    processed_episode_count = len(summaries)
    summary = {
        "schema": CORPUS_SCHEMA,
        "row_schema": SCHEMA,
        "status": "PASS" if processed_episode_count == source_episode_count and not missing else "FAIL",
        "dataset_root": str(dataset_root),
        "repair_ledger": str(repair_ledger_path),
        "repair_ledger_sha256": repair_ledger_sha,
        "task_instruction_repairs": str(task_instruction_repairs_path),
        "task_instruction_repairs_sha256": task_instruction_repairs_sha,
        "task_instruction_repair_count": sum(
            int(bool(summary.get("task_instruction_repair")))
            for summary in summaries
        ),
        "split_manifest": str(split_manifest_path),
        "split_manifest_sha256": split_manifest_sha,
        "source_episode_count": source_episode_count,
        "processed_episode_count": processed_episode_count,
        "missing_or_failed_episode_count": len(missing),
        "missing_or_failed_episodes": missing,
        "task_count": len(task_counts),
        "task_episode_counts": dict(sorted(task_counts.items())),
        "train_episode_count": sum(summary.get("split") == "train" for summary in summaries),
        "val_episode_count": sum(summary.get("split") == "val" for summary in summaries),
        "effective_subgoal_count": sum(
            int(summary.get("effective_subgoal_count") or 0) for summary in summaries
        ),
        "excluded_zero_frame_segment_count": sum(
            int(summary.get("excluded_zero_frame_segment_count") or 0)
            for summary in summaries
        ),
        "applied_repair_count": sum(
            int(summary.get("applied_repair_count") or 0) for summary in summaries
        ),
        "action_chunk_count": sum(
            int(summary.get("action_chunk_count") or 0) for summary in summaries
        ),
        "horizon_histogram": {
            str(key): value for key, value in sorted(horizon_counts.items())
        },
        "scheduler_target_counts": dict(sorted(target_counts.items())),
        "row_counts": {
            f"{split}_{kind}": count
            for (split, kind), count in sorted(split_kind_counts.items())
        },
        "runtime_call_count": sum(
            int(summary.get("runtime_calls") or 0) for summary in summaries
        ),
        "image_manifest_entry_count": sum(
            int(summary.get("images") or 0) for summary in summaries
        ),
        "max_prompt_chars": max(
            (int(summary.get("max_prompt_chars") or 0) for summary in summaries),
            default=0,
        ),
        "max_recent_loop_history": max(
            (
                int(summary.get("max_recent_loop_history") or 0)
                for summary in summaries
            ),
            default=0,
        ),
        "shard_size_episodes": shard_size,
        "engineering_coverage": {
            "included_in_dataset_info": engineering_summary is not None,
            "directory": str(engineering_dir),
            "status": engineering_summary.get("status")
            if engineering_summary is not None
            else None,
            "scheduler_gold_rows": engineering_summary.get("counts", {}).get(
                "scheduler_gold_rows"
            )
            if engineering_summary is not None
            else 0,
            "component_gold_rows": engineering_summary.get("counts", {}).get(
                "component_gold_rows"
            )
            if engineering_summary is not None
            else 0,
        },
    }
    (output_dir / "corpus_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        _corpus_readme(summary), encoding="utf-8"
    )
    return summary


def _corpus_readme(summary: dict[str, Any]) -> str:
    return f"""# RoboTwin Agent skill SFT corpus

This corpus expands the production-context contiguous replay collector to {summary.get('processed_episode_count')} expert episodes across {summary.get('task_count')} tasks.

- Effective positive-frame subgoals: {summary.get('effective_subgoal_count')}
- Excluded zero-frame failed source segments: {summary.get('excluded_zero_frame_segment_count')}
- Accepted repair-ledger annotations applied at collection time: {summary.get('applied_repair_count')}
- Contiguous action chunks: {summary.get('action_chunk_count')}
- Train episodes: {summary.get('train_episode_count')}
- Validation episodes: {summary.get('val_episode_count')}

Every episode runs through the production `AgentLoop`, uses production `Blackboard.compact_context()`, commits consecutive expert cursors only at `motion.execute_action`, loads the matching real expert endpoint RGB/state, verifies every chunk, keeps unfinished subgoals active, and advances only at the recorded expert segment boundary.

Training rows are sharded under `shards/train`; validation rows are kept separately under `shards/val`. Per-episode exact runtime calls, action audits, image manifests, and completion summaries live under `episodes/<task>/episodeXXXX/`.

Physical-disturbance recovery such as knocked-over or dropped objects remains intentionally excluded. The separate multi-task r8 engineering coverage set contains synthetic occlusion, unchanged-state recovery, plan-repair, stale-preflight, and diagnostic-only skill branches across ten source tasks.
"""


def _resolve_hdf5_path(segment: dict[str, Any]) -> Path:
    path = Path(str(segment.get("hdf5_path") or "")).expanduser().resolve()
    if not path.is_file():
        provenance = (
            segment.get("merge_provenance")
            if isinstance(segment.get("merge_provenance"), dict)
            else {}
        )
        path = Path(str(provenance.get("source_hdf5_path") or "")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _segment_sort_key(path: Path) -> tuple[str, int]:
    stem = path.stem
    episode_text = stem.removeprefix("episode")
    try:
        episode_index = int(episode_text)
    except ValueError:
        episode_index = 0
    return path.parent.name, episode_index


def _segment_identity(path: Path) -> tuple[str, int]:
    return _segment_sort_key(path)


def _parse_identity(value: str) -> tuple[str, int]:
    text = str(value).strip()
    if ":" not in text:
        raise ValueError(f"identity must use task:episode format: {value!r}")
    task_name, episode_text = text.rsplit(":", 1)
    task_name = task_name.strip()
    if not task_name:
        raise ValueError(f"identity has an empty task name: {value!r}")
    episode_index = int(episode_text)
    if episode_index < 0:
        raise ValueError(f"identity episode must be non-negative: {value!r}")
    return task_name, episode_index


def _episode_dir(output_dir: Path, source: dict[str, Any]) -> Path:
    return (
        output_dir
        / "episodes"
        / str(source.get("task_name") or "unknown_task")
        / f"episode{int(source.get('episode_index', 0)):04d}"
    )


def _completed_summary_matches(
    summary_path: Path,
    *,
    segment_path: Path,
    repair_ledger_sha: str,
    task_instruction_repairs_sha: str,
    split_manifest_sha: str,
) -> bool:
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        summary.get("status") == "PASS"
        and summary.get("schema") == CORPUS_SCHEMA
        and summary.get("source_segment_sha256") == _file_sha(segment_path)
        and summary.get("repair_ledger_sha256") == repair_ledger_sha
        and summary.get("task_instruction_repairs_sha256")
        == task_instruction_repairs_sha
        and summary.get("split_manifest_sha256") == split_manifest_sha
    )


if __name__ == "__main__":
    main()
