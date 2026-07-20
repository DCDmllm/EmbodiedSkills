from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

from clawvla.action_backends.factory import build_action_backend
from clawvla.artifacts import _jsonable
from clawvla.config import AgentConfig, load_config
from clawvla.envs import build_env_adapter
from clawvla.runtime import AgentRuntime
from clawvla.schema import MotionGoal, WorldState


INFRASTRUCTURE_FAILURE_TYPES = {"environment", "http_or_action_backend", "agent_runtime"}


@dataclass(frozen=True)
class SequenceSpec:
    sequence_id: str
    official_sequence_index: int
    official_sequence_pool_size: int
    expected_subtasks: tuple[str, ...]
    expected_initial_state: dict[str, Any] | None
    seeds: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate X-VLA baseline and/or the full Agent on persistent CALVIN task sequences."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Override environment.params.dataset_path without changing the source config.",
    )
    parser.add_argument(
        "--validation-dir",
        default=None,
        help="Override environment.params.validation_dir; defaults to DATASET_PATH/validation when omitted.",
    )
    parser.add_argument("--output-dir", default="runs/eval/calvin_sequence")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runner", action="append", choices=["baseline", "agent"], default=[])
    parser.add_argument("--sequence-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-env-steps-per-subtask", type=int, default=120)
    parser.add_argument("--max-agent-steps", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_positive_args(args)
    manifest_path = Path(args.manifest).resolve()
    specs, manifest_metadata = load_sequence_manifest(manifest_path)
    specs = _filter_specs(specs, args.sequence_id, args.limit)
    runners = tuple(args.runner or ["baseline"])
    run_id = args.run_id or time.strftime("calvin_sequence_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    plan = build_run_plan(specs, runners)
    _write_json(
        run_dir / "run_config.json",
        {
            "config": str(Path(args.config).resolve()),
            "manifest": str(manifest_path),
            "manifest_metadata": manifest_metadata,
            "dataset_path": str(Path(args.dataset_path).expanduser().resolve()) if args.dataset_path else None,
            "validation_dir": (
                str(Path(args.validation_dir).expanduser().resolve())
                if args.validation_dir
                else (
                    str((Path(args.dataset_path).expanduser().resolve() / "validation"))
                    if args.dataset_path
                    else None
                )
            ),
            "runners": list(runners),
            "max_env_steps_per_subtask": args.max_env_steps_per_subtask,
            "max_agent_steps": args.max_agent_steps,
            "horizon": args.horizon,
            "inference_steps": args.inference_steps,
            "resume": args.resume,
        },
    )
    _write_json(run_dir / "plan.json", {"jobs": plan, "job_count": len(plan)})
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "run_dir": str(run_dir), "job_count": len(plan)}, ensure_ascii=False))
        return

    base_config = override_calvin_dataset_paths(
        load_config(args.config),
        dataset_path=args.dataset_path,
        validation_dir=args.validation_dir,
    )
    results_path = run_dir / "sequence_results.jsonl"
    completed = load_completed_results(results_path) if args.resume else {}
    infrastructure_failed = False
    for job in plan:
        key = result_key(job["runner"], job["sequence_id"], job["seed"])
        if key in completed:
            continue
        spec = next(item for item in specs if item.sequence_id == job["sequence_id"])
        started = time.perf_counter()
        try:
            record = run_sequence_job(
                base_config,
                spec,
                runner=job["runner"],
                seed=job["seed"],
                artifact_root=run_dir / "artifacts",
                max_env_steps_per_subtask=args.max_env_steps_per_subtask,
                max_agent_steps=args.max_agent_steps,
                horizon=args.horizon,
                inference_steps=args.inference_steps,
            )
        except Exception as exc:
            failure_type = _exception_failure_type(exc)
            record = {
                **job,
                "status": "sequence_infrastructure_failed",
                "success": False,
                "completed_subtasks": 0,
                "sequence_length": len(spec.expected_subtasks),
                "failure_type": failure_type,
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "subtasks": [],
            }
        record["elapsed_seconds"] = time.perf_counter() - started
        record = _jsonable(record)
        _append_jsonl(results_path, record)
        completed[key] = record
        if record.get("failure_type") in INFRASTRUCTURE_FAILURE_TYPES:
            infrastructure_failed = True
        write_sequence_summary(run_dir, list(completed.values()))
    write_sequence_summary(run_dir, list(completed.values()))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"status": "completed", "run_dir": str(run_dir), **summary["overall"]}, ensure_ascii=False, indent=2))
    if infrastructure_failed:
        raise SystemExit(1)


def load_sequence_manifest(path: Path) -> tuple[list[SequenceSpec], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("calvin_sequence_manifest_must_be_object")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"calvin_sequence_manifest_schema_unsupported:{payload.get('schema_version')}")
    default_seeds = _integer_list(payload.get("default_seeds", [0]), "default_seeds")
    official_pool_size = int(payload.get("official_sequence_pool_size", 0))
    if official_pool_size <= 0:
        raise ValueError(
            f"calvin_official_sequence_pool_size_must_be_positive:{official_pool_size}"
        )
    raw_sequences = payload.get("sequences")
    if not isinstance(raw_sequences, list) or not raw_sequences:
        raise ValueError("calvin_sequence_manifest_sequences_empty")
    specs: list[SequenceSpec] = []
    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    for raw in raw_sequences:
        if not isinstance(raw, dict):
            raise TypeError("calvin_sequence_manifest_entry_must_be_object")
        sequence_id = str(raw.get("sequence_id") or "").strip()
        if not sequence_id:
            raise ValueError("calvin_sequence_id_missing")
        if sequence_id in seen_ids:
            raise ValueError(f"calvin_sequence_id_duplicate:{sequence_id}")
        seen_ids.add(sequence_id)
        official_index = raw.get("official_sequence_index")
        if official_index is None:
            raise ValueError(f"calvin_official_sequence_index_missing:{sequence_id}")
        official_index = int(official_index)
        if official_index < 0:
            raise ValueError(f"calvin_official_sequence_index_invalid:{sequence_id}:{official_index}")
        if official_index >= official_pool_size:
            raise ValueError(
                f"calvin_official_sequence_index_out_of_range:{sequence_id}:"
                f"index={official_index}:pool_size={official_pool_size}"
            )
        if official_index in seen_indices:
            raise ValueError(f"calvin_official_sequence_index_duplicate:{official_index}")
        seen_indices.add(official_index)
        subtasks = tuple(str(item).strip() for item in raw.get("expected_subtasks", []) if str(item).strip())
        if len(subtasks) != 5:
            raise ValueError(f"calvin_expected_five_subtasks:{sequence_id}:found={len(subtasks)}")
        initial_state = raw.get("expected_initial_state")
        if initial_state is not None and not isinstance(initial_state, dict):
            raise TypeError(f"calvin_expected_initial_state_must_be_object:{sequence_id}")
        seeds = tuple(_integer_list(raw.get("seeds", default_seeds), f"seeds:{sequence_id}"))
        specs.append(
            SequenceSpec(
                sequence_id=sequence_id,
                official_sequence_index=official_index,
                official_sequence_pool_size=official_pool_size,
                expected_subtasks=subtasks,
                expected_initial_state=dict(initial_state) if isinstance(initial_state, dict) else None,
                seeds=seeds,
            )
        )
    metadata = {key: value for key, value in payload.items() if key != "sequences"}
    return specs, metadata


def build_run_plan(specs: list[SequenceSpec], runners: tuple[str, ...]) -> list[dict[str, Any]]:
    jobs = []
    for runner in runners:
        for spec in specs:
            for seed in spec.seeds:
                jobs.append(
                    {
                        "runner": runner,
                        "sequence_id": spec.sequence_id,
                        "official_sequence_index": spec.official_sequence_index,
                        "official_sequence_pool_size": spec.official_sequence_pool_size,
                        "seed": seed,
                    }
                )
    return jobs


def override_calvin_dataset_paths(
    base_config: AgentConfig,
    *,
    dataset_path: str | None,
    validation_dir: str | None,
) -> AgentConfig:
    """Return an evaluation config with explicit dataset paths and no source-config mutation."""
    if dataset_path is None and validation_dir is None:
        return base_config
    config = copy.deepcopy(base_config)
    params = dict(config.environment.params)
    resolved_dataset = Path(dataset_path).expanduser().resolve() if dataset_path else None
    if resolved_dataset is not None:
        params["dataset_path"] = str(resolved_dataset)
    resolved_validation = (
        Path(validation_dir).expanduser().resolve()
        if validation_dir
        else resolved_dataset / "validation"
        if resolved_dataset is not None
        else None
    )
    if resolved_validation is not None:
        params["validation_dir"] = str(resolved_validation)
    config.environment.params = params
    return config


def run_sequence_job(
    base_config: AgentConfig,
    spec: SequenceSpec,
    *,
    runner: str,
    seed: int,
    artifact_root: Path,
    max_env_steps_per_subtask: int,
    max_agent_steps: int,
    horizon: int | None,
    inference_steps: int | None,
) -> dict[str, Any]:
    config = config_for_sequence(base_config, spec, seed, artifact_root)
    adapter = build_env_adapter(config)
    try:
        if runner == "baseline":
            record = _run_baseline_sequence(
                config,
                adapter,
                spec,
                max_env_steps_per_subtask=max_env_steps_per_subtask,
                horizon=horizon,
                inference_steps=inference_steps,
            )
        elif runner == "agent":
            record = _run_agent_sequence(config, adapter, spec, max_agent_steps=max_agent_steps)
        else:
            raise ValueError(f"calvin_sequence_runner_unsupported:{runner}")
        return {
            "runner": runner,
            "sequence_id": spec.sequence_id,
            "official_sequence_index": spec.official_sequence_index,
            "official_sequence_pool_size": spec.official_sequence_pool_size,
            "seed": seed,
            **record,
        }
    finally:
        adapter.close()


def config_for_sequence(
    base_config: AgentConfig,
    spec: SequenceSpec,
    seed: int,
    artifact_root: Path,
) -> AgentConfig:
    config = copy.deepcopy(base_config)
    config.environment.seed = int(seed)
    config.environment.artifact_dir = str(artifact_root)
    params = dict(config.environment.params)
    for key in ("initial_state", "eval_sequence", "subtask"):
        params.pop(key, None)
    params["sequence_index"] = int(spec.official_sequence_index)
    params["sequence_pool_size"] = int(spec.official_sequence_pool_size)
    params["subtask_index"] = 0
    config.environment.params = params
    return config


def _run_baseline_sequence(
    config: AgentConfig,
    adapter: Any,
    spec: SequenceSpec,
    *,
    max_env_steps_per_subtask: int,
    horizon: int | None,
    inference_steps: int | None,
) -> dict[str, Any]:
    backend = build_action_backend(config)
    observation = adapter.capture_views(
        setup=True,
        artifact_prefix=f"{spec.sequence_id}/seed_{int(config.environment.seed or 0):04d}/subtask_00/reset",
    )
    resolved = validate_resolved_sequence(adapter, spec)
    subtask_records: list[dict[str, Any]] = []
    failure_type = None
    failure_reason = None
    for subtask_index, subtask in enumerate(resolved["subtasks"]):
        if subtask_index > 0:
            adapter.advance_sequence_subtask()
            observation = adapter.capture_views(
                artifact_prefix=(
                    f"{spec.sequence_id}/seed_{int(config.environment.seed or 0):04d}/"
                    f"subtask_{subtask_index:02d}/start"
                ),
                instruction=adapter.task_language(),
            )
        instruction = str(adapter.task_language() or "").strip()
        start_step = int(adapter.step_count)
        chunk_records: list[dict[str, Any]] = []
        while int(adapter.step_count) - start_step < max_env_steps_per_subtask:
            remaining = max_env_steps_per_subtask - (int(adapter.step_count) - start_step)
            request: dict[str, Any] = {
                "horizon": min(int(horizon or backend.action_spec()["horizon"]), remaining),
                "motion_plan": {"status": "image_grounded_motion_plan_built", "vla_prompt": instruction},
            }
            if inference_steps is not None:
                request["inference_steps"] = int(inference_steps)
            result = backend.build_action_chunk(
                MotionGoal(skill="act", motion_hint=instruction),
                WorldState(task_instruction=instruction),
                observation,
                request,
            )
            if not result.success or result.action_chunk is None:
                failure_type = "http_or_action_backend"
                failure_reason = result.errors[0] if result.errors else result.status
                break
            result.action_chunk.metadata["artifact_prefix"] = (
                f"{spec.sequence_id}/seed_{int(config.environment.seed or 0):04d}/"
                f"subtask_{subtask_index:02d}/chunk_{len(chunk_records):03d}"
            )
            execution = adapter.execute_action(result.action_chunk)
            chunk_records.append(
                {
                    "chunk_index": len(chunk_records),
                    "action_count": len(result.action_chunk.commands),
                    "execution_status": execution.get("status"),
                    "executed_steps": execution.get("executed_steps"),
                    "success": execution.get("success"),
                }
            )
            if execution.get("status") != "action_executed":
                failure_type = "environment"
                failure_reason = str(execution.get("reason") or execution.get("status"))
                break
            observation = adapter.last_observation
            if execution.get("success"):
                break
        status = adapter.task_status()
        success = status.get("success") is True
        subtask_records.append(
            {
                "subtask_index": subtask_index,
                "subtask": subtask,
                "instruction": instruction,
                "success": success,
                "environment_steps": int(adapter.step_count) - start_step,
                "chunk_count": len(chunk_records),
                "stalled_loop": False,
                "premature_finish": False,
                "chunks": chunk_records,
                "task_status": _compact_task_status(status),
            }
        )
        if not success:
            if failure_type is None:
                failure_type = "task_failure"
                failure_reason = "subtask_step_budget_exhausted"
            break
    return _sequence_record(resolved, subtask_records, failure_type, failure_reason)


def _run_agent_sequence(
    config: AgentConfig,
    adapter: Any,
    spec: SequenceSpec,
    *,
    max_agent_steps: int,
) -> dict[str, Any]:
    first_observation = adapter.capture_views(
        setup=True,
        artifact_prefix=f"{spec.sequence_id}/seed_{int(config.environment.seed or 0):04d}/subtask_00/reset",
    )
    resolved = validate_resolved_sequence(adapter, spec)
    subtask_records: list[dict[str, Any]] = []
    failure_type = None
    failure_reason = None
    observation = first_observation
    for subtask_index, subtask in enumerate(resolved["subtasks"]):
        if subtask_index > 0:
            adapter.advance_sequence_subtask()
            observation = adapter.capture_views(
                artifact_prefix=(
                    f"{spec.sequence_id}/seed_{int(config.environment.seed or 0):04d}/"
                    f"subtask_{subtask_index:02d}/start"
                ),
                instruction=adapter.task_language(),
            )
        instruction = str(adapter.task_language() or "").strip()
        runtime = AgentRuntime(config)
        runtime.blackboard.task_instruction = instruction
        runtime.blackboard.write("env_adapter", adapter)
        runtime.blackboard.write("run_environment", True)
        runtime.blackboard.write(
            "artifact_prefix",
            f"{spec.sequence_id}/seed_{int(config.environment.seed or 0):04d}/subtask_{subtask_index:02d}",
        )
        runtime.blackboard.write("observation", observation)
        start_step = int(adapter.step_count)
        loop_result = runtime.run_loop(max_steps=max_agent_steps, initial_stage="observe")
        task_status = adapter.task_status()
        success = task_status.get("success") is True
        action_chunk_count = sum(
            step.decision.next_component == "motion"
            and step.decision.next_skill == "execute_action"
            for step in loop_result.steps
        )
        stalled_loop = loop_result.status in {
            "stalled_loop",
            "max_steps_reached",
            "max_steps_reached_with_failures",
        }
        premature_finish = loop_result.status == "finished" and not success
        subtask_records.append(
            {
                "subtask_index": subtask_index,
                "subtask": subtask,
                "instruction": instruction,
                "success": success,
                "environment_steps": int(adapter.step_count) - start_step,
                "chunk_count": action_chunk_count,
                "agent_steps": len(loop_result.steps),
                "loop_status": loop_result.status,
                "loop_reason": loop_result.reason,
                "stalled_loop": stalled_loop,
                "premature_finish": premature_finish,
                "task_status": _compact_task_status(task_status),
            }
        )
        if not success:
            if premature_finish:
                failure_type = "premature_finish"
            elif stalled_loop:
                failure_type = "stalled_loop"
            elif loop_result.status.endswith("failures"):
                failure_type = "agent_runtime"
            else:
                failure_type = "task_failure"
            failure_reason = loop_result.reason or loop_result.status
            break
    return _sequence_record(resolved, subtask_records, failure_type, failure_reason)


def validate_resolved_sequence(adapter: Any, spec: SequenceSpec) -> dict[str, Any]:
    actual_pool_size = int(getattr(adapter, "sequence_pool_size", 0))
    if actual_pool_size != spec.official_sequence_pool_size:
        raise ValueError(
            f"calvin_official_sequence_pool_drift:{spec.sequence_id}:"
            f"expected={spec.official_sequence_pool_size}:actual={actual_pool_size}"
        )
    subtasks = tuple(str(item) for item in (adapter.eval_sequence or ()))
    if subtasks != spec.expected_subtasks:
        raise ValueError(
            f"calvin_official_sequence_drift:{spec.sequence_id}:"
            f"expected={list(spec.expected_subtasks)}:actual={list(subtasks)}"
        )
    initial_state = dict(adapter.initial_state or {})
    if spec.expected_initial_state is not None and initial_state != spec.expected_initial_state:
        raise ValueError(
            f"calvin_official_initial_state_drift:{spec.sequence_id}:"
            f"expected={spec.expected_initial_state}:actual={initial_state}"
        )
    return {
        "official_sequence_index": spec.official_sequence_index,
        "official_sequence_pool_size": spec.official_sequence_pool_size,
        "subtasks": list(subtasks),
        "initial_state": initial_state,
    }


def _sequence_record(
    resolved: dict[str, Any],
    subtasks: list[dict[str, Any]],
    failure_type: str | None,
    failure_reason: str | None,
) -> dict[str, Any]:
    completed = sum(1 for item in subtasks if item.get("success") is True)
    sequence_length = len(resolved["subtasks"])
    success = completed == sequence_length
    return {
        "status": "sequence_succeeded" if success else "sequence_failed",
        "success": success,
        "completed_subtasks": completed,
        "sequence_length": sequence_length,
        "failure_type": None if success else failure_type,
        "failure_reason": None if success else failure_reason,
        "resolved_sequence": resolved,
        "subtasks": subtasks,
    }


def sequence_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {
            "sequence_count": 0,
            "full_sequence_success_rate": 0.0,
            "average_completed_length": 0.0,
            "average_environment_steps": 0.0,
            "average_action_chunks": 0.0,
            "completion_rates": {str(index): 0.0 for index in range(1, 6)},
            "stalled_loop_count": 0,
            "stalled_loop_rate": 0.0,
            "premature_finish_count": 0,
            "premature_finish_rate": 0.0,
            "environment_or_http_error_count": 0,
            "failure_type_counts": {},
        }
    completed = [int(record.get("completed_subtasks", 0)) for record in records]
    lengths = [int(record.get("sequence_length", 0)) for record in records]
    rates = {}
    for index in range(1, 6):
        eligible = [value for value, length in zip(completed, lengths) if length >= index]
        rates[str(index)] = sum(value >= index for value in eligible) / len(eligible) if eligible else 0.0
    failure_counts = Counter(str(record.get("failure_type")) for record in records if record.get("failure_type"))
    environment_steps = [
        sum(int(subtask.get("environment_steps", 0)) for subtask in record.get("subtasks", []))
        for record in records
    ]
    action_chunks = [
        sum(int(subtask.get("chunk_count", 0)) for subtask in record.get("subtasks", []))
        for record in records
    ]
    stalled_count = sum(
        any(bool(subtask.get("stalled_loop")) for subtask in record.get("subtasks", []))
        for record in records
    )
    premature_finish_count = sum(
        any(bool(subtask.get("premature_finish")) for subtask in record.get("subtasks", []))
        for record in records
    )
    return {
        "sequence_count": total,
        "full_sequence_success_count": sum(bool(record.get("success")) for record in records),
        "full_sequence_success_rate": sum(bool(record.get("success")) for record in records) / total,
        "average_completed_length": sum(completed) / total,
        "average_environment_steps": sum(environment_steps) / total,
        "average_action_chunks": sum(action_chunks) / total,
        "completion_rates": rates,
        "stalled_loop_count": stalled_count,
        "stalled_loop_rate": stalled_count / total,
        "premature_finish_count": premature_finish_count,
        "premature_finish_rate": premature_finish_count / total,
        "environment_or_http_error_count": sum(
            record.get("failure_type") in INFRASTRUCTURE_FAILURE_TYPES for record in records
        ),
        "failure_type_counts": dict(sorted(failure_counts.items())),
    }


def write_sequence_summary(run_dir: Path, records: list[dict[str, Any]]) -> None:
    by_runner: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_runner.setdefault(str(record.get("runner")), []).append(record)
    payload = {
        "overall": sequence_metrics(records),
        "by_runner": {runner: sequence_metrics(items) for runner, items in sorted(by_runner.items())},
    }
    _write_json(run_dir / "summary.json", payload)
    with (run_dir / "sequence_results.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "runner",
            "sequence_id",
            "official_sequence_index",
            "official_sequence_pool_size",
            "seed",
            "status",
            "success",
            "completed_subtasks",
            "sequence_length",
            "failure_type",
            "failure_reason",
            "elapsed_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: result_key(item.get("runner"), item.get("sequence_id"), item.get("seed"))):
            writer.writerow({name: record.get(name) for name in fieldnames})


def load_completed_results(path: Path) -> dict[str, dict[str, Any]]:
    completed = {}
    if not path.exists():
        return completed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise TypeError(f"calvin_sequence_result_not_object:line={line_number}")
        completed[result_key(record.get("runner"), record.get("sequence_id"), record.get("seed"))] = record
    return completed


def result_key(runner: object, sequence_id: object, seed: object) -> str:
    return f"{runner}|{sequence_id}|{int(seed)}"


def _filter_specs(specs: list[SequenceSpec], selected: list[str], limit: int | None) -> list[SequenceSpec]:
    if selected:
        selected_set = set(selected)
        missing = selected_set - {spec.sequence_id for spec in specs}
        if missing:
            raise ValueError(f"calvin_sequence_ids_unknown:{sorted(missing)}")
        specs = [spec for spec in specs if spec.sequence_id in selected_set]
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"calvin_sequence_limit_must_be_positive:{limit}")
        specs = specs[:limit]
    return specs


def _integer_list(value: object, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"calvin_sequence_{name}_must_be_nonempty_list")
    items = [int(item) for item in value]
    if len(items) != len(set(items)):
        raise ValueError(f"calvin_sequence_{name}_must_be_unique")
    return items


def _validate_positive_args(args: argparse.Namespace) -> None:
    for name in ("max_env_steps_per_subtask", "max_agent_steps"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"calvin_sequence_{name}_must_be_positive")


def _exception_failure_type(exc: Exception) -> str:
    message = str(exc)
    if message.startswith("calvin_official_"):
        return "protocol"
    return "environment"


def _compact_task_status(status: dict[str, Any]) -> dict[str, Any]:
    """Keep benchmark records small without discarding scoring evidence."""
    keys = (
        "backend",
        "task_name",
        "task_language",
        "subtask",
        "sequence_index",
        "sequence_pool_size",
        "subtask_index",
        "success",
        "done",
        "step_count",
        "reward",
    )
    return {key: status.get(key) for key in keys if key in status}
    for name in ("horizon", "inference_steps"):
        value = getattr(args, name)
        if value is not None and int(value) <= 0:
            raise ValueError(f"calvin_sequence_{name}_must_be_positive")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
