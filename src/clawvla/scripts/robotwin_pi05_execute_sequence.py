from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from clawvla.action_backends.pi05 import Pi05ActionBackend
from clawvla.artifacts import _jsonable
from clawvla.config import load_config
from clawvla.envs import RoboTwinAdapter
from clawvla.scripts.run_loop import _openpi_worker_lifecycle


@dataclass(frozen=True)
class FixedStage:
    instruction: str
    rounds: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute a fixed sequence of PI0.5 prompts in RoboTwin without a planner or verifier."
    )
    parser.add_argument("--config", default="configs/robotwin_pi05_subtasks_25k.json")
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--now-ep-num", type=int, default=0)
    parser.add_argument("--artifact-prefix", default="pi05_fixed_sequence")
    parser.add_argument(
        "--stage",
        action="append",
        required=True,
        metavar="ROUNDS::INSTRUCTION",
        help="Fixed VLA stage and repetition count, for example 3::Use both arms to grasp the bottles.",
    )
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--openpi-gpu", default="2")
    parser.add_argument("--openpi-port", type=int, default=8865)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = [_parse_stage(value) for value in args.stage]
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    config.robotwin.task_name = args.task_name
    config.robotwin.seed = args.seed
    config.robotwin.now_ep_num = args.now_ep_num
    config.robotwin.is_test = True
    config.robotwin.eval_mode = True
    config.robotwin.render_freq = 0
    config.robotwin.need_plan = True
    config.environment.task_name = args.task_name
    config.environment.seed = args.seed

    backend_config = config.metadata.get("action_backend")
    if not isinstance(backend_config, dict):
        raise ValueError("metadata.action_backend must be configured")
    runtime = backend_config.setdefault("openpi_runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("metadata.action_backend.openpi_runtime must be a mapping")
    runtime.update(
        {
            "mode": "worker",
            "auto_start": True,
            "config_path": str(config_path),
            "cuda_visible_devices": str(args.openpi_gpu),
            "host": "127.0.0.1",
            "port": int(args.openpi_port),
        }
    )

    adapter = RoboTwinAdapter(config.robotwin)
    reports: list[dict[str, Any]] = []
    setup = True
    try:
        with _openpi_worker_lifecycle(config, str(config_path), args.artifact_prefix):
            backend = Pi05ActionBackend(backend_config)
            for stage_index, stage in enumerate(stages):
                for round_index in range(stage.rounds):
                    prefix = f"{args.artifact_prefix}/stage_{stage_index:02d}/round_{round_index:02d}"
                    observation = adapter.capture_views(
                        setup=setup,
                        instruction=stage.instruction,
                        artifact_prefix=f"{prefix}/before",
                    )
                    setup = False
                    result = backend.build_action_chunk(
                        motion_goal=None,
                        world_state=None,
                        observation=observation,
                        request={
                            "motion_plan": {"vla_prompt": stage.instruction},
                            "num_steps": args.num_steps,
                            "horizon": args.horizon,
                        },
                    )
                    if not result.success or result.action_chunk is None:
                        raise RuntimeError(f"{result.status}: {';'.join(result.errors)}")
                    result.action_chunk.metadata["artifact_prefix"] = f"{prefix}/execute"
                    execution = adapter.execute_action(result.action_chunk)
                    reports.append(
                        {
                            "stage_index": stage_index,
                            "round_index": round_index,
                            "instruction": stage.instruction,
                            "backend": result.to_dict(),
                            "execution": execution,
                        }
                    )
            adapter.capture_views(
                setup=False,
                instruction=stages[-1].instruction,
                artifact_prefix=f"{args.artifact_prefix}/final",
            )
            payload = {
                "task_name": args.task_name,
                "seed": args.seed,
                "stages": [stage.__dict__ for stage in stages],
                "horizon": args.horizon,
                "num_steps": args.num_steps,
                "reports": reports,
                "task_status": adapter.task_status(),
            }
            output_path = Path(config.robotwin.artifact_dir) / args.artifact_prefix / "sequence_result.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
            print(json.dumps({"output": str(output_path), **_jsonable(payload["task_status"])}, ensure_ascii=True))
    finally:
        _close_adapter(adapter)


def _parse_stage(value: str) -> FixedStage:
    rounds_text, separator, instruction = value.partition("::")
    if not separator:
        raise ValueError(f"stage must use ROUNDS::INSTRUCTION syntax: {value!r}")
    rounds = int(rounds_text)
    instruction = instruction.strip()
    if rounds < 1:
        raise ValueError(f"stage rounds must be positive: {rounds}")
    if not instruction:
        raise ValueError("stage instruction must not be empty")
    return FixedStage(instruction=instruction, rounds=rounds)


def _close_adapter(adapter: RoboTwinAdapter) -> None:
    task_env = getattr(getattr(adapter, "session", None), "task_env", None)
    if task_env is None:
        return
    for method_name in ("close_env", "close"):
        method = getattr(task_env, method_name, None)
        if callable(method):
            method()
            return


if __name__ == "__main__":
    main()
