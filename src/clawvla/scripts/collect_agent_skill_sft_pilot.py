from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from PIL import Image

from clawvla.action_backends.base import ActionBackendResult
from clawvla.agent_loop import AgentLoop, AgentLoopConfig
from clawvla.blackboard import MAX_CONTEXT_LOOP_HISTORY, Blackboard
from clawvla.components.factory import build_component_registry
from clawvla.config import load_config
from clawvla.loop_types import LoopDecision, LoopStepRecord
from clawvla.phase_policy import DEFAULT_ALLOWED_SKILLS
from clawvla.runtime import AgentRuntime
from clawvla.schema import (
    ActionChunk,
    CameraView,
    ObservationBundle,
    RobotArmState,
    SafetyReport,
    SkillResult,
    TaskPlan,
)
from clawvla.task_semantics import task_requires_target


SCHEMA = "clawvla-agent-skill-sft-contiguous-v3"
FULL_TASK_SCENARIO = "direct_vla_full_task_all_subgoals_complete"
EXECUTION_FAILURE_RECOVERY_SCENARIOS = (
    "direct_vla_execution_failure_recover_retry",
    "direct_vla_controller_timeout_recover_retry",
    "direct_vla_controller_disconnect_recover_retry",
)
OPENAI_STYLE_SHAREGPT_TAGS = {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
    "observation_tag": "observation",
    "function_tag": "function",
    "system_tag": "system",
}
# RoboTwin Camera.get_config() inserts the two wrist cameras first, followed by
# the embodiment static-camera order (head, then front for aloha-agilex).
CAMERA_ROLES = ("left_camera", "right_camera", "head_camera", "front_camera")


@dataclass
class TraceCall:
    component: str
    call_index: int
    event_index: int
    runtime_messages: list[dict[str, Any]]
    image_paths: list[str]
    target: dict[str, Any]
    supervision: str
    teacher_rule: str
    context: dict[str, Any]
    teacher_evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceSequence:
    next_index: int = 0

    def claim(self) -> int:
        value = self.next_index
        self.next_index += 1
        return value


@dataclass
class TeacherTraceRuntime:
    component: str
    scenario: str
    oracle_candidates: list[dict[str, Any]]
    sequence: TraceSequence
    oracle_plan: dict[str, Any] = field(default_factory=dict)
    verification_outcome: str | None = None
    verification_outcomes: list[str] = field(default_factory=list)
    recovery_directive: dict[str, Any] = field(default_factory=dict)
    action_budgets: list[int] = field(default_factory=list)
    optional_scheduler_decisions: list[dict[str, Any]] = field(default_factory=list)
    replay_timeline: ExpertSubtaskTimeline | None = None
    wrong_scheduler_call_indexes: set[int] = field(default_factory=set)
    calls: list[TraceCall] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return True

    def generate_text(
        self,
        messages: list[dict[str, Any]],
        image_paths: list[str] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        del max_new_tokens, temperature
        context = _context_from_runtime_messages(messages)
        call_index = len(self.calls)
        event_index = self.sequence.claim()
        prompt = _runtime_prompt_text(messages)
        if self.component == "scheduler" and prompt.startswith("Build a complete ordered manipulation subgoal plan"):
            if not self.oracle_plan:
                raise RuntimeError(f"teacher_missing_oracle_plan:{self.scenario}")
            target = deepcopy(self.oracle_plan)
            supervision = "gold"
            teacher_rule = "expert_subtask_segments:oracle_task_plan"
        elif self.component == "scheduler":
            target, supervision, teacher_rule = self._scheduler_target(context, call_index)
        elif self.component == "vision":
            target, supervision, teacher_rule = self._vision_target(messages)
        elif self.component == "verifier":
            target, supervision, teacher_rule = self._verifier_target(messages)
        elif self.component == "recovery":
            target, supervision, teacher_rule = self._recovery_target(messages)
        else:
            raise RuntimeError(f"unexpected_teacher_component:{self.component}")
        teacher_evidence: dict[str, Any] = {}
        if (
            self.component == "verifier"
            and self.replay_timeline is not None
            and isinstance(self.replay_timeline.last_execution, dict)
        ):
            teacher_evidence = {
                "source": "expert_contiguous_replay_private_supervision",
                "replay_execution": deepcopy(self.replay_timeline.last_execution),
                "all_expert_segments_complete": all(
                    self.replay_timeline.current_complete(subgoal_id)
                    for subgoal_id in self.replay_timeline.commands_by_subgoal
                ),
            }
        self.calls.append(
            TraceCall(
                component=self.component,
                call_index=call_index,
                event_index=event_index,
                runtime_messages=deepcopy(messages),
                image_paths=list(image_paths or []),
                target=deepcopy(target),
                supervision=supervision,
                teacher_rule=teacher_rule,
                context=context,
                teacher_evidence=teacher_evidence,
            )
        )
        return json.dumps(target, ensure_ascii=False)

    def _scheduler_target(
        self, context: dict[str, Any], call_index: int
    ) -> tuple[dict[str, Any], str, str]:
        runtime_state = context.get("runtime_state") if isinstance(context, dict) else None
        runtime_state = runtime_state if isinstance(runtime_state, dict) else {}
        current_stage = str(context.get("current_stage") or "observe")
        if call_index in self.wrong_scheduler_call_indexes:
            return (
                _canonical_loop_decision(
                    {
                        "control": "run_skill",
                        "stage": current_stage,
                        "next_component": "state",
                        "next_skill": "update_world_state",
                        "payload": {},
                        "reason": "injected_wrong_update_before_perception",
                    },
                    state_summary="An observation exists but perception has not been produced.",
                    expected_result="The runtime should reject this premature world-state update.",
                ),
                "rejected",
                "fault_injection:premature_update_world_state",
            )
        required = runtime_state.get("next_required_decision")
        if current_stage == "preflight" and not isinstance(required, dict):
            blackboard = context.get("blackboard") if isinstance(context, dict) else None
            blackboard = blackboard if isinstance(blackboard, dict) else {}
            report = blackboard.get("preflight_report")
            report = report if isinstance(report, dict) else {}
            preflight_errors = {str(item) for item in report.get("errors") or []}
            plan_errors = {
                "missing_task_plan",
                "missing_current_subgoal",
                "current_subgoal_mismatch_task_plan",
                "missing_source_candidate",
                "source_candidate_not_found",
            }
            if preflight_errors & plan_errors:
                required = {
                    "control": "run_skill",
                    "stage": "preflight",
                    "next_component": "scheduler",
                    "next_skill": "repair_stage_transition",
                    "payload": {
                        "target_stage": "plan",
                        "reason": "preflight_invalid_plan_state",
                    },
                    "reason": "preflight_invalid_plan_state",
                }
        if current_stage == "recover" and not isinstance(required, dict):
            blackboard = context.get("blackboard") if isinstance(context, dict) else None
            blackboard = blackboard if isinstance(blackboard, dict) else {}
            if not blackboard.get("last_recovery_directive"):
                required = {
                    "control": "run_skill",
                    "stage": "recover",
                    "next_component": "recovery",
                    "next_skill": "decide_recovery",
                    "payload": {},
                    "reason": "missing_recovery_directive",
                }
            elif not blackboard.get("last_retry_request"):
                required = {
                    "control": "run_skill",
                    "stage": "recover",
                    "next_component": "recovery",
                    "next_skill": "build_retry_request",
                    "payload": {},
                    "reason": "missing_retry_request",
                }
            else:
                retry_stage = str(blackboard["last_retry_request"].get("stage") or "preflight")
                required = {
                    "control": "run_skill",
                    "stage": "recover",
                    "next_component": "scheduler",
                    "next_skill": "repair_stage_transition",
                    "payload": {
                        "target_stage": retry_stage,
                        "reason": "recovery_retry_request_ready",
                    },
                    "reason": "recovery_retry_request_ready",
                }
        if not isinstance(required, dict):
            raise RuntimeError(
                f"teacher_missing_next_required_decision:{self.scenario}:scheduler_call={call_index}"
            )
        required = deepcopy(required)
        if self.optional_scheduler_decisions:
            optional = self.optional_scheduler_decisions[0]
            required_identity = (
                str(required.get("next_component") or ""),
                str(required.get("next_skill") or required.get("control") or ""),
            )
            expected_identity = (
                str(optional.get("when_required_component") or ""),
                str(optional.get("when_required_skill") or ""),
            )
            expected_stage = optional.get("when_stage")
            if required_identity == expected_identity and (
                expected_stage is None or str(expected_stage) == current_stage
            ):
                self.optional_scheduler_decisions.pop(0)
                injected = {
                    "control": "run_skill",
                    "stage": current_stage,
                    "next_component": str(optional["next_component"]),
                    "next_skill": str(optional["next_skill"]),
                    "payload": deepcopy(optional.get("payload") or {}),
                    "reason": str(optional.get("reason") or "optional_skill_evidence_requested"),
                }
                return (
                    _canonical_loop_decision(
                        injected,
                        state_summary=_teacher_state_summary(
                            runtime_state, str(injected["reason"])
                        ),
                        expected_result=_teacher_expected_result(injected),
                    ),
                    str(optional.get("supervision") or "gold"),
                    str(
                        optional.get("teacher_rule")
                        or f"production_optional_skill:{injected['next_component']}.{injected['next_skill']}"
                    ),
                )
        teacher_rule_suffix = ""
        if (
            required.get("control") == "run_skill"
            and required.get("next_component") == "motion"
            and required.get("next_skill") == "emit_action_chunk"
        ):
            emitted_chunks = sum(
                call.target.get("next_component") == "motion"
                and call.target.get("next_skill") == "emit_action_chunk"
                for call in self.calls
            )
            if not self.action_budgets:
                raise RuntimeError(f"teacher_missing_action_budgets:{self.scenario}")
            if emitted_chunks >= len(self.action_budgets):
                raise RuntimeError(
                    f"teacher_exhausted_action_budgets:{self.scenario}:"
                    f"emitted={emitted_chunks}:available={len(self.action_budgets)}"
                )
            horizon = int(self.action_budgets[emitted_chunks])
            if horizon < 15 or horizon > 32:
                raise RuntimeError(
                    f"teacher_action_budget_out_of_range:{self.scenario}:{horizon}"
                )
            required["payload"] = {**dict(required.get("payload") or {}), "horizon": horizon}
            teacher_rule_suffix = f":expert_contiguous_horizon_{horizon}"
        reason = str(required.get("reason") or "runtime_next_required_decision")
        return (
            _canonical_loop_decision(
                required,
                state_summary=_teacher_state_summary(runtime_state, reason),
                expected_result=_teacher_expected_result(required),
            ),
            "gold",
            f"runtime_state.next_required_decision:{reason}{teacher_rule_suffix}",
        )

    def _vision_target(
        self, messages: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], str, str]:
        prompt = _runtime_prompt_text(messages)
        target_candidate_id = (
            "C2"
            if len(self.oracle_candidates) > 1
            else None
        )
        if prompt.startswith("Detect task-relevant visual candidates"):
            return (
                {
                    "candidates": deepcopy(self.oracle_candidates),
                    "uncertainty": {"needs_reobserve": False, "reasons": []},
                },
                "gold",
                "expert_episode_metadata:oracle_perception",
            )
        if prompt.startswith("Bind the task source object"):
            return (
                {
                    "candidates": deepcopy(self.oracle_candidates),
                    "source_candidate_id": "C1",
                    "target_candidate_id": target_candidate_id,
                    "uncertainty": {"needs_reobserve": False, "reasons": []},
                },
                "gold",
                "expert_episode_metadata:oracle_source_target_binding",
            )
        if prompt.startswith("Estimate whether current visual state is reliable enough for scheduling"):
            return (
                {
                    "needs_reobserve": False,
                    "reasons": [],
                    "notes": [
                        "All four replayed expert RGB views are present and the semantic candidates remain visible."
                    ],
                },
                "gold",
                "expert_replay_views:visible_candidates_low_uncertainty",
            )
        raise RuntimeError(f"unrecognized_vision_teacher_prompt:{prompt[:120]}")

    def _verifier_target(
        self, messages: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], str, str]:
        prompt = _runtime_prompt_text(messages)
        if not prompt.startswith("Verify only the current robot manipulation subgoal"):
            raise RuntimeError(f"unrecognized_verifier_teacher_prompt:{prompt[:120]}")
        context = _context_from_runtime_messages(messages)
        outcome = (
            self.verification_outcomes[len(self.calls)]
            if len(self.calls) < len(self.verification_outcomes)
            else self.verification_outcome
        )
        timeline_execution = (
            dict(self.replay_timeline.last_execution)
            if self.replay_timeline is not None
            and isinstance(self.replay_timeline.last_execution, dict)
            else None
        )
        if outcome == "timeline":
            if timeline_execution is None:
                raise RuntimeError(f"teacher_missing_timeline_execution:{self.scenario}")
            outcome = (
                "complete"
                if bool(timeline_execution.get("expert_segment_complete_after_chunk"))
                else "incomplete"
            )
        current_subgoal = context.get("current_subgoal")
        current_subgoal = current_subgoal if isinstance(current_subgoal, dict) else {}
        criteria = current_subgoal.get("completion_criteria")
        criteria = criteria if isinstance(criteria, dict) else {}
        criterion = str(criteria.get("natural_language") or "the current subgoal completion criterion").strip()
        if outcome == "complete":
            notes = [f"The fresh verification images satisfy: {criterion}"]
            teacher_rule = "expert_segment_end:subgoal_complete"
            if timeline_execution is not None:
                teacher_rule = "expert_contiguous_segment_end:subgoal_complete"
            task_success = bool(
                self.replay_timeline is not None
                and all(
                    self.replay_timeline.current_complete(subgoal_id)
                    for subgoal_id in self.replay_timeline.commands_by_subgoal
                )
            )
            return (
                {
                    "subgoal_success": True,
                    "task_success": task_success,
                    "partial_progress": True,
                    "failure_type": "none",
                    "progress_score": 1.0,
                    "should_reobserve": False,
                    "next_action": "advance_subgoal",
                    "notes": notes,
                },
                "gold",
                teacher_rule,
            )
        if outcome == "incomplete":
            progress_score = 0.5
            notes = [f"The fresh verification images do not yet satisfy: {criterion}"]
            teacher_rule = "expert_segment_prefix:subgoal_incomplete"
            if timeline_execution is not None:
                end = int(timeline_execution.get("expert_cursor_end_exclusive") or 0)
                length = max(1, int(timeline_execution.get("expert_segment_length") or 1))
                # The private expert cursor determines the label, but is never
                # exposed in the model answer.  At deployment the verifier has
                # fresh RGB evidence, not an expert-replay cursor.
                progress_score = round(min(0.99, max(0.01, end / length)), 3)
                teacher_rule = "expert_contiguous_segment_prefix:continue_current_subgoal"
            return (
                {
                    "subgoal_success": False,
                    "task_success": False,
                    "partial_progress": True,
                    "failure_type": "not_done",
                    "progress_score": progress_score,
                    "should_reobserve": False,
                    "next_action": "continue_execute",
                    "notes": notes,
                },
                "gold",
                teacher_rule,
            )
        if outcome == "reobserve":
            return (
                {
                    "subgoal_success": False,
                    "task_success": False,
                    "partial_progress": False,
                    "failure_type": "ambiguous",
                    "progress_score": 0.0,
                    "should_reobserve": True,
                    "next_action": "reobserve",
                    "notes": ["The fresh verification images are fully occluded, so the subgoal state is ambiguous."],
                },
                "gold",
                "synthetic_occluded_verify_views:reobserve",
            )
        raise RuntimeError(f"teacher_missing_verification_outcome:{self.scenario}")

    def _recovery_target(
        self, messages: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], str, str]:
        prompt = _runtime_prompt_text(messages)
        if not prompt.startswith("Diagnose a true robot manipulation failure"):
            raise RuntimeError(f"unrecognized_recovery_teacher_prompt:{prompt[:120]}")
        if not self.recovery_directive:
            raise RuntimeError(f"teacher_missing_recovery_directive:{self.scenario}")
        return (
            deepcopy(self.recovery_directive),
            "gold",
            "fault_injection:action_backend_failure_retry_current_subgoal",
        )


@dataclass
class UnexpectedModelRuntime:
    name: str

    @property
    def enabled(self) -> bool:
        return True

    def generate_text(self, **_: Any) -> str:
        raise RuntimeError(f"unexpected_model_call:{self.name}")


@dataclass
class ReplayObservationAdapter:
    initial_observation: ObservationBundle
    post_execution_observation: ObservationBundle
    last_observation: ObservationBundle = field(init=False)

    def __post_init__(self) -> None:
        self.last_observation = self.initial_observation

    def capture_views(self, **_: Any) -> ObservationBundle:
        return self.last_observation

    def execute_action(self, action_chunk: ActionChunk | None) -> dict[str, Any]:
        self.last_observation = self.post_execution_observation
        return {
            "backend": "robotwin_replay",
            "status": "action_executed",
            "success": False,
            "executed_steps": len(action_chunk.commands) if action_chunk is not None else 0,
            "observation": self.last_observation.to_dict(),
            "action_chunk": action_chunk.to_dict() if action_chunk is not None else None,
            "task_env_bound": False,
            "replay": True,
        }

    def status(self) -> dict[str, Any]:
        return {
            "backend": "robotwin_replay",
            "ready": True,
            "needs_setup": False,
            "live_env_bound": False,
            "last_observation_present": self.last_observation is not None,
            "source": "expert_hdf5_replay",
        }

    def metadata(self) -> dict[str, Any]:
        return {"backend": "robotwin", "source": "expert_hdf5_replay"}


@dataclass
class ReplayActionBackend:
    commands: list[list[float]]
    name: str = "expert_hdf5_replay"
    requires_candidate_bindings: bool = False

    def health(self) -> dict[str, Any]:
        return {"ok": True, "status": "replay_ready", "command_count": len(self.commands)}

    def action_spec(self) -> dict[str, Any]:
        return {"types": {"qpos": 14}, "source": "expert_hdf5_replay"}

    def build_action_chunk(
        self,
        motion_goal: object | None,
        world_state: object | None,
        observation: ObservationBundle | None,
        request: dict[str, Any],
    ) -> ActionBackendResult:
        del motion_goal, world_state, observation, request
        chunk = ActionChunk(
            action_type="qpos",
            commands=deepcopy(self.commands),
            control_horizon=len(self.commands),
            metadata={"source": "expert_hdf5_replay"},
        )
        return ActionBackendResult(
            success=True,
            status="replay_action_chunk_built",
            action_chunk=chunk,
            metadata={"command_count": len(self.commands)},
        )


@dataclass
class SubgoalReplayObservationAdapter:
    initial_observation: ObservationBundle
    post_observations: dict[str, ObservationBundle]
    last_observation: ObservationBundle = field(init=False)

    def __post_init__(self) -> None:
        self.last_observation = self.initial_observation

    def capture_views(self, **_: Any) -> ObservationBundle:
        return self.last_observation

    def execute_action(self, action_chunk: ActionChunk | None) -> dict[str, Any]:
        metadata = action_chunk.metadata if action_chunk is not None else {}
        subgoal_id = str(metadata.get("subgoal_id") or "")
        if subgoal_id not in self.post_observations:
            return {
                "backend": "robotwin_replay",
                "status": "execution_unavailable",
                "reason": f"missing_post_observation_for_subgoal:{subgoal_id}",
                "success": False,
                "executed_steps": 0,
                "task_env_bound": False,
                "replay": True,
            }
        self.last_observation = self.post_observations[subgoal_id]
        return {
            "backend": "robotwin_replay",
            "status": "action_executed",
            "success": False,
            "executed_steps": len(action_chunk.commands) if action_chunk is not None else 0,
            "observation": self.last_observation.to_dict(),
            "action_chunk": action_chunk.to_dict() if action_chunk is not None else None,
            "task_env_bound": False,
            "replay": True,
            "subgoal_id": subgoal_id,
        }

    def status(self) -> dict[str, Any]:
        return {
            "backend": "robotwin_replay",
            "ready": True,
            "needs_setup": False,
            "live_env_bound": False,
            "last_observation_present": self.last_observation is not None,
            "source": "expert_hdf5_subgoal_replay",
        }

    def metadata(self) -> dict[str, Any]:
        return {"backend": "robotwin", "source": "expert_hdf5_subgoal_replay"}


@dataclass
class SubgoalReplayActionBackend:
    commands_by_subgoal: dict[str, list[list[float]]]
    name: str = "expert_hdf5_subgoal_replay"
    requires_candidate_bindings: bool = False

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "status": "replay_ready",
            "subgoal_command_counts": {
                key: len(commands) for key, commands in self.commands_by_subgoal.items()
            },
        }

    def action_spec(self) -> dict[str, Any]:
        return {"types": {"qpos": 14}, "source": "expert_hdf5_subgoal_replay"}

    def build_action_chunk(
        self,
        motion_goal: object | None,
        world_state: object | None,
        observation: ObservationBundle | None,
        request: dict[str, Any],
    ) -> ActionBackendResult:
        del world_state, observation, request
        metadata = getattr(motion_goal, "metadata", {}) if motion_goal is not None else {}
        subgoal_id = str(metadata.get("subgoal_id") or "") if isinstance(metadata, dict) else ""
        commands = self.commands_by_subgoal.get(subgoal_id)
        if commands is None:
            return ActionBackendResult(
                success=False,
                status="replay_subgoal_commands_unavailable",
                errors=[f"missing_replay_commands_for_subgoal:{subgoal_id}"],
            )
        chunk = ActionChunk(
            action_type="qpos",
            commands=deepcopy(commands),
            control_horizon=len(commands),
            metadata={"source": "expert_hdf5_subgoal_replay", "subgoal_id": subgoal_id},
        )
        return ActionBackendResult(
            success=True,
            status="replay_action_chunk_built",
            action_chunk=chunk,
            metadata={"command_count": len(commands), "subgoal_id": subgoal_id},
        )


@dataclass
class ExpertSubtaskTimeline:
    """Mutable cursor over contiguous expert subtask segments.

    A PI0.5 inference call may execute only 15--32 actions.  The cursor advances
    only when the corresponding action chunk is actually executed; emitting a
    chunk alone never changes replay state.
    """

    commands_by_subgoal: dict[str, list[list[float]]]
    absolute_start_by_subgoal: dict[str, int]
    cursor_by_subgoal: dict[str, int] = field(default_factory=dict)
    last_execution: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        for subgoal_id in self.commands_by_subgoal:
            self.cursor_by_subgoal.setdefault(str(subgoal_id), 0)

    def build_chunk(self, subgoal_id: str, horizon: int) -> tuple[list[list[float]], dict[str, Any]]:
        if subgoal_id not in self.commands_by_subgoal:
            raise ValueError(f"missing_expert_segment_for_subgoal:{subgoal_id}")
        if horizon < 15 or horizon > 32:
            raise ValueError(f"expert_replay_horizon_out_of_range:{horizon}:expected_15_to_32")
        commands = self.commands_by_subgoal[subgoal_id]
        cursor = int(self.cursor_by_subgoal.get(subgoal_id, 0))
        if cursor >= len(commands):
            raise ValueError(
                f"expert_segment_already_complete:{subgoal_id}:cursor={cursor}:length={len(commands)}"
            )
        end = min(cursor + horizon, len(commands))
        valid = [list(command) for command in commands[cursor:end]]
        padded = [*valid]
        if len(padded) < horizon:
            padded.extend([list(valid[-1]) for _ in range(horizon - len(padded))])
        absolute_start = int(self.absolute_start_by_subgoal[subgoal_id]) + cursor
        absolute_end = int(self.absolute_start_by_subgoal[subgoal_id]) + end
        metadata = {
            "expert_subgoal_id": subgoal_id,
            "expert_cursor_start": cursor,
            "expert_cursor_end_exclusive": end,
            "expert_segment_length": len(commands),
            "expert_valid_steps": len(valid),
            "requested_horizon": horizon,
            "absolute_frame_start": absolute_start,
            "absolute_frame_end_exclusive": absolute_end,
            "expert_segment_complete_after_chunk": end == len(commands),
            "padding_steps": horizon - len(valid),
        }
        return padded, metadata

    def commit(self, chunk_metadata: dict[str, Any]) -> dict[str, Any]:
        subgoal_id = str(chunk_metadata.get("expert_subgoal_id") or "")
        expected = int(self.cursor_by_subgoal.get(subgoal_id, 0))
        start = int(chunk_metadata.get("expert_cursor_start", -1))
        end = int(chunk_metadata.get("expert_cursor_end_exclusive", -1))
        if start != expected:
            raise ValueError(
                f"expert_replay_cursor_mismatch:{subgoal_id}:chunk_start={start}:expected={expected}"
            )
        if end <= start:
            raise ValueError(f"expert_replay_empty_commit:{subgoal_id}:{start}->{end}")
        self.cursor_by_subgoal[subgoal_id] = end
        self.last_execution = dict(chunk_metadata)
        return dict(self.last_execution)

    def current_complete(self, subgoal_id: str) -> bool:
        commands = self.commands_by_subgoal.get(subgoal_id) or []
        return bool(commands) and int(self.cursor_by_subgoal.get(subgoal_id, 0)) >= len(commands)


@dataclass
class CursorReplayActionBackend:
    timeline: ExpertSubtaskTimeline
    name: str = "expert_hdf5_contiguous_replay"
    requires_candidate_bindings: bool = False

    def health(self) -> dict[str, Any]:
        return {"ok": True, "status": "contiguous_replay_ready"}

    def action_spec(self) -> dict[str, Any]:
        return {"types": {"qpos": 14}, "horizon": 32, "source": self.name}

    def build_action_chunk(
        self,
        motion_goal: object | None,
        world_state: object | None,
        observation: ObservationBundle | None,
        request: dict[str, Any],
    ) -> ActionBackendResult:
        del motion_goal, world_state, observation
        motion_plan = request.get("motion_plan") if isinstance(request.get("motion_plan"), dict) else {}
        current_subgoal = (
            motion_plan.get("current_subgoal")
            if isinstance(motion_plan.get("current_subgoal"), dict)
            else {}
        )
        subgoal_id = str(current_subgoal.get("subgoal_id") or "")
        horizon = int(request.get("horizon") or 32)
        commands, metadata = self.timeline.build_chunk(subgoal_id, horizon)
        chunk = ActionChunk(
            action_type="qpos",
            commands=commands,
            control_horizon=horizon,
            metadata={"source": self.name, **metadata},
        )
        return ActionBackendResult(
            success=True,
            status="replay_contiguous_action_chunk_built",
            action_chunk=chunk,
            metadata=dict(metadata),
        )


@dataclass
class CursorReplayObservationAdapter:
    timeline: ExpertSubtaskTimeline
    initial_observation: ObservationBundle
    observation_for_frame: Callable[[int], ObservationBundle]
    last_observation: ObservationBundle = field(init=False)

    def __post_init__(self) -> None:
        self.last_observation = self.initial_observation

    def capture_views(self, **_: Any) -> ObservationBundle:
        return self.last_observation

    def execute_action(self, action_chunk: ActionChunk | None) -> dict[str, Any]:
        if action_chunk is None:
            return {
                "backend": "robotwin_replay",
                "status": "execution_unavailable",
                "reason": "missing_action_chunk",
                "success": False,
                "executed_steps": 0,
                "task_env_bound": False,
                "replay": True,
            }
        committed = self.timeline.commit(dict(action_chunk.metadata))
        post_frame = int(committed["absolute_frame_end_exclusive"]) - 1
        self.last_observation = self.observation_for_frame(post_frame)
        full_task_success = all(
            self.timeline.current_complete(subgoal_id)
            for subgoal_id in self.timeline.commands_by_subgoal
        )
        return {
            "backend": "robotwin_replay",
            "status": "action_executed",
            "success": full_task_success,
            "executed_steps": len(action_chunk.commands),
            "expert_valid_steps": int(committed["expert_valid_steps"]),
            "expert_cursor_start": int(committed["expert_cursor_start"]),
            "expert_cursor_end_exclusive": int(committed["expert_cursor_end_exclusive"]),
            "expert_segment_length": int(committed["expert_segment_length"]),
            "expert_segment_complete_after_chunk": bool(
                committed["expert_segment_complete_after_chunk"]
            ),
            "requested_horizon": int(committed["requested_horizon"]),
            "padding_steps": int(committed["padding_steps"]),
            "post_frame": post_frame,
            "observation": self.last_observation.to_dict(),
            "action_chunk": action_chunk.to_dict(),
            "task_env_bound": False,
            "replay": True,
            "subgoal_id": str(committed["expert_subgoal_id"]),
        }

    def status(self) -> dict[str, Any]:
        return {
            "backend": "robotwin_replay",
            "ready": True,
            "needs_setup": False,
            "live_env_bound": False,
            "last_observation_present": self.last_observation is not None,
            "source": "expert_hdf5_contiguous_replay",
        }

    def metadata(self) -> dict[str, Any]:
        return {"backend": "robotwin", "source": "expert_hdf5_contiguous_replay"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect an exact-production-context Agent skill SFT pilot with a deterministic teacher."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/runtime/robotwin.json"))
    parser.add_argument("--segment-json", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--qwen-model-path", type=Path, required=True)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=65536,
        help="Shared runtime and SFT context limit used by the token-parity audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    max_model_len = int(args.max_model_len)
    segment_path = args.segment_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing_to_reuse_nonempty_output_dir:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    segment = json.loads(segment_path.read_text(encoding="utf-8"))
    instruction = _canonical_task_instruction(segment)
    if not instruction:
        raise ValueError("segment_missing_task_instruction")
    source_segments = list(segment.get("segments") or [])
    if len(source_segments) < 3:
        raise ValueError(f"expanded_pilot_requires_three_subtasks:actual={len(source_segments)}")
    commands_by_subgoal: dict[str, list[list[float]]] = {}
    frames_by_subgoal: dict[str, list[int]] = {}
    for segment_index in range(len(source_segments)):
        commands, frames = _contiguous_expert_segment_commands(
            segment, segment_index=segment_index
        )
        subgoal_id = f"S{segment_index + 1}"
        commands_by_subgoal[subgoal_id] = commands
        frames_by_subgoal[subgoal_id] = frames

    replay_observations: dict[int, ObservationBundle] = {}
    image_manifest: list[dict[str, Any]] = []

    def replay_observation(frame: int) -> ObservationBundle:
        if frame not in replay_observations:
            paths, manifest, arms = _extract_replay_observation(segment, output_dir, frame)
            replay_observations[frame] = _observation_from_replay(
                segment, paths, arms, frame
            )
            image_manifest.extend(manifest)
        return replay_observations[frame]

    start_observations = {
        subgoal_id: replay_observation(int(frames[0]))
        for subgoal_id, frames in frames_by_subgoal.items()
    }
    complete_observations = {
        subgoal_id: replay_observation(int(frames[-1]))
        for subgoal_id, frames in frames_by_subgoal.items()
    }
    absolute_start_by_subgoal = {
        subgoal_id: int(frames[0])
        for subgoal_id, frames in frames_by_subgoal.items()
    }
    full_task_budget_groups = _full_task_action_budget_groups(
        {
            subgoal_id: len(commands)
            for subgoal_id, commands in commands_by_subgoal.items()
        }
    )
    full_task_action_budgets = [
        budget
        for _, budgets in full_task_budget_groups
        for budget in budgets
    ]
    observation = start_observations["S1"]
    occluded_observation, occluded_manifest = _occluded_replay_observation(
        observation, output_dir, scenario="verify_reobserve"
    )
    image_manifest.extend(occluded_manifest)
    oracle_candidates = _oracle_candidates(segment, task_instruction=instruction)
    oracle_target_candidate_id = "C2" if len(oracle_candidates) > 1 else None
    oracle_plan = _oracle_task_plan(segment, task_instruction=instruction)
    grounded_oracle_plan = deepcopy(oracle_plan)
    for subgoal in grounded_oracle_plan["subgoals"]:
        subgoal["source_candidate_id"] = "C1"
        if str(subgoal.get("type") or "").lower() == "place":
            subgoal["target_candidate_id"] = oracle_target_candidate_id
    recovery_subgoal = deepcopy(oracle_plan["subgoals"][0])
    recovery_subgoal["status"] = "pending"
    recovery_subgoal["instruction"] = (
        "Retry the current subgoal from the current view: "
        + str(recovery_subgoal.get("instruction") or "execute the current subgoal").strip()
    )
    def execution_failure_recovery_directive(diagnosis: str) -> dict[str, Any]:
        return {
            "recoverable": True,
            "failure_diagnosis": diagnosis,
            "patch_type": "retry_current_subgoal",
            "next_stage": "preflight",
            "repaired_subgoal": deepcopy(recovery_subgoal),
            "notes": [
                "The current images show a safe unchanged state for a fresh retry after the execution failure."
            ],
        }

    recovery_directive = execution_failure_recovery_directive(
        "The action backend rejected the command before executing any robot action."
    )

    scenario_specs = [
        {
            "name": "direct_vla_normal",
            "candidate_bindings_required": False,
            "wrong_scheduler_call_indexes": set(),
            "max_steps": 2,
            "observation": observation,
            "oracle_plan": oracle_plan,
            "commands": commands_by_subgoal["S1"],
            "post_observation": observation,
            "verification_outcome": None,
        },
        {
            "name": "grounded_normal",
            "candidate_bindings_required": True,
            "wrong_scheduler_call_indexes": set(),
            "max_steps": 4,
            "observation": observation,
            "oracle_plan": oracle_plan,
            "commands": commands_by_subgoal["S1"],
            "post_observation": observation,
            "verification_outcome": None,
        },
        {
            "name": "grounded_error_then_correction",
            "candidate_bindings_required": True,
            "wrong_scheduler_call_indexes": {1},
            "max_steps": 3,
            "observation": observation,
            "oracle_plan": oracle_plan,
            "commands": commands_by_subgoal["S1"],
            "post_observation": observation,
            "verification_outcome": None,
        },
        {
            "name": "grounded_optional_skill_diagnostics_contiguous",
            "candidate_bindings_required": True,
            "wrong_scheduler_call_indexes": set(),
            "max_steps": 30,
            "observation": observation,
            "oracle_plan": grounded_oracle_plan,
            "commands": commands_by_subgoal["S1"],
            "post_observation": observation,
            "verification_outcome": "timeline",
            "contiguous_replay": True,
            "action_budgets": [32],
            # These skills are legal under PhasePolicy but conflict with the
            # production scheduler contract whenever next_required_decision is
            # non-null.  Execute them in a diagnostic branch so the following
            # gold row observes the real result and returns to the authoritative
            # required decision; never place the injected call in SFT gold.
            "optional_scheduler_decisions": [
                {
                    "when_stage": "observe",
                    "when_required_component": "vision",
                    "when_required_skill": "localize_task_objects",
                    "next_component": "vision",
                    "next_skill": "estimate_uncertainty",
                    "reason": "diagnostic_optional_uncertainty_before_semantic_binding",
                    "supervision": "rejected",
                    "teacher_rule": "counterfactual_optional_skill:vision.estimate_uncertainty",
                },
                {
                    "when_stage": "observe",
                    "when_required_component": "",
                    "when_required_skill": "advance_stage",
                    "next_component": "vision",
                    "next_skill": "bind_arm",
                    "payload": {"binding": {"image_left": "right", "image_right": "left"}},
                    "reason": "diagnostic_optional_arm_binding",
                    "supervision": "rejected",
                    "teacher_rule": "counterfactual_optional_skill:vision.bind_arm",
                },
                {
                    "when_stage": "observe",
                    "when_required_component": "",
                    "when_required_skill": "advance_stage",
                    "next_component": "vision",
                    "next_skill": "lift_depth_cluster",
                    "reason": "diagnostic_missing_depth_capability_probe",
                    "supervision": "rejected",
                    "teacher_rule": "counterfactual_missing_metric_input:vision.lift_depth_cluster",
                },
                {
                    "when_stage": "observe",
                    "when_required_component": "",
                    "when_required_skill": "advance_stage",
                    "next_component": "vision",
                    "next_skill": "lift_geometry",
                    "reason": "diagnostic_missing_depth_alias_probe",
                    "supervision": "rejected",
                    "teacher_rule": "counterfactual_missing_metric_input:vision.lift_geometry",
                },
                {
                    "when_stage": "observe",
                    "when_required_component": "",
                    "when_required_skill": "advance_stage",
                    "next_component": "state",
                    "next_skill": "summarize_state",
                    "reason": "diagnostic_optional_observe_summary",
                    "supervision": "rejected",
                    "teacher_rule": "counterfactual_optional_skill:state.summarize_state",
                },
                {
                    "when_stage": "plan",
                    "when_required_component": "",
                    "when_required_skill": "advance_stage",
                    "next_component": "scheduler",
                    "next_skill": "allocate_budget",
                    "payload": {"budget_steps": 20},
                    "reason": "diagnostic_legacy_budget_metadata_not_action_horizon",
                    "supervision": "rejected",
                    "teacher_rule": "counterfactual_legacy_skill:scheduler.allocate_budget",
                },
                {
                    "when_stage": "plan",
                    "when_required_component": "",
                    "when_required_skill": "advance_stage",
                    "next_component": "state",
                    "next_skill": "summarize_state",
                    "reason": "diagnostic_optional_plan_summary",
                    "supervision": "rejected",
                    "teacher_rule": "counterfactual_optional_skill:state.summarize_state",
                },
                {
                    "when_stage": "execute",
                    "when_required_component": "motion",
                    "when_required_skill": "execute_action",
                    "next_component": "motion",
                    "next_skill": "validate_action_chunk",
                    "reason": "diagnostic_explicit_validation_before_execute",
                    "supervision": "rejected",
                    "teacher_rule": "counterfactual_runtime_already_validates:motion.validate_action_chunk",
                },
                {
                    "when_stage": "execute",
                    "when_required_component": "motion",
                    "when_required_skill": "execute_action",
                    "next_component": "state",
                    "next_skill": "summarize_state",
                    "reason": "diagnostic_optional_execute_summary",
                    "supervision": "rejected",
                    "teacher_rule": "counterfactual_optional_skill:state.summarize_state",
                },
            ],
        },
        {
            "name": "direct_vla_plan_preflight",
            "candidate_bindings_required": False,
            "wrong_scheduler_call_indexes": set(),
            "max_steps": 7,
            "observation": observation,
            "oracle_plan": oracle_plan,
            "commands": commands_by_subgoal["S1"],
            "post_observation": observation,
            "verification_outcome": None,
        },
        {
            "name": "subgoal_1_multichunk_contiguous",
            "candidate_bindings_required": False,
            "wrong_scheduler_call_indexes": set(),
            "max_steps": 32,
            "observation": observation,
            "oracle_plan": oracle_plan,
            "commands": commands_by_subgoal["S1"],
            "post_observation": observation,
            "verification_outcome": "timeline",
            "contiguous_replay": True,
            "action_budgets": [30, 32, 31],
        },
        {
            "name": "subgoal_1_budget15_prefix",
            "candidate_bindings_required": False,
            "wrong_scheduler_call_indexes": set(),
            "max_steps": 14,
            "observation": observation,
            "oracle_plan": oracle_plan,
            "commands": commands_by_subgoal["S1"],
            "post_observation": observation,
            "verification_outcome": "timeline",
            "contiguous_replay": True,
            "action_budgets": [15],
        },
        {
            "name": FULL_TASK_SCENARIO,
            "candidate_bindings_required": False,
            "wrong_scheduler_call_indexes": set(),
            "max_steps": (
                20
                + 12 * len(full_task_action_budgets)
                + 8 * len(commands_by_subgoal)
            ),
            "observation": observation,
            "oracle_plan": oracle_plan,
            "commands": commands_by_subgoal["S1"],
            "post_observation": complete_observations["S1"],
            "verification_outcome": "timeline",
            "verification_outcomes": [],
            "contiguous_replay": True,
            "action_budgets": full_task_action_budgets,
            "action_budget_groups": full_task_budget_groups,
        },
        {
            "name": "subgoal_2_short_segment_padded_complete",
            "candidate_bindings_required": False,
            "wrong_scheduler_call_indexes": set(),
            "max_steps": 14,
            "observation": start_observations["S2"],
            "oracle_plan": _oracle_task_plan(segment, current_subgoal_index=1),
            "commands": commands_by_subgoal["S2"],
            "post_observation": start_observations["S2"],
            "verification_outcome": "timeline",
            "contiguous_replay": True,
            "action_budgets": [32],
        },
        {
            "name": "subgoal_3_multichunk_contiguous",
            "candidate_bindings_required": False,
            "wrong_scheduler_call_indexes": set(),
            "max_steps": 25,
            "observation": start_observations["S3"],
            "oracle_plan": _oracle_task_plan(segment, current_subgoal_index=2),
            "commands": commands_by_subgoal["S3"],
            "post_observation": start_observations["S3"],
            "verification_outcome": "timeline",
            "contiguous_replay": True,
            "action_budgets": [29, 32],
        },
        {
            "name": "direct_vla_verify_occluded_reobserve",
            "candidate_bindings_required": False,
            "wrong_scheduler_call_indexes": set(),
            "max_steps": 16,
            "observation": observation,
            "oracle_plan": oracle_plan,
            "commands": commands_by_subgoal["S1"][:32],
            "post_observation": occluded_observation,
            "verification_outcome": "reobserve",
            "action_budgets": [32],
        },
        {
            "name": "direct_vla_execution_failure_recover_retry",
            "candidate_bindings_required": False,
            "wrong_scheduler_call_indexes": set(),
            "max_steps": 7,
            "observation": observation,
            "oracle_plan": oracle_plan,
            "commands": commands_by_subgoal["S1"][:32],
            "post_observation": observation,
            "verification_outcome": None,
            "recovery_directive": recovery_directive,
            "initial_stage": "recover",
            "seed_execution_failure_recovery": True,
            "execution_failure_reason": "action_backend_rejected_command",
        },
        {
            "name": "direct_vla_controller_timeout_recover_retry",
            "candidate_bindings_required": False,
            "wrong_scheduler_call_indexes": set(),
            "max_steps": 7,
            "observation": observation,
            "oracle_plan": oracle_plan,
            "commands": commands_by_subgoal["S1"][:32],
            "post_observation": observation,
            "verification_outcome": None,
            "recovery_directive": execution_failure_recovery_directive(
                "The robot controller timed out before executing any robot action."
            ),
            "initial_stage": "recover",
            "seed_execution_failure_recovery": True,
            "execution_failure_reason": "robot_controller_timeout_before_motion",
        },
        {
            "name": "direct_vla_controller_disconnect_recover_retry",
            "candidate_bindings_required": False,
            "wrong_scheduler_call_indexes": set(),
            "max_steps": 7,
            "observation": observation,
            "oracle_plan": oracle_plan,
            "commands": commands_by_subgoal["S1"][:32],
            "post_observation": observation,
            "verification_outcome": None,
            "recovery_directive": execution_failure_recovery_directive(
                "The robot controller connection reset before executing any robot action."
            ),
            "initial_stage": "recover",
            "seed_execution_failure_recovery": True,
            "execution_failure_reason": "robot_controller_disconnected_before_motion",
        },
        {
            "name": "grounded_preflight_stale_visual_refresh",
            "candidate_bindings_required": True,
            "wrong_scheduler_call_indexes": set(),
            "max_steps": 3,
            "observation": observation,
            "oracle_plan": oracle_plan,
            "commands": commands_by_subgoal["S1"],
            "post_observation": observation,
            "verification_outcome": None,
            "initial_stage": "preflight",
            "seed_preflight_stale": True,
        },
        {
            "name": "direct_vla_preflight_invalid_plan_replan",
            "candidate_bindings_required": False,
            "wrong_scheduler_call_indexes": set(),
            "max_steps": 4,
            "observation": observation,
            "oracle_plan": oracle_plan,
            "commands": commands_by_subgoal["S1"],
            "post_observation": observation,
            "verification_outcome": None,
            "initial_stage": "preflight",
            "seed_preflight_invalid_plan": True,
        },
    ]
    runtime_calls: list[dict[str, Any]] = []
    scenario_results: list[dict[str, Any]] = []
    for spec in scenario_specs:
        result, calls = _run_teacher_scenario(
            config_path=config_path,
            instruction=instruction,
            observation=spec["observation"],
            oracle_candidates=oracle_candidates,
            oracle_plan=spec["oracle_plan"],
            scenario=str(spec["name"]),
            candidate_bindings_required=bool(spec["candidate_bindings_required"]),
            wrong_scheduler_call_indexes=set(spec["wrong_scheduler_call_indexes"]),
            max_steps=int(spec["max_steps"]),
            action_commands=[list(command) for command in spec["commands"]],
            post_execution_observation=spec["post_observation"],
            verification_outcome=spec["verification_outcome"],
            verification_outcomes=list(spec.get("verification_outcomes") or []),
            action_commands_by_subgoal=spec.get("commands_by_subgoal"),
            post_execution_observations_by_subgoal=spec.get("post_observations_by_subgoal"),
            recovery_directive=dict(spec.get("recovery_directive") or {}),
            initial_stage=str(spec.get("initial_stage") or "observe"),
            seed_preflight_stale=bool(spec.get("seed_preflight_stale", False)),
            seed_preflight_invalid_plan=bool(
                spec.get("seed_preflight_invalid_plan", False)
            ),
            seed_execution_failure_recovery=bool(
                spec.get("seed_execution_failure_recovery", False)
            ),
            execution_failure_reason=str(
                spec.get("execution_failure_reason")
                or "action_backend_rejected_command"
            ),
            contiguous_commands_by_subgoal=commands_by_subgoal
            if spec.get("contiguous_replay", False)
            else None,
            absolute_start_by_subgoal=absolute_start_by_subgoal
            if spec.get("contiguous_replay", False)
            else None,
            observation_for_frame=replay_observation
            if spec.get("contiguous_replay", False)
            else None,
            action_budgets=list(spec.get("action_budgets") or []),
            optional_scheduler_decisions=deepcopy(
                spec.get("optional_scheduler_decisions") or []
            ),
        )
        result["action_budget_groups"] = [
            [str(subgoal_id), [int(item) for item in budgets]]
            for subgoal_id, budgets in spec.get("action_budget_groups", [])
        ]
        runtime_calls.extend(calls)
        scenario_results.append(result)

    scheduler_rows = [
        _training_row(call, segment, segment_path)
        for call in runtime_calls
        if call["component"] == "scheduler" and call["supervision"] == "gold"
    ]
    component_rows = [
        _training_row(call, segment, segment_path)
        for call in runtime_calls
        if call["component"] != "scheduler" and call["supervision"] == "gold"
    ]
    rejected_rows = [call for call in runtime_calls if call["supervision"] == "rejected"]

    _write_jsonl(output_dir / "scheduler_train.jsonl", scheduler_rows)
    _write_jsonl(output_dir / "component_train.jsonl", component_rows)
    _write_jsonl(output_dir / "runtime_calls.jsonl", runtime_calls)
    _write_jsonl(output_dir / "rejected_decisions.jsonl", rejected_rows)
    contiguous_audit = _contiguous_replay_audit(runtime_calls, scenario_results)
    (output_dir / "contiguous_replay_audit.json").write_text(
        json.dumps(contiguous_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    examples = _representative_training_examples(scheduler_rows, component_rows)
    (output_dir / "representative_examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    dataset_info = {
        "robotwin_agent_skill_scheduler": {
            "file_name": "scheduler_train.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": dict(OPENAI_STYLE_SHAREGPT_TAGS),
        },
        "robotwin_agent_skill_components": {
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
    compression_audit = _history_compression_audit()
    (output_dir / "history_compression_audit.json").write_text(
        json.dumps(compression_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    token_audit = _qwen_token_parity_audit(
        runtime_calls=runtime_calls,
        model_path=args.qwen_model_path.expanduser().resolve(),
        max_model_len=max_model_len,
    )
    (output_dir / "qwen_token_parity_audit.json").write_text(
        json.dumps(token_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    coverage_report = _coverage_report(runtime_calls, scenario_results)
    (output_dir / "coverage_report.json").write_text(
        json.dumps(coverage_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "image_manifest.json").write_text(
        json.dumps(image_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        _dataset_readme(
            scheduler_rows=scheduler_rows,
            component_rows=component_rows,
            rejected_rows=rejected_rows,
            scenario_results=scenario_results,
            coverage_report=coverage_report,
            source_task=str(segment.get("task_name") or "unknown_task"),
            source_episode=int(segment.get("episode_index", -1)),
        ),
        encoding="utf-8",
    )
    summary = _validate_and_summarize(
        config_path=config_path,
        segment_path=segment_path,
        segment=segment,
        output_dir=output_dir,
        image_manifest=image_manifest,
        runtime_calls=runtime_calls,
        scheduler_rows=scheduler_rows,
        component_rows=component_rows,
        rejected_rows=rejected_rows,
        scenario_results=scenario_results,
        compression_audit=compression_audit,
        token_audit=token_audit,
        coverage_report=coverage_report,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    if summary["status"] != "PASS":
        raise SystemExit(1)


def _run_teacher_scenario(
    *,
    config_path: Path,
    instruction: str,
    observation: ObservationBundle,
    oracle_candidates: list[dict[str, Any]],
    oracle_plan: dict[str, Any],
    scenario: str,
    candidate_bindings_required: bool,
    wrong_scheduler_call_indexes: set[int],
    max_steps: int,
    action_commands: list[list[float]],
    post_execution_observation: ObservationBundle,
    verification_outcome: str | None,
    verification_outcomes: list[str] | None = None,
    action_commands_by_subgoal: dict[str, list[list[float]]] | None = None,
    post_execution_observations_by_subgoal: dict[str, ObservationBundle] | None = None,
    recovery_directive: dict[str, Any] | None = None,
    initial_stage: str = "observe",
    seed_preflight_stale: bool = False,
    seed_preflight_invalid_plan: bool = False,
    seed_execution_failure_recovery: bool = False,
    execution_failure_reason: str = "action_backend_rejected_command",
    contiguous_commands_by_subgoal: dict[str, list[list[float]]] | None = None,
    absolute_start_by_subgoal: dict[str, int] | None = None,
    observation_for_frame: Callable[[int], ObservationBundle] | None = None,
    action_budgets: list[int] | None = None,
    optional_scheduler_decisions: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_config(config_path)
    sequence = TraceSequence()
    replay_timeline = None
    if contiguous_commands_by_subgoal:
        replay_timeline = ExpertSubtaskTimeline(
            commands_by_subgoal={
                str(key): [list(command) for command in commands]
                for key, commands in contiguous_commands_by_subgoal.items()
            },
            absolute_start_by_subgoal={
                str(key): int(value)
                for key, value in (absolute_start_by_subgoal or {}).items()
            },
        )
    scheduler_teacher = TeacherTraceRuntime(
        component="scheduler",
        scenario=scenario,
        oracle_candidates=oracle_candidates,
        sequence=sequence,
        oracle_plan=oracle_plan,
        verification_outcome=verification_outcome,
        verification_outcomes=list(verification_outcomes or []),
        recovery_directive=deepcopy(recovery_directive or {}),
        action_budgets=list(action_budgets or []),
        optional_scheduler_decisions=deepcopy(optional_scheduler_decisions or []),
        replay_timeline=replay_timeline,
        wrong_scheduler_call_indexes=wrong_scheduler_call_indexes,
    )
    vision_teacher = TeacherTraceRuntime(
        component="vision",
        scenario=scenario,
        oracle_candidates=oracle_candidates,
        sequence=sequence,
        oracle_plan=oracle_plan,
        verification_outcome=verification_outcome,
        verification_outcomes=list(verification_outcomes or []),
        recovery_directive=deepcopy(recovery_directive or {}),
        replay_timeline=replay_timeline,
    )
    verifier_teacher = TeacherTraceRuntime(
        component="verifier",
        scenario=scenario,
        oracle_candidates=oracle_candidates,
        sequence=sequence,
        oracle_plan=oracle_plan,
        verification_outcome=verification_outcome,
        verification_outcomes=list(verification_outcomes or []),
        recovery_directive=deepcopy(recovery_directive or {}),
        replay_timeline=replay_timeline,
    )
    recovery_teacher = TeacherTraceRuntime(
        component="recovery",
        scenario=scenario,
        oracle_candidates=oracle_candidates,
        sequence=sequence,
        oracle_plan=oracle_plan,
        verification_outcome=verification_outcome,
        verification_outcomes=list(verification_outcomes or []),
        recovery_directive=deepcopy(recovery_directive or {}),
        replay_timeline=replay_timeline,
    )
    model_runtimes: dict[str, Any] = {
        name: UnexpectedModelRuntime(name) for name in config.models
    }
    model_runtimes["scheduler"] = scheduler_teacher
    model_runtimes["vision"] = vision_teacher
    model_runtimes["verifier"] = verifier_teacher
    model_runtimes["recovery"] = recovery_teacher
    components = build_component_registry(config, model_runtimes=model_runtimes)
    runtime = AgentRuntime(config, components=components)
    runtime.blackboard.task_instruction = instruction
    if replay_timeline is not None:
        action_backend = CursorReplayActionBackend(
            replay_timeline,
            requires_candidate_bindings=candidate_bindings_required,
        )
    elif candidate_bindings_required:
        action_backend: Any = SimpleNamespace(
            name="teacher_grounded",
            requires_candidate_bindings=True,
            health=lambda: {"ok": True, "status": "teacher_grounded_ready"},
            action_spec=lambda: {"types": {"qpos": 14}},
        )
    elif action_commands_by_subgoal:
        action_backend = SubgoalReplayActionBackend(
            {
                str(key): [list(command) for command in commands]
                for key, commands in action_commands_by_subgoal.items()
            }
        )
    else:
        action_backend = ReplayActionBackend(action_commands)
    runtime.blackboard.write("action_backend", action_backend)
    if replay_timeline is not None:
        if observation_for_frame is None:
            raise ValueError(f"missing_observation_loader_for_contiguous_replay:{scenario}")
        env_adapter = CursorReplayObservationAdapter(
            timeline=replay_timeline,
            initial_observation=observation,
            observation_for_frame=observation_for_frame,
        )
    elif post_execution_observations_by_subgoal:
        env_adapter: Any = SubgoalReplayObservationAdapter(
            initial_observation=observation,
            post_observations=dict(post_execution_observations_by_subgoal),
        )
    else:
        env_adapter = ReplayObservationAdapter(
            initial_observation=observation,
            post_execution_observation=post_execution_observation,
        )
    runtime.blackboard.write("env_adapter", env_adapter)
    runtime.blackboard.write("run_environment", False)
    runtime.blackboard.write("artifact_prefix", f"agent_skill_sft/{scenario}")
    if seed_preflight_stale:
        task_plan = TaskPlan.from_payload(oracle_plan)
        current_subgoal = task_plan.current_subgoal()
        if current_subgoal is None:
            raise ValueError(f"seed_preflight_stale_missing_current_subgoal:{scenario}")
        current_subgoal.status = "running"
        task_plan.current_subgoal_id = current_subgoal.subgoal_id
        runtime.blackboard.write("observation", observation)
        runtime.blackboard.write("task_plan", task_plan)
        runtime.blackboard.write("current_subgoal", current_subgoal)
        stale_report = SafetyReport(
            allowed=False,
            status="preflight_failed",
            errors=["stale_perception", "stale_world_state"],
            metadata={"observation_id": "stale_observation_before_refresh"},
        )
        runtime.blackboard.write("preflight_report", stale_report)
        runtime.blackboard.write("safety_report", stale_report)
    if seed_preflight_invalid_plan:
        task_plan = TaskPlan.from_payload(oracle_plan)
        runtime.blackboard.write("observation", observation)
        runtime.blackboard.write("task_plan", task_plan)
        runtime.blackboard.write("current_subgoal", None)
        invalid_plan_report = SafetyReport(
            allowed=False,
            status="preflight_failed",
            errors=["missing_current_subgoal"],
            metadata={"observation_id": observation.observation_id},
        )
        runtime.blackboard.write("preflight_report", invalid_plan_report)
        runtime.blackboard.write("safety_report", invalid_plan_report)
    if seed_execution_failure_recovery:
        task_plan = TaskPlan.from_payload(oracle_plan)
        current_subgoal = task_plan.current_subgoal()
        if current_subgoal is None:
            raise ValueError(f"seed_execution_failure_missing_current_subgoal:{scenario}")
        current_subgoal.status = "running"
        task_plan.current_subgoal_id = current_subgoal.subgoal_id
        runtime.blackboard.write("observation", observation)
        runtime.blackboard.write("task_plan", task_plan)
        runtime.blackboard.write("current_subgoal", current_subgoal)
        runtime.blackboard.write(
            "execution_report",
            {
                "backend": "teacher_fault_injection",
                "status": "execution_failed",
                "success": False,
                "executed_steps": 0,
                "task_env_bound": False,
                "reason": execution_failure_reason,
                "errors": [execution_failure_reason],
                "observation": observation.to_dict(),
                "action_chunk": None,
            },
            event_type="motion.execute_action_failed",
        )
        runtime.blackboard.write(
            "last_cleared_verify_observation",
            {
                "reason": "execution_failed_before_verify",
                "observation_id": observation.observation_id,
                "source_execution_observation_id": observation.observation_id,
                "image_paths": [
                    view.rgb_path
                    for view in observation.camera_views.values()
                    if view.rgb_path
                ],
            },
            event_type="verify_observation.cleared",
        )
    loop_result = AgentLoop(
        runtime,
        config=AgentLoopConfig(max_steps=max_steps, initial_stage=initial_stage, stop_on_skill_error=False),
    ).run()

    calls = []
    for trace in [
        *scheduler_teacher.calls,
        *vision_teacher.calls,
        *verifier_teacher.calls,
        *recovery_teacher.calls,
    ]:
        calls.append(_trace_call_payload(trace, scenario))
    calls.sort(key=lambda item: int(item["event_index"]))
    return (
        {
            "scenario": scenario,
            "candidate_bindings_required": candidate_bindings_required,
            "injected_wrong_scheduler_calls": sorted(wrong_scheduler_call_indexes),
            "loop_result": loop_result.to_dict(),
            "final_compact_context": runtime.blackboard.compact_context(),
            "runtime_history_length": len(runtime.history),
            "scheduler_calls": len(scheduler_teacher.calls),
            "vision_calls": len(vision_teacher.calls),
            "verifier_calls": len(verifier_teacher.calls),
            "recovery_calls": len(recovery_teacher.calls),
            "action_command_count": len(action_commands),
            "action_command_counts_by_subgoal": {
                str(key): len(commands)
                for key, commands in (action_commands_by_subgoal or {}).items()
            },
            "verification_outcome": verification_outcome,
            "verification_outcomes": list(verification_outcomes or []),
            "initial_stage": initial_stage,
            "seed_preflight_stale": seed_preflight_stale,
            "seed_preflight_invalid_plan": seed_preflight_invalid_plan,
            "seed_execution_failure_recovery": seed_execution_failure_recovery,
            "execution_failure_reason": execution_failure_reason,
            "action_budgets": list(action_budgets or []),
            "contiguous_replay": replay_timeline is not None,
            "final_timeline_cursors": dict(replay_timeline.cursor_by_subgoal)
            if replay_timeline is not None
            else {},
            "segment_lengths": {
                key: len(value)
                for key, value in (contiguous_commands_by_subgoal or {}).items()
            },
            "absolute_start_by_subgoal": {
                str(key): int(value)
                for key, value in (absolute_start_by_subgoal or {}).items()
            },
            "remaining_optional_scheduler_decisions": deepcopy(
                scheduler_teacher.optional_scheduler_decisions
            ),
        },
        calls,
    )


def _trace_call_payload(trace: TraceCall, scenario: str) -> dict[str, Any]:
    prompt_text = _runtime_prompt_text(trace.runtime_messages)
    request_hash = sha256(
        json.dumps(
            {"messages": trace.runtime_messages, "image_paths": trace.image_paths},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    blackboard = trace.context.get("blackboard") if isinstance(trace.context, dict) else None
    blackboard = blackboard if isinstance(blackboard, dict) else {}
    recent_history = blackboard.get("recent_loop_history")
    recent_history = recent_history if isinstance(recent_history, list) else []
    return {
        "schema": SCHEMA,
        "scenario": scenario,
        "component": trace.component,
        "call_index": trace.call_index,
        "event_index": trace.event_index,
        "supervision": trace.supervision,
        "teacher_rule": trace.teacher_rule,
        "teacher_evidence": deepcopy(trace.teacher_evidence),
        "runtime_messages": trace.runtime_messages,
        "image_paths": trace.image_paths,
        "target": trace.target,
        "rendered_prompt_sha256": sha256(prompt_text.encode("utf-8")).hexdigest(),
        "runtime_request_sha256": request_hash,
        "context_audit": {
            "current_stage": trace.context.get("current_stage") if isinstance(trace.context, dict) else None,
            "allowed_skills": trace.context.get("allowed_skills") if isinstance(trace.context, dict) else None,
            "runtime_state": trace.context.get("runtime_state") if isinstance(trace.context, dict) else None,
            "recent_loop_history_count": len(recent_history),
            "recent_loop_history": recent_history,
            "blackboard_event_count": blackboard.get("event_count"),
        },
    }


def _training_row(
    call: dict[str, Any], segment: dict[str, Any], segment_path: Path
) -> dict[str, Any]:
    runtime_messages = call["runtime_messages"]
    prompt = _runtime_prompt_text(runtime_messages)
    image_paths = list(call.get("image_paths") or [])
    user_content = "<image>" * len(image_paths)
    user_content += prompt
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(call["target"], ensure_ascii=False)},
        ],
        "images": image_paths,
        "metadata": {
            "schema_version": SCHEMA,
            "scenario": call["scenario"],
            "component": call["component"],
            "call_index": call["call_index"],
            "event_index": call["event_index"],
            "teacher_rule": call["teacher_rule"],
            "teacher_evidence": deepcopy(call.get("teacher_evidence") or {}),
            "runtime_request_sha256": call["runtime_request_sha256"],
            "rendered_prompt_sha256": call["rendered_prompt_sha256"],
            "task_name": segment.get("task_name"),
            "episode_index": segment.get("episode_index"),
            "seed": segment.get("seed"),
            "source_segment_path": str(segment_path),
            "source_segment_sha256": _file_sha(segment_path),
            "history_compaction": {
                "max_recent_loop_steps": MAX_CONTEXT_LOOP_HISTORY,
                "effective_recent_loop_steps": call["context_audit"]["recent_loop_history_count"],
            },
        },
    }


def _extract_replay_observation(
    segment: dict[str, Any], output_dir: Path, frame: int
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, RobotArmState]]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py_required_run_with_robotwin_python") from exc
    hdf5_path = Path(str(segment.get("hdf5_path") or "")).expanduser().resolve()
    if not hdf5_path.is_file():
        provenance = segment.get("merge_provenance") if isinstance(segment.get("merge_provenance"), dict) else {}
        hdf5_path = Path(str(provenance.get("source_hdf5_path") or "")).expanduser().resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)
    image_root = (
        output_dir
        / "images"
        / str(segment.get("task_name"))
        / f"episode{int(segment.get('episode_index', 0)):04d}"
        / f"frame{frame:04d}"
    )
    image_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    manifest: list[dict[str, Any]] = []
    with h5py.File(hdf5_path, "r") as handle:
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
        robot_arms = _robot_arms_from_hdf5(handle, frame)
    return paths, manifest, robot_arms


def _occluded_replay_observation(
    observation: ObservationBundle,
    output_dir: Path,
    *,
    scenario: str,
) -> tuple[ObservationBundle, list[dict[str, Any]]]:
    occluded = deepcopy(observation)
    occluded.observation_id = f"{observation.observation_id}_{scenario}_occluded"
    image_root = output_dir / "images" / "synthetic" / scenario
    image_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for camera, view in occluded.camera_views.items():
        source_path = Path(str(view.rgb_path)).resolve()
        with Image.open(source_path) as source:
            rendered = Image.new("RGB", source.size, color=(0, 0, 0))
        destination = image_root / f"{camera}.jpg"
        rendered.save(destination, format="JPEG", quality=95, subsampling=0)
        view.rgb_path = str(destination.resolve())
        view.metadata.update(
            {
                "synthetic_corruption": "full_black_occlusion",
                "source_image": str(source_path),
            }
        )
        manifest.append(
            {
                "camera": camera,
                "frame": observation.raw.get("source_frame"),
                "source_path": str(source_path),
                "source_sha256": _file_sha(source_path),
                "output_path": str(destination.resolve()),
                "output_sha256": _file_sha(destination),
                "synthetic_transformation": "full_black_occlusion",
                "scenario": scenario,
            }
        )
    occluded.raw.update(
        {
            "synthetic_branch": True,
            "synthetic_transformation": "full_black_occlusion",
            "source_observation_id": observation.observation_id,
        }
    )
    occluded.metadata.update(
        {
            "synthetic_branch": True,
            "synthetic_transformation": "full_black_occlusion",
        }
    )
    return occluded, manifest


def _observation_from_replay(
    segment: dict[str, Any],
    image_paths: dict[str, str],
    robot_arms: dict[str, RobotArmState],
    frame: int,
) -> ObservationBundle:
    instruction = _canonical_task_instruction(segment)
    left = robot_arms["left"]
    right = robot_arms["right"]
    joint_action_vector = [
        *list(left.joint_positions or []),
        float(left.gripper_value),
        *list(right.joint_positions or []),
        float(right.gripper_value),
    ]
    if len(joint_action_vector) != 14:
        raise ValueError(f"replay_joint_action_vector_is_not_14d:{len(joint_action_vector)}")
    return ObservationBundle(
        observation_id=(
            f"expert_replay_{segment.get('task_name')}_episode{int(segment.get('episode_index', 0)):04d}_frame{frame:04d}"
        ),
        task_instruction=instruction,
        camera_views={
            camera: CameraView(name=camera, rgb_path=image_paths[camera])
            for camera in CAMERA_ROLES
        },
        robot_arms=robot_arms,
        raw={
            "source": "expert_hdf5_replay",
            "source_frame": frame,
            "source_episode_success": True,
            "joint_action_vector": joint_action_vector,
        },
        metadata={
            "backend": "robotwin",
            "task_name": segment.get("task_name"),
            "task_config": segment.get("task_config"),
            "seed": segment.get("seed"),
            "now_ep_num": segment.get("episode_index"),
            "planner_image_mode": "current_rgb_4",
            "replay_source": True,
        },
    )


def _robot_arms_from_hdf5(handle: Any, frame: int) -> dict[str, RobotArmState]:
    arms: dict[str, RobotArmState] = {}
    for arm_name, image_side in (("left", "right"), ("right", "left")):
        joint_positions = [
            float(value) for value in handle[f"joint_action/{arm_name}_arm"][frame].tolist()
        ]
        gripper_value = float(handle[f"joint_action/{arm_name}_gripper"][frame])
        eef_pose = [float(value) for value in handle[f"endpose/{arm_name}_endpose"][frame].tolist()]
        arms[arm_name] = RobotArmState(
            arm_name=arm_name,
            eef_pose=eef_pose,
            gripper_state="open" if gripper_value > 0.5 else "closed",
            gripper_value=gripper_value,
            joint_positions=joint_positions,
            image_side=image_side,
            metadata={"source": "expert_hdf5_replay", "frame": frame},
        )
    return arms


def _contiguous_expert_segment_commands(
    segment: dict[str, Any], *, segment_index: int
) -> tuple[list[list[float]], list[int]]:
    """Return every recorded action in one subtask without temporal resampling."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py_required_run_with_robotwin_python") from exc
    segments = list(segment.get("segments") or [])
    if segment_index < 0 or segment_index >= len(segments):
        raise ValueError(f"invalid_segment_index:{segment_index}")
    selected = segments[segment_index]
    start = int(selected["frame_start"])
    end = int(selected["frame_end_exclusive"])
    if end <= start:
        raise ValueError(f"empty_expert_segment:start={start}:end={end}")
    hdf5_path = Path(str(segment.get("hdf5_path") or "")).expanduser().resolve()
    if not hdf5_path.is_file():
        provenance = (
            segment.get("merge_provenance")
            if isinstance(segment.get("merge_provenance"), dict)
            else {}
        )
        hdf5_path = Path(str(provenance.get("source_hdf5_path") or "")).expanduser().resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)
    frame_indexes = list(range(start, end))
    with h5py.File(hdf5_path, "r") as handle:
        action_dataset = handle["joint_action/vector"]
        commands = [
            [float(value) for value in action_dataset[frame].tolist()]
            for frame in frame_indexes
        ]
    if any(len(command) != 14 for command in commands):
        raise ValueError("contiguous_expert_command_is_not_14d")
    return commands, frame_indexes


def _full_task_action_budget_groups(
    segment_lengths: dict[str, int],
) -> list[tuple[str, list[int]]]:
    """Allocate long-biased 15--32 step chunks until every real segment ends."""

    preferred_budgets = (32, 31, 30, 32, 29, 32)
    budget_index = 0
    groups: list[tuple[str, list[int]]] = []
    for raw_subgoal_id, raw_length in segment_lengths.items():
        subgoal_id = str(raw_subgoal_id)
        length = int(raw_length)
        if not subgoal_id or length <= 0:
            raise ValueError(
                f"invalid_full_task_segment_length:{subgoal_id}:{length}"
            )
        covered = 0
        budgets: list[int] = []
        while covered < length:
            budget = preferred_budgets[budget_index % len(preferred_budgets)]
            budget_index += 1
            budgets.append(budget)
            covered += budget
        groups.append((subgoal_id, budgets))
    if not groups:
        raise ValueError("full_task_requires_at_least_one_segment")
    return groups


def _oracle_candidates(
    segment: dict[str, Any], *, task_instruction: str | None = None
) -> list[dict[str, Any]]:
    task_name = str(segment.get("task_name") or "")
    episode_info = segment.get("episode_info") if isinstance(segment.get("episode_info"), dict) else {}
    info = episode_info.get("info") if isinstance(episode_info.get("info"), dict) else {}
    labels_by_task = {
        "adjust_bottle": ("blue-cap bottle", "upright placement target"),
        "beat_block_hammer": ("hammer", "target block"),
        "click_alarmclock": ("alarm clock top button", "alarm clock"),
        "dump_bin_bigbin": ("small bin", "big bin"),
        "handover_block": ("blue block", "center handover pose"),
        "move_can_pot": ("can", "position beside the pot"),
        "move_pillbottle_pad": ("pill bottle", "colored pad"),
        "move_stapler_pad": ("stapler", "colored mat"),
        "open_microwave": ("microwave handle", "microwave door"),
        "pick_diverse_bottles": (
            "green and blue bottles",
            "left and right target locations",
        ),
        "pick_dual_bottles": (
            "bottle1 and bottle2",
            "left and right target locations",
        ),
        "place_a2b_left": ("movable object", "left side of the target object"),
        "place_container_plate": ("gray metallic cup", "gray plate"),
        "press_stapler": ("compact stapler", "stapler pressing surface"),
        "shake_bottle_horizontally": ("blue-cap bottle", "horizontal shaking workspace"),
        "stack_bowls_three": ("right bowl", "center stack position"),
    }
    if task_name == "place_mouse_pad":
        pad_color = str(info.get("{B}") or "colored").strip().lower()
        source_label, target_label = "mouse", f"{pad_color} mouse pad"
    elif task_name in labels_by_task:
        source_label, target_label = labels_by_task[task_name]
    else:
        raise ValueError(f"pilot_oracle_unsupported_task:{task_name}")
    candidates = [
        {
            "candidate_id": "C1",
            "label": source_label,
            "visibility": "yes",
            "confidence": 1.0,
            "status": "oracle_visible_source",
        },
    ]
    instruction = str(task_instruction or _canonical_task_instruction(segment))
    if _oracle_task_requires_target(instruction):
        candidates.append({
            "candidate_id": "C2",
            "label": target_label,
            "visibility": "yes",
            "confidence": 1.0,
            "status": "oracle_visible_target",
        })
    return candidates


def _oracle_task_requires_target(instruction: str) -> bool:
    normalized = " ".join(str(instruction).strip().lower().split())
    return task_requires_target(normalized) or any(
        token in normalized
        for token in ("target", "destination", "goal marker")
    )


def _canonical_task_instruction(segment: dict[str, Any]) -> str:
    """Return the exact episode instruction used by the production Runtime.

    ``instruction`` is the sampled instruction for this recorded episode.
    ``task_instruction_from_config`` is only a generic task template and may
    have different arm semantics, so it must never override the episode text.
    """
    instruction = str(
        segment.get("instruction")
        or segment.get("task_instruction_from_config")
        or ""
    ).strip()
    if not instruction:
        raise ValueError("segment_missing_task_instruction")
    return instruction


def _oracle_task_plan(
    segment: dict[str, Any], *, current_subgoal_index: int = 0,
    task_instruction: str | None = None,
) -> dict[str, Any]:
    subgoals = []
    for index, source in enumerate(segment.get("segments") or [], start=1):
        instruction = str(
            source.get("polished_instruction")
            or source.get("canonical_instruction")
            or source.get("raw_canonical_instruction")
            or ""
        ).strip()
        completion = str(source.get("completion_criteria") or "").strip()
        subgoal_type = str(source.get("subgoal_type") or "move").strip().lower()
        if not instruction or not completion:
            raise ValueError(f"expert_segment_missing_subgoal_label:{index - 1}")
        subgoals.append(
            {
                "subgoal_id": f"S{index}",
                "type": subgoal_type,
                "instruction": instruction,
                "status": "succeeded" if index - 1 < current_subgoal_index else "pending",
                "completion_criteria": {"natural_language": completion},
                "source_candidate_id": None,
                "target_candidate_id": None,
                # The deployed planner cannot know private expert segment ids.
                # Keep source provenance in row metadata/teacher_evidence, not
                # in the assistant answer or later Runtime context.
                "metadata": {},
            }
        )
    if current_subgoal_index < 0 or current_subgoal_index >= len(subgoals):
        raise ValueError(
            f"invalid_current_subgoal_index:{current_subgoal_index}:count={len(subgoals)}"
        )
    return {
        "task": str(task_instruction or _canonical_task_instruction(segment)),
        "subgoals": subgoals,
        "current_subgoal_id": f"S{current_subgoal_index + 1}",
        "status": "pending",
    }


def _canonical_loop_decision(
    payload: dict[str, Any], *, state_summary: str, expected_result: str
) -> dict[str, Any]:
    control = str(payload.get("control") or "run_skill")
    return {
        "control": control,
        "stage": payload.get("stage") if control == "run_skill" else None,
        "next_component": payload.get("next_component") if control == "run_skill" else None,
        "next_skill": payload.get("next_skill") if control == "run_skill" else None,
        "payload": dict(payload.get("payload") or {}),
        "reason": str(payload.get("reason") or "teacher_decision"),
        "narration": _teacher_narration(payload),
        "state_summary": state_summary,
        "expected_result": expected_result,
        "budget_steps": None,
    }


def _teacher_narration(payload: dict[str, Any]) -> str:
    if payload.get("control") == "advance_stage":
        return "Advance to the next runtime stage."
    component = payload.get("next_component")
    skill = payload.get("next_skill")
    return f"Run {component}.{skill}."


def _teacher_state_summary(runtime_state: dict[str, Any], reason: str) -> str:
    stage = runtime_state.get("current_stage") or runtime_state.get("stage") or "current"
    return f"The {stage} stage requires the decision identified by runtime state: {reason}."


def _teacher_expected_result(payload: dict[str, Any]) -> str:
    if payload.get("control") == "advance_stage":
        return "The runtime enters the default next stage."
    return f"{payload.get('next_component')}.{payload.get('next_skill')} produces its required stage artifact."


def _context_from_runtime_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = _runtime_prompt_text(messages)
    marker = "\n\n{"
    start = prompt.find(marker)
    if start < 0:
        return {}
    return json.loads(prompt[start + 2 :])


def _runtime_prompt_text(messages: list[dict[str, Any]]) -> str:
    if not messages:
        raise ValueError("missing_runtime_messages")
    content = messages[0].get("content")
    if not isinstance(content, list):
        raise ValueError("runtime_message_content_is_not_multimodal_list")
    text_blocks = [item.get("text") for item in content if isinstance(item, dict) and item.get("type") == "text"]
    if len(text_blocks) != 1 or not isinstance(text_blocks[0], str):
        raise ValueError("runtime_message_missing_single_text_block")
    return text_blocks[0]


def _history_compression_audit() -> dict[str, Any]:
    blackboard = Blackboard(task_instruction="history compression audit")
    records = []
    for step_index in range(MAX_CONTEXT_LOOP_HISTORY + 5):
        decision = LoopDecision(
            control="run_skill",
            stage="observe",
            next_component="vision",
            next_skill="capture_views",
            reason=f"audit_step_{step_index}",
        )
        result = SkillResult(
            success=step_index % 2 == 0,
            status="observation_captured" if step_index % 2 == 0 else "observation_unavailable",
            errors=[] if step_index % 2 == 0 else ["synthetic audit error " + "x" * 1000],
        )
        records.append(
            LoopStepRecord(step_index, "observe", decision, result.status, result.to_dict())
        )
    blackboard.write("loop_history", records)
    compact = blackboard.compact_context()["recent_loop_history"]
    return {
        "status": "PASS"
        if len(compact) == MAX_CONTEXT_LOOP_HISTORY
        and compact[0]["step_index"] == 5
        and compact[-1]["step_index"] == 24
        else "FAIL",
        "production_constant": MAX_CONTEXT_LOOP_HISTORY,
        "input_steps": len(records),
        "output_steps": len(compact),
        "first_output_step": compact[0]["step_index"],
        "last_output_step": compact[-1]["step_index"],
        "max_compact_error_chars": max(
            [len(error) for item in compact for error in item.get("errors", [])] or [0]
        ),
    }


def _qwen_token_parity_audit(
    *,
    runtime_calls: list[dict[str, Any]],
    model_path: Path,
    max_model_len: int,
) -> dict[str, Any]:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    visual_extra_cache: dict[tuple[str, ...], int] = {}
    rows: list[dict[str, Any]] = []
    for call in runtime_calls:
        if call.get("supervision") != "gold":
            continue
        converted_messages = []
        for message in call.get("runtime_messages") or []:
            converted_content = []
            for item in message.get("content") or []:
                if item.get("type") == "image":
                    converted_content.append({"type": "image"})
                elif item.get("type") == "text":
                    converted_content.append(
                        {"type": "text", "text": str(item.get("text") or "")}
                    )
            converted_messages.append(
                {"role": str(message.get("role") or "user"), "content": converted_content}
            )
        online_rendered = str(
            processor.apply_chat_template(
                converted_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        prompt = _runtime_prompt_text(call["runtime_messages"])
        image_paths = tuple(str(path) for path in call.get("image_paths") or [])
        sharegpt_user = "<image>" * len(image_paths) + prompt
        simulated_sft_rendered = (
            "<|im_start|>user\n"
            + sharegpt_user.replace(
                "<image>",
                "<|vision_start|><|image_pad|><|vision_end|>",
            )
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        render_parity = simulated_sft_rendered == online_rendered
        placeholder_tokens = len(
            processor.tokenizer(online_rendered, add_special_tokens=False)["input_ids"]
        )
        if image_paths not in visual_extra_cache:
            if image_paths:
                images = [Image.open(path).convert("RGB") for path in image_paths]
                processed = processor(
                    text=[online_rendered],
                    images=images,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                visual_extra_cache[image_paths] = (
                    int(processed["input_ids"].shape[1]) - placeholder_tokens
                )
            else:
                visual_extra_cache[image_paths] = 0
        prompt_tokens = placeholder_tokens + visual_extra_cache[image_paths]
        target_text = json.dumps(call.get("target") or {}, ensure_ascii=False)
        target_tokens = len(
            processor.tokenizer(
                target_text + "<|im_end|>\n",
                add_special_tokens=False,
            )["input_ids"]
        )
        total_tokens = prompt_tokens + target_tokens
        rows.append(
            {
                "scenario": call.get("scenario"),
                "component": call.get("component"),
                "call_index": call.get("call_index"),
                "event_index": call.get("event_index"),
                "image_count": len(image_paths),
                "render_parity": render_parity,
                "prompt_tokens": prompt_tokens,
                "target_tokens": target_tokens,
                "total_tokens": total_tokens,
                "fits_max_model_len": total_tokens <= max_model_len,
            }
        )
    parity_failures = [row for row in rows if not row["render_parity"]]
    overflows = [row for row in rows if not row["fits_max_model_len"]]
    return {
        "status": "PASS"
        if rows and not parity_failures and not overflows
        else "FAIL",
        "model_path": str(model_path),
        "template": "qwen3_vl_nothink",
        "max_model_len": max_model_len,
        "gold_rows": len(rows),
        "render_parity_failures": len(parity_failures),
        "context_overflows": len(overflows),
        "max_prompt_tokens": max((row["prompt_tokens"] for row in rows), default=0),
        "max_total_tokens": max((row["total_tokens"] for row in rows), default=0),
        "failure_rows": [*parity_failures, *overflows],
        "rows": rows,
    }


def _contiguous_replay_audit(
    runtime_calls: list[dict[str, Any]],
    scenario_results: list[dict[str, Any]],
) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    for scenario in scenario_results:
        if not scenario.get("contiguous_replay"):
            continue
        name = str(scenario.get("scenario") or "")
        reports = _executed_action_reports(scenario)
        verifier_calls = _matching_calls(
            runtime_calls, scenario=name, component="verifier"
        )
        chunks = []
        for index, report in enumerate(reports):
            verifier = verifier_calls[index] if index < len(verifier_calls) else {}
            chunks.append(
                {
                    "chunk_index": index,
                    "subgoal_id": report.get("subgoal_id"),
                    "requested_horizon": report.get("requested_horizon"),
                    "executed_steps": report.get("executed_steps"),
                    "expert_valid_steps": report.get("expert_valid_steps"),
                    "padding_steps": report.get("padding_steps"),
                    "expert_cursor_start": report.get("expert_cursor_start"),
                    "expert_cursor_end_exclusive": report.get(
                        "expert_cursor_end_exclusive"
                    ),
                    "expert_segment_length": report.get("expert_segment_length"),
                    "expert_segment_complete_after_chunk": report.get(
                        "expert_segment_complete_after_chunk"
                    ),
                    "post_frame": report.get("post_frame"),
                    "post_observation_id": (
                        report.get("observation", {}).get("observation_id")
                        if isinstance(report.get("observation"), dict)
                        else None
                    ),
                    "verifier_subgoal_success": verifier.get("target", {}).get(
                        "subgoal_success"
                    ),
                    "verifier_failure_type": verifier.get("target", {}).get(
                        "failure_type"
                    ),
                    "verifier_next_action": verifier.get("target", {}).get(
                        "next_action"
                    ),
                    "verifier_image_paths": list(verifier.get("image_paths") or []),
                }
            )
        scenarios.append(
            {
                "scenario": name,
                "action_budgets": list(scenario.get("action_budgets") or []),
                "segment_lengths": dict(scenario.get("segment_lengths") or {}),
                "absolute_start_by_subgoal": dict(
                    scenario.get("absolute_start_by_subgoal") or {}
                ),
                "final_timeline_cursors": dict(
                    scenario.get("final_timeline_cursors") or {}
                ),
                "chunks": chunks,
            }
        )
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "semantics": {
            "horizon_range": [15, 32],
            "ordinary_horizon_preference": "28-32, with 32 preferred",
            "action_selection": "consecutive expert qpos frames from the current subtask cursor",
            "cursor_commit": "only after motion.execute_action succeeds",
            "segment_boundary": "pad with the final valid action only when fewer than horizon steps remain",
            "post_observation": "expert frame absolute_frame_end_exclusive - 1",
            "completion": "only when the cursor reaches the recorded expert subtask boundary",
        },
        "scenarios": scenarios,
    }


def _representative_training_examples(
    scheduler_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def pick(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        for row in rows:
            if predicate(row):
                return row
        raise ValueError("representative_training_example_missing")

    continuation_emit_rows = [
        row
        for row in scheduler_rows
        if row.get("metadata", {}).get("scenario")
        == FULL_TASK_SCENARIO
        and json.loads(row["messages"][1]["content"]).get("next_skill")
        == "emit_action_chunk"
    ]
    if len(continuation_emit_rows) < 2:
        raise ValueError("missing_second_round_continuation_emit_example")
    return {
        "note": (
            "These are exact copies of released gold rows. The full prompt includes the production compact "
            "blackboard, runtime_state, ordered recent_loop_history, and the same image list used by AgentLoop."
        ),
        "second_round_same_subgoal_emit": continuation_emit_rows[1],
        "error_conditioned_scheduler_correction": pick(
            scheduler_rows,
            lambda row: row.get("metadata", {}).get("scenario")
            == "grounded_error_then_correction"
            and row.get("metadata", {})
            .get("history_compaction", {})
            .get("effective_recent_loop_steps", 0)
            > 0,
        ),
        "incomplete_subgoal_verifier": pick(
            component_rows,
            lambda row: str(row.get("metadata", {}).get("teacher_rule") or "").startswith(
                "expert_contiguous_segment_prefix"
            ),
        ),
        "completed_subgoal_verifier": pick(
            component_rows,
            lambda row: str(row.get("metadata", {}).get("teacher_rule") or "").startswith(
                "expert_contiguous_segment_end"
            ),
        ),
    }


def _dataset_readme(
    *,
    scheduler_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    scenario_results: list[dict[str, Any]],
    coverage_report: dict[str, Any],
    source_task: str,
    source_episode: int,
) -> str:
    return f"""# RoboTwin Agent skill SFT contiguous replay set

This is a production-context engineering set generated through the real EmbodiedSkills `AgentLoop`. It contains {len(scheduler_rows)} scheduler gold rows, {len(component_rows)} model-component gold rows, {len(rejected_rows)} diagnostic/rejected decisions, and {len(scenario_results)} scenarios.

## Gold contract

- Prompts are the exact multimodal messages rendered by production components.
- Context uses production `Blackboard.compact_context()` and its {MAX_CONTEXT_LOOP_HISTORY}-step ordered history limit.
- Camera order is `left_camera`, `right_camera`, `head_camera`, `front_camera`.
- `motion.emit_action_chunk` is supervised through `payload.horizon` in the inclusive range 15--32; ordinary chunks favor 28--32.
- Expert actions are consecutive from the current subtask cursor. They are never uniformly resampled to 32 steps.
- Cursor state commits only at `motion.execute_action`; the post image/state is the matching real expert endpoint frame.
- An unfinished chunk keeps the same subgoal and re-enters preflight with prior execute/verify/repair skills in history.
- Physical-disturbance recovery such as knocked-over or dropped objects is intentionally absent.

## Files

- `scheduler_train.jsonl`: scheduler SFT gold only.
- `component_train.jsonl`: vision/verifier/recovery model-output gold only.
- `runtime_calls.jsonl`: every teacher model call with exact runtime messages and context audits.
- `rejected_decisions.jsonl`: deliberately wrong or diagnostic-only scheduler calls; excluded from SFT gold.
- `contiguous_replay_audit.json`: compact chunk/cursor/frame/verifier audit.
- `representative_examples.json`: four directly inspectable full training examples.
- `coverage_report.json`: gold and diagnostic skill coverage.
- `history_compression_audit.json`: production history-limit gate.
- `qwen_token_parity_audit.json`: prompt render/token/cutoff gate.
- `summary.json`: release result, provenance, hashes, limitations, and scenario traces.
- `images/`: exact replay RGB endpoints plus explicitly named synthetic counterfactual views.

## Coverage interpretation

All {coverage_report.get('configured_phase_policy_skills')} configured PhasePolicy skills are exercised across gold and diagnostic branches. {coverage_report.get('positively_targeted_skills')} are authoritative production scheduler targets. The remaining optional/legacy/metric-only skills are diagnostic-only because production says `runtime_state.next_required_decision` is authoritative; they are not allowed to leak into SFT gold. `state.update_world_state` is also invoked automatically by the production loop after relevant vision/verifier skills.

This release is an engineering scenario set from one successful `{source_task}` episode (episode index {source_episode}), not yet a population-scale multi-task training corpus. See `summary.json` for the exact limitations.
"""


def _coverage_report(
    runtime_calls: list[dict[str, Any]],
    scenario_results: list[dict[str, Any]],
) -> dict[str, Any]:
    configured_skills = sorted(
        {
            f"{component}.{skill}"
            for stage in DEFAULT_ALLOWED_SKILLS.values()
            for component, skills in stage.items()
            for skill in skills
        }
    )
    target_counts: dict[str, int] = {}
    diagnostic_target_counts: dict[str, int] = {}
    control_counts: dict[str, int] = {}
    component_output_counts: dict[str, int] = {}
    for call in runtime_calls:
        component = str(call.get("component") or "")
        target = call.get("target") if isinstance(call.get("target"), dict) else {}
        if component == "scheduler" and target.get("control") in {
            "run_skill",
            "advance_stage",
            "finish_run",
        }:
            control = str(target.get("control"))
            if call.get("supervision") == "gold":
                control_counts[control] = control_counts.get(control, 0) + 1
            if control == "run_skill":
                skill_name = f"{target.get('next_component')}.{target.get('next_skill')}"
                counts = (
                    target_counts
                    if call.get("supervision") == "gold"
                    else diagnostic_target_counts
                )
                counts[skill_name] = counts.get(skill_name, 0) + 1
        else:
            if call.get("supervision") != "gold":
                continue
            key = f"{component}:{call.get('teacher_rule')}"
            component_output_counts[key] = component_output_counts.get(key, 0) + 1
    covered = sorted(set(target_counts) & set(configured_skills))
    uncovered = sorted(set(configured_skills) - set(covered))
    exercised = sorted(
        (set(target_counts) | set(diagnostic_target_counts)) & set(configured_skills)
    )
    unexercised = sorted(set(configured_skills) - set(exercised))
    real_history_counts = [
        int(call.get("context_audit", {}).get("recent_loop_history_count") or 0)
        for call in runtime_calls
        if call.get("supervision") == "gold"
    ]
    return {
        "status": "PASS" if not unexercised else "FAIL",
        "scope": "one-episode engineering coverage; not population coverage",
        "configured_phase_policy_skills": len(configured_skills),
        "positively_targeted_skills": len(covered),
        "skill_name_coverage_fraction": len(covered) / len(configured_skills),
        "covered_skills": covered,
        "uncovered_skills": uncovered,
        "exercised_skills": exercised,
        "unexercised_skills": unexercised,
        "diagnostic_only_skills": sorted(set(exercised) - set(covered)),
        "target_counts": dict(sorted(target_counts.items())),
        "diagnostic_target_counts": dict(sorted(diagnostic_target_counts.items())),
        "control_counts": dict(sorted(control_counts.items())),
        "component_output_counts": dict(sorted(component_output_counts.items())),
        "scenarios": [str(item.get("scenario")) for item in scenario_results],
        "real_context_max_history_steps": max(real_history_counts, default=0),
        "contexts_at_production_history_limit": sum(
            count == MAX_CONTEXT_LOOP_HISTORY for count in real_history_counts
        ),
        "source_tasks": 1,
        "source_episodes": 1,
        "source_seeds": 1,
        "synthetic_counterfactual_scenarios": [
            "direct_vla_verify_occluded_reobserve",
            *EXECUTION_FAILURE_RECOVERY_SCENARIOS,
            "grounded_preflight_stale_visual_refresh",
            "direct_vla_preflight_invalid_plan_replan",
        ],
    }


def _validate_and_summarize(
    *,
    config_path: Path,
    segment_path: Path,
    segment: dict[str, Any],
    output_dir: Path,
    image_manifest: list[dict[str, Any]],
    runtime_calls: list[dict[str, Any]],
    scheduler_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    scenario_results: list[dict[str, Any]],
    compression_audit: dict[str, Any],
    token_audit: dict[str, Any],
    coverage_report: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    correction_calls = [
        call
        for call in runtime_calls
        if call["scenario"] == "grounded_error_then_correction"
        and call["component"] == "scheduler"
        and call["supervision"] == "gold"
        and call["context_audit"]["recent_loop_history_count"] > 0
    ]
    if not correction_calls:
        errors.append("missing_error_conditioned_correction_row")
    elif not any(
        item.get("success") is False
        and (
            "missing_perception_before_world_state_update" in str(item.get("reason") or "")
            or "missing_perception_before_world_state_update"
            in " ".join(str(error) for error in item.get("errors", []))
        )
        for call in correction_calls
        for item in call["context_audit"]["recent_loop_history"]
    ):
        errors.append("correction_row_missing_actual_runtime_error_history")
    if not any(call["image_paths"] for call in correction_calls):
        errors.append("correction_row_missing_current_images")
    if not any(
        row.get("scenario") == "grounded_error_then_correction"
        and row.get("teacher_rule") == "fault_injection:premature_update_world_state"
        for row in rejected_rows
    ):
        errors.append("missing_premature_world_state_rejected_decision")
    rejected_identities = {
        (
            str(row.get("scenario") or ""),
            str(row.get("component") or ""),
            int(row.get("call_index", -1)),
        )
        for row in rejected_rows
    }
    gold_identities = {
        (
            str(row.get("metadata", {}).get("scenario") or ""),
            str(row.get("metadata", {}).get("component") or ""),
            int(row.get("metadata", {}).get("call_index", -1)),
        )
        for row in [*scheduler_rows, *component_rows]
    }
    if rejected_identities & gold_identities:
        errors.append("rejected_decision_leaked_into_sft_gold")
    if any(
        result.get("remaining_optional_scheduler_decisions")
        for result in scenario_results
    ):
        errors.append("optional_scheduler_diagnostic_not_exercised")
    if compression_audit.get("status") != "PASS":
        errors.append("history_compression_audit_failed")
    if token_audit.get("status") != "PASS":
        errors.append("qwen_token_parity_or_length_audit_failed")
    if coverage_report.get("status") != "PASS":
        errors.append("coverage_report_failed")
    if any(call["context_audit"]["recent_loop_history_count"] > MAX_CONTEXT_LOOP_HISTORY for call in runtime_calls):
        errors.append("runtime_call_exceeds_history_compaction_limit")
    for row in [*scheduler_rows, *component_rows]:
        image_count = len(row.get("images") or [])
        if row["messages"][0]["content"].count("<image>") != image_count:
            errors.append("training_row_image_placeholder_mismatch")
            break
    errors.extend(
        _validate_plan_and_execution_scenarios(
            runtime_calls=runtime_calls,
            scenario_results=scenario_results,
        )
    )
    return {
        "schema": SCHEMA,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "claim": (
            "Replay-backed contiguous expert-subtask engineering dataset. Runtime messages, compact blackboard, "
            "allowed skills, runtime state, loop history, validation errors, prompts, and 15--32 step horizon "
            "decisions come from the production AgentLoop path; Qwen3-VL prompt rendering and token budgets are "
            "release-gated."
        ),
        "limitations": [
            "RGB inputs are replayed from an existing successful expert HDF5 rather than a newly reset live simulator.",
            "Action chunks use consecutive expert qpos frames with cursor commits on execute; short final chunks are padded only at the expert segment boundary.",
            "Subgoal completion labels are derived from reaching the recorded successful expert segment boundary, not from a newly run online verifier audit.",
            "The occluded-view branch is a synthetic visual corruption; the recovery branch is an explicit action-backend failure injection. Physical disturbance failures are not fabricated.",
            (
                "Only one successful "
                f"{segment.get('task_name')} episode (index {segment.get('episode_index')}) is covered by this "
                "engineering set; use the multi-task aggregate for diversity."
            ),
            "The source contains RGB and calibration but no depth or pointcloud samples; metric-geometry skills are diagnostic-only and excluded from SFT gold.",
            "PhasePolicy-legal optional skills that conflict with authoritative next_required_decision are diagnostic-only and excluded from SFT gold.",
            "This expanded dataset is still an engineering coverage set, not a statistically independent train/validation release.",
        ],
        "source": {
            "config": _file_info(config_path),
            "segment": _file_info(segment_path),
            "task_name": segment.get("task_name"),
            "episode_index": segment.get("episode_index"),
            "seed": segment.get("seed"),
            "hdf5_path": segment.get("hdf5_path"),
        },
        "history_compaction": compression_audit,
        "qwen_token_parity": {
            key: value for key, value in token_audit.items() if key != "rows"
        },
        "coverage": coverage_report,
        "counts": {
            "scenarios": len(scenario_results),
            "runtime_calls": len(runtime_calls),
            "scheduler_gold_rows": len(scheduler_rows),
            "component_gold_rows": len(component_rows),
            "rejected_decisions": len(rejected_rows),
            "images": len(image_manifest),
            "error_conditioned_correction_rows": len(correction_calls),
        },
        "scenarios": scenario_results,
        "image_manifest": image_manifest,
        "artifacts": {
            name: _file_info(output_dir / name)
            for name in (
                "scheduler_train.jsonl",
                "component_train.jsonl",
                "runtime_calls.jsonl",
                "rejected_decisions.jsonl",
                "history_compression_audit.json",
                "qwen_token_parity_audit.json",
                "dataset_info.json",
                "coverage_report.json",
                "contiguous_replay_audit.json",
                "representative_examples.json",
                "README.md",
            )
        },
    }


def _validate_plan_and_execution_scenarios(
    *,
    runtime_calls: list[dict[str, Any]],
    scenario_results: list[dict[str, Any]],
) -> list[str]:
    """Release-gate production-history and contiguous expert replay semantics."""

    errors: list[str] = []
    scenarios = {
        str(result.get("scenario") or ""): result for result in scenario_results
    }
    required_names = (
        "direct_vla_plan_preflight",
        "grounded_optional_skill_diagnostics_contiguous",
        "subgoal_1_multichunk_contiguous",
        "subgoal_1_budget15_prefix",
        FULL_TASK_SCENARIO,
        "subgoal_2_short_segment_padded_complete",
        "subgoal_3_multichunk_contiguous",
        "direct_vla_verify_occluded_reobserve",
        *EXECUTION_FAILURE_RECOVERY_SCENARIOS,
        "grounded_preflight_stale_visual_refresh",
        "direct_vla_preflight_invalid_plan_replan",
    )
    for name in required_names:
        scenario = scenarios.get(name)
        if scenario is None:
            errors.append(f"missing_required_scenario:{name}")
            continue
        for step in _loop_steps(scenario):
            result = step.get("result") if isinstance(step.get("result"), dict) else {}
            expected_grounded_stale_preflight = (
                name == "grounded_optional_skill_diagnostics_contiguous"
                and step.get("status") == "preflight_failed"
                and {
                    str(item)
                    for item in result.get("errors", [])
                }
                & {"stale_perception", "stale_world_state"}
            )
            if step.get("error") is not None or result.get("success") is not True:
                if expected_grounded_stale_preflight and step.get("error") is None:
                    continue
                errors.append(
                    f"unexpected_skill_failure:{name}:step={step.get('step_index')}:"
                    f"status={step.get('status')}"
                )

    if not any(
        call.get("supervision") == "gold"
        and call.get("teacher_rule") == "expert_subtask_segments:oracle_task_plan"
        for call in runtime_calls
    ):
        errors.append("missing_expert_subtask_task_plan_teacher_row")

    preflight = scenarios.get("direct_vla_plan_preflight")
    if preflight is not None:
        if _loop_result(preflight).get("final_stage") != "execute":
            errors.append(
                "preflight_final_stage_not_execute:"
                f"actual={_loop_result(preflight).get('final_stage')}"
            )
        steps = _loop_steps(preflight)
        if not steps or steps[-1].get("status") != "stage_advanced":
            errors.append("preflight_missing_successful_stage_advance")

    def expected_contiguous_contract(
        scenario: dict[str, Any],
        budget_groups: list[tuple[str, list[int]]],
    ) -> dict[str, Any]:
        segment_lengths = {
            str(key): int(value)
            for key, value in dict(scenario.get("segment_lengths") or {}).items()
        }
        absolute_starts = {
            str(key): int(value)
            for key, value in dict(
                scenario.get("absolute_start_by_subgoal") or {}
            ).items()
        }
        cursors = {subgoal_id: 0 for subgoal_id in segment_lengths}
        reports: list[tuple[Any, ...]] = []
        final_subgoal: str | None = None
        subgoal_order = list(segment_lengths)
        for subgoal_id, budgets in budget_groups:
            length = segment_lengths[subgoal_id]
            for horizon in budgets:
                cursor_start = cursors[subgoal_id]
                if cursor_start >= length:
                    break
                cursor_end = min(length, cursor_start + int(horizon))
                valid_steps = cursor_end - cursor_start
                padding_steps = int(horizon) - valid_steps
                complete = cursor_end == length
                post_frame = absolute_starts[subgoal_id] + cursor_end - 1
                reports.append(
                    (
                        subgoal_id,
                        cursor_start,
                        cursor_end,
                        int(horizon),
                        valid_steps,
                        padding_steps,
                        post_frame,
                        complete,
                    )
                )
                cursors[subgoal_id] = cursor_end
            if cursors[subgoal_id] < length:
                final_subgoal = subgoal_id
                break
            index = subgoal_order.index(subgoal_id)
            final_subgoal = (
                subgoal_order[index + 1]
                if index + 1 < len(subgoal_order)
                else None
            )
        return {
            "reports": reports,
            "verifier_success": [bool(item[-1]) for item in reports],
            "final_cursors": cursors,
            "final_subgoal": final_subgoal,
        }

    expected_contiguous: dict[str, dict[str, Any]] = {}
    for name in (
        "grounded_optional_skill_diagnostics_contiguous",
        "subgoal_1_multichunk_contiguous",
        "subgoal_1_budget15_prefix",
        "subgoal_2_short_segment_padded_complete",
        "subgoal_3_multichunk_contiguous",
    ):
        scenario = scenarios.get(name)
        if scenario is None:
            continue
        subgoal_id = {
            "subgoal_2_short_segment_padded_complete": "S2",
            "subgoal_3_multichunk_contiguous": "S3",
        }.get(name, "S1")
        expected_contiguous[name] = expected_contiguous_contract(
            scenario,
            [(subgoal_id, [int(item) for item in scenario.get("action_budgets") or []])],
        )

    full_task = scenarios.get(FULL_TASK_SCENARIO)
    if full_task is not None:
        budget_groups = [
            (str(item[0]), [int(budget) for budget in item[1]])
            for item in full_task.get("action_budget_groups") or []
            if isinstance(item, list) and len(item) == 2
        ]
        expected_ids = list(dict(full_task.get("segment_lengths") or {}))
        if [subgoal_id for subgoal_id, _ in budget_groups] != expected_ids:
            errors.append(
                "full_task_budget_group_subgoals_mismatch:"
                f"actual={[item[0] for item in budget_groups]}:expected={expected_ids}"
            )
        expected_contiguous[FULL_TASK_SCENARIO] = expected_contiguous_contract(
            full_task,
            budget_groups,
        )
    for name, expected in expected_contiguous.items():
        scenario = scenarios.get(name)
        if scenario is None:
            continue
        reports = _executed_action_reports(scenario)
        expected_reports = expected["reports"]
        if len(reports) != len(expected_reports):
            errors.append(
                f"contiguous_report_count_mismatch:{name}:"
                f"actual={len(reports)}:expected={len(expected_reports)}"
            )
            continue
        emitted = _matching_calls(
            runtime_calls,
            scenario=name,
            component="scheduler",
            next_skill="emit_action_chunk",
        )
        emitted_horizons = [
            call.get("target", {}).get("payload", {}).get("horizon") for call in emitted
        ]
        expected_horizons = [item[3] for item in expected_reports]
        if emitted_horizons != expected_horizons:
            errors.append(
                f"contiguous_emit_horizons_mismatch:{name}:"
                f"actual={emitted_horizons}:expected={expected_horizons}"
            )
        previous_end: dict[str, int] = {}
        for index, (report, contract) in enumerate(zip(reports, expected_reports)):
            (
                subgoal_id,
                cursor_start,
                cursor_end,
                horizon,
                valid_steps,
                padding_steps,
                post_frame,
                complete,
            ) = contract
            actual = (
                report.get("subgoal_id"),
                report.get("expert_cursor_start"),
                report.get("expert_cursor_end_exclusive"),
                report.get("requested_horizon"),
                report.get("expert_valid_steps"),
                report.get("padding_steps"),
                report.get("post_frame"),
                report.get("expert_segment_complete_after_chunk"),
            )
            if actual != contract:
                errors.append(
                    f"contiguous_report_contract_mismatch:{name}:index={index}:"
                    f"actual={actual}:expected={contract}"
                )
            if report.get("executed_steps") != horizon:
                errors.append(
                    f"contiguous_executed_horizon_mismatch:{name}:index={index}:"
                    f"actual={report.get('executed_steps')}:expected={horizon}"
                )
            if not 15 <= int(horizon) <= 32:
                errors.append(f"contiguous_horizon_out_of_range:{name}:index={index}:{horizon}")
            if cursor_end - cursor_start != valid_steps:
                errors.append(f"contiguous_valid_span_mismatch:{name}:index={index}")
            if horizon - valid_steps != padding_steps:
                errors.append(f"contiguous_padding_mismatch:{name}:index={index}")
            if cursor_start != previous_end.get(subgoal_id, 0):
                errors.append(
                    f"contiguous_cursor_gap_or_overlap:{name}:{subgoal_id}:"
                    f"actual={cursor_start}:expected={previous_end.get(subgoal_id, 0)}"
                )
            previous_end[subgoal_id] = cursor_end
            segment_length = int(scenario.get("segment_lengths", {}).get(subgoal_id, -1))
            if cursor_end > segment_length or bool(complete) != (cursor_end == segment_length):
                errors.append(f"contiguous_segment_boundary_mismatch:{name}:index={index}")
            absolute_start = int(
                scenario.get("absolute_start_by_subgoal", {}).get(subgoal_id, -1)
            )
            if post_frame != absolute_start + cursor_end - 1:
                errors.append(f"contiguous_post_frame_mismatch:{name}:index={index}")
            observation = report.get("observation")
            observation = observation if isinstance(observation, dict) else {}
            expected_suffix = f"_frame{post_frame:04d}"
            if not str(observation.get("observation_id") or "").endswith(expected_suffix):
                errors.append(f"contiguous_post_observation_mismatch:{name}:index={index}")
        if scenario.get("final_timeline_cursors") != expected["final_cursors"]:
            errors.append(
                f"contiguous_final_cursor_mismatch:{name}:"
                f"actual={scenario.get('final_timeline_cursors')}"
            )
        if _current_subgoal_id(scenario) != expected["final_subgoal"]:
            errors.append(
                f"contiguous_final_subgoal_mismatch:{name}:"
                f"actual={_current_subgoal_id(scenario)}:expected={expected['final_subgoal']}"
            )
        verifier_calls = _matching_calls(
            runtime_calls, scenario=name, component="verifier"
        )
        verifier_success = [
            call.get("target", {}).get("subgoal_success") for call in verifier_calls
        ]
        if verifier_success != expected["verifier_success"]:
            errors.append(
                f"contiguous_verifier_sequence_mismatch:{name}:"
                f"actual={verifier_success}:expected={expected['verifier_success']}"
            )
        if any(not call.get("image_paths") for call in verifier_calls):
            errors.append(f"contiguous_verifier_missing_images:{name}")

        emits_by_subgoal: dict[str, list[dict[str, Any]]] = {}
        for call in emitted:
            current = call.get("context_audit", {}).get("runtime_state", {}).get(
                "current_subgoal"
            )
            current = current if isinstance(current, dict) else {}
            emits_by_subgoal.setdefault(str(current.get("subgoal_id") or ""), []).append(call)
        for subgoal_id, calls in emits_by_subgoal.items():
            for continuation in calls[1:]:
                history = continuation.get("context_audit", {}).get(
                    "recent_loop_history", []
                )
                history_skills = {
                    str(item.get("next_skill") or "")
                    for item in history
                    if isinstance(item, dict)
                }
                required_history = {
                    "execute_action",
                    "capture_verify_views",
                    "verify_progress",
                    "repair_stage_transition",
                }
                if not required_history <= history_skills:
                    errors.append(
                        f"contiguous_continuation_history_incomplete:{name}:{subgoal_id}:"
                        f"actual={sorted(history_skills)}"
                    )

    full_task = scenarios.get(FULL_TASK_SCENARIO)
    if full_task is not None:
        if _loop_result(full_task).get("status") != "finished":
            errors.append(
                f"full_task_loop_not_finished:actual={_loop_result(full_task).get('status')}"
            )
        compact_plan = full_task.get("final_compact_context", {}).get("task_plan", {})
        if compact_plan.get("status") != "succeeded":
            errors.append(f"full_task_plan_not_succeeded:actual={compact_plan.get('status')}")
        full_calls = [
            call
            for call in runtime_calls
            if call.get("scenario") == FULL_TASK_SCENARIO
        ]
        if not any(
            call.get("context_audit", {}).get("recent_loop_history_count")
            == MAX_CONTEXT_LOOP_HISTORY
            for call in full_calls
        ):
            errors.append("full_task_missing_real_max_history_context")

    diagnostics = scenarios.get("grounded_optional_skill_diagnostics_contiguous")
    diagnostic_skills = {
        "vision.estimate_uncertainty",
        "vision.bind_arm",
        "vision.lift_depth_cluster",
        "vision.lift_geometry",
        "state.summarize_state",
        "scheduler.allocate_budget",
        "motion.validate_action_chunk",
    }
    diagnostic_calls = [
        call
        for call in runtime_calls
        if call.get("scenario") == "grounded_optional_skill_diagnostics_contiguous"
        and call.get("component") == "scheduler"
        and call.get("supervision") == "rejected"
    ]
    actual_diagnostic_skills = {
        f"{call.get('target', {}).get('next_component')}."
        f"{call.get('target', {}).get('next_skill')}"
        for call in diagnostic_calls
    }
    if actual_diagnostic_skills != diagnostic_skills:
        errors.append(
            "diagnostic_skill_coverage_mismatch:"
            f"actual={sorted(actual_diagnostic_skills)}:expected={sorted(diagnostic_skills)}"
        )
    if diagnostics is not None:
        step_statuses = {
            str((step.get("decision") or {}).get("next_skill") or ""): step.get("status")
            for step in _loop_steps(diagnostics)
        }
        for skill in ("lift_depth_cluster", "lift_geometry"):
            if step_statuses.get(skill) != "metric_geometry_unavailable":
                errors.append(
                    f"diagnostic_geometry_did_not_report_real_unavailable:{skill}:"
                    f"actual={step_statuses.get(skill)}"
                )
        if step_statuses.get("validate_action_chunk") != "action_chunk_validated":
            errors.append("diagnostic_explicit_action_validation_not_exercised")
        diagnostic_steps = _loop_steps(diagnostics)
        stale_failures = [
            step
            for step in diagnostic_steps
            if step.get("status") == "preflight_failed"
            and {
                str(item)
                for item in (
                    step.get("result", {}).get("errors", [])
                    if isinstance(step.get("result"), dict)
                    else []
                )
            }
            & {"stale_perception", "stale_world_state"}
        ]
        if len(stale_failures) != 1:
            errors.append(
                "diagnostic_grounded_stale_preflight_failure_count_mismatch:"
                f"actual={len(stale_failures)}"
            )
        refresh_indexes = [
            int(step.get("step_index", -1))
            for step in diagnostic_steps
            if (step.get("decision") or {}).get("next_skill")
            == "refresh_preflight_observation"
            and step.get("result", {}).get("success") is True
        ]
        passed_indexes = [
            int(step.get("step_index", -1))
            for step in diagnostic_steps
            if (step.get("decision") or {}).get("next_skill") == "preflight_action"
            and step.get("status") == "preflight_passed"
        ]
        failure_index = int(stale_failures[0].get("step_index", -1)) if stale_failures else -1
        if not refresh_indexes or not passed_indexes or not (
            failure_index < refresh_indexes[-1] < passed_indexes[-1]
        ):
            errors.append("diagnostic_grounded_stale_preflight_not_repaired_in_order")

    reobserve = scenarios.get("direct_vla_verify_occluded_reobserve")
    if reobserve is not None:
        verifier_calls = _matching_calls(
            runtime_calls,
            scenario="direct_vla_verify_occluded_reobserve",
            component="verifier",
        )
        if len(verifier_calls) != 1 or verifier_calls[0].get("target", {}).get(
            "next_action"
        ) != "reobserve":
            errors.append("reobserve_branch_missing_reobserve_verifier_target")
        repairs = _matching_calls(
            runtime_calls,
            scenario="direct_vla_verify_occluded_reobserve",
            component="scheduler",
            next_skill="repair_stage_transition",
        )
        if not repairs or repairs[-1].get("target", {}).get("payload", {}).get(
            "target_stage"
        ) != "observe":
            errors.append("reobserve_branch_missing_observe_repair_target")

    for recovery_scenario in EXECUTION_FAILURE_RECOVERY_SCENARIOS:
        recover = scenarios.get(recovery_scenario)
        if recover is None:
            continue
        recovery_calls = _matching_calls(
            runtime_calls,
            scenario=recovery_scenario,
            component="recovery",
        )
        if len(recovery_calls) != 1:
            errors.append("recover_branch_missing_recovery_component_target")
        for skill in ("decide_recovery", "build_retry_request"):
            if not _matching_calls(
                runtime_calls,
                scenario=recovery_scenario,
                component="scheduler",
                next_skill=skill,
            ):
                errors.append(f"recover_branch_missing_scheduler_target:{skill}")
        retry_repairs = _matching_calls(
            runtime_calls,
            scenario=recovery_scenario,
            component="scheduler",
            next_skill="repair_stage_transition",
        )
        if not any(
            call.get("target", {}).get("payload", {}).get("target_stage") == "preflight"
            and call.get("target", {}).get("reason") == "recovery_retry_request_ready"
            for call in retry_repairs
        ):
            errors.append("recover_branch_missing_retry_to_preflight_target")

    stale_refresh = scenarios.get("grounded_preflight_stale_visual_refresh")
    if stale_refresh is not None:
        refresh_targets = _matching_calls(
            runtime_calls,
            scenario="grounded_preflight_stale_visual_refresh",
            component="scheduler",
            next_skill="refresh_preflight_observation",
        )
        if len(refresh_targets) != 1:
            errors.append(
                f"stale_preflight_refresh_target_count_mismatch:{len(refresh_targets)}"
            )
        if _loop_result(stale_refresh).get("final_stage") != "execute":
            errors.append("stale_preflight_refresh_final_stage_not_execute")

    replan = scenarios.get("direct_vla_preflight_invalid_plan_replan")
    if replan is not None:
        repairs = _matching_calls(
            runtime_calls,
            scenario="direct_vla_preflight_invalid_plan_replan",
            component="scheduler",
            next_skill="repair_stage_transition",
        )
        if len(repairs) != 1 or repairs[0].get("target", {}).get("payload", {}).get(
            "target_stage"
        ) != "plan":
            errors.append("invalid_plan_branch_missing_repair_to_plan_target")
        if _current_subgoal_id(replan) != "S1":
            errors.append("invalid_plan_branch_current_subgoal_not_s1")

    return errors


def _loop_result(scenario: dict[str, Any]) -> dict[str, Any]:
    value = scenario.get("loop_result")
    return value if isinstance(value, dict) else {}


def _loop_steps(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    value = _loop_result(scenario).get("steps")
    return value if isinstance(value, list) else []


def _executed_action_steps(scenario: dict[str, Any]) -> int | None:
    reports = _executed_action_reports(scenario)
    if reports:
        value = reports[0].get("executed_steps")
        return int(value) if isinstance(value, int) else None
    return None


def _executed_action_reports(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for step in _loop_steps(scenario):
        decision = step.get("decision") if isinstance(step.get("decision"), dict) else {}
        if decision.get("next_skill") != "execute_action":
            continue
        result = step.get("result") if isinstance(step.get("result"), dict) else {}
        output = result.get("output") if isinstance(result.get("output"), dict) else {}
        execution_report = output.get("execution_report")
        execution_report = execution_report if isinstance(execution_report, dict) else output
        reports.append(execution_report)
    return reports


def _current_subgoal_id(scenario: dict[str, Any]) -> str | None:
    compact = scenario.get("final_compact_context")
    compact = compact if isinstance(compact, dict) else {}
    task_plan = compact.get("task_plan")
    task_plan = task_plan if isinstance(task_plan, dict) else {}
    value = task_plan.get("current_subgoal_id")
    return str(value) if value is not None else None


def _matching_calls(
    runtime_calls: list[dict[str, Any]],
    *,
    scenario: str,
    component: str,
    next_skill: str | None = None,
) -> list[dict[str, Any]]:
    calls = [
        call
        for call in runtime_calls
        if call.get("scenario") == scenario and call.get("component") == component
    ]
    if next_skill is not None:
        calls = [
            call for call in calls if call.get("target", {}).get("next_skill") == next_skill
        ]
    return calls


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_info(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _file_sha(path)}


if __name__ == "__main__":
    main()
