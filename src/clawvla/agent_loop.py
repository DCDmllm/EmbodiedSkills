from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from .blackboard_utils import current_observation_id, mark_motion_artifacts_stale, metadata_value
from .loop_types import (
    ADVANCE_STAGE,
    FINISH_RUN,
    MAX_ACTION_HORIZON,
    MIN_ACTION_HORIZON,
    RUN_SKILL,
    LoopDecision,
    LoopRunResult,
    LoopStepRecord,
)
from .notices import emit_human_trace, emit_runtime_event, emit_status_notice
from .phase_policy import PhasePolicy
from .runtime import AgentRuntime
from .schema import SkillResult, Subgoal, TaskPlan
from .task_semantics import (
    action_backend_requires_candidate_bindings,
    action_backend_task_plan_contract,
    task_requires_target,
)


REPAIR_TARGET_STAGES = {"observe", "plan", "preflight", "recover"}


@dataclass
class AgentLoopConfig:
    max_steps: int = 12
    initial_stage: str = "observe"
    stop_on_skill_error: bool = False
    allow_same_decision_repeats: int = 3
    failed_skill_stall_limit: int = 5
    no_progress_action_stall_limit: int = 6
    scheduler_payload: dict[str, Any] = field(default_factory=dict)


class AgentLoop:
    def __init__(
        self,
        runtime: AgentRuntime,
        policy: PhasePolicy | None = None,
        config: AgentLoopConfig | None = None,
    ):
        self.runtime = runtime
        self.policy = policy or PhasePolicy()
        self.config = config or AgentLoopConfig()

    def run(self) -> LoopRunResult:
        stage = self.policy.normalize_stage(self.runtime.blackboard.read("stage") or self.config.initial_stage)
        self.runtime.blackboard.write("stage", stage, event_type="loop.stage_initialized")
        records: list[LoopStepRecord] = []
        stall_state: dict[str, Any] = {
            "last_failed_key": None,
            "failed_count": 0,
            "no_progress_action_count": 0,
            "last_action_subgoal": None,
        }

        for step_index in range(self.config.max_steps):
            stage_before = stage
            decision_result = self._choose_decision(stage)
            if not decision_result.success:
                decision = LoopDecision(
                    control=RUN_SKILL,
                    stage=stage,
                    next_component="scheduler",
                    next_skill="choose_next_skill",
                    reason="scheduler_failed_before_decision",
                )
                emit_runtime_event(
                    "clawvla_scheduler_failure_recorded",
                    {
                        "step_index": step_index,
                        "stage": stage,
                        "status": decision_result.status,
                        "errors": decision_result.errors[:3],
                    },
                )
                records.append(LoopStepRecord(step_index, stage_before, decision, decision_result.status, decision_result.to_dict()))
                self._write_loop_history(records)
                stall_reason = self._stall_reason(decision, decision_result, stall_state)
                if stall_reason:
                    return LoopRunResult("stalled_loop", stage, records, reason=stall_reason)
                if self.config.stop_on_skill_error:
                    return LoopRunResult("scheduler_failed", stage, records, reason=decision_result.status)
                continue
            decision = self._decision_from_result(decision_result)
            decision = self._enforce_completion_authority(stage, decision)
            if decision.control == RUN_SKILL:
                decision.stage = self.policy.normalize_stage(decision.stage or stage)
            elif decision.stage is not None:
                decision.stage = self.policy.normalize_stage(decision.stage)
            emit_runtime_event(
                "clawvla_loop_decision",
                {
                    "step_index": step_index,
                    "stage_before": stage_before,
                    "control": decision.control,
                    "next_component": decision.next_component,
                    "next_skill": decision.next_skill,
                    "decision_stage": decision.stage,
                    "reason": decision.reason,
                    "narration": decision.narration,
                    "state_summary": decision.state_summary,
                    "expected_result": decision.expected_result,
                    "decision_status": decision_result.status,
                    "decision_success": decision_result.success,
                },
            )
            if decision.narration or decision.state_summary or decision.expected_result:
                emit_human_trace(
                    "scheduler",
                    decision.narration or decision.reason or "scheduler selected next action",
                    detail=_trace_detail(decision),
                )
            self.runtime.blackboard.write("last_loop_decision", decision, event_type="loop.decision")

            if decision.control == FINISH_RUN:
                if self._environment_oracle_blocks_completion():
                    self._restore_oracle_owned_subgoal()
                    self._clear_verify_observation("finish_blocked_by_environment_oracle")
                    self.runtime.blackboard.write(
                        "last_verification_report",
                        None,
                        event_type="loop.environment_oracle_rejected_finish",
                    )
                    mark_motion_artifacts_stale(
                        self.runtime.blackboard,
                        "environment_oracle_rejected_finish",
                        include_goal=True,
                    )
                    stage = "preflight"
                    self.runtime.blackboard.write(
                        "stage",
                        stage,
                        event_type="loop.environment_oracle_rejected_finish",
                    )
                    records.append(
                        LoopStepRecord(
                            step_index,
                            stage_before,
                            decision,
                            "finish_blocked_by_environment_oracle",
                            decision_result.to_dict(),
                        )
                    )
                    self._write_loop_history(records)
                    continue
                if stage == "verify":
                    self._clear_verify_observation("finish_run")
                records.append(LoopStepRecord(step_index, stage_before, decision, "finished", decision_result.to_dict()))
                self._write_loop_history(records)
                return LoopRunResult("finished", stage, records, reason=decision.reason)

            if decision.control == ADVANCE_STAGE:
                error = self._validate_advance_stage(stage, decision)
                if error is not None:
                    emit_runtime_event(
                        "clawvla_decision_blocked",
                        {
                            "step_index": step_index,
                            "stage": stage,
                            "control": decision.control,
                            "requested_stage": decision.stage,
                            "reason": error,
                        },
                    )
                    records.append(LoopStepRecord(step_index, stage_before, decision, "invalid_decision", decision_result.to_dict(), error))
                    self._write_loop_history(records)
                    if self.config.stop_on_skill_error:
                        return LoopRunResult("invalid_decision", stage, records, reason=error)
                    continue
                stage = self._advance_stage(stage, decision)
                records.append(LoopStepRecord(step_index, stage_before, decision, "stage_advanced", decision_result.to_dict()))
                self._write_loop_history(records)
                continue

            error = self._validate_run_skill_decision(decision)
            if error is not None:
                emit_runtime_event(
                    "clawvla_decision_blocked",
                    {
                        "step_index": step_index,
                        "stage": stage,
                        "control": decision.control,
                        "next_component": decision.next_component,
                        "next_skill": decision.next_skill,
                        "reason": error,
                    },
                )
                records.append(LoopStepRecord(step_index, stage_before, decision, "invalid_decision", decision_result.to_dict(), error))
                self._write_loop_history(records)
                invalid_result = SkillResult(success=False, status="invalid_decision", errors=[error])
                stall_reason = self._stall_reason(decision, invalid_result, stall_state)
                if stall_reason:
                    return LoopRunResult("stalled_loop", stage, records, reason=stall_reason)
                if self.config.stop_on_skill_error:
                    return LoopRunResult("invalid_decision", stage, records, reason=error)
                continue

            self._before_skill(decision, step_index)
            result = self.runtime.run_skill(
                str(decision.next_component),
                str(decision.next_skill),
                self._prepare_payload(decision),
                stage=decision.stage,
                budget_steps=decision.budget_steps,
                metadata={"loop_step": step_index, "loop_stage": stage},
            )
            self._after_skill(decision, step_index, result)
            record = LoopStepRecord(step_index, stage_before, decision, result.status, result.to_dict())
            records.append(record)
            self._write_loop_history(records)
            if self._environment_oracle_succeeded(decision, result):
                self._mark_environment_task_succeeded()
                emit_status_notice(
                    "environment_task_succeeded",
                    success=True,
                    source="agent_loop",
                    reason="environment_oracle_success",
                    always=True,
                )
                emit_runtime_event(
                    "clawvla_environment_task_succeeded",
                    {
                        "step_index": step_index,
                        "stage": stage,
                        "component": decision.next_component,
                        "skill": decision.next_skill,
                    },
                )
                return LoopRunResult("finished", stage, records, reason="environment_oracle_success")
            stall_reason = self._stall_reason(decision, result, stall_state)
            if stall_reason:
                return LoopRunResult("stalled_loop", stage, records, reason=stall_reason)
            if not result.success:
                emit_runtime_event(
                    "clawvla_skill_failure_recorded",
                    {
                        "step_index": step_index,
                        "stage": stage,
                        "component": decision.next_component,
                        "skill": decision.next_skill,
                        "status": result.status,
                        "errors": result.errors[:3],
                    },
                )
                if self.config.stop_on_skill_error:
                    return LoopRunResult("skill_failed", stage, records, reason=result.status)
                continue
            self._post_skill_update(decision, result)
            self._apply_stage(decision)
            stage = self.policy.normalize_stage(self.runtime.blackboard.read("stage") or decision.stage or stage)

        return self._max_steps_result(stage, records)

    def _environment_oracle_succeeded(self, decision: LoopDecision, result: SkillResult) -> bool:
        if not result.success:
            return False
        if decision.next_component != "motion" or decision.next_skill != "execute_action":
            return False
        report = self.runtime.blackboard.read("execution_report")
        if isinstance(report, dict) and report.get("status") == "action_executed" and report.get("success") is True:
            return True
        env = self.runtime.blackboard.read("env_adapter")
        task_status = env.task_status() if env is not None and hasattr(env, "task_status") else None
        return isinstance(task_status, dict) and task_status.get("success") is True

    def _mark_environment_task_succeeded(self) -> None:
        task_plan = self.runtime.blackboard.read("task_plan")
        if isinstance(task_plan, TaskPlan):
            task_plan.status = "succeeded"
            for subgoal in task_plan.subgoals:
                if subgoal.status in {"pending", "running"}:
                    subgoal.status = "succeeded"
            self.runtime.blackboard.write("task_plan", task_plan, event_type="loop.environment_oracle_task_succeeded")
        current_subgoal = self.runtime.blackboard.read("current_subgoal")
        if isinstance(current_subgoal, Subgoal) and current_subgoal.status in {"pending", "running"}:
            current_subgoal.status = "succeeded"
            self.runtime.blackboard.write(
                "current_subgoal",
                current_subgoal,
                event_type="loop.environment_oracle_subgoal_succeeded",
            )

    def _stall_reason(
        self,
        decision: LoopDecision,
        result: SkillResult,
        state: dict[str, Any],
    ) -> str | None:
        if not result.success:
            key = self._failed_skill_signature(decision, result)
            state["failed_count"] = int(state.get("failed_count", 0)) + 1 if state.get("last_failed_key") == key else 1
            state["last_failed_key"] = key
            if int(state["failed_count"]) >= int(self.config.failed_skill_stall_limit):
                return (
                    "repeated_failed_skill:"
                    f"{decision.next_component}.{decision.next_skill}:"
                    f"count={state['failed_count']}"
                )
            return None

        state["last_failed_key"] = None
        state["failed_count"] = 0
        if decision.next_component != "motion" or decision.next_skill != "execute_action":
            return None
        subgoal = self.runtime.blackboard.read("current_subgoal")
        subgoal_id = getattr(subgoal, "subgoal_id", None)
        if state.get("last_action_subgoal") not in {None, subgoal_id}:
            state["no_progress_action_count"] = 0
        state["last_action_subgoal"] = subgoal_id
        progress_payload = self.runtime.blackboard.read("rl_last_reward_progress")
        progressed = bool(progress_payload.get("progress")) if isinstance(progress_payload, dict) else True
        state["no_progress_action_count"] = 0 if progressed else int(state.get("no_progress_action_count", 0)) + 1
        if int(state["no_progress_action_count"]) >= int(self.config.no_progress_action_stall_limit):
            return (
                "successful_actions_without_progress:"
                f"subgoal={subgoal_id}:count={state['no_progress_action_count']}"
            )
        return None

    @staticmethod
    def _failed_skill_signature(decision: LoopDecision, result: SkillResult) -> str:
        payload = json.dumps(decision.payload, sort_keys=True, ensure_ascii=True, default=str)
        errors = tuple(str(error) for error in (result.errors or [])[:3])
        return json.dumps(
            [decision.stage, decision.next_component, decision.next_skill, payload, result.status, errors],
            ensure_ascii=True,
            default=str,
        )

    def _choose_decision(self, stage: str) -> SkillResult:
        allowed_skills = self._state_gated_allowed_skills(
            stage,
            self._enabled_allowed_skills(self.policy.allowed_for_stage(stage)),
        )
        payload = {
            "use_model": True,
            "loop_mode": True,
            "current_stage": stage,
            "stage_order": list(self.policy.stage_order),
            "allowed_skills": allowed_skills,
            "runtime_state": self._runtime_state_summary(),
            "image_paths": self._current_vlm_image_paths(),
            **self.config.scheduler_payload,
        }
        return self.runtime.run_skill("scheduler", "choose_next_skill", payload, stage=stage)

    def _max_steps_result(self, stage: str, records: list[LoopStepRecord]) -> LoopRunResult:
        reason = f"max_steps={self.config.max_steps}"
        failure_statuses = {
            "invalid_decision",
            "scheduler_failed",
            "skill_exception",
            "skill_failed",
        }
        failed_records: list[str] = []
        for record in records:
            result = record.result if isinstance(record.result, dict) else {}
            success = result.get("success")
            if record.status in failure_statuses or success is False:
                failed_records.append(f"step={record.step_index}:status={record.status}")
        if failed_records:
            return LoopRunResult(
                "max_steps_reached_with_failures",
                stage,
                records,
                reason=f"{reason}; failures={';'.join(failed_records[-3:])}",
            )
        return LoopRunResult("max_steps_reached", stage, records, reason=reason)

    def _enabled_allowed_skills(self, allowed_skills: dict[str, list[str]]) -> dict[str, list[str]]:
        enabled_components = set(self.runtime.components.names())
        return {
            component: list(skills)
            for component, skills in allowed_skills.items()
            if component in enabled_components
        }

    def _state_gated_allowed_skills(self, stage: str, allowed_skills: dict[str, list[str]]) -> dict[str, list[str]]:
        if stage == "verify" and self.runtime.blackboard.read("last_verification_report") is not None:
            if self._task_plan_complete():
                return {}
            verification = self.runtime.blackboard.read("last_verification_report")
            if self._verification_next_action(verification) == "advance_subgoal":
                return {"scheduler": ["advance_subgoal"]} if "scheduler" in self.runtime.components.names() else {}
            if self._verification_next_action(verification) in {"continue_execute", "reobserve", "replan", "recover"}:
                return {"scheduler": ["repair_stage_transition"]} if "scheduler" in self.runtime.components.names() else {}
            return {}
        gated = {component: list(skills) for component, skills in allowed_skills.items()}
        for component, skills in list(gated.items()):
            gated[component] = [
                skill
                for skill in skills
                if self._skill_prerequisite_error(component, skill) is None
            ]
        return {component: skills for component, skills in gated.items() if skills}

    def _decision_from_result(self, result: SkillResult) -> LoopDecision:
        payload = result.output.get("loop_decision") or result.output.get("decision") or {}
        if not isinstance(payload, dict):
            payload = {}
        return LoopDecision.from_payload(payload)

    def _advance_stage(self, current_stage: str, decision: LoopDecision) -> str:
        next_stage = self.policy.normalize_stage(self.policy.next_stage(current_stage))
        if current_stage == "verify":
            self._clear_verify_observation(f"advance_stage:{current_stage}->{next_stage}")
        emit_status_notice(
            "stage_advanced",
            success=True,
            source="agent_loop.advance_stage",
            reason=f"{current_stage}->{next_stage}",
            always=True,
        )
        self.runtime.blackboard.write("stage", next_stage, event_type="loop.stage_advanced")
        return next_stage

    def _apply_stage(self, decision: LoopDecision) -> None:
        if decision.next_component == "motion" and decision.next_skill == "execute_action":
            return
        if decision.next_component == "scheduler" and decision.next_skill in {"repair_stage_transition", "advance_subgoal"}:
            return
        if decision.stage:
            self.runtime.blackboard.write("stage", decision.stage, event_type="loop.stage_set")

    def _post_skill_update(self, decision: LoopDecision, result: SkillResult) -> None:
        is_execute_action = decision.next_component == "motion" and decision.next_skill == "execute_action"
        should_update_world_state = (
            decision.next_component in {"motion", "verifier"} and not is_execute_action
        ) or (
            decision.next_component == "vision"
            and decision.next_skill
            in {
                "perceive_scene",
                "localize_task_objects",
                "lift_depth_cluster",
                "lift_geometry",
                "bind_arm",
                "estimate_uncertainty",
            }
        )
        should_update_world_state = (
            should_update_world_state and self.runtime.blackboard.read("perception") is not None
        )
        if should_update_world_state:
            self.runtime.run_skill("state", "update_world_state", {"stage": decision.stage}, stage=decision.stage)
        if is_execute_action and result.status == "action_executed":
            self.runtime.blackboard.write("stage", "verify", event_type="loop.stage_forced_after_execute_action")
        if (
            decision.stage == "verify"
            and decision.next_component == "scheduler"
            and decision.next_skill in {"repair_stage_transition", "advance_subgoal"}
            and result.success
        ):
            self._clear_verify_observation(f"verify_{decision.next_skill}_completed")

    def _prepare_payload(self, decision: LoopDecision) -> dict[str, Any]:
        payload = dict(decision.payload)
        if decision.next_component == "vision" and decision.next_skill == "capture_views":
            payload.setdefault("artifact_prefix", self.runtime.blackboard.read("artifact_prefix") or "agent_loop")
            if self.runtime.blackboard.read("run_environment") and self._environment_needs_setup():
                payload.setdefault("setup", True)
                payload.setdefault("instruction", self.runtime.blackboard.task_instruction)
        if decision.next_component == "vision" and decision.next_skill == "refresh_preflight_observation":
            payload.setdefault("artifact_prefix", self.runtime.blackboard.read("artifact_prefix") or "agent_loop")
            payload.setdefault("instruction", self.runtime.blackboard.task_instruction)
        if decision.next_component == "vision" and decision.next_skill == "capture_verify_views":
            payload.setdefault("artifact_prefix", self.runtime.blackboard.read("artifact_prefix") or "agent_loop")
            payload.setdefault("instruction", self.runtime.blackboard.task_instruction)
        if decision.next_component == "vision" and decision.next_skill in {
            "perceive_scene",
            "localize_task_objects",
            "render_grounding_overlay",
            "estimate_uncertainty",
        }:
            payload.setdefault("use_model", True)
            payload.setdefault("image_paths", self._current_vlm_image_paths())
        if decision.next_component == "scheduler" and decision.next_skill == "build_task_plan":
            payload.setdefault("use_model", True)
            payload.setdefault("image_paths", self._current_vlm_image_paths())
        if decision.next_component == "verifier" and decision.next_skill == "verify_progress":
            payload.setdefault("use_model", True)
            payload.setdefault("image_paths", self._current_verify_image_paths())
        if decision.next_component == "recovery" and decision.next_skill == "decide_recovery":
            payload.setdefault("use_model", True)
            payload.setdefault("image_paths", self._recovery_evidence_image_paths())
        if decision.next_component == "motion" and decision.next_skill == "emit_action_chunk":
            payload.setdefault("horizon", MAX_ACTION_HORIZON)
        return payload

    def _before_skill(self, decision: LoopDecision, step_index: int) -> None:
        tracker = self.runtime.blackboard.read("rl_reward_tracker")
        hook = getattr(tracker, "before_skill", None)
        if callable(hook):
            hook(
                blackboard=self.runtime.blackboard,
                component=str(decision.next_component),
                skill=str(decision.next_skill),
                step_index=step_index,
            )

    def _after_skill(self, decision: LoopDecision, step_index: int, result: SkillResult) -> None:
        tracker = self.runtime.blackboard.read("rl_reward_tracker")
        hook = getattr(tracker, "after_skill", None)
        if callable(hook):
            hook(
                blackboard=self.runtime.blackboard,
                component=str(decision.next_component),
                skill=str(decision.next_skill),
                step_index=step_index,
                result=result,
            )

    def _environment_needs_setup(self) -> bool:
        env = self.runtime.blackboard.read("env_adapter")
        status = env.status() if env is not None and hasattr(env, "status") else {}
        if isinstance(status, dict) and "needs_setup" in status:
            return bool(status["needs_setup"])
        return env is None

    def _current_image_paths(self) -> list[str]:
        observation = self.runtime.blackboard.read("observation")
        camera_views = getattr(observation, "camera_views", {})
        if not isinstance(camera_views, dict):
            return []
        return [view.rgb_path for view in camera_views.values() if getattr(view, "rgb_path", None)]

    def _write_loop_history(self, records: list[LoopStepRecord]) -> None:
        self.runtime.blackboard.write("loop_history", list(records), event_type="loop.history_updated")

    def _validate_run_skill_decision(self, decision: LoopDecision) -> str | None:
        if decision.control != RUN_SKILL:
            return f"unsupported_control:{decision.control}"
        if not decision.next_component or not decision.next_skill:
            return "missing_component_or_skill"
        if decision.next_component == "scheduler" and decision.next_skill == "choose_next_skill":
            return "scheduler_choose_next_skill_is_internal_only"
        current_stage = self.policy.normalize_stage(self.runtime.blackboard.read("stage") or decision.stage)
        requested_stage = self.policy.normalize_stage(decision.stage or current_stage)
        if requested_stage != current_stage:
            return f"run_skill_stage_must_equal_current_stage:{requested_stage}!={current_stage}"
        stage = current_stage
        if not self.policy.is_allowed(stage, decision.next_component, decision.next_skill, self.runtime.components):
            return f"skill_not_allowed:{stage}.{decision.next_component}.{decision.next_skill}"
        verify_route_error = self._verify_post_report_run_skill_error(
            stage,
            str(decision.next_component),
            str(decision.next_skill),
        )
        if verify_route_error is not None:
            return verify_route_error
        if decision.next_component == "scheduler" and decision.next_skill == "repair_stage_transition":
            repair_error = self._repair_stage_transition_error(decision)
            if repair_error is not None:
                return repair_error
        if decision.next_component == "vision" and decision.next_skill == "refresh_preflight_observation":
            refresh_error = self._preflight_observation_refresh_error(stage)
            if refresh_error is not None:
                return refresh_error
        prerequisite_error = self._skill_prerequisite_error(str(decision.next_component), str(decision.next_skill))
        if prerequisite_error is not None:
            return prerequisite_error
        if decision.next_component == "motion" and decision.next_skill == "emit_action_chunk":
            horizon_error = self._emit_action_horizon_error(decision)
            if horizon_error is not None:
                return horizon_error
        return None

    def _skill_prerequisite_error(self, component: str, skill: str) -> str | None:
        blackboard = self.runtime.blackboard
        if component == "vision":
            if skill in {"perceive_scene", "localize_task_objects", "estimate_uncertainty"} and blackboard.read("observation") is None:
                return f"missing_observation_before_{skill}"
            if skill == "capture_verify_views":
                execution_report = blackboard.read("execution_report")
                if not isinstance(execution_report, dict) or execution_report.get("status") != "action_executed":
                    return "missing_action_executed_report_before_capture_verify_views"
                if self._verify_observation_fresh():
                    return "verify_observation_already_captured_run_verify_progress"
            if skill == "render_grounding_overlay":
                perception = blackboard.read("perception")
                if blackboard.read("observation") is None:
                    return "missing_observation_before_render_grounding_overlay"
                if perception is None or not getattr(perception, "candidates", []):
                    return "missing_perception_candidates_before_render_grounding_overlay"
                if not _perception_has_bbox(perception):
                    return "missing_bbox_before_render_grounding_overlay"
        if component == "scheduler":
            if skill == "build_task_plan":
                task_plan = blackboard.read("task_plan")
                if getattr(task_plan, "subgoals", None):
                    return "task_plan_already_built"
                if self._world_state_ready_error() is not None:
                    return self._world_state_ready_error()
            if skill == "select_current_subgoal" and blackboard.read("task_plan") is None:
                return "missing_task_plan_before_select_current_subgoal"
            if skill == "select_current_subgoal" and blackboard.read("current_subgoal") is not None:
                return "current_subgoal_already_selected"
            if skill == "advance_subgoal":
                verification = blackboard.read("last_verification_report")
                if blackboard.read("task_plan") is None:
                    return "missing_task_plan_before_advance_subgoal"
                if blackboard.read("current_subgoal") is None:
                    return "missing_current_subgoal_before_advance_subgoal"
                if verification is None or not getattr(verification, "success", False):
                    return "missing_successful_verification_before_advance_subgoal"
            if skill == "repair_stage_transition" and not self._repair_stage_transition_available():
                return "no_repair_condition_for_stage_transition"
        if component == "motion":
            preflight_error = self._preflight_ready_error()
            if preflight_error is not None:
                return preflight_error
            if skill == "build_motion_goal":
                if blackboard.read("current_subgoal") is None:
                    return "missing_current_subgoal_before_build_motion_goal"
                if self._candidate_bindings_required() and blackboard.read("world_state") is None:
                    return "missing_world_state_before_build_motion_goal"
            if skill == "plan_motion" and not self._motion_goal_fresh():
                return "missing_fresh_motion_goal_before_plan_motion"
            if skill == "emit_action_chunk" and not self._motion_plan_fresh():
                return "missing_fresh_motion_plan_before_emit_action_chunk"
            if skill == "validate_action_chunk" and not self._action_chunk_fresh():
                return "missing_fresh_action_chunk_before_validate_action_chunk"
            if skill == "execute_action" and not self._action_chunk_fresh():
                return "missing_fresh_action_chunk_before_execute_action"
        if component == "verifier" and skill == "verify_progress":
            if blackboard.read("current_subgoal") is None:
                return "missing_current_subgoal_before_verify_progress"
            if blackboard.read("execution_report") is None:
                return "missing_execution_report_before_verify_progress"
            if not self._verify_observation_fresh():
                return "missing_fresh_verify_observation_before_verify_progress"
            if not self._current_verify_image_paths():
                return "missing_verify_images_before_verify_progress"
        if component == "recovery":
            if skill == "decide_recovery" and blackboard.read("last_verification_report") is None and blackboard.read("execution_report") is None:
                return "missing_failure_report_before_decide_recovery"
            if skill == "build_retry_request" and blackboard.read("last_recovery_directive") is None:
                return "missing_recovery_directive_before_build_retry_request"
        return None

    def _validate_advance_stage(self, current_stage: str, decision: LoopDecision) -> str | None:
        if decision.stage is not None:
            return f"advance_stage_must_not_set_destination_stage:{decision.stage}"
        next_stage = self.policy.normalize_stage(self.policy.next_stage(current_stage))
        if next_stage == current_stage:
            return f"advance_stage_noop:{current_stage}"
        if current_stage == "observe":
            return self._world_state_ready_error()
        if current_stage == "plan":
            if next_stage != "preflight":
                return f"plan_must_advance_to_preflight_before_execute:{next_stage}"
            if self.runtime.blackboard.read("task_plan") is None:
                return "missing_task_plan_before_leaving_plan"
            if self.runtime.blackboard.read("current_subgoal") is None:
                return "missing_current_subgoal_before_leaving_plan"
        if current_stage == "preflight" and next_stage == "execute":
            if self.runtime.blackboard.read("current_subgoal") is None:
                return "missing_current_subgoal_before_execute"
            preflight_error = self._preflight_ready_error()
            if preflight_error is not None:
                return preflight_error
        if current_stage == "execute" and next_stage == "verify":
            report = self.runtime.blackboard.read("execution_report")
            if not isinstance(report, dict) or report.get("status") != "action_executed":
                return "missing_successful_execution_report_before_verify"
        if current_stage == "verify":
            verification = self.runtime.blackboard.read("last_verification_report")
            if verification is None:
                return "missing_verification_report_before_leaving_verify"
            next_action = self._verification_next_action(verification)
            if next_stage == "recover" and next_action != "recover":
                return f"verify_next_action_mismatch:{next_action}->recover"
        if current_stage == "recover":
            return "recover_has_no_default_next_stage_use_repair_stage_transition"
        return None

    def _world_state_ready_error(self) -> str | None:
        if not self._candidate_bindings_required():
            if self.runtime.blackboard.read("observation") is None:
                return "missing_observation"
            return None
        world_state = self.runtime.blackboard.read("world_state")
        if world_state is None:
            return "missing_world_state"
        if getattr(world_state, "needs_reobserve", False):
            return "world_state_requires_reobserve"
        if not getattr(world_state, "source_candidate_id", None):
            return "missing_source_candidate"
        if task_requires_target(self.runtime.blackboard.task_instruction) and not getattr(
            world_state,
            "target_candidate_id",
            None,
        ):
            return "missing_target_candidate"
        return None

    def _candidate_bindings_required(self) -> bool:
        return action_backend_requires_candidate_bindings(
            self.runtime.blackboard.read("action_backend")
        )

    def _runtime_state_summary(self) -> dict[str, Any]:
        obs_id = current_observation_id(self.runtime.blackboard)
        perception = self.runtime.blackboard.read("perception")
        world_state = self.runtime.blackboard.read("world_state")
        perception_obs_id = getattr(perception, "observation_id", None)
        world_state_obs_id = metadata_value(world_state, "observation_id")
        perception_fresh = perception is not None and obs_id is not None and perception_obs_id == obs_id
        world_state_fresh = world_state is not None and obs_id is not None and world_state_obs_id == obs_id
        world_state_ready_error = self._world_state_ready_error()
        motion_plan = self.runtime.blackboard.read("motion_plan")
        action_chunk = self.runtime.blackboard.read("action_chunk")
        task_plan = self.runtime.blackboard.read("task_plan")
        current_subgoal = self.runtime.blackboard.read("current_subgoal")
        overlay = self.runtime.blackboard.read("grounding_overlay")
        policy = getattr(self, "policy", PhasePolicy())
        current_stage = policy.normalize_stage(self.runtime.blackboard.read("stage"))
        motion_goal = self.runtime.blackboard.read("motion_goal")
        verify_observation = self.runtime.blackboard.read("verify_observation")
        verify_observation_id = getattr(verify_observation, "observation_id", None)
        last_budget_steps = self.runtime.blackboard.read("last_budget_steps")
        preflight_error = self._preflight_ready_error()
        preflight_errors = list(getattr(self.runtime.blackboard.read("preflight_report"), "errors", []) or [])
        stale_visual_errors = {"stale_perception", "stale_world_state"} & {str(error) for error in preflight_errors}
        next_required_decision = self._next_required_decision_summary(current_stage, preflight_error)
        candidate_bindings_required = self._candidate_bindings_required()
        target_candidate_required = candidate_bindings_required and task_requires_target(
            self.runtime.blackboard.task_instruction
        )
        visual_state_fresh = (
            perception_fresh and world_state_fresh
            if candidate_bindings_required
            else obs_id is not None
        )
        return {
            "observation_id": obs_id,
            "observation_present": obs_id is not None,
            "verify_observation_id": str(verify_observation_id) if verify_observation_id is not None else None,
            "verify_observation_present": verify_observation is not None,
            "verify_observation_fresh": self._verify_observation_fresh(),
            "verify_image_count": len(self._current_verify_image_paths()),
            "current_stage": current_stage,
            "perception_observation_id": str(perception_obs_id) if perception_obs_id is not None else None,
            "world_state_observation_id": str(world_state_obs_id) if world_state_obs_id is not None else None,
            "perception_fresh_for_current_observation": perception_fresh,
            "world_state_fresh_for_current_observation": world_state_fresh,
            "visual_state_fresh_for_current_observation": visual_state_fresh,
            "stale_visual_state_unresolved": bool(stale_visual_errors) and not visual_state_fresh,
            "perception_source_candidate_id": getattr(perception, "source_candidate_id", None),
            "perception_target_candidate_id": getattr(perception, "target_candidate_id", None),
            "world_state_source_candidate_id": getattr(world_state, "source_candidate_id", None),
            "world_state_target_candidate_id": getattr(world_state, "target_candidate_id", None),
            "target_candidate_required": target_candidate_required,
            "candidate_bindings_required": candidate_bindings_required,
            "world_state_needs_reobserve": getattr(world_state, "needs_reobserve", None),
            "world_state_ready": world_state_ready_error is None,
            "world_state_ready_error": world_state_ready_error,
            "observe_complete": world_state_ready_error is None and visual_state_fresh,
            "grounding_overlay_fresh": self._grounding_overlay_fresh(),
            "task_plan_present": task_plan is not None,
            "task_plan_status": getattr(task_plan, "status", None),
            "task_plan_complete": self._task_plan_complete(),
            "current_subgoal": current_subgoal.to_dict() if hasattr(current_subgoal, "to_dict") else current_subgoal,
            "current_subgoal_present": current_subgoal is not None,
            "plan_ready": task_plan is not None and current_subgoal is not None,
            "last_budget_steps": last_budget_steps,
            "budget_allocated": last_budget_steps is not None,
            "motion_artifacts_relevant": current_stage == "execute",
            "motion_goal_missing_expected_before_execute": current_stage in {"observe", "plan", "preflight"}
            and motion_goal is None,
            "motion_goal_present": motion_goal is not None,
            "motion_goal_fresh": self._motion_goal_fresh(),
            "motion_plan_present": motion_plan is not None,
            "motion_plan_fresh": self._motion_plan_fresh(),
            "motion_plan_status": getattr(motion_plan, "status", None) if not isinstance(motion_plan, dict) else motion_plan.get("status"),
            "action_chunk_present": action_chunk is not None,
            "action_chunk_fresh": self._action_chunk_fresh(),
            "action_chunk_type": getattr(action_chunk, "action_type", None),
            "action_chunk_command_count": len(getattr(action_chunk, "commands", []) or []),
            "grounding_overlay_stale": getattr(overlay, "stale", None),
            "preflight_status": getattr(self.runtime.blackboard.read("preflight_report"), "status", None),
            "preflight_allowed": getattr(self.runtime.blackboard.read("preflight_report"), "allowed", None),
            "preflight_errors": preflight_errors,
            "preflight_ready": preflight_error is None,
            "preflight_next_required_control": "advance_stage" if preflight_error is None else None,
            "preflight_error": preflight_error,
            "next_required_decision": next_required_decision,
            "verification_present": self.runtime.blackboard.read("last_verification_report") is not None,
            "verification_success": getattr(self.runtime.blackboard.read("last_verification_report"), "success", None),
            "verification_failure_type": getattr(self.runtime.blackboard.read("last_verification_report"), "failure_type", None),
            "verification_should_reobserve": getattr(
                self.runtime.blackboard.read("last_verification_report"),
                "should_reobserve",
                None,
            ),
            "verification_next_action": self._verification_next_action(self.runtime.blackboard.read("last_verification_report")),
        }

    def _next_required_decision_summary(self, current_stage: str, preflight_error: str | None) -> dict[str, Any] | None:
        blackboard = self.runtime.blackboard
        if current_stage == "observe":
            observation = blackboard.read("observation")
            perception = blackboard.read("perception")
            if observation is None:
                return _required_decision("run_skill", current_stage, "vision", "capture_views", reason="missing_observation")
            if not self._candidate_bindings_required():
                return _required_decision(
                    "advance_stage",
                    None,
                    reason="current_images_ready_for_direct_vla",
                )
            if perception is None:
                return _required_decision("run_skill", current_stage, "vision", "perceive_scene", reason="missing_perception")
            target_required = task_requires_target(self.runtime.blackboard.task_instruction)
            if not getattr(perception, "source_candidate_id", None) or (
                target_required and not getattr(perception, "target_candidate_id", None)
            ):
                reason = (
                    "missing_source_binding"
                    if not getattr(perception, "source_candidate_id", None)
                    else "missing_required_target_binding"
                )
                return _required_decision(
                    "run_skill",
                    current_stage,
                    "vision",
                    "localize_task_objects",
                    reason=reason,
                )
            if self._world_state_ready_error() is not None:
                return _required_decision("run_skill", current_stage, "state", "update_world_state", reason=self._world_state_ready_error())
            return _required_decision("advance_stage", None, reason="observe_complete")

        if current_stage == "plan":
            if blackboard.read("task_plan") is None:
                return _required_decision("run_skill", current_stage, "scheduler", "build_task_plan", reason="missing_task_plan")
            if blackboard.read("current_subgoal") is None:
                return _required_decision(
                    "run_skill",
                    current_stage,
                    "scheduler",
                    "select_current_subgoal",
                    reason="missing_current_subgoal",
                )
            return _required_decision("advance_stage", None, reason="plan_ready")

        if current_stage == "preflight":
            if preflight_error is None:
                return _required_decision("advance_stage", None, reason="preflight_passed_enter_execute")
            preflight_errors = self._preflight_error_codes()
            visual_errors = {
                "stale_perception",
                "stale_world_state",
                "world_state_requires_reobserve",
                "missing_observation",
                "missing_observation_id",
            }
            if preflight_errors & visual_errors:
                obs_id = current_observation_id(blackboard)
                perception = blackboard.read("perception")
                world_state = blackboard.read("world_state")
                perception_fresh = perception is not None and getattr(perception, "observation_id", None) == obs_id
                world_state_fresh = world_state is not None and metadata_value(world_state, "observation_id") == obs_id
                if perception_fresh and world_state_fresh:
                    return _required_decision(
                        "run_skill",
                        current_stage,
                        "safety",
                        "preflight_action",
                        reason="visual_state_refreshed_after_preflight_failure",
                    )
                return _required_decision(
                    "run_skill",
                    current_stage,
                    "vision",
                    "refresh_preflight_observation",
                    reason="preflight_visual_state_error",
                )
            if any(error.startswith("camera_") for error in preflight_errors):
                return _required_decision(
                    "run_skill",
                    current_stage,
                    "vision",
                    "refresh_preflight_observation",
                    reason="preflight_camera_error",
                )
            if preflight_error in {"missing_preflight_report_before_execute"} or preflight_error.startswith("stale_preflight_report:"):
                return _required_decision(
                    "run_skill",
                    current_stage,
                    "safety",
                    "preflight_action",
                    reason=preflight_error,
                )
            return None

        if current_stage == "execute":
            if preflight_error is not None:
                return _required_decision(
                    "run_skill",
                    current_stage,
                    "scheduler",
                    "repair_stage_transition",
                    payload={"target_stage": "preflight", "reason": preflight_error},
                    reason="preflight_not_ready_in_execute",
                )
            if not self._motion_goal_fresh():
                return _required_decision("run_skill", current_stage, "motion", "build_motion_goal", reason="missing_or_stale_motion_goal")
            if not self._motion_plan_fresh():
                return _required_decision("run_skill", current_stage, "motion", "plan_motion", reason="missing_or_stale_motion_plan")
            if not self._action_chunk_fresh():
                return _required_decision(
                    "run_skill",
                    current_stage,
                    "motion",
                    "emit_action_chunk",
                    reason="missing_or_stale_action_chunk",
                )
            return _required_decision("run_skill", current_stage, "motion", "execute_action", reason="fresh_action_chunk_ready")

        if current_stage == "verify":
            if self._task_plan_complete():
                if self._environment_oracle_blocks_completion():
                    self._restore_oracle_owned_subgoal()
                    return _required_decision(
                        "run_skill",
                        current_stage,
                        "scheduler",
                        "repair_stage_transition",
                        payload={
                            "target_stage": "preflight",
                            "reason": "environment_oracle_not_successful",
                        },
                        reason="environment_oracle_not_successful",
                    )
                return _required_decision("finish_run", None, reason="task_plan_complete")
            verification = blackboard.read("last_verification_report")
            if verification is None:
                if not self._verify_observation_fresh():
                    return _required_decision(
                        "run_skill",
                        current_stage,
                        "vision",
                        "capture_verify_views",
                        reason="missing_fresh_verify_observation",
                    )
                return _required_decision("run_skill", current_stage, "verifier", "verify_progress", reason="missing_verification_report")
            next_action = self._verification_next_action(verification)
            if next_action in {"advance_subgoal", "finish"} and self._environment_oracle_blocks_completion():
                return _required_decision(
                    "run_skill",
                    current_stage,
                    "scheduler",
                    "repair_stage_transition",
                    payload={
                        "target_stage": "preflight",
                        "reason": "environment_oracle_not_successful",
                    },
                    reason="environment_oracle_not_successful",
                )
            if next_action == "advance_subgoal":
                return _required_decision("run_skill", current_stage, "scheduler", "advance_subgoal", reason="verification_advance_subgoal")
            if next_action in {"continue_execute", "reobserve", "replan", "recover"}:
                target_stage = {
                    "continue_execute": "preflight",
                    "reobserve": "observe",
                    "replan": "plan",
                    "recover": "recover",
                }[next_action]
                return _required_decision(
                    "run_skill",
                    current_stage,
                    "scheduler",
                    "repair_stage_transition",
                    payload={"target_stage": target_stage, "reason": f"verification_next_action:{next_action}"},
                    reason=f"verification_next_action:{next_action}",
                )
            if next_action == "finish":
                return _required_decision("finish_run", None, reason="verification_requested_finish")
        return None

    def _task_plan_complete(self) -> bool:
        task_plan = self.runtime.blackboard.read("task_plan")
        return getattr(task_plan, "status", None) == "succeeded" and self.runtime.blackboard.read("current_subgoal") is None

    def _environment_oracle_blocks_completion(self) -> bool:
        blackboard = self.runtime.blackboard
        contract = action_backend_task_plan_contract(
            blackboard.read("action_backend"),
            blackboard.task_instruction,
        )
        if contract.get("completion_authority") != "environment_oracle":
            return False
        env = blackboard.read("env_adapter")
        if env is None or not hasattr(env, "task_status"):
            return False
        status = env.task_status()
        return isinstance(status, dict) and status.get("success") is False

    def _enforce_completion_authority(self, stage: str, decision: LoopDecision) -> LoopDecision:
        if stage != "verify" or not self._environment_oracle_blocks_completion():
            return decision
        attempts_completion = decision.control == FINISH_RUN or (
            decision.control == RUN_SKILL
            and decision.next_component == "scheduler"
            and decision.next_skill == "advance_subgoal"
        )
        if not attempts_completion:
            return decision
        return LoopDecision(
            control=RUN_SKILL,
            stage="verify",
            next_component="scheduler",
            next_skill="repair_stage_transition",
            payload={
                "target_stage": "preflight",
                "reason": "environment_oracle_not_successful",
            },
            reason="environment_oracle_not_successful",
            narration="CALVIN oracle has not confirmed success; continue the current atomic task.",
            state_summary=decision.state_summary,
            expected_result="Return to preflight and continue bounded execution until the environment oracle succeeds.",
            metadata={
                **dict(decision.metadata),
                "completion_decision_overridden": True,
                "original_control": decision.control,
                "original_next_component": decision.next_component,
                "original_next_skill": decision.next_skill,
            },
        )

    def _restore_oracle_owned_subgoal(self) -> None:
        blackboard = self.runtime.blackboard
        task_plan = blackboard.read("task_plan")
        if not isinstance(task_plan, TaskPlan) or not task_plan.subgoals:
            return
        current = blackboard.read("current_subgoal")
        if not isinstance(current, Subgoal):
            current = task_plan.subgoals[-1]
        current.status = "running"
        task_plan.status = "running"
        task_plan.current_subgoal_id = current.subgoal_id
        blackboard.write(
            "task_plan",
            task_plan,
            event_type="loop.environment_oracle_restored_task_plan",
        )
        blackboard.write(
            "current_subgoal",
            current,
            event_type="loop.environment_oracle_restored_subgoal",
        )

    def _preflight_ready_error(self) -> str | None:
        report = self.runtime.blackboard.read("preflight_report")
        if report is None:
            return "missing_preflight_report_before_execute"
        if not getattr(report, "allowed", False):
            errors = getattr(report, "errors", []) or []
            reason = str(errors[0]) if errors else getattr(report, "status", "preflight_not_allowed")
            return f"preflight_not_allowed:{reason}"
        if getattr(report, "status", None) != "preflight_passed":
            return f"preflight_status_not_passed:{getattr(report, 'status', None)}"
        report_obs_id = metadata_value(report, "observation_id")
        obs_id = current_observation_id(self.runtime.blackboard)
        if report_obs_id != obs_id:
            return f"stale_preflight_report:{report_obs_id}->{obs_id}"
        return None

    def _verification_next_action(self, verification: object | None) -> str | None:
        metadata = getattr(verification, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("next_action"):
            return str(metadata["next_action"])
        if getattr(verification, "success", False):
            return "advance_subgoal"
        if getattr(verification, "should_reobserve", False):
            return "reobserve"
        return None

    def _repair_stage_transition_available(self) -> bool:
        return bool(self._repair_allowed_target_stages())

    def _repair_stage_transition_error(self, decision: LoopDecision) -> str | None:
        target_stage = decision.payload.get("target_stage")
        if target_stage not in REPAIR_TARGET_STAGES:
            return f"invalid_repair_target_stage:{target_stage}:expected_{sorted(REPAIR_TARGET_STAGES)}"
        reason = str(decision.payload.get("reason") or "").strip()
        if not reason:
            return "missing_reason_for_repair_stage_transition"
        allowed_targets = self._repair_allowed_target_stages()
        if target_stage not in allowed_targets:
            return f"repair_stage_transition_target_not_allowed:{target_stage}:allowed_{sorted(allowed_targets)}"
        return None

    def _preflight_observation_refresh_error(self, stage: str) -> str | None:
        if stage != "preflight":
            return f"refresh_preflight_observation_only_allowed_in_preflight:{stage}"
        errors = self._preflight_error_codes()
        visual_errors = errors & {
            "stale_perception",
            "stale_world_state",
            "world_state_requires_reobserve",
            "missing_observation",
            "missing_observation_id",
        }
        if visual_errors:
            obs_id = current_observation_id(self.runtime.blackboard)
            perception = self.runtime.blackboard.read("perception")
            world_state = self.runtime.blackboard.read("world_state")
            perception_fresh = perception is not None and getattr(perception, "observation_id", None) == obs_id
            world_state_fresh = world_state is not None and metadata_value(world_state, "observation_id") == obs_id
            if perception_fresh and world_state_fresh:
                return "preflight_visual_state_already_refreshed_run_preflight_action"
            return None
        if any(error.startswith("camera_") for error in errors):
            return None
        return f"refresh_preflight_observation_requires_visual_preflight_error:{sorted(errors)}"

    def _repair_allowed_target_stages(self) -> set[str]:
        targets: set[str] = set()
        blackboard = self.runtime.blackboard
        current_stage = self.policy.normalize_stage(blackboard.read("stage"))
        preflight_error = self._preflight_ready_error()
        preflight_errors = self._preflight_error_codes()

        if current_stage == "execute" and preflight_error is not None:
            targets.add("preflight")
        if preflight_error and preflight_error.startswith("stale_preflight_report:"):
            targets.add("preflight")
        if preflight_errors & {
            "missing_task_plan",
            "missing_current_subgoal",
            "current_subgoal_mismatch_task_plan",
            "missing_source_candidate",
            "source_candidate_not_found",
            "source_label_missing",
            "source_visibility_no",
            "source_target_same_candidate",
        }:
            targets.add("plan")
        if any(error.startswith("missing_target_candidate") or error.startswith("target_") for error in preflight_errors):
            targets.add("plan")
        if preflight_errors & {
            "env_unavailable",
            "action_backend_missing",
            "action_backend_disabled",
            "action_backend_pretrained_path_missing",
        } or any(error.startswith(("openpi_worker_", "action_backend_")) for error in preflight_errors):
            targets.add("recover")

        verification = blackboard.read("last_verification_report")
        next_action = self._verification_next_action(verification)
        if verification is not None and not getattr(verification, "success", False):
            if next_action == "continue_execute":
                targets.add("preflight")
            elif next_action == "reobserve":
                targets.add("observe")
            elif next_action == "replan":
                targets.add("plan")
            elif next_action == "recover":
                targets.add("recover")
        if verification is not None and self._environment_oracle_blocks_completion():
            if next_action in {"advance_subgoal", "finish"}:
                targets.add("preflight")

        retry_request = blackboard.read("last_retry_request")
        if isinstance(retry_request, dict):
            retry_stage = retry_request.get("stage")
            if retry_stage in REPAIR_TARGET_STAGES:
                targets.add(str(retry_stage))

        world_state_error = self._world_state_ready_error()
        if current_stage == "plan" and world_state_error in {
            "missing_world_state",
            "world_state_requires_reobserve",
            "missing_observation",
            "missing_source_candidate",
            "missing_target_candidate",
        }:
            targets.add("observe")
        return targets

    def _preflight_error_codes(self) -> set[str]:
        report = self.runtime.blackboard.read("preflight_report")
        return {str(error) for error in (getattr(report, "errors", []) or [])}

    def _emit_action_horizon_error(self, decision: LoopDecision) -> str | None:
        horizon = decision.payload.get("horizon")
        if horizon is None:
            return None
        try:
            horizon_value = int(horizon)
        except (TypeError, ValueError):
            return f"invalid_horizon_before_emit_action_chunk:{horizon}"
        if horizon_value < MIN_ACTION_HORIZON or horizon_value > MAX_ACTION_HORIZON:
            return (
                f"horizon_out_of_range_before_emit_action_chunk:{horizon_value}:"
                f"expected_{MIN_ACTION_HORIZON}_to_{MAX_ACTION_HORIZON}"
            )
        return None

    def _verify_post_report_run_skill_error(self, stage: str, component: str, skill: str) -> str | None:
        if stage != "verify":
            return None
        verification = self.runtime.blackboard.read("last_verification_report")
        if verification is None:
            return None
        next_action = self._verification_next_action(verification)
        if (
            next_action in {"advance_subgoal", "finish"}
            and self._environment_oracle_blocks_completion()
            and component == "scheduler"
            and skill == "repair_stage_transition"
        ):
            return None
        if next_action == "advance_subgoal" and component == "scheduler" and skill == "advance_subgoal":
            return None
        if next_action in {"continue_execute", "reobserve", "replan", "recover"} and component == "scheduler" and skill == "repair_stage_transition":
            return None
        return f"verify_report_requires_next_action:{next_action}:not_{component}.{skill}"

    def _verification_allows_execute(self, verification: object) -> bool:
        next_action = self._verification_next_action(verification)
        if next_action == "continue_execute":
            return True
        if next_action != "advance_subgoal":
            return False
        metadata = getattr(verification, "metadata", None)
        verified_subgoal_id = metadata.get("current_subgoal_id") if isinstance(metadata, dict) else None
        current_subgoal = self.runtime.blackboard.read("current_subgoal")
        current_subgoal_id = getattr(current_subgoal, "subgoal_id", None)
        return current_subgoal_id is not None and current_subgoal_id != verified_subgoal_id

    def _current_vlm_image_paths(self) -> list[str]:
        if self._grounding_overlay_fresh():
            overlay = self.runtime.blackboard.read("grounding_overlay")
            paths = getattr(overlay, "image_paths", {})
            if isinstance(paths, dict):
                return [path for _, path in sorted(paths.items()) if path]
        return self._current_image_paths()

    def _current_verify_image_paths(self) -> list[str]:
        verify_observation = self.runtime.blackboard.read("verify_observation")
        if verify_observation is None:
            return []
        camera_views = getattr(verify_observation, "camera_views", {})
        if not isinstance(camera_views, dict):
            return []
        return [view.rgb_path for view in camera_views.values() if getattr(view, "rgb_path", None)]

    def _recovery_evidence_image_paths(self) -> list[str]:
        current_paths = self._current_verify_image_paths()
        if current_paths:
            return current_paths
        cleared = self.runtime.blackboard.read("last_cleared_verify_observation")
        if isinstance(cleared, dict) and isinstance(cleared.get("image_paths"), list):
            return [str(path) for path in cleared["image_paths"] if path]
        return []

    def _verify_observation_fresh(self) -> bool:
        verify_observation = self.runtime.blackboard.read("verify_observation")
        if verify_observation is None:
            return False
        if not metadata_value(verify_observation, "verify_active", False):
            return False
        execution_report = self.runtime.blackboard.read("execution_report")
        if not isinstance(execution_report, dict) or execution_report.get("status") != "action_executed":
            return False
        expected_obs_id = None
        observation = execution_report.get("observation")
        if isinstance(observation, dict) and observation.get("observation_id") is not None:
            expected_obs_id = str(observation["observation_id"])
        captured_for = metadata_value(verify_observation, "source_execution_observation_id")
        if expected_obs_id is not None and captured_for != expected_obs_id:
            return False
        return bool(self._current_verify_image_paths())

    def _clear_verify_observation(self, reason: str) -> None:
        verify_observation = self.runtime.blackboard.read("verify_observation")
        if verify_observation is not None:
            self.runtime.blackboard.write(
                "last_cleared_verify_observation",
                {
                    "reason": reason,
                    "observation_id": getattr(verify_observation, "observation_id", None),
                    "source_execution_observation_id": metadata_value(verify_observation, "source_execution_observation_id"),
                    "image_paths": self._current_verify_image_paths(),
                },
                event_type="verify_observation.cleared",
            )
        self.runtime.blackboard.write("verify_observation", None, event_type="verify_observation.cleared")

    def _grounding_overlay_fresh(self) -> bool:
        overlay = self.runtime.blackboard.read("grounding_overlay")
        if overlay is None or getattr(overlay, "stale", False):
            return False
        return getattr(overlay, "observation_id", None) == current_observation_id(self.runtime.blackboard)

    def _motion_goal_fresh(self) -> bool:
        goal = self.runtime.blackboard.read("motion_goal")
        if goal is None or metadata_value(goal, "stale", False):
            return False
        subgoal = self.runtime.blackboard.read("current_subgoal")
        expected = getattr(subgoal, "subgoal_id", None)
        if expected and metadata_value(goal, "subgoal_id") != expected:
            return False
        obs_id = current_observation_id(self.runtime.blackboard)
        return not obs_id or metadata_value(goal, "observation_id") == obs_id

    def _motion_plan_fresh(self) -> bool:
        plan = self.runtime.blackboard.read("motion_plan")
        if not isinstance(plan, dict) or plan.get("status") == "motion_plan_unavailable":
            return False
        if metadata_value(plan, "stale", False):
            return False
        subgoal = self.runtime.blackboard.read("current_subgoal")
        expected = getattr(subgoal, "subgoal_id", None)
        if expected and metadata_value(plan, "subgoal_id") != expected:
            return False
        obs_id = current_observation_id(self.runtime.blackboard)
        return not obs_id or metadata_value(plan, "observation_id") == obs_id

    def _action_chunk_fresh(self) -> bool:
        chunk = self.runtime.blackboard.read("action_chunk")
        if chunk is None or getattr(chunk, "action_type", None) == "unavailable":
            return False
        if metadata_value(chunk, "stale", False) or metadata_value(chunk, "consumed", False):
            return False
        if not getattr(chunk, "commands", []):
            return False
        subgoal = self.runtime.blackboard.read("current_subgoal")
        expected = getattr(subgoal, "subgoal_id", None)
        if expected and metadata_value(chunk, "subgoal_id") != expected:
            return False
        obs_id = current_observation_id(self.runtime.blackboard)
        return not obs_id or metadata_value(chunk, "observation_id") == obs_id

    def _check_repeat(
        self,
        decision: LoopDecision,
        repeat_state: dict[str, Any],
    ) -> str | None:
        key = (decision.next_component, decision.next_skill, decision.control)
        repeat_state["count"] = int(repeat_state.get("count", 0)) + 1 if repeat_state.get("last_key") == key else 1
        repeat_state["last_key"] = key
        if int(repeat_state["count"]) > self.config.allow_same_decision_repeats:
            emit_runtime_event(
                "clawvla_repeated_decision_notice",
                {
                    "component": decision.next_component,
                    "skill": decision.next_skill,
                    "count": repeat_state["count"],
                    "limit": self.config.allow_same_decision_repeats,
                },
            )
            return (
                f"repeated_decision_limit_exceeded:"
                f"{decision.next_component}.{decision.next_skill}:"
                f"{repeat_state['count']}>{self.config.allow_same_decision_repeats}"
            )
        return None


def _perception_has_bbox(perception: object) -> bool:
    for candidate in getattr(perception, "candidates", []) or []:
        if getattr(candidate, "bbox_by_view", None):
            return True
    return False


def _required_decision(
    control: str,
    stage: str | None,
    next_component: str | None = None,
    next_skill: str | None = None,
    *,
    payload: dict[str, Any] | None = None,
    reason: str,
) -> dict[str, Any]:
    return {
        "control": control,
        "stage": stage,
        "next_component": next_component,
        "next_skill": next_skill,
        "payload": dict(payload or {}),
        "reason": reason,
    }


def _trace_detail(decision: LoopDecision) -> str | None:
    parts = []
    if decision.state_summary:
        parts.append(f"state={decision.state_summary}")
    if decision.next_component and decision.next_skill:
        parts.append(f"next={decision.next_component}.{decision.next_skill}")
    if decision.expected_result:
        parts.append(f"expect={decision.expected_result}")
    return " | ".join(parts) if parts else None
