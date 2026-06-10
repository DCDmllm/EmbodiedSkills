from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .blackboard_utils import current_observation_id, metadata_value
from .loop_types import ADVANCE_STAGE, FINISH_RUN, RUN_SKILL, LoopDecision, LoopRunResult, LoopStepRecord
from .notices import emit_human_trace, emit_runtime_event, emit_status_notice
from .phase_policy import PhasePolicy
from .runtime import AgentRuntime
from .schema import SkillResult


@dataclass
class AgentLoopConfig:
    max_steps: int = 12
    initial_stage: str = "observe"
    stop_on_skill_error: bool = False
    allow_same_decision_repeats: int = 3
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
        repeat_state: dict[str, Any] = {"last_key": None, "count": 0}

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
                if self.config.stop_on_skill_error:
                    return LoopRunResult("scheduler_failed", stage, records, reason=decision_result.status)
                continue
            decision = self._decision_from_result(decision_result)
            decision.stage = self.policy.normalize_stage(decision.stage or stage)
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
            if error is None:
                error = self._check_repeat(decision, repeat_state)
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

    def _choose_decision(self, stage: str) -> SkillResult:
        allowed_skills = self._state_gated_allowed_skills(
            stage,
            self._enabled_allowed_skills(self.policy.allowed_for_stage(stage)),
        )
        payload = {
            "use_model": True,
            "loop_mode": True,
            "current_stage": stage,
            "phase_policy": {
                name: self._enabled_allowed_skills(skills)
                for name, skills in self.policy.full_allowed_skills().items()
            },
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
        next_stage = self.policy.normalize_stage(decision.stage or self.policy.next_stage(current_stage))
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
                "ground_task_objects",
                "lift_depth_cluster",
                "lift_geometry",
                "bind_arm",
                "estimate_uncertainty",
            }
        )
        if should_update_world_state:
            self.runtime.run_skill("state", "update_world_state", {"stage": decision.stage}, stage=decision.stage)
        if is_execute_action and result.status == "action_executed":
            self.runtime.blackboard.write("stage", "verify", event_type="loop.stage_forced_after_execute_action")

    def _prepare_payload(self, decision: LoopDecision) -> dict[str, Any]:
        payload = dict(decision.payload)
        if decision.next_component == "vision" and decision.next_skill == "capture_views":
            payload.setdefault("artifact_prefix", self.runtime.blackboard.read("artifact_prefix") or "agent_loop")
            if self.runtime.blackboard.read("run_robotwin") and self._robotwin_needs_setup():
                payload.setdefault("setup", True)
                payload.setdefault("instruction", self.runtime.blackboard.task_instruction)
        if decision.next_component == "vision" and decision.next_skill in {
            "perceive_scene",
            "localize_task_objects",
            "ground_task_objects",
            "render_grounding_overlay",
            "estimate_uncertainty",
        }:
            payload.setdefault("use_model", True)
            payload.setdefault("image_paths", self._current_vlm_image_paths())
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

    def _robotwin_needs_setup(self) -> bool:
        env = self.runtime.blackboard.read("env_adapter")
        session = getattr(env, "session", None)
        if session is None:
            return True
        return getattr(session, "task_env", None) is None

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
        stage = self.policy.normalize_stage(decision.stage or self.runtime.blackboard.read("stage"))
        if not self.policy.is_allowed(stage, decision.next_component, decision.next_skill, self.runtime.components):
            return f"skill_not_allowed:{stage}.{decision.next_component}.{decision.next_skill}"
        prerequisite_error = self._skill_prerequisite_error(str(decision.next_component), str(decision.next_skill))
        if prerequisite_error is not None:
            return prerequisite_error
        return None

    def _skill_prerequisite_error(self, component: str, skill: str) -> str | None:
        blackboard = self.runtime.blackboard
        if component == "vision":
            if skill in {"perceive_scene", "localize_task_objects", "estimate_uncertainty"} and blackboard.read("observation") is None:
                return f"missing_observation_before_{skill}"
            if skill == "ground_task_objects" and blackboard.read("perception") is None:
                return "missing_perception_before_ground_task_objects"
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
        if component == "motion":
            if skill == "build_motion_goal":
                if blackboard.read("current_subgoal") is None:
                    return "missing_current_subgoal_before_build_motion_goal"
                if blackboard.read("world_state") is None:
                    return "missing_world_state_before_build_motion_goal"
            if skill == "plan_motion" and not self._motion_goal_fresh():
                return "missing_fresh_motion_goal_before_plan_motion"
            if skill == "emit_action_chunk" and not self._motion_plan_fresh():
                return "missing_fresh_motion_plan_before_emit_action_chunk"
            if skill == "execute_action" and not self._action_chunk_fresh():
                return "missing_fresh_action_chunk_before_execute_action"
        if component == "verifier" and skill == "verify_progress":
            if blackboard.read("current_subgoal") is None:
                return "missing_current_subgoal_before_verify_progress"
            if blackboard.read("execution_report") is None:
                return "missing_execution_report_before_verify_progress"
        if component == "recovery":
            if skill == "decide_recovery" and blackboard.read("last_verification_report") is None and blackboard.read("execution_report") is None:
                return "missing_failure_report_before_decide_recovery"
            if skill == "build_retry_request" and blackboard.read("last_recovery_directive") is None:
                return "missing_recovery_directive_before_build_retry_request"
        return None

    def _validate_advance_stage(self, current_stage: str, decision: LoopDecision) -> str | None:
        next_stage = self.policy.normalize_stage(decision.stage or self.policy.next_stage(current_stage))
        if current_stage == "observe" and next_stage != "observe":
            return self._world_state_ready_error()
        if current_stage == "plan" and next_stage in {"preflight", "execute"}:
            if self.runtime.blackboard.read("task_plan") is None:
                return "missing_task_plan_before_leaving_plan"
            if self.runtime.blackboard.read("current_subgoal") is None:
                return "missing_current_subgoal_before_leaving_plan"
        if current_stage == "preflight" and next_stage == "execute":
            if self.runtime.blackboard.read("current_subgoal") is None:
                return "missing_current_subgoal_before_execute"
        if current_stage == "execute" and next_stage == "verify":
            report = self.runtime.blackboard.read("execution_report")
            if not isinstance(report, dict) or report.get("status") != "action_executed":
                return "missing_successful_execution_report_before_verify"
        if current_stage == "verify" and next_stage in {"execute", "plan", "observe", "recover"}:
            verification = self.runtime.blackboard.read("last_verification_report")
            if verification is None:
                return "missing_verification_report_before_leaving_verify"
            next_action = self._verification_next_action(verification)
            if next_stage == "observe" and next_action != "reobserve":
                return f"verify_next_action_mismatch:{next_action}->observe"
            if next_stage == "plan" and next_action != "replan":
                return f"verify_next_action_mismatch:{next_action}->plan"
            if next_stage == "recover" and next_action != "recover":
                return f"verify_next_action_mismatch:{next_action}->recover"
            if next_stage == "execute" and not self._verification_allows_execute(verification):
                return f"verify_next_action_mismatch:{next_action}->execute"
        if current_stage == "recover" and next_stage in {"observe", "plan", "execute"}:
            if self.runtime.blackboard.read("last_retry_request") is None:
                return "missing_retry_request_before_leaving_recover"
        return None

    def _world_state_ready_error(self) -> str | None:
        world_state = self.runtime.blackboard.read("world_state")
        if world_state is None:
            return "missing_world_state"
        if getattr(world_state, "needs_reobserve", False):
            return "world_state_requires_reobserve"
        if not getattr(world_state, "source_candidate_id", None):
            return "missing_source_candidate"
        if not getattr(world_state, "target_candidate_id", None):
            return "missing_target_candidate"
        return None

    def _runtime_state_summary(self) -> dict[str, Any]:
        motion_plan = self.runtime.blackboard.read("motion_plan")
        action_chunk = self.runtime.blackboard.read("action_chunk")
        task_plan = self.runtime.blackboard.read("task_plan")
        current_subgoal = self.runtime.blackboard.read("current_subgoal")
        overlay = self.runtime.blackboard.read("grounding_overlay")
        return {
            "observation_id": current_observation_id(self.runtime.blackboard),
            "grounding_overlay_fresh": self._grounding_overlay_fresh(),
            "task_plan_present": task_plan is not None,
            "task_plan_status": getattr(task_plan, "status", None),
            "current_subgoal": current_subgoal.to_dict() if hasattr(current_subgoal, "to_dict") else current_subgoal,
            "motion_goal_present": self.runtime.blackboard.read("motion_goal") is not None,
            "motion_goal_fresh": self._motion_goal_fresh(),
            "motion_plan_present": motion_plan is not None,
            "motion_plan_fresh": self._motion_plan_fresh(),
            "motion_plan_status": getattr(motion_plan, "status", None) if not isinstance(motion_plan, dict) else motion_plan.get("status"),
            "action_chunk_present": action_chunk is not None,
            "action_chunk_fresh": self._action_chunk_fresh(),
            "action_chunk_type": getattr(action_chunk, "action_type", None),
            "action_chunk_command_count": len(getattr(action_chunk, "commands", []) or []),
            "grounding_overlay_stale": getattr(overlay, "stale", None),
            "verification_next_action": self._verification_next_action(self.runtime.blackboard.read("last_verification_report")),
        }

    def _verification_next_action(self, verification: object | None) -> str | None:
        metadata = getattr(verification, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("next_action"):
            return str(metadata["next_action"])
        if getattr(verification, "success", False):
            return "advance_subgoal"
        if getattr(verification, "should_reobserve", False):
            return "reobserve"
        return None

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
        return None


def _perception_has_bbox(perception: object) -> bool:
    for candidate in getattr(perception, "candidates", []) or []:
        if getattr(candidate, "bbox_by_view", None):
            return True
    return False


def _trace_detail(decision: LoopDecision) -> str | None:
    parts = []
    if decision.state_summary:
        parts.append(f"state={decision.state_summary}")
    if decision.next_component and decision.next_skill:
        parts.append(f"next={decision.next_component}.{decision.next_skill}")
    if decision.expected_result:
        parts.append(f"expect={decision.expected_result}")
    return " | ".join(parts) if parts else None
