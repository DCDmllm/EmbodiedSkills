from __future__ import annotations

from dataclasses import dataclass, field
import traceback
from typing import Any

from .action_backends import build_action_backend
from .blackboard import Blackboard
from .components import ComponentRegistry, build_component_registry
from .config import AgentConfig
from .notices import emit_human_trace, emit_runtime_event, emit_status_notice
from .schema import SkillRequest, SkillResult


@dataclass
class RuntimeRecord:
    request: SkillRequest
    result: SkillResult


class AgentRuntime:
    def __init__(self, config: AgentConfig, components: ComponentRegistry | None = None):
        self.config = config
        self.components = components or build_component_registry(config)
        self.blackboard = Blackboard(task_instruction=config.task.get("instruction"))
        self.blackboard.write("action_backend", build_action_backend(config))
        self.history: list[RuntimeRecord] = []

    def reset(self, task_instruction: str | None = None) -> None:
        self.blackboard = Blackboard(task_instruction=task_instruction or self.config.task.get("instruction"))
        self.blackboard.write("action_backend", build_action_backend(self.config))
        self.history = []

    def run_skill(
        self,
        component: str,
        skill: str,
        payload: dict[str, Any] | None = None,
        stage: str | None = None,
        budget_steps: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SkillResult:
        request = SkillRequest(
            component=component,
            skill=skill,
            payload=dict(payload or {}),
            stage=stage,
            budget_steps=budget_steps,
            metadata=dict(metadata or {}),
        )
        emit_runtime_event(
            "clawvla_skill_start",
            {
                "component": component,
                "skill": skill,
                "stage": stage,
                "budget_steps": budget_steps,
                "loop_step": request.metadata.get("loop_step"),
                "loop_stage": request.metadata.get("loop_stage"),
                "payload_keys": sorted(str(key) for key in request.payload.keys()),
            },
        )
        emit_human_trace(
            "skill",
            f"run {component}.{skill}",
            detail=f"stage={stage} payload={sorted(str(key) for key in request.payload.keys())}",
        )
        try:
            result = self.components.run(request, self.blackboard)
        except Exception as exc:
            reason = _short_exception(exc)
            report = {
                "component": component,
                "skill": skill,
                "exception_type": type(exc).__name__,
                "message": reason,
                "traceback_tail": traceback.format_exc().splitlines()[-8:],
            }
            self.blackboard.write("last_skill_exception", report, event_type="runtime.skill_exception")
            emit_status_notice(
                "skill_exception",
                success=False,
                source=f"{component}.{skill}",
                reason=reason,
                payload=report,
                always=True,
            )
            result = SkillResult(
                success=False,
                status="skill_exception",
                output={"exception": report, "retryable": True},
                errors=[reason],
            )
        result.request_id = request.request_id
        result.component = component
        result.skill = skill
        self.history.append(RuntimeRecord(request=request, result=result))
        self.blackboard.append_event(
            "runtime.skill_finished",
            {
                "component": component,
                "skill": skill,
                "success": result.success,
                "status": result.status,
            },
        )
        emit_runtime_event(
            "clawvla_skill_finish",
            {
                "component": component,
                "skill": skill,
                "stage": stage,
                "success": result.success,
                "status": result.status,
                "errors": result.errors[:3],
                "loop_step": request.metadata.get("loop_step"),
                "loop_stage": request.metadata.get("loop_stage"),
                "output_keys": sorted(str(key) for key in result.output.keys()),
            },
        )
        emit_human_trace(
            "success" if result.success else "failure",
            f"{component}.{skill} -> {result.status}",
            detail=f"errors={result.errors[:2]}" if result.errors else None,
        )
        return result

    def inspect(self) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "components": self.components.summaries(),
            "stages": [stage.to_dict() for stage in self.config.stages],
            "history_length": len(self.history),
            "blackboard": self.blackboard.compact_context(),
        }

    def run_loop(self, max_steps: int = 12, initial_stage: str = "observe"):
        from .agent_loop import AgentLoop, AgentLoopConfig

        return AgentLoop(self, config=AgentLoopConfig(max_steps=max_steps, initial_stage=initial_stage)).run()


def _short_exception(exc: Exception, limit: int = 1200) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else text[:limit] + "...<truncated>"
