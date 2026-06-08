from __future__ import annotations

import json
import os
import re
from typing import Any

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class TerminalRenderer:
    def __init__(self) -> None:
        self.console = Console(stderr=True, highlight=False)

    def render_agent_line(self, line: str) -> None:
        text = line.rstrip("\n")
        if not text:
            return
        plain = _strip_ansi(text).strip()
        if _looks_like_legacy_trace(plain):
            return
        if not plain.startswith("{"):
            self.console.print(Text.from_ansi(text))
            return
        try:
            event = json.loads(plain)
        except json.JSONDecodeError:
            self.console.print(Text.from_ansi(text))
            return
        self.render_event(event)

    def render_event(self, event: dict[str, Any]) -> None:
        name = str(event.get("event") or event.get("status") or "")
        if name == "clawvla_loop_decision":
            self._render_decision(event)
        elif name == "clawvla_skill_start":
            self._render_skill_start(event)
        elif name == "clawvla_skill_finish":
            self._render_skill_finish(event)
        elif name == "clawvla_decision_blocked":
            self._render_blocked(event)
        elif name == "clawvla_invalid_model_output":
            self._render_invalid_model_output(event)
        elif name == "clawvla_repeated_decision_notice":
            self._render_repeated_decision(event)
        elif name == "clawvla_status_notice":
            self._render_status_notice(event)
        elif name == "clawvla_result_written":
            self._render_result(event)
        elif event.get("status") in {"vllm_ready", "vllm_stopped", "pi05_worker_started", "pi05_worker_stopped"}:
            self._render_service_status(event)

    def _render_decision(self, event: dict[str, Any]) -> None:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()
        next_skill = _join_skill(event.get("next_component"), event.get("next_skill"))
        table.add_row("step", f"{event.get('step_index')}  stage={event.get('stage_before')}  control={event.get('control')}")
        if next_skill:
            table.add_row("next", next_skill)
        if event.get("narration"):
            table.add_row("say", str(event["narration"]))
        if event.get("state_summary"):
            table.add_row("state", str(event["state_summary"]))
        if event.get("expected_result"):
            table.add_row("expect", str(event["expected_result"]))
        if event.get("reason"):
            table.add_row("reason", str(event["reason"]))
        self.console.print(Panel(table, title="scheduler", border_style="cyan", box=box.ROUNDED))

    def _render_skill_start(self, event: dict[str, Any]) -> None:
        skill = _join_skill(event.get("component"), event.get("skill")) or "unknown"
        payload = ", ".join(str(item) for item in event.get("payload_keys") or [])
        detail = f"stage={event.get('stage')}"
        if event.get("loop_step") is not None:
            detail += f" step={event.get('loop_step')}"
        if payload:
            detail += f" payload=[{payload}]"
        self.console.print(f"[magenta]run[/]  [bold]{escape(skill)}[/] [dim]{escape(detail)}[/]")

    def _render_skill_finish(self, event: dict[str, Any]) -> None:
        skill = _join_skill(event.get("component"), event.get("skill")) or "unknown"
        success = bool(event.get("success"))
        label = "ok" if success else "fail"
        style = "green" if success else "red"
        status = event.get("status")
        errors = event.get("errors") or []
        suffix = f" -> {status}"
        if errors:
            suffix += f" errors={errors[:2]}"
        self.console.print(f"[{style}]{label:<4}[/] [bold]{escape(skill)}[/]{escape(suffix)}")

    def _render_blocked(self, event: dict[str, Any]) -> None:
        target = event.get("next_skill") or event.get("requested_stage") or event.get("control")
        self.console.print(
            Panel(
                f"target={target}\nreason={event.get('reason')}",
                title="decision blocked",
                border_style="red",
                box=box.ROUNDED,
            )
        )

    def _render_invalid_model_output(self, event: dict[str, Any]) -> None:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold red", no_wrap=True)
        table.add_column()
        table.add_row("source", str(event.get("source")))
        table.add_row("reason", str(event.get("reason")))
        table.add_row("keys", str(event.get("raw_keys")))
        table.add_row("raw", str(event.get("raw_preview")))
        self.console.print(Panel(table, title="invalid model output", border_style="red", box=box.ROUNDED))

    def _render_repeated_decision(self, event: dict[str, Any]) -> None:
        self.console.print(
            f"[yellow]repeat[/] {escape(str(_join_skill(event.get('component'), event.get('skill'))))} "
            f"count={escape(str(event.get('count')))} limit={escape(str(event.get('limit')))}"
        )

    def _render_status_notice(self, event: dict[str, Any]) -> None:
        markers = event.get("markers") or []
        if event.get("success") and not markers:
            return
        style = "yellow" if event.get("success") else "red"
        self.console.print(
            f"[{style}]notice[/] {escape(str(event.get('source')))} status={escape(str(event.get('status')))} "
            f"reason={escape(str(event.get('reason')))} markers={escape(str(markers))}"
        )

    def _render_result(self, event: dict[str, Any]) -> None:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold green", no_wrap=True)
        table.add_column()
        table.add_row("status", str(event.get("loop_status")))
        table.add_row("stage", str(event.get("final_stage")))
        table.add_row("reason", str(event.get("reason")))
        table.add_row("result", str(event.get("path")))
        self.console.print(Panel(table, title="run result", border_style="green", box=box.ROUNDED))

    def _render_service_status(self, event: dict[str, Any]) -> None:
        status = str(event.get("status"))
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold yellow", no_wrap=True)
        table.add_column()
        table.add_row("status", status)
        for key in ["log", "agent_log", "result", "config"]:
            if event.get(key):
                table.add_row(key, str(event[key]))
        self.console.print(Panel(table, title="service", border_style="yellow", box=box.ROUNDED))


def _join_skill(component: object, skill: object) -> str | None:
    if component and skill:
        return f"{component}.{skill}"
    return None


def _looks_like_legacy_trace(text: str) -> bool:
    stripped = text.lstrip()
    if stripped.startswith(("==>", "$", "OK", "!!", "pi", ">>")):
        return True
    if stripped.startswith(("[skill]", "[success]", "[failure]", "[scheduler]", "[openpi]", "[execute]")):
        return True
    return text.startswith(("    state=", "    next=", "    expect=", "    reason="))


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)
