from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class RLTerminal:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.console = Console(stderr=True, highlight=False)

    def event(self, name: str, **payload: Any) -> None:
        if not self.enabled:
            return
        failure_status = payload.get("status") in {"infra_failure", "adapter_error"}
        if name.endswith("failed") or payload.get("success") is False or failure_status:
            self._panel("rl failure", payload, "red")
        elif name in {"episode_start", "episode_finish"}:
            self._panel(name.replace("_", " "), payload, "cyan")
        elif name.startswith("reward"):
            self._panel("reward", payload, "green")
        elif name.startswith("service") or name.endswith("ready") or name.endswith("stopped"):
            self._panel("service", payload, "yellow")
        elif name.startswith("checkpoint") or name.startswith("archive"):
            self._panel("archive", payload, "magenta")
        else:
            self.console.print(Text(f"[rl] {name} {payload}", style="dim"))

    def _panel(self, title: str, payload: dict[str, Any], style: str) -> None:
        table = Table.grid(padding=(0, 1))
        table.add_column(style=f"bold {style}", no_wrap=True)
        table.add_column()
        for key, value in payload.items():
            if value is None:
                continue
            table.add_row(str(key), str(value))
        self.console.print(Panel(table, title=title, border_style=style, box=box.ROUNDED))
