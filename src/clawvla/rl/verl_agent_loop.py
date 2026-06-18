from __future__ import annotations

"""Compatibility wrapper for the archived VERL AgentLoop."""

from .legacy_verl import verl_agent_loop as _legacy_verl_agent_loop


globals().update(
    {
        name: getattr(_legacy_verl_agent_loop, name)
        for name in dir(_legacy_verl_agent_loop)
        if not name.startswith("__")
    }
)
