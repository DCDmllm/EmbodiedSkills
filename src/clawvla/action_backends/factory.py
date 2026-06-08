from __future__ import annotations

from typing import Any

from ..config import AgentConfig
from .base import ActionBackend
from .pi05 import Pi05ActionBackend


def build_action_backend(config: AgentConfig) -> ActionBackend:
    payload = config.metadata.get("action_backend")
    if not isinstance(payload, dict):
        payload = {"type": "pi05", "enabled": False}
    backend_type = str(payload.get("type", "pi05")).lower()
    if backend_type in {"pi05", "pi0.5", "pi_05"}:
        return Pi05ActionBackend(payload)
    return Pi05ActionBackend({"type": backend_type, "enabled": False, "reason": "unsupported_action_backend"})
