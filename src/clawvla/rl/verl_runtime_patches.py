from __future__ import annotations

"""Compatibility wrapper for archived VERL runtime patches."""

from .legacy_verl import verl_runtime_patches as _legacy_verl_runtime_patches


globals().update(
    {
        name: getattr(_legacy_verl_runtime_patches, name)
        for name in dir(_legacy_verl_runtime_patches)
        if not name.startswith("__")
    }
)
