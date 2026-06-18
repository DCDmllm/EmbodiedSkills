from __future__ import annotations

"""Compatibility wrapper for the archived VERL runner.

New training should use :mod:`clawvla.rl.openrlhf_runner`. The old VERL
implementation lives under :mod:`clawvla.rl.legacy_verl.runner` so existing
imports and old commands do not break abruptly.
"""

from .legacy_verl import runner as _legacy_runner


globals().update(
    {
        name: getattr(_legacy_runner, name)
        for name in dir(_legacy_runner)
        if not name.startswith("__")
    }
)


if __name__ == "__main__":
    _legacy_runner.main()
