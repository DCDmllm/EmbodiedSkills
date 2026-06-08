from __future__ import annotations

import argparse
import json
import sys

from clawvla.config import load_config
from clawvla.action_backends.factory import build_action_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose the configured pi05/LeRobot action backend.")
    parser.add_argument("--config", default="configs/robotwin_default.json")
    parser.add_argument("--load-policy", action="store_true", help="Actually instantiate the LeRobot policy.")
    parser.add_argument(
        "--pythonpath",
        action="append",
        default=[],
        help="Prepend an extra import path before probing, e.g. RoboTwin/policy/pi05/src.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in reversed(args.pythonpath):
        if path and path not in sys.path:
            sys.path.insert(0, path)
    backend = build_action_backend(load_config(args.config))
    if hasattr(backend, "diagnose"):
        report = backend.diagnose(load_policy=args.load_policy)
    else:
        report = {
            "status": "action_backend_diagnosis_unavailable",
            "backend": getattr(backend, "name", type(backend).__name__),
        }
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
