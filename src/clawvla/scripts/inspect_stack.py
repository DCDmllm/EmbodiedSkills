from __future__ import annotations

import argparse
import json

from clawvla.components.factory import build_component_registry, build_model_runtimes, build_skill_registry
from clawvla.config import load_config
from clawvla.envs import RoboTwinAdapter, robotwin_runtime_environment
from clawvla.runtime import AgentRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ClawVLA config, components, and skills.")
    parser.add_argument("--config", default="configs/robotwin_default.json")
    parser.add_argument("--lightweight-models", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    skill_registry = build_skill_registry()
    model_runtimes = build_model_runtimes(config)
    components = build_component_registry(
        config,
        skill_registry=skill_registry,
        model_runtimes=model_runtimes,
    )
    runtime = AgentRuntime(config, components=components)
    runtime.blackboard.write("env_adapter", RoboTwinAdapter(config.robotwin))
    model_reports = {
        name: runtime_obj.validate_config(lightweight=True)
        for name, runtime_obj in model_runtimes.items()
    }
    payload = {
        "config_name": config.name,
        "components": components.summaries(),
        "skills": [spec.to_dict() for spec in skill_registry.all_specs()],
        "stages": [stage.to_dict() for stage in config.stages],
        "models": model_reports,
        "robotwin": config.robotwin.__dict__,
        "runtime_environment": robotwin_runtime_environment(config.runtime_environment),
        "runtime": runtime.inspect(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
