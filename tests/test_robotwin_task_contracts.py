from __future__ import annotations

import ast
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
ROBOTWIN_ENVS = WORKSPACE_ROOT / "RoboTwin" / "envs"
TASK_CONFIG = PROJECT_ROOT / "configs" / "rl" / "tasks" / "robotwin_all.yaml"


def _self_attributes(node: ast.AST, context: type[ast.expr_context]) -> set[str]:
    return {
        item.attr
        for item in ast.walk(node)
        if isinstance(item, ast.Attribute)
        and isinstance(item.value, ast.Name)
        and item.value.id == "self"
        and isinstance(item.ctx, context)
    }


def _self_method_calls(node: ast.AST) -> set[str]:
    return {
        item.func.attr
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and isinstance(item.func.value, ast.Name)
        and item.func.value.id == "self"
    }


def test_robotwin_success_state_does_not_depend_on_expert_play_once() -> None:
    payload = yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))
    task_names = [row["task_name"] for row in payload["rollout"]["tasks"]]
    assert len(task_names) == 50

    violations: dict[str, list[str]] = {}
    for task_name in task_names:
        source = (ROBOTWIN_ENVS / f"{task_name}.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        task_class = next(
            item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == task_name
        )
        methods = {
            item.name: item
            for item in task_class.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert "play_once" in methods, task_name
        assert "check_success" in methods, task_name

        reachable = {"check_success"}
        pending = ["check_success"]
        while pending:
            method_name = pending.pop()
            for called_name in _self_method_calls(methods[method_name]):
                if called_name in methods and called_name not in reachable:
                    reachable.add(called_name)
                    pending.append(called_name)

        success_reads = set().union(
            *(_self_attributes(methods[name], ast.Load) for name in reachable)
        )
        expert_writes = _self_attributes(methods["play_once"], ast.Store)
        initialization_writes = set().union(
            *(
                _self_attributes(methods[name], ast.Store)
                for name in ("__init__", "setup_demo", "load_actors")
                if name in methods
            )
        )
        expert_only_success_state = sorted((expert_writes & success_reads) - initialization_writes)
        if expert_only_success_state:
            violations[task_name] = expert_only_success_state

    assert violations == {}
