from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from clawvla.action_backends.calvin import CalvinHttpActionBackend, _commands_from_response
from clawvla.scripts.calvin_xvla_baseline import _fixed_seeds
from clawvla.scripts.calvin_sequence_eval import (
    build_run_plan,
    config_for_sequence,
    load_completed_results,
    load_sequence_manifest,
    override_calvin_dataset_paths,
    result_key,
    sequence_metrics,
    validate_resolved_sequence,
    write_sequence_summary,
)
from clawvla.artifacts import ArtifactStore
from clawvla.config import EnvironmentConfig, PROJECT_ROOT, WORKSPACE_ROOT, load_config
from clawvla.envs.calvin import CalvinAdapter, _calvin_env_action, normalize_calvin_observation
from clawvla.rl.config import load_rl_config
from clawvla.rl.reward_registry import build_reward_registry
from clawvla.rl.rollout_worker import _populate_episode_from_result, _should_run_environment
from clawvla.rl.trajectory import EpisodeRecord
from clawvla.rewards.robotwin_reward import RewardSnapshot
from clawvla.schema import ActionChunk


def _raw_observation(step: int = 0) -> dict[str, object]:
    return {
        "rgb_obs": {
            "rgb_static": np.full((4, 4, 3), step, dtype=np.uint8),
            "rgb_gripper": np.full((4, 4, 3), step + 1, dtype=np.uint8),
        },
        "depth_obs": {
            "depth_static": np.zeros((4, 4), dtype=np.float32),
            "depth_gripper": np.ones((4, 4), dtype=np.float32),
        },
        "robot_obs": [0.1 + step, 0.2, 0.3, 0.0, 0.0, 0.0, 0.08, *([0.0] * 7), 1.0],
        "scene_obs": [0.0] * 24,
    }


def _observation(tmp_path: Path):
    return normalize_calvin_observation(
        _raw_observation(),
        task_instruction="move the slider left",
        artifacts=ArtifactStore(tmp_path),
        artifact_prefix="contract",
    )


def _backend(**overrides) -> CalvinHttpActionBackend:
    config = {
        "enabled": True,
        "url": "http://127.0.0.1:8000/act",
        "serialization": "list",
        "horizon": 2,
    }
    config.update(overrides)
    return CalvinHttpActionBackend(config)


def test_calvin_http_declares_direct_vla_atomic_planning_contract() -> None:
    backend = _backend()

    contract = backend.task_plan_contract("push the sliding door to the left side")

    assert backend.requires_candidate_bindings is False
    assert contract == {
        "mode": "atomic_instruction_passthrough",
        "instruction": "push the sliding door to the left side",
        "max_subgoals": 1,
        "candidate_bindings_required": False,
        "completion_authority": "environment_oracle",
    }


def test_calvin_task_status_treats_oracle_success_as_done() -> None:
    from clawvla.config import EnvironmentConfig
    from clawvla.envs.calvin import CalvinAdapter

    adapter = CalvinAdapter(EnvironmentConfig(type="calvin"))
    adapter.env = object()
    adapter.current_subtask = "move_slider_left"
    adapter.start_info = {"start": True}
    adapter.last_info = {"current": True}
    adapter.last_done = False
    adapter._success_from_info = lambda info: True

    status = adapter.task_status()

    assert status["success"] is True
    assert status["done"] is True


def test_calvin_baseline_parses_fixed_seed_matrix() -> None:
    assert _fixed_seeds("0,1,2,3,4,5,6,7,8,9", 1, 99) == list(range(10))
    assert _fixed_seeds(None, 3, 7) == [7, 7, 7]
    with pytest.raises(ValueError, match="seeds_must_be_unique"):
        _fixed_seeds("1,1", 2, 0)


def test_calvin_sequence_manifest_freezes_ten_official_five_step_sequences() -> None:
    path = Path(PROJECT_ROOT) / "configs" / "calvin" / "stage4_official_pilot.json"

    specs, metadata = load_sequence_manifest(path)
    jobs = build_run_plan(specs, ("baseline", "agent"))

    assert metadata["source"] == "calvin_agent.evaluation.multistep_sequences.get_sequences"
    assert metadata["official_sequence_pool_size"] == 1000
    assert len(specs) == 10
    assert [spec.official_sequence_index for spec in specs] == list(range(10))
    assert all(spec.official_sequence_pool_size == 1000 for spec in specs)
    assert all(len(spec.expected_subtasks) == 5 for spec in specs)
    assert len(jobs) == 20
    assert jobs[0] == {
        "runner": "baseline",
        "sequence_id": "official_000",
        "official_sequence_index": 0,
        "official_sequence_pool_size": 1000,
        "seed": 0,
    }


def test_calvin_sequence_config_removes_single_task_override(tmp_path) -> None:
    base = load_config("configs/calvin_xvla_enabled_probe.json")
    spec = load_sequence_manifest(
        Path(PROJECT_ROOT) / "configs" / "calvin" / "stage4_official_pilot.json"
    )[0][2]

    configured = config_for_sequence(base, spec, 7, tmp_path / "artifacts")

    assert configured.environment.seed == 7
    assert configured.environment.artifact_dir == str(tmp_path / "artifacts")
    assert configured.environment.params["sequence_index"] == 2
    assert configured.environment.params["sequence_pool_size"] == 1000
    assert configured.environment.params["subtask_index"] == 0
    assert "initial_state" not in configured.environment.params
    assert "eval_sequence" not in configured.environment.params
    assert "subtask" not in configured.environment.params
    assert "initial_state" in base.environment.params


def test_calvin_sequence_dataset_override_preserves_source_config(tmp_path) -> None:
    base = load_config("configs/calvin_xvla_enabled_probe.json")
    dataset_path = tmp_path / "task_ABC_D"

    configured = override_calvin_dataset_paths(
        base,
        dataset_path=str(dataset_path),
        validation_dir=None,
    )

    assert configured.environment.params["dataset_path"] == str(dataset_path.resolve())
    assert configured.environment.params["validation_dir"] == str((dataset_path / "validation").resolve())
    assert base.environment.params["dataset_path"] != configured.environment.params["dataset_path"]


def test_calvin_sequence_advance_preserves_scene_and_rebases_oracle(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    adapter.env = object()
    adapter.eval_sequence = ("move_slider_left", "open_drawer")
    adapter.current_subtask = "move_slider_left"
    adapter.subtask_index = 0
    adapter.start_info = {"phase": "reset"}
    adapter.last_info = {"phase": "slider_complete"}
    adapter.last_done = False
    adapter.last_reward = 1.0
    adapter.val_annotations = {
        "move_slider_left": ["push the sliding door to the left side"],
        "open_drawer": ["pull the handle to open the drawer"],
    }
    adapter._success_from_info = lambda info: True

    advanced = adapter.advance_sequence_subtask()

    assert advanced["sequence_complete"] is False
    assert adapter.env is not None
    assert adapter.subtask_index == 1
    assert adapter.current_subtask == "open_drawer"
    assert adapter.start_info == {"phase": "slider_complete"}
    assert advanced["task_language"] == "pull the handle to open the drawer"
    assert adapter.last_reward == 0.0

    complete = adapter.advance_sequence_subtask()
    assert complete["sequence_complete"] is True
    assert complete["sequence_length"] == 2


def test_calvin_sequence_advance_rejects_failed_subtask(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    adapter.env = object()
    adapter.eval_sequence = ("move_slider_left", "open_drawer")
    adapter.current_subtask = "move_slider_left"
    adapter.start_info = {"phase": "reset"}
    adapter.last_info = {"phase": "unchanged"}
    adapter._success_from_info = lambda info: False

    with pytest.raises(RuntimeError, match="sequence_advance_requires_success"):
        adapter.advance_sequence_subtask()


def test_calvin_sequence_resolution_detects_official_protocol_drift(tmp_path) -> None:
    spec = load_sequence_manifest(
        Path(PROJECT_ROOT) / "configs" / "calvin" / "stage4_official_pilot.json"
    )[0][0]
    adapter = SimpleNamespace(
        sequence_pool_size=spec.official_sequence_pool_size,
        eval_sequence=spec.expected_subtasks,
        initial_state=spec.expected_initial_state,
    )

    resolved = validate_resolved_sequence(adapter, spec)

    assert resolved["subtasks"] == list(spec.expected_subtasks)
    assert resolved["official_sequence_pool_size"] == 1000
    adapter.eval_sequence = tuple(reversed(spec.expected_subtasks))
    with pytest.raises(ValueError, match="official_sequence_drift"):
        validate_resolved_sequence(adapter, spec)


def test_calvin_sequence_metrics_report_length_one_to_five_and_infra_errors() -> None:
    records = [
        {"success": True, "completed_subtasks": 5, "sequence_length": 5, "failure_type": None},
        {
            "success": False,
            "completed_subtasks": 2,
            "sequence_length": 5,
            "failure_type": "http_or_action_backend",
        },
    ]

    metrics = sequence_metrics(records)

    assert metrics["sequence_count"] == 2
    assert metrics["full_sequence_success_rate"] == 0.5
    assert metrics["average_completed_length"] == 3.5
    assert metrics["average_environment_steps"] == 0.0
    assert metrics["average_action_chunks"] == 0.0
    assert metrics["completion_rates"] == {"1": 1.0, "2": 1.0, "3": 0.5, "4": 0.5, "5": 0.5}
    assert metrics["environment_or_http_error_count"] == 1
    assert metrics["failure_type_counts"] == {"http_or_action_backend": 1}


def test_calvin_sequence_metrics_report_stall_and_premature_finish() -> None:
    records = [
        {
            "success": False,
            "completed_subtasks": 0,
            "sequence_length": 5,
            "failure_type": "stalled_loop",
            "subtasks": [
                {
                    "environment_steps": 40,
                    "chunk_count": 2,
                    "stalled_loop": True,
                    "premature_finish": False,
                }
            ],
        },
        {
            "success": False,
            "completed_subtasks": 1,
            "sequence_length": 5,
            "failure_type": "premature_finish",
            "subtasks": [
                {
                    "environment_steps": 20,
                    "chunk_count": 1,
                    "stalled_loop": False,
                    "premature_finish": True,
                }
            ],
        },
    ]

    metrics = sequence_metrics(records)

    assert metrics["average_environment_steps"] == 30.0
    assert metrics["average_action_chunks"] == 1.5
    assert metrics["stalled_loop_count"] == 1
    assert metrics["stalled_loop_rate"] == 0.5
    assert metrics["premature_finish_count"] == 1
    assert metrics["premature_finish_rate"] == 0.5


def test_calvin_sequence_results_are_resumable_and_export_csv(tmp_path) -> None:
    record = {
        "runner": "baseline",
        "sequence_id": "official_000",
        "official_sequence_index": 0,
        "seed": 0,
        "status": "sequence_succeeded",
        "success": True,
        "completed_subtasks": 5,
        "sequence_length": 5,
        "failure_type": None,
        "failure_reason": None,
        "elapsed_seconds": 1.0,
    }
    results_path = tmp_path / "sequence_results.jsonl"
    results_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    completed = load_completed_results(results_path)
    write_sequence_summary(tmp_path, list(completed.values()))

    assert completed[result_key("baseline", "official_000", 0)]["success"] is True
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["overall"]["full_sequence_success_rate"] == 1.0
    csv_text = (tmp_path / "sequence_results.csv").read_text(encoding="utf-8")
    assert "official_000" in csv_text


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({}, "calvin_http_response_missing_action"),
        ({"action": []}, "calvin_http_empty_action_response"),
        ({"action": [[0.0] * 9]}, "calvin_http_action_shape_invalid"),
        ({"action": [[0.0] * 9 + [float("nan")]]}, "calvin_http_action_contains_nonfinite"),
        ({"action": [[0.0] * 9 + [float("inf")]]}, "calvin_http_action_contains_nonfinite"),
    ],
)
def test_xvla_response_contract_rejects_missing_short_empty_and_nonfinite(response, message) -> None:
    with pytest.raises((KeyError, ValueError), match=message):
        _commands_from_response(response, horizon=2)


def test_xvla_response_contract_consumes_bounded_prefix_from_fixed_model_horizon() -> None:
    actions = [[float(index)] * 20 for index in range(30)]

    commands = _commands_from_response({"action": actions}, horizon=2)

    assert commands == [[0.0] * 10, [1.0] * 10]


def test_xvla_sampling_steps_are_distinct_from_execution_horizon(tmp_path) -> None:
    observation = _observation(tmp_path)
    backend = _backend(horizon=4, inference_steps=12)

    payload, metadata = backend._request_payload(observation, {})

    assert payload["steps"] == 12
    assert metadata["inference_steps"] == 12
    assert metadata["requested_horizon"] == 4
    assert metadata["configured_horizon"] == 4
    assert metadata["execution_horizon"] == 4


def test_xvla_execution_horizon_is_capped_by_backend_contract(monkeypatch, tmp_path) -> None:
    actions = [[float(index)] * 20 for index in range(30)]
    backend = _backend(horizon=20, inference_steps=10)
    monkeypatch.setattr(backend, "_post", lambda _url, _payload: {"action": actions})

    result = backend.build_action_chunk(
        None,
        None,
        _observation(tmp_path),
        {"horizon": 30},
    )

    assert result.success is True
    assert result.action_chunk.control_horizon == 20
    assert len(result.action_chunk.commands) == 20
    assert result.metadata["requested_horizon"] == 30
    assert result.metadata["configured_horizon"] == 20
    assert result.metadata["execution_horizon"] == 20


def test_xvla_non_object_response_is_an_explicit_backend_failure(monkeypatch, tmp_path) -> None:
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return []

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=lambda *args, **kwargs: Response()))
    result = _backend().build_action_chunk(None, None, _observation(tmp_path), {})

    assert result.success is False
    assert result.errors == ["calvin_http_inference_failed"]
    assert "calvin_http_response_must_be_object:list" in result.metadata["exception"]


def test_xvla_timeout_is_an_explicit_backend_failure(monkeypatch, tmp_path) -> None:
    def timeout(*args, **kwargs):
        raise TimeoutError("policy server timed out")

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=timeout))
    result = _backend(timeout=0.01).build_action_chunk(None, None, _observation(tmp_path), {})

    assert result.success is False
    assert result.errors == ["calvin_http_inference_failed"]
    assert result.metadata["exception"] == "TimeoutError: policy server timed out"


@pytest.mark.parametrize(("missing", "message"), [("proprio", "calvin_http_calvin_proprio_missing"), ("static", "calvin_http_missing_rgb_artifact:static"), ("gripper", "calvin_http_missing_rgb_artifact:gripper")])
def test_xvla_request_requires_20d_proprio_and_both_rgb_views(tmp_path, missing, message) -> None:
    observation = _observation(tmp_path)
    if missing == "proprio":
        observation.raw["calvin_proprio"] = [0.0] * 19
        observation.raw["summary_ref"] = None
    else:
        observation.camera_views.pop(missing)

    result = _backend().build_action_chunk(None, None, observation, {})

    assert result.success is False
    assert message in result.metadata["exception"]


def test_calvin_10d_action_maps_pose_rotation_and_gripper_threshold() -> None:
    identity_rot6d = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    position, quaternion, gripper = _calvin_env_action(
        [0.1, -0.2, 0.3, *identity_rot6d, 0.799],
        gripper_close_threshold=0.8,
    )
    _, _, threshold_gripper = _calvin_env_action(
        [0.1, -0.2, 0.3, *identity_rot6d, 0.8],
        gripper_close_threshold=0.8,
    )

    assert position.tolist() == pytest.approx([0.1, -0.2, 0.3])
    assert np.abs(quaternion).tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert gripper == 1
    assert threshold_gripper == -1


class _Oracle:
    def __init__(self):
        self.calls = []

    def get_task_info_for_set(self, start, current, tasks):
        self.calls.append((start, current, tasks))
        return {"move_slider_left"} if current.get("oracle_success") else set()


class _FakeEnv:
    def __init__(self, *, done: bool = False, oracle_success: bool = False):
        self.done = done
        self.oracle_success = oracle_success
        self.step_calls = 0
        self.closed = False
        self.ownsPhysicsClient = True

    def step(self, action):
        self.step_calls += 1
        return _raw_observation(self.step_calls), 0.25, self.done, {"oracle_success": self.oracle_success}

    def close(self):
        self.closed = True


def _adapter(tmp_path: Path, *, max_episode_steps: int = 10) -> CalvinAdapter:
    return CalvinAdapter(
        EnvironmentConfig(
            type="calvin",
            artifact_dir=str(tmp_path),
            task_name="calvin_move_slider_left",
            params={
                "subtask": "move_slider_left",
                "max_episode_steps": max_episode_steps,
                "camera_names": ["rgb_static", "rgb_gripper"],
            },
        )
    )


@pytest.mark.parametrize(
    ("mode", "max_steps"),
    [("success", 10), ("done", 10), ("budget", 1)],
)
def test_calvin_execute_stops_on_oracle_success_env_done_or_step_budget(tmp_path, mode, max_steps) -> None:
    adapter = _adapter(tmp_path, max_episode_steps=max_steps)
    adapter.env = _FakeEnv(done=mode == "done", oracle_success=mode == "success")
    adapter.task_oracle = _Oracle()
    adapter.start_info = {"oracle_success": False}
    adapter.last_raw_observation = _raw_observation()
    adapter.last_info = dict(adapter.start_info)
    command = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.7]

    result = adapter.execute_action(ActionChunk("calvin_ee_pose_10d", [command, command], 2))

    assert result["status"] == "action_executed"
    assert result["executed_steps"] == 1
    assert result["done"] is True
    assert adapter.env.step_calls == 1
    assert result["success"] is (mode == "success")


def test_calvin_oracle_compares_reset_info_with_current_info(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    oracle = _Oracle()
    adapter.task_oracle = oracle
    adapter.start_info = {"oracle_success": False, "marker": "reset"}
    current = {"oracle_success": True, "marker": "current"}

    assert adapter._success_from_info(current) is True
    assert oracle.calls == [(adapter.start_info, current, {"move_slider_left"})]


def test_calvin_close_clears_dynamic_episode_state(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    env = _FakeEnv()
    adapter.env = env
    adapter.start_info = {"reset": True}
    adapter.last_raw_observation = _raw_observation()
    adapter.last_observation = _observation(tmp_path / "observation")
    adapter.last_reward = 1.0
    adapter.last_done = True
    adapter.last_info = {"old": True}
    adapter.step_count = 7

    adapter.close()

    assert env.closed is True
    assert env.ownsPhysicsClient is False
    assert adapter.env is None
    assert adapter.start_info is None
    assert adapter.last_raw_observation is None
    assert adapter.last_observation is None
    assert adapter.last_reward is None
    assert adapter.last_done is None
    assert adapter.last_info == {}
    assert adapter.step_count == 0


def test_calvin_sequence_and_subtask_bounds_fail_before_external_imports(tmp_path) -> None:
    negative_sequence = _adapter(tmp_path / "sequence")
    negative_sequence.sequence_index = -1
    negative_sequence.initial_state = None
    negative_sequence.eval_sequence = None
    with pytest.raises(ValueError, match="calvin_sequence_index_out_of_range:-1"):
        negative_sequence._load_task_context()

    invalid_pool = _adapter(tmp_path / "pool")
    invalid_pool.sequence_pool_size = 0
    invalid_pool.initial_state = None
    invalid_pool.eval_sequence = None
    with pytest.raises(ValueError, match="calvin_sequence_pool_size_must_be_positive:0"):
        invalid_pool._load_task_context()

    outside_pool = _adapter(tmp_path / "outside_pool")
    outside_pool.sequence_pool_size = 1000
    outside_pool.sequence_index = 1000
    outside_pool.initial_state = None
    outside_pool.eval_sequence = None
    with pytest.raises(ValueError, match="calvin_sequence_index_out_of_range:1000:available=1000"):
        outside_pool._load_task_context()

    bad_subtask = _adapter(tmp_path / "subtask")
    bad_subtask.initial_state = {"ready": True}
    bad_subtask.eval_sequence = ("move_slider_left",)
    bad_subtask.current_subtask = None
    bad_subtask.subtask_index = 2
    with pytest.raises(ValueError, match="calvin_subtask_index_out_of_range:2:sequence_len=1"):
        bad_subtask._load_task_context()


def test_calvin_missing_annotation_is_explicit(monkeypatch, tmp_path) -> None:
    class FakeOmegaConf:
        @staticmethod
        def load(path):
            return {} if str(path).endswith("new_playtable_validation.yaml") else {"task": "config"}

    monkeypatch.setitem(sys.modules, "omegaconf", SimpleNamespace(OmegaConf=FakeOmegaConf))
    monkeypatch.setitem(sys.modules, "hydra", SimpleNamespace(utils=SimpleNamespace(instantiate=lambda cfg: object())))
    adapter = _adapter(tmp_path)
    adapter.initial_state = {"ready": True}
    adapter.eval_sequence = ("missing_annotation_task",)
    adapter.current_subtask = None

    with pytest.raises(KeyError, match="calvin_subtask_annotation_missing:missing_annotation_task"):
        adapter._load_task_context()


@pytest.mark.parametrize(
    ("before_success", "after_success", "expected"),
    [(False, False, -0.05), (False, True, 9.95), (True, True, 1.95)],
)
def test_calvin_reward_exact_numeric_contract(before_success, after_success, expected) -> None:
    registry = build_reward_registry(
        ["clawvla.rl.reward_registry:register_builtin_calvin"],
        {"calvin_move_slider_left": "calvin"},
    )
    handler = registry.handler_for_task("calvin_move_slider_left")
    before = RewardSnapshot("calvin_move_slider_left", before_success, metadata={"step_count": 0})
    after = RewardSnapshot(
        "calvin_move_slider_left",
        after_success,
        metadata={"step_count": 1, "done": after_success},
    )

    result = handler.compute(before, after, {"task_name": "calvin_move_slider_left", "step_cost": 0.05})

    assert result.reward == pytest.approx(expected)
    assert result.family == "calvin_terminal"


def test_finish_run_status_cannot_override_calvin_oracle_failure() -> None:
    episode = EpisodeRecord.new(
        task_name="calvin_move_slider_left",
        instruction="move the slider left",
        seed=0,
    )

    _populate_episode_from_result(
        episode,
        {
            "loop": {"status": "finished", "reason": "finish_run", "steps": []},
            "task_status": {"available": True, "backend": "calvin", "success": False},
        },
    )

    assert episode.status == "finished"
    assert episode.metadata["official_task_success"] is False


def test_calvin_configs_expand_project_paths_and_prefer_run_environment() -> None:
    agent = load_config("configs/calvin_xvla_enabled_probe.json")
    rl = load_rl_config("configs/rl/qwen3vl_calvin_xvla_1update.yaml")

    assert agent.environment.artifact_dir == f"{PROJECT_ROOT}/tmp_artifacts/calvin"
    assert agent.runtime_environment.pythonpath_prefix[0] == f"{PROJECT_ROOT}/src"
    assert agent.runtime_environment.env["NO_PROXY"] == "127.0.0.1,localhost"
    assert "http_proxy" not in agent.runtime_environment.env
    assert agent.metadata["action_backend"]["horizon"] == 20
    assert agent.metadata["action_backend"]["inference_steps"] == 10
    backend_health = CalvinHttpActionBackend(agent.metadata["action_backend"]).health()
    assert backend_health["checkpoint_id"] == "2toINF/X-VLA-Calvin-ABC_D"
    assert backend_health["checkpoint_sha256"].startswith("98813bb2")
    assert rl.rollout.base_config == f"{PROJECT_ROOT}/configs/calvin_xvla_enabled_probe.json"
    assert rl.environment.cwd == PROJECT_ROOT
    assert rl.environment.env["NO_PROXY"] == "127.0.0.1,localhost"
    assert _should_run_environment(rl) is True
    assert WORKSPACE_ROOT == str(Path(PROJECT_ROOT).parent)


@pytest.mark.calvin_integration
def test_real_calvin_reset_and_capture_via_configured_python() -> None:
    if os.environ.get("CLAWVLA_RUN_CALVIN_INTEGRATION") != "1":
        pytest.skip("set CLAWVLA_RUN_CALVIN_INTEGRATION=1 to enable the real CALVIN reset/capture test")
    python = Path(os.environ.get("CLAWVLA_CALVIN_PYTHON", ""))
    config = Path(os.environ.get("CLAWVLA_CALVIN_CONFIG", ""))
    if not python.is_file() or not config.is_file():
        pytest.skip("CLAWVLA_CALVIN_PYTHON and CLAWVLA_CALVIN_CONFIG must point to existing files")

    code = """
import json
import sys
from clawvla.config import load_config
from clawvla.envs.factory import build_env_adapter

adapter = build_env_adapter(load_config(sys.argv[1]))
try:
    observation = adapter.capture_views(setup=True, artifact_prefix='pytest_calvin_integration')
    status = adapter.task_status()
    print(json.dumps({
        'camera_names': sorted(observation.camera_views),
        'proprio_dim': len(observation.raw['calvin_proprio']),
        'task_env_bound': adapter.env is not None,
        'status_backend': status['backend'],
    }))
finally:
    adapter.close()
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(PROJECT_ROOT) / "src")
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    completed = subprocess.run(
        [str(python), "-c", code, str(config)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert completed.returncode == 0, completed.stderr
    payload = next(
        (
            candidate
            for line in completed.stdout.splitlines()
            if line.lstrip().startswith("{")
            for candidate in [json.loads(line)]
            if "proprio_dim" in candidate
        ),
        None,
    )
    assert payload == {
        "camera_names": ["gripper", "static"],
        "proprio_dim": 20,
        "task_env_bound": True,
        "status_backend": "calvin",
    }
