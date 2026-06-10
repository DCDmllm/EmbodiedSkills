from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen

from clawvla.agent_loop import AgentLoop, AgentLoopConfig
from clawvla.config import load_config
from clawvla.loop_types import LoopDecision, LoopStepRecord
from clawvla.notices import _collect_markers
from clawvla.blackboard import Blackboard
from clawvla.components.state import update_world_state
from clawvla.rl.config import load_rl_config
from clawvla.rl.rollout_worker import _append_episode_terminal_reward
from clawvla.rl.verl_agent_loop import _episode_reward
from clawvla.rl.policy_proxy import PolicyProxy, StaticPolicyBackend
from clawvla.rl.reward_registry import build_reward_registry
from clawvla.scripts.run_loop import _apply_runtime_environment
from clawvla.schema import ObservationBundle, SkillRequest, SkillResult
from clawvla.skills.base import SkillContext
from clawvla.rl.trajectory import (
    EpisodeRecord,
    PolicyCallTrace,
    SkillCallTrace,
    TrajectoryWriter,
    build_agent_loop_adapter_from_calls,
    build_response_mask_from_calls,
)


def test_load_default_rl_config() -> None:
    config = load_rl_config("configs/rl/qwen3vl_pi05_grpo.yaml")
    assert config.policy.model_path.endswith("Qwen3-VL-8B-Instruct")
    assert config.reward.task_map["place_container_plate"] == "robotwin"
    assert config.verl.algorithm == "grpo"
    assert config.verl.train_mode == "full"
    assert config.verl.lora_merge_for_rollout is False
    assert config.verl.force_full_gpu_workers is False
    assert config.verl.lora_target_modules == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_reward_registry_requires_configured_task() -> None:
    config = load_rl_config("configs/rl/qwen3vl_pi05_grpo.yaml")
    registry = build_reward_registry(config.reward.registry, config.reward.task_map)
    assert registry.handler_for_task("place_container_plate").name == "robotwin"


def test_policy_proxy_static_backend(tmp_path) -> None:
    writer = TrajectoryWriter(tmp_path / "events.jsonl")
    proxy = PolicyProxy(
        host="127.0.0.1",
        port=0,
        backend=StaticPolicyBackend('{"ok": true}'),
        trajectory_writer=writer,
    )
    proxy.start()
    try:
        payload = {
            "model": "clawvla-policy:scheduler",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "choose"}]}],
        }
        request = Request(
            f"{proxy.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer test"},
        )
        with urlopen(request, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))
    finally:
        proxy.stop()
    assert body["choices"][0]["message"]["content"] == '{"ok": true}'
    assert proxy.calls[0].role == "scheduler"
    assert proxy.calls[0].parsed_json == {"ok": True}


def test_response_mask_trains_only_model_tokens() -> None:
    first = PolicyCallTrace.new(role="scheduler", model="m:scheduler", messages=[], image_refs=[])
    first.prompt_ids = [1, 2]
    first.response_ids = [3, 4]
    second = PolicyCallTrace.new(role="vision", model="m:vision", messages=[], image_refs=[])
    second.prompt_ids = [5]
    second.response_ids = [6, 7]
    adapter = build_response_mask_from_calls([first, second], separator_ids=[99])
    assert adapter["prompt_ids"] == [1, 2]
    assert adapter["response_ids"] == [3, 4, 99, 5, 6, 7]
    assert adapter["response_mask"] == [1, 1, 0, 0, 1, 1]


def test_agent_loop_adapter_carries_multimodal_payloads_in_call_order() -> None:
    first = PolicyCallTrace.new(role="vision", model="m:vision", messages=[], image_refs=["first.png"])
    first.prompt_ids = [1, 2]
    first.response_ids = [3]
    first._clawvla_multi_modal_data = {"images": ["image-first"]}
    first._clawvla_mm_processor_kwargs = {"max_pixels": 1024}

    second = PolicyCallTrace.new(role="scheduler", model="m:scheduler", messages=[], image_refs=["second.png"])
    second.prompt_ids = [4, 5]
    second.response_ids = [6]
    second._clawvla_multi_modal_data = {"images": ["image-second"]}
    second._clawvla_mm_processor_kwargs = {"max_pixels": 1024}

    adapter = build_agent_loop_adapter_from_calls([first, second], separator_ids=[99])

    assert adapter["prompt_ids"] == [1, 2]
    assert adapter["response_ids"] == [3, 99, 4, 5, 6]
    assert adapter["response_mask"] == [1, 0, 0, 0, 1]
    assert adapter["multi_modal_data"] == {"images": ["image-first", "image-second"]}
    assert adapter["mm_processor_kwargs"] == {"max_pixels": 1024}


def test_agent_loop_adapter_rejects_image_refs_without_training_payload() -> None:
    call = PolicyCallTrace.new(role="vision", model="m:vision", messages=[], image_refs=["view.png"])
    call.prompt_ids = [1]
    call.response_ids = [2]
    try:
        build_agent_loop_adapter_from_calls([call])
    except ValueError as exc:
        assert "did not carry training multi_modal_data" in str(exc)
    else:
        raise AssertionError("expected missing multimodal payload to fail loudly")


def test_max_steps_result_keeps_failure_visible() -> None:
    loop = AgentLoop.__new__(AgentLoop)
    loop.config = AgentLoopConfig(max_steps=1)
    result = loop._max_steps_result(
        "observe",
        [
            LoopStepRecord(
                step_index=0,
                stage_before="observe",
                decision=LoopDecision(next_component="scheduler", next_skill="choose_next_skill"),
                status="skill_exception",
                result={"success": False, "status": "skill_exception"},
            )
        ],
    )
    assert result.status == "max_steps_reached_with_failures"
    assert "step=0:status=skill_exception" in str(result.reason)


def test_terminal_reward_is_archived_for_failed_episode(tmp_path) -> None:
    config = load_rl_config("configs/rl/qwen3vl_pi05_grpo.yaml")
    episode = EpisodeRecord.new(task_name="place_container_plate", instruction="place the container on the plate")
    episode.status = "max_steps_reached_with_failures"
    episode.skill_calls = [
        SkillCallTrace(
            step_index=0,
            stage="observe",
            component="scheduler",
            skill="choose_next_skill",
            status="invalid_decision",
            success=False,
        ),
        SkillCallTrace(
            step_index=1,
            stage="observe",
            component="vision",
            skill="ground_task_objects",
            status="task_grounding_invalid_model_output",
            success=False,
        ),
    ]
    reward_path = tmp_path / "episode_reward.jsonl"

    _append_episode_terminal_reward(episode, reward_path, config)

    lines = reward_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "clawvla_rl_reward_record"
    assert payload["reward"]["reward"] == -4.0
    assert episode.reward_score == -4.0
    assert episode.rewards[0].reason == (
        "episode_status=max_steps_reached_with_failures;incomplete_episode=1;invalid_decisions=1;failed_skills=1"
    )


def test_terminal_reward_penalizes_incomplete_episode_without_skill_failure(tmp_path) -> None:
    config = load_rl_config("configs/rl/qwen3vl_pi05_grpo.yaml")
    episode = EpisodeRecord.new(task_name="place_container_plate", instruction="place the container on the plate")
    episode.status = "max_steps_reached"
    episode.skill_calls = [
        SkillCallTrace(
            step_index=0,
            stage="observe",
            component="vision",
            skill="capture_views",
            status="observation_captured",
            success=True,
        )
    ]
    reward_path = tmp_path / "episode_reward.jsonl"

    _append_episode_terminal_reward(episode, reward_path, config)

    assert episode.reward_score == -1.0
    payload = json.loads(reward_path.read_text(encoding="utf-8"))
    assert payload["reward"]["events"]["episode_incomplete"] is True
    assert payload["reward"]["metrics"]["incomplete_episode"] == 1.0


def test_episode_reward_uses_archived_score_without_double_counting() -> None:
    episode = EpisodeRecord.new(task_name="place_container_plate", instruction="place the container on the plate")
    episode.status = "max_steps_reached_with_failures"
    episode.reward_score = -3.0
    episode.skill_calls = [
        SkillCallTrace(
            step_index=0,
            stage="observe",
            component="scheduler",
            skill="choose_next_skill",
            status="invalid_decision",
            success=False,
        )
    ]

    assert _episode_reward(episode, invalid_decision_penalty=-2.0, skill_failure_penalty=-1.0) == -3.0


def test_run_loop_applies_runtime_environment(monkeypatch) -> None:
    config = load_config("configs/robotwin_pi05_worker_probe.json")
    monkeypatch.delenv("VK_ICD_FILENAMES", raising=False)
    monkeypatch.delenv("__EGL_VENDOR_LIBRARY_DIRS", raising=False)

    _apply_runtime_environment(config)

    assert os.environ["VK_ICD_FILENAMES"] == "/etc/vulkan/icd.d/nvidia_icd.json"
    assert os.environ["__EGL_VENDOR_LIBRARY_DIRS"] == "/usr/share/glvnd/egl_vendor.d"


def test_update_world_state_does_not_create_placeholder_perception() -> None:
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("observation", ObservationBundle(observation_id="obs_test"))
    request = SkillRequest(component="state", skill="update_world_state", payload={"stage": "observe"})

    result = update_world_state(request, SkillContext(component_name="state", blackboard=blackboard))

    assert result.success is False
    assert result.status == "world_state_unavailable"
    assert result.errors == ["missing_perception_before_world_state_update"]
    assert blackboard.read("world_state") is None
    assert blackboard.read("last_state_error")["reason"] == "missing_perception_before_world_state_update"


def test_loop_does_not_auto_update_world_state_after_capture_only() -> None:
    class RuntimeStub:
        def __init__(self) -> None:
            self.blackboard = Blackboard(task_instruction="place the container on the plate")
            self.calls: list[tuple[str, str]] = []

        def run_skill(self, component: str, skill: str, payload: dict, **kwargs):
            self.calls.append((component, skill))
            return SkillResult(success=True, status="ok")

    runtime = RuntimeStub()
    loop = AgentLoop.__new__(AgentLoop)
    loop.runtime = runtime

    loop._post_skill_update(
        LoopDecision(next_component="vision", next_skill="capture_views", stage="observe"),
        SkillResult(success=True, status="observation_captured"),
    )

    assert runtime.calls == []


def test_notice_marker_ignores_inactive_placeholder_flag() -> None:
    assert _collect_markers({"metadata": {"placeholder": False, "placeholder_reason": None}}) == set()
    assert _collect_markers({"metadata": {"placeholder": True}}) == {"placeholder"}
