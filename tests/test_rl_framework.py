from __future__ import annotations

import importlib
import json
import os
import sys
import types
from types import SimpleNamespace
from urllib.request import Request, urlopen

import pytest

from clawvla.agent_loop import AgentLoop, AgentLoopConfig
from clawvla.action_backends.pi05 import _resolve_prompt
from clawvla.config import RobotwinConfig, load_config
from clawvla.components.motion import _vla_prompt, execute_action
from clawvla.components.recovery import build_retry_request, decide_recovery
from clawvla.components.safety import preflight_action
from clawvla.components.scheduler import (
    _task_plan_instruction,
    _validate_task_plan_completeness,
    advance_subgoal,
    build_task_plan,
    choose_next_skill,
    repair_stage_transition,
)
from clawvla.components.verifier import _subgoal_verification_contract, _verifier_blackboard_context, verify_progress
from clawvla.envs.robotwin_session import prepare_task_args
from clawvla.components.vision import localize_task_objects
from clawvla.loop_types import LoopDecision, LoopStepRecord
from clawvla.notices import _collect_markers
from clawvla.blackboard import Blackboard
from clawvla.components.state import update_world_state
from clawvla.phase_policy import PhasePolicy
from clawvla.rl.config import build_rollout_episode_specs, load_rl_config
from clawvla.rl.openrlhf_runner import _openrlhf_env, _openrlhf_train_command, _write_prompt_dataset
from clawvla.rl.openrlhf_runtime_patches import (
    _flatten_agent_outputs,
    _patch_samples_generator_preserve_clawvla_rollout_batches,
    _shape_episode_rewards,
)
from clawvla.rl.rollout_worker import _append_episode_terminal_reward
from clawvla.rl.policy_proxy import PolicyProxy, StaticPolicyBackend
from clawvla.rl.reward_registry import build_reward_registry
from clawvla.scripts.run_loop import _apply_runtime_environment
from clawvla.schema import (
    ActionChunk,
    CameraView,
    MotionGoal,
    ObservationBundle,
    PerceptionResult,
    RobotArmState,
    SceneCandidate,
    SkillRequest,
    SkillResult,
    Subgoal,
    TaskPlan,
    VerificationReport,
    WorldState,
)
from clawvla.skills.base import SkillContext
from clawvla.rl.trajectory import (
    EpisodeRecord,
    PolicyCallTrace,
    SkillCallTrace,
    TrajectoryWriter,
    build_policy_call_adapter,
)


def _import_openrlhf_agent_with_fakes(monkeypatch):
    openrlhf_module = types.ModuleType("openrlhf")
    openrlhf_module.__path__ = []
    utils_module = types.ModuleType("openrlhf.utils")
    utils_module.__path__ = []
    agent_module = types.ModuleType("openrlhf.utils.agent")
    vlm_module = types.ModuleType("openrlhf.utils.vlm_utils")

    class AgentExecutorBase:
        pass

    agent_module.AgentExecutorBase = AgentExecutorBase
    vlm_module.process_prompt_with_images = lambda *args, **kwargs: ([1], None, [])
    monkeypatch.setitem(sys.modules, "openrlhf", openrlhf_module)
    monkeypatch.setitem(sys.modules, "openrlhf.utils", utils_module)
    monkeypatch.setitem(sys.modules, "openrlhf.utils.agent", agent_module)
    monkeypatch.setitem(sys.modules, "openrlhf.utils.vlm_utils", vlm_module)
    sys.modules.pop("clawvla.rl.openrlhf_agent", None)
    return importlib.import_module("clawvla.rl.openrlhf_agent")


def test_load_default_rl_config() -> None:
    config = load_rl_config("configs/rl/qwen3vl_pi05_grpo.yaml")
    assert config.policy.model_path.endswith("Qwen3-VL-8B-Instruct")
    assert config.reward.task_map["place_container_plate"] == "robotwin"
    assert config.openrlhf.algorithm == "grpo"
    assert config.openrlhf.train_mode == "full"
    assert config.openrlhf.lora_merge_for_rollout is False
    assert config.openrlhf.force_full_gpu_workers is False
    assert config.openrlhf.lora_target_modules == [
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


def test_policy_proxy_preserves_raw_image_refs_for_training(tmp_path) -> None:
    data_url = "data:image/png;base64,iVBORw0KGgo="
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
            "model": "clawvla-policy:vision",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        }
        request = Request(
            f"{proxy.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer test"},
        )
        with urlopen(request, timeout=5) as response:
            response.read()
    finally:
        proxy.stop()

    call = proxy.calls[0]
    assert call.image_refs == [data_url]
    assert call.messages[0]["content"][1]["image_url"]["url"] == f"data_url:{len(data_url)}"


def test_policy_call_adapter_keeps_single_call_context_only() -> None:
    call = PolicyCallTrace.new(role="scheduler", model="m:scheduler", messages=[], image_refs=[])
    call.prompt_ids = [10, 11]
    call.response_ids = [12, 13]
    call.response_logprobs = [-0.1, -0.2]

    adapter = build_policy_call_adapter(call)

    assert adapter["prompt_ids"] == [10, 11]
    assert adapter["response_ids"] == [12, 13]
    assert adapter["response_mask"] == [1, 1]
    assert adapter["response_logprobs"] == [-0.1, -0.2]


def test_policy_call_adapter_requires_image_training_payload() -> None:
    call = PolicyCallTrace.new(role="vision", model="m:vision", messages=[], image_refs=["view.png"])
    call.prompt_ids = [1]
    call.response_ids = [2]

    with pytest.raises(ValueError, match="did not carry training multi_modal_data"):
        build_policy_call_adapter(call)


def test_openrlhf_outputs_use_one_training_item_per_policy_call(monkeypatch) -> None:
    openrlhf_agent = _import_openrlhf_agent_with_fakes(monkeypatch)
    episode = EpisodeRecord.new(
        task_name="place_container_plate",
        instruction="place the container on the plate",
        seed=123,
    )
    episode.status = "finished"

    first = PolicyCallTrace.new(role="vision", model="m:vision", messages=[], image_refs=["first.png"])
    first.prompt_ids = [10, 11]
    first.response_ids = [12]
    first.response_logprobs = [-0.12]
    first._clawvla_multi_modal_data = {"images": ["image-first"]}
    first._clawvla_openrlhf_mm_train_inputs = {"pixel_values": "first-mm"}

    second = PolicyCallTrace.new(role="scheduler", model="m:scheduler", messages=[], image_refs=[])
    second.prompt_ids = [20, 21, 22]
    second.response_ids = [23, 24]

    episode.policy_calls = [first, second]

    samples = openrlhf_agent._episode_to_call_samples(
        prompt="run episode",
        label='{"index": 0}',
        episode=episode,
        reward_score=1.25,
        group_uid=7,
    )

    assert len(samples) == 2
    assert samples[0]["observation_tokens"] == [10, 11, 12]
    assert samples[0]["action_ranges"] == [(2, 3)]
    assert samples[0]["rollout_log_probs"] == [0.0, 0.0, -0.12]
    assert samples[0]["images"] == ["first.png"]
    assert samples[0]["mm_train_inputs"] == {"pixel_values": "first-mm"}
    assert samples[0]["extra_logs"]["clawvla_group_uid"] == 7
    assert samples[0]["extra_logs"]["clawvla_call_index"] == 0
    assert samples[0]["extra_logs"]["clawvla_policy_calls"] == 2
    assert samples[1]["observation_tokens"] == [20, 21, 22, 23, 24]
    assert samples[1]["action_ranges"] == [(3, 5)]
    assert samples[1]["rollout_log_probs"] is None
    assert samples[1]["images"] is None
    assert samples[1]["reward"] == 1.25
    assert samples[1]["scores"] == 1.25


def test_openrlhf_sample_builder_fails_on_untrainable_policy_call(monkeypatch) -> None:
    openrlhf_agent = _import_openrlhf_agent_with_fakes(monkeypatch)
    episode = EpisodeRecord.new(task_name="place_container_plate", instruction="place the container on the plate")
    episode.status = "finished"
    call = PolicyCallTrace.new(role="scheduler", model="m:scheduler", messages=[], image_refs=[])
    call.response_ids = [2]
    episode.policy_calls = [call]

    with pytest.raises(ValueError, match="without prompt_ids"):
        openrlhf_agent._episode_to_call_samples(
            prompt="run episode",
            label="{}",
            episode=episode,
            reward_score=0.0,
            group_uid=1,
        )


def test_openrlhf_sample_builder_requires_image_mm_train_inputs(monkeypatch) -> None:
    openrlhf_agent = _import_openrlhf_agent_with_fakes(monkeypatch)
    episode = EpisodeRecord.new(task_name="place_container_plate", instruction="place the container on the plate")
    episode.status = "finished"
    call = PolicyCallTrace.new(role="vision", model="m:vision", messages=[], image_refs=["first.png"])
    call.prompt_ids = [1]
    call.response_ids = [2]
    call._clawvla_multi_modal_data = {"images": ["image-first"]}
    episode.policy_calls = [call]

    with pytest.raises(ValueError, match="did not carry mm_train_inputs"):
        openrlhf_agent._episode_to_call_samples(
            prompt="run episode",
            label="{}",
            episode=episode,
            reward_score=0.0,
            group_uid=1,
        )


def test_openrlhf_label_parser_is_strict(monkeypatch) -> None:
    openrlhf_agent = _import_openrlhf_agent_with_fakes(monkeypatch)

    with pytest.raises(ValueError, match="label must be valid JSON"):
        openrlhf_agent._parse_label("not-json")
    with pytest.raises(ValueError, match="label is required"):
        openrlhf_agent._parse_label("")


def test_openrlhf_logprob_extraction_is_strict(monkeypatch) -> None:
    openrlhf_agent = _import_openrlhf_agent_with_fakes(monkeypatch)
    completion = SimpleNamespace(logprobs=[{11: SimpleNamespace(logprob=-0.5)}])

    assert openrlhf_agent._extract_logprobs(completion, [11]) == [-0.5]
    with pytest.raises(ValueError, match="unexpected length"):
        openrlhf_agent._extract_logprobs(completion, [11, 12])
    with pytest.raises(ValueError, match="missing sampled token"):
        openrlhf_agent._extract_logprobs(completion, [12])


def test_openrlhf_runtime_flattens_nested_agent_outputs() -> None:
    assert _flatten_agent_outputs({"a": 1}) == [{"a": 1}]
    assert _flatten_agent_outputs([[{"a": 1}], ({"b": 2}, [{"c": 3}])]) == [
        {"a": 1},
        {"b": 2},
        {"c": 3},
    ]
    with pytest.raises(TypeError, match="dict or nested list"):
        _flatten_agent_outputs("bad")


def test_openrlhf_episode_group_advantage_is_copied_to_each_call() -> None:
    torch = pytest.importorskip("torch")

    shaped = _shape_episode_rewards(
        raw_rewards=torch.tensor([1.0, 1.0, 3.0, 3.0]),
        group_uids=torch.tensor([7, 7, 7, 7]),
        episode_uids=torch.tensor([10, 10, 20, 20]),
        estimator="group_norm",
    )

    assert shaped[0].item() == pytest.approx(shaped[1].item())
    assert shaped[2].item() == pytest.approx(shaped[3].item())
    assert shaped[0].item() == pytest.approx(-0.70710677, rel=1e-5)
    assert shaped[2].item() == pytest.approx(0.70710677, rel=1e-5)


def test_openrlhf_episode_group_advantage_rejects_mismatched_call_rewards() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="same episode disagree on reward"):
        _shape_episode_rewards(
            raw_rewards=torch.tensor([1.0, 2.0]),
            group_uids=torch.tensor([7, 7]),
            episode_uids=torch.tensor([10, 10]),
            estimator="group_norm",
        )


def test_openrlhf_runtime_patch_does_not_split_generated_call_batch(monkeypatch) -> None:
    samples_module = types.ModuleType("openrlhf.trainer.ppo_utils.samples_generator")
    sleep_calls = []
    log_messages = []

    class SamplesGenerator:
        def __init__(self):
            self.prompts_dataloader = iter(["prompt"])
            self.vllm_engines = ["engine"]
            self.args = SimpleNamespace(
                vllm=SimpleNamespace(enable_sleep=True),
                rollout=SimpleNamespace(vllm_generate_batch_size=1, batch_size=1),
                algo=SimpleNamespace(dynamic_filtering_enable=False),
            )

        def _generate_vllm(self, **kwargs):
            assert kwargs["num_prompts"] == 1
            return ["call-0", "call-1", "call-2"], 1, True

    samples_module.SamplesGenerator = SamplesGenerator
    samples_module.batch_vllm_engine_call = lambda engines, method: sleep_calls.append((engines, method))
    samples_module.logger = SimpleNamespace(info=log_messages.append)
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ppo_utils.samples_generator", samples_module)

    _patch_samples_generator_preserve_clawvla_rollout_batches()

    generator = SamplesGenerator()
    batch, filter_pass_rate, prompts_consumed, exhausted = generator.generate_samples()
    assert batch == ["call-0", "call-1", "call-2"]
    assert filter_pass_rate is None
    assert prompts_consumed == 1
    assert exhausted is False
    assert sleep_calls == [(["engine"], "wake_up"), (["engine"], "sleep")]
    assert log_messages == ["Prompt dataloader is exhausted."]

    batch, filter_pass_rate, prompts_consumed, exhausted = generator.generate_samples()
    assert batch == []
    assert filter_pass_rate is None
    assert prompts_consumed == 0
    assert exhausted is True


def test_rollout_episode_specs_expand_multitask_config() -> None:
    config = load_rl_config("configs/rl/qwen3vl_pi05_multitask_1update.yaml")
    specs = build_rollout_episode_specs(config)

    assert len(specs) == 50
    assert specs[0].index == 0
    assert specs[0].task_name == "beat_block_hammer"
    assert specs[0].seed == 0
    assert specs[-1].index == 49
    assert specs[-1].task_name == "put_object_cabinet"


def test_openrlhf_prompt_dataset_expands_multitask_tasks(tmp_path) -> None:
    config = load_rl_config("configs/rl/qwen3vl_pi05_multitask_1update.yaml")

    path = _write_prompt_dataset(config, tmp_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 50
    first_label = json.loads(rows[0]["label"])
    last_label = json.loads(rows[-1]["label"])
    assert first_label["task_name"] == "beat_block_hammer"
    assert first_label["task_index"] == 0
    assert first_label["seed"] == 0
    assert "beat the block" in rows[0]["input"]
    assert last_label["task_name"] == "put_object_cabinet"


def test_openrlhf_train_command_uses_agent_entrypoint_and_full_zero3(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CLAWVLA_OPENRLHF_ADAM_OFFLOAD", raising=False)
    monkeypatch.delenv("CLAWVLA_OPENRLHF_ATTN_IMPLEMENTATION", raising=False)
    monkeypatch.delenv("CLAWVLA_OPENRLHF_DS_TENSOR_PARALLEL_SIZE", raising=False)
    monkeypatch.delenv("CLAWVLA_OPENRLHF_ZERO_STAGE", raising=False)
    config = load_rl_config("configs/rl/qwen3vl_pi05_train_smoke.yaml")
    command = _openrlhf_train_command(
        config,
        python=tmp_path / "python",
        run_dir=tmp_path,
        train_file=tmp_path / "train.jsonl",
    )

    assert command[:3] == [str(tmp_path / "python"), "-m", "openrlhf.cli.train_ppo_ray"]
    assert command[command.index("--train.agent_func_path") + 1].endswith("src/clawvla/rl/openrlhf_agent.py")
    assert command[command.index("--ds.zero_stage") + 1] == "3"
    assert command[command.index("--ds.attn_implementation") + 1] == "flash_attention_2"
    assert command[command.index("--ds.tensor_parallel_size") + 1] == "1"
    assert command[command.index("--train.max_tokens_per_gpu") + 1] == "12288"
    assert "--vllm.enable_sleep" in command
    assert "--ds.enable_sleep" in command
    assert "--ds.adam_offload" in command
    assert "--actor.gradient_checkpointing_enable" in command


def test_openrlhf_train_command_uses_multitask_prompt_count(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CLAWVLA_OPENRLHF_ADAM_OFFLOAD", raising=False)
    monkeypatch.delenv("CLAWVLA_OPENRLHF_ATTN_IMPLEMENTATION", raising=False)
    monkeypatch.delenv("CLAWVLA_OPENRLHF_DS_TENSOR_PARALLEL_SIZE", raising=False)
    config = load_rl_config("configs/rl/qwen3vl_pi05_multitask_1update.yaml")
    command = _openrlhf_train_command(
        config,
        python=tmp_path / "python",
        run_dir=tmp_path,
        train_file=tmp_path / "train.jsonl",
    )

    assert command[command.index("--data.max_samples") + 1] == "50"
    assert command[command.index("--rollout.batch_size") + 1] == "1"
    assert command[command.index("--rollout.n_samples_per_prompt") + 1] == "4"
    assert command[command.index("--algo.advantage.estimator") + 1] == "group_norm"


def test_openrlhf_train_command_allows_adam_offload_opt_out(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CLAWVLA_OPENRLHF_ADAM_OFFLOAD", "0")
    config = load_rl_config("configs/rl/qwen3vl_pi05_train_smoke.yaml")
    command = _openrlhf_train_command(
        config,
        python=tmp_path / "python",
        run_dir=tmp_path,
        train_file=tmp_path / "train.jsonl",
    )

    assert "--ds.adam_offload" not in command


def test_openrlhf_train_command_allows_attention_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CLAWVLA_OPENRLHF_ATTN_IMPLEMENTATION", "sdpa")
    config = load_rl_config("configs/rl/qwen3vl_pi05_train_smoke.yaml")
    command = _openrlhf_train_command(
        config,
        python=tmp_path / "python",
        run_dir=tmp_path,
        train_file=tmp_path / "train.jsonl",
    )

    assert command[command.index("--ds.attn_implementation") + 1] == "sdpa"


def test_openrlhf_real_5step_uses_single_gpu_vllm_engines(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CLAWVLA_OPENRLHF_ADAM_OFFLOAD", raising=False)
    monkeypatch.delenv("CLAWVLA_OPENRLHF_ATTN_IMPLEMENTATION", raising=False)
    monkeypatch.delenv("CLAWVLA_OPENRLHF_DS_TENSOR_PARALLEL_SIZE", raising=False)
    config = load_rl_config("configs/rl/qwen3vl_pi05_real_5step_1update.yaml")
    command = _openrlhf_train_command(
        config,
        python=tmp_path / "python",
        run_dir=tmp_path,
        train_file=tmp_path / "train.jsonl",
    )

    assert command[command.index("--vllm.tensor_parallel_size") + 1] == "1"
    assert command[command.index("--vllm.num_engines") + 1] == "4"
    assert command[command.index("--vllm.gpu_memory_utilization") + 1] == "0.45"
    assert command[command.index("--data.max_len") + 1] == "16384"
    assert command[command.index("--actor.num_gpus_per_node") + 1] == "4"
    assert command[command.index("--ds.zero_stage") + 1] == "3"
    assert command[command.index("--ds.attn_implementation") + 1] == "flash_attention_2"
    assert command[command.index("--ds.tensor_parallel_size") + 1] == "1"
    assert command[command.index("--train.max_tokens_per_gpu") + 1] == "8192"
    assert "--ds.adam_offload" in command
    assert "--actor.gradient_checkpointing_enable" in command


def test_openrlhf_env_removes_expandable_segments_when_vllm_sleep_is_enabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.delenv("CLAWVLA_OPENRLHF_VLLM_ENABLE_SLEEP", raising=False)
    config = load_rl_config("configs/rl/qwen3vl_pi05_train_smoke.yaml")

    env = _openrlhf_env(config, tmp_path, tmp_path / "resolved_config.yaml")

    assert "PYTORCH_CUDA_ALLOC_CONF" not in env
    assert env["CLAWVLA_ENABLE_OPENRLHF_RUNTIME_PATCHES"] == "1"
    assert env["CLAWVLA_OPENRLHF_RL_CONFIG"] == str(tmp_path / "resolved_config.yaml")
    assert env["RAY_CGRAPH_submit_timeout"] == "300"
    assert env["RAY_CGRAPH_get_timeout"] == "300"


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
            skill="localize_task_objects",
            status="localization_invalid_model_output",
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


def test_terminal_reward_lightly_penalizes_recoverable_preflight_refresh(tmp_path) -> None:
    config = load_rl_config("configs/rl/qwen3vl_pi05_grpo.yaml")
    episode = EpisodeRecord.new(task_name="place_container_plate", instruction="place the container on the plate")
    episode.status = "finished"
    episode.skill_calls = [
        SkillCallTrace(
            step_index=0,
            stage="preflight",
            component="safety",
            skill="preflight_action",
            status="preflight_failed",
            success=False,
            errors=["stale_perception", "stale_world_state"],
        ),
        SkillCallTrace(
            step_index=1,
            stage="preflight",
            component="vision",
            skill="refresh_preflight_observation",
            status="preflight_observation_refreshed",
            success=True,
        ),
    ]
    reward_path = tmp_path / "episode_reward.jsonl"

    _append_episode_terminal_reward(episode, reward_path, config)

    payload = json.loads(reward_path.read_text(encoding="utf-8"))
    assert episode.reward_score == -0.1
    assert payload["reward"]["reward"] == -0.1
    assert payload["reward"]["metrics"]["failed_skills"] == 0.0
    assert payload["reward"]["metrics"]["recoverable_preflight_failures"] == 1.0
    assert payload["reward"]["events"]["recoverable_preflight_failure_seen"] is True


def test_run_loop_applies_runtime_environment(monkeypatch) -> None:
    config = load_config("configs/robotwin_pi05_worker_probe.json")
    monkeypatch.delenv("VK_ICD_FILENAMES", raising=False)
    monkeypatch.delenv("__EGL_VENDOR_LIBRARY_DIRS", raising=False)

    _apply_runtime_environment(config)

    assert os.environ["VK_ICD_FILENAMES"] == "/etc/vulkan/icd.d/nvidia_icd.json"
    assert os.environ["__EGL_VENDOR_LIBRARY_DIRS"] == "/usr/share/glvnd/egl_vendor.d"


def test_robotwin_camera_profile_applies_to_all_observation_cameras() -> None:
    config = RobotwinConfig(camera_profile="Large_D435_Wide")

    args = prepare_task_args(config)

    assert args["camera"]["head_camera_type"] == "Large_D435_Wide"
    assert args["camera"]["wrist_camera_type"] == "Large_D435_Wide"
    static_types = {
        camera["name"]: camera["type"]
        for camera in args["left_embodiment_config"]["static_camera_list"]
        if camera["name"] in {"head_camera", "front_camera"}
    }
    assert static_types == {
        "head_camera": "Large_D435_Wide",
        "front_camera": "Large_D435_Wide",
    }
    assert args["head_camera_h"] == 540
    assert args["head_camera_w"] == 960


def test_vla_prompt_uses_subgoal_instruction_without_agent_schema() -> None:
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write(
        "current_subgoal",
        Subgoal("S1", "approach", instruction="move close to the container", source_candidate_id="C1"),
    )

    prompt = _vla_prompt(blackboard)

    assert prompt == "move close to the container."
    assert "place the container on the plate" not in prompt
    assert "Task:" not in prompt
    assert "Current action:" not in prompt
    assert "Source object:" not in prompt
    assert "Target object:" not in prompt
    assert "bbox" not in prompt
    assert "criteria" not in prompt
    assert "C1" not in prompt
    assert "S1" not in prompt


def test_pi05_prompt_prefers_motion_plan_subgoal_over_request_prompt() -> None:
    prompt = _resolve_prompt(
        SimpleNamespace(config={}),
        None,
        None,
        None,
        {
            "prompt": "place the container on the plate",
            "motion_plan": {"vla_prompt": "grasp the container."},
        },
    )

    assert prompt == "grasp the container."


def test_pi05_prompt_requires_motion_plan_vla_prompt() -> None:
    with pytest.raises(ValueError, match="motion_plan.vla_prompt is required for VLA execution"):
        _resolve_prompt(
            SimpleNamespace(config={"default_prompt": "perform the task"}),
            SimpleNamespace(motion_hint="grasp the container"),
            SimpleNamespace(task_instruction="place the container on the plate"),
            SimpleNamespace(task_instruction="place the container on the plate"),
            {"prompt": "place the container on the plate"},
        )


def test_task_plan_validation_does_not_apply_keyword_completeness_rules() -> None:
    task_plan = TaskPlan(
        task="place the container on the plate",
        subgoals=[
            Subgoal(
                "S1",
                "grasp",
                instruction="grasp the container",
                source_candidate_id="C1",
                target_candidate_id="C2",
                completion_criteria={"natural_language": "the container is visibly held by the gripper"},
            )
        ],
        current_subgoal_id="S1",
    )

    errors = _validate_task_plan_completeness(task_plan, "place the container on the plate", "C1", "C2")

    assert "incomplete_task_plan_single_subgoal" not in errors
    assert not any(error.startswith("missing_terminal_subgoal:") for error in errors)
    assert errors == []


def test_task_plan_accepts_non_place_terminal_action() -> None:
    task_plan = TaskPlan(
        task="press the button",
        subgoals=[
            Subgoal(
                "S1",
                "approach",
                instruction="move close to the button",
                source_candidate_id="C1",
                completion_criteria={"natural_language": "the gripper is close to the button"},
            ),
            Subgoal(
                "S2",
                "press",
                instruction="press the button",
                source_candidate_id="C1",
                completion_criteria={"natural_language": "the button is visibly pressed"},
            ),
        ],
        current_subgoal_id="S1",
    )

    errors = _validate_task_plan_completeness(task_plan, "press the button", "C1", "C1")

    assert errors == []


def test_task_plan_instruction_prefers_atomic_multi_step_subgoals() -> None:
    instruction = _task_plan_instruction()

    assert "four or more subgoals" in instruction
    assert "reasonably atomic" in instruction
    assert "Do not combine transport" in instruction
    assert "released or otherwise visibly stable" in instruction
    assert "not merely held above" in instruction
    assert "completion_criteria.natural_language" in instruction


def test_task_plan_rejects_missing_natural_language_completion_criteria() -> None:
    task_plan = TaskPlan(
        task="press the button",
        subgoals=[
            Subgoal(
                "S1",
                "press",
                instruction="press the button",
                source_candidate_id="C1",
                completion_criteria={"button_pressed": "C1"},
            )
        ],
        current_subgoal_id="S1",
    )

    errors = _validate_task_plan_completeness(task_plan, "press the button", "C1", "C1")

    assert "missing_natural_language_completion_criteria:S1" in errors


def test_place_verification_requires_stable_release_not_held_above_target() -> None:
    contract = _subgoal_verification_contract(
        Subgoal(
            "S4",
            "place",
            instruction="place the white bowl on the light green plate",
            source_candidate_id="C1",
            target_candidate_id="C2",
            completion_criteria={"natural_language": "the white bowl is resting on the light green plate"},
        )
    )

    success_condition = str(contract["success_condition"])
    not_required = " ".join(str(item) for item in contract.get("not_required", []))

    assert "resting stably" in success_condition
    assert "closed gripper" in success_condition
    assert "full release if the current subgoal is place rather than release" not in not_required


def test_three_step_target_relation_plan_is_not_rejected_by_special_case_rule() -> None:
    task_plan = TaskPlan(
        task="place the container on the plate",
        subgoals=[
            Subgoal(
                "S1",
                "approach",
                instruction="move close to the container",
                source_candidate_id="C1",
                completion_criteria={"natural_language": "the gripper is near the container"},
            ),
            Subgoal(
                "S2",
                "grasp",
                instruction="grasp the container",
                source_candidate_id="C1",
                completion_criteria={"natural_language": "the container is visibly held by the gripper"},
            ),
            Subgoal(
                "S3",
                "place",
                instruction="place the container on the plate",
                source_candidate_id="C1",
                target_candidate_id="C2",
                completion_criteria={"natural_language": "the container is resting on the plate"},
            ),
        ],
        current_subgoal_id="S1",
    )

    errors = _validate_task_plan_completeness(task_plan, "place the container on the plate", "C1", "C2")

    assert "missing_intermediate_subgoal:target_approach" not in errors


def test_target_relation_plan_accepts_transport_before_terminal_place() -> None:
    task_plan = TaskPlan(
        task="place the container on the plate",
        subgoals=[
            Subgoal(
                "S1",
                "approach",
                instruction="move close to the container",
                source_candidate_id="C1",
                completion_criteria={"natural_language": "the gripper is near the container"},
            ),
            Subgoal(
                "S2",
                "grasp",
                instruction="grasp the container",
                source_candidate_id="C1",
                completion_criteria={"natural_language": "the container is visibly held by the gripper"},
            ),
            Subgoal(
                "S3",
                "transport",
                instruction="move the held container above the plate",
                source_candidate_id="C1",
                target_candidate_id="C2",
                completion_criteria={"natural_language": "the container is above the plate"},
            ),
            Subgoal(
                "S4",
                "place",
                instruction="place the container down on the plate and release it",
                source_candidate_id="C1",
                target_candidate_id="C2",
                completion_criteria={"natural_language": "the container is resting on the plate"},
            ),
        ],
        current_subgoal_id="S1",
    )

    errors = _validate_task_plan_completeness(task_plan, "place the container on the plate", "C1", "C2")

    assert errors == []


def test_advance_subgoal_consumes_verification_and_returns_to_preflight() -> None:
    blackboard = Blackboard(task_instruction="place the container on the plate")
    task_plan = TaskPlan(
        task="place the container on the plate",
        subgoals=[
            Subgoal("S1", "approach", source_candidate_id="C1", status="running"),
            Subgoal("S2", "grasp", source_candidate_id="C1", status="pending"),
        ],
        current_subgoal_id="S1",
    )
    blackboard.write("stage", "verify")
    blackboard.write("task_plan", task_plan)
    blackboard.write("current_subgoal", task_plan.subgoals[0])
    blackboard.write("preflight_report", {"status": "preflight_passed"})
    blackboard.write("safety_report", {"status": "preflight_passed"})
    blackboard.write(
        "last_verification_report",
        VerificationReport(success=True, metadata={"next_action": "advance_subgoal", "current_subgoal_id": "S1"}),
    )

    result = advance_subgoal(
        SkillRequest(component="scheduler", skill="advance_subgoal", stage="verify"),
        SkillContext("scheduler", blackboard),
    )

    current = blackboard.read("current_subgoal")
    assert result.success is True
    assert result.status == "subgoal_advanced"
    assert blackboard.read("stage") == "preflight"
    assert blackboard.read("last_verification_report") is None
    assert blackboard.read("last_resolved_verification_report")["consumed_subgoal_id"] == "S1"
    assert blackboard.read("preflight_report") is None
    assert blackboard.read("safety_report") is None
    assert current.subgoal_id == "S2"
    assert current.status == "running"


def test_advance_subgoal_rejects_verification_for_previous_subgoal() -> None:
    blackboard = Blackboard(task_instruction="place the container on the plate")
    task_plan = TaskPlan(
        task="place the container on the plate",
        subgoals=[
            Subgoal("S1", "approach", source_candidate_id="C1", status="succeeded"),
            Subgoal("S2", "grasp", source_candidate_id="C1", status="running"),
        ],
        current_subgoal_id="S2",
    )
    blackboard.write("task_plan", task_plan)
    blackboard.write("current_subgoal", task_plan.subgoals[1])
    blackboard.write(
        "last_verification_report",
        VerificationReport(success=True, metadata={"next_action": "advance_subgoal", "current_subgoal_id": "S1"}),
    )

    result = advance_subgoal(
        SkillRequest(component="scheduler", skill="advance_subgoal", stage="verify"),
        SkillContext("scheduler", blackboard),
    )

    assert result.success is False
    assert result.status == "advance_subgoal_unavailable"
    assert result.errors == ["verification_current_subgoal_mismatch"]
    assert blackboard.read("current_subgoal").subgoal_id == "S2"
    assert blackboard.read("last_verification_report") is not None


def test_preflight_action_passes_complete_execute_contract(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "clawvla.components.safety._openpi_worker_status",
        lambda runtime_cfg: {"ok": True, "mode": "worker", "reason": "ok"},
    )
    blackboard = _preflight_blackboard(tmp_path, subgoal_type="place", include_target=True)

    result = preflight_action(SkillRequest(component="safety", skill="preflight_action"), SkillContext("safety", blackboard))

    report = blackboard.read("preflight_report")
    assert result.success is True
    assert result.status == "preflight_passed"
    assert report.allowed is True
    assert report.checks["camera_inputs"]["status"] == "passed"
    assert report.checks["robot_state"]["status"] == "passed"


def test_preflight_action_blocks_place_without_target(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "clawvla.components.safety._openpi_worker_status",
        lambda runtime_cfg: {"ok": True, "mode": "worker", "reason": "ok"},
    )
    blackboard = _preflight_blackboard(tmp_path, subgoal_type="place", include_target=False)

    result = preflight_action(SkillRequest(component="safety", skill="preflight_action"), SkillContext("safety", blackboard))

    assert result.success is False
    assert result.status == "preflight_failed"
    assert "missing_target_candidate_for_place" in result.errors


def test_localize_task_objects_rejects_bad_source_target_contract() -> None:
    existing = PerceptionResult(
        observation_id="obs_test",
        candidates=[
            SceneCandidate(
                candidate_id="C1",
                label="container",
                bbox_by_view={"head_camera": [10, 20, 80, 120]},
                visibility="yes",
                confidence=0.9,
            ),
            SceneCandidate(
                candidate_id="C2",
                label="plate",
                bbox_by_view={"head_camera": [100, 140, 220, 260]},
                visibility="yes",
                confidence=0.9,
            ),
        ],
        source_candidate_id="C1",
        target_candidate_id="C2",
    )
    bad_response = json.dumps(
        {
            "candidates": [
                {"candidate_id": "C1", "label": "container", "bbox_by_view": {}, "visibility": "no"},
                {"candidate_id": "C2", "label": "plate", "bbox_by_view": {}, "visibility": "no"},
            ],
            "source_candidate_id": "C1",
            "target_candidate_id": "C2",
            "uncertainty": {"needs_reobserve": False, "reasons": []},
        }
    )
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("observation", ObservationBundle(observation_id="obs_test"))
    blackboard.write("perception", existing)
    model_runtime = SimpleNamespace(enabled=True, generate_text=lambda **kwargs: bad_response)

    result = localize_task_objects(
        SkillRequest(component="vision", skill="localize_task_objects", payload={"use_model": True}),
        SkillContext("vision", blackboard, model_runtime=model_runtime),
    )

    assert result.success is False
    assert result.status == "localization_invalid_model_output"
    assert result.output["errors"] == [
        "source_visibility_no",
        "target_visibility_no",
    ]
    assert blackboard.read("perception") is existing
    assert blackboard.read("last_localization_error")["reason"] == "localization_contract_failed"


def test_localize_task_objects_accepts_semantic_binding_without_bbox() -> None:
    existing = PerceptionResult(
        observation_id="obs_test",
        candidates=[
            SceneCandidate(candidate_id="C1", label="container", visibility="yes", confidence=0.9),
            SceneCandidate(candidate_id="C2", label="plate", visibility="yes", confidence=0.9),
        ],
        source_candidate_id="C1",
        target_candidate_id="C2",
    )
    good_response = json.dumps(
        {
            "candidates": [
                {"candidate_id": "C1", "label": "container", "visibility": "yes", "confidence": 0.9},
                {"candidate_id": "C2", "label": "plate", "visibility": "yes", "confidence": 0.9},
            ],
            "source_candidate_id": "C1",
            "target_candidate_id": "C2",
            "uncertainty": {"needs_reobserve": False, "reasons": []},
        }
    )
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("observation", ObservationBundle(observation_id="obs_test"))
    blackboard.write("perception", existing)
    model_runtime = SimpleNamespace(enabled=True, generate_text=lambda **kwargs: good_response)

    result = localize_task_objects(
        SkillRequest(component="vision", skill="localize_task_objects", payload={"use_model": True}),
        SkillContext("vision", blackboard, model_runtime=model_runtime),
    )

    assert result.success is True
    assert result.status == "task_objects_localized_by_model"
    perception = blackboard.read("perception")
    assert perception.source_candidate_id == "C1"
    assert perception.target_candidate_id == "C2"
    assert all(candidate.bbox_by_view == {} for candidate in perception.candidates)


def test_localize_task_objects_model_prompt_has_no_bbox_contract() -> None:
    captured: dict[str, object] = {}

    def generate_text(**kwargs):
        captured.update(kwargs)
        return json.dumps(
            {
                "candidates": [
                    {"candidate_id": "C1", "label": "container", "visibility": "yes", "confidence": 0.9},
                    {"candidate_id": "C2", "label": "plate", "visibility": "yes", "confidence": 0.9},
                ],
                "source_candidate_id": "C1",
                "target_candidate_id": "C2",
                "uncertainty": {"needs_reobserve": False, "reasons": []},
            }
        )

    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("observation", ObservationBundle(observation_id="obs_test"))
    blackboard.write(
        "perception",
        PerceptionResult(
            observation_id="obs_test",
            candidates=[
                SceneCandidate(candidate_id="C1", label="container", visibility="yes"),
                SceneCandidate(candidate_id="C2", label="plate", visibility="yes"),
            ],
            source_candidate_id="C1",
            target_candidate_id="C2",
        ),
    )
    model_runtime = SimpleNamespace(enabled=True, generate_text=generate_text)

    result = localize_task_objects(
        SkillRequest(component="vision", skill="localize_task_objects", payload={"use_model": True}),
        SkillContext("vision", blackboard, model_runtime=model_runtime),
    )

    assert result.success is True
    messages = captured["messages"]
    prompt = messages[0]["content"][-1]["text"]
    assert "bbox_by_view" not in prompt
    assert "[0, 0, 0, 0]" not in prompt
    assert "bounding box" not in prompt.lower()


def test_blackboard_compact_context_limits_loop_history_and_large_reports() -> None:
    blackboard = Blackboard(task_instruction="place the container on the plate")
    long_error = "context overflow " * 500
    records = []
    for step_index in range(25):
        decision = LoopDecision(
            control="run_skill",
            stage="verify",
            next_component="scheduler",
            next_skill="choose_next_skill",
            narration="verify loop " * 200,
            state_summary="state summary " * 200,
            expected_result="expected result " * 200,
        )
        result = SkillResult(
            success=False,
            status="skill_exception",
            errors=[long_error],
            output={
                "exception": {
                    "exception_type": "BadRequestError",
                    "message": long_error,
                    "traceback": "traceback frame\n" * 1000,
                }
            },
        )
        records.append(LoopStepRecord(step_index, "verify", decision, result.status, result.to_dict()))
    action_chunk = ActionChunk(
        action_type="qpos",
        commands=[[float(item) for item in range(14)] for _ in range(100)],
        control_horizon=100,
        metadata={"subgoal_id": "S1", "observation_id": "obs_test", "consumed": True},
    )
    execution_report = {
        "backend": "pi05",
        "status": "action_executed",
        "success": True,
        "executed_steps": 2,
        "observation": {"observation_id": "obs_test", "raw": {"large": "x" * 10000}},
        "action_chunk": action_chunk.to_dict(),
        "task_env_bound": True,
    }
    blackboard.write("loop_history", records)
    blackboard.write(
        "last_skill_exception",
        {"exception_type": "BadRequestError", "message": long_error, "traceback": "traceback frame\n" * 1000},
    )
    blackboard.write(
        "last_verification_report",
        VerificationReport(
            success=False,
            failure_type="not_done",
            metadata={"next_action": "continue_execute", "execution_report": execution_report},
        ),
    )

    compact = blackboard.compact_context()

    assert len(compact["recent_loop_history"]) == 20
    assert compact["recent_loop_history"][0]["step_index"] == 5
    assert len(compact["recent_loop_history"][-1]["errors"][0]) < 600
    assert compact["last_skill_exception"]["traceback_available"] is True
    assert "traceback frame" not in json.dumps(compact, ensure_ascii=False)
    verification_execution = compact["last_verification_report"]["metadata"]["execution_report"]
    assert verification_execution["action_chunk"]["command_count"] == 100
    assert "commands" not in verification_execution["action_chunk"]
    assert len(json.dumps(compact, ensure_ascii=False)) < 25000


def test_verifier_context_filters_scheduler_narration_claims() -> None:
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write(
        "last_scheduler_decision",
        {
            "next_component": "vision",
            "next_skill": "capture_verify_views",
            "reason": "The robot has grasped the bowl.",
        },
    )
    blackboard.write(
        "loop_history",
        [
            LoopStepRecord(
                24,
                "verify",
                LoopDecision(
                    control="run_skill",
                    stage="verify",
                    next_component="vision",
                    next_skill="capture_verify_views",
                    narration="Capturing images after the robot has grasped the bowl.",
                    state_summary="The robot has grasped the bowl.",
                    expected_result="Images will confirm the grasped bowl.",
                ),
                "verify_observation_captured",
                SkillResult(success=True, status="verify_observation_captured").to_dict(),
            )
        ],
    )

    context = _verifier_blackboard_context(blackboard)

    assert context["last_scheduler_decision"] is None
    assert context["recent_loop_history"] == []
    assert "has grasped" not in json.dumps(context, ensure_ascii=False)


def test_agent_loop_requires_preflight_report_before_execute() -> None:
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("current_subgoal", Subgoal("S1", "approach", source_candidate_id="C1"))
    loop = AgentLoop.__new__(AgentLoop)
    loop.policy = PhasePolicy()
    loop.runtime = SimpleNamespace(blackboard=blackboard)

    error = loop._validate_advance_stage("preflight", LoopDecision(control="advance_stage"))

    assert error == "missing_preflight_report_before_execute"


def test_agent_loop_blocks_plan_to_execute_directly() -> None:
    subgoal = Subgoal("S1", "approach", source_candidate_id="C1")
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("task_plan", TaskPlan(subgoals=[subgoal], current_subgoal_id="S1"))
    blackboard.write("current_subgoal", subgoal)
    loop = AgentLoop.__new__(AgentLoop)
    loop.policy = PhasePolicy()
    loop.runtime = SimpleNamespace(blackboard=blackboard)

    explicit_error = loop._validate_advance_stage("plan", LoopDecision(control="advance_stage", stage="execute"))
    default_error = loop._validate_advance_stage("plan", LoopDecision(control="advance_stage"))

    assert explicit_error == "advance_stage_must_not_set_destination_stage:execute"
    assert default_error is None


def test_agent_loop_blocks_noop_stage_advance() -> None:
    loop = AgentLoop.__new__(AgentLoop)
    loop.policy = PhasePolicy()
    loop.runtime = SimpleNamespace(blackboard=Blackboard(task_instruction="place the container on the plate"))

    error = loop._validate_advance_stage("preflight", LoopDecision(control="advance_stage", stage="preflight"))

    assert error == "advance_stage_must_not_set_destination_stage:preflight"


def test_agent_loop_rejects_destination_for_stage_advance() -> None:
    loop = AgentLoop.__new__(AgentLoop)
    loop.policy = PhasePolicy()
    loop.runtime = SimpleNamespace(blackboard=Blackboard(task_instruction="place the container on the plate"))

    error = loop._validate_advance_stage("verify", LoopDecision(control="advance_stage", stage="recover"))

    assert error == "advance_stage_must_not_set_destination_stage:recover"


def test_scheduler_payload_only_exposes_current_stage_allowed_skills() -> None:
    class ComponentsStub:
        def names(self):
            return ["scheduler", "vision", "state", "safety"]

    class RuntimeStub:
        def __init__(self) -> None:
            self.blackboard = Blackboard(task_instruction="place the container on the plate")
            self.components = ComponentsStub()
            self.payload = None

        def run_skill(self, component: str, skill: str, payload: dict | None = None, **kwargs):
            assert component == "scheduler"
            assert skill == "choose_next_skill"
            self.payload = payload
            return SkillResult(
                success=True,
                status="next_skill_chosen_by_model",
                output={"loop_decision": {"control": "finish_run", "reason": "captured payload"}},
            )

    runtime = RuntimeStub()
    loop = AgentLoop(runtime, config=AgentLoopConfig(max_steps=1))

    loop.run()

    assert runtime.payload is not None
    assert "phase_policy" not in runtime.payload
    assert "safety" not in runtime.payload["allowed_skills"]
    assert "stage_order" in runtime.payload


def test_scheduler_without_model_is_unavailable_not_fallback() -> None:
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("stage", "observe")

    result = choose_next_skill(
        SkillRequest(
            component="scheduler",
            skill="choose_next_skill",
            payload={"loop_mode": True, "current_stage": "observe", "allowed_skills": {"vision": ["capture_views"]}},
            stage="observe",
        ),
        SkillContext("scheduler", blackboard),
    )

    assert result.success is False
    assert result.status == "scheduler_model_unavailable"
    assert result.errors
    assert "scheduler_model_required_no_fallback" in result.errors[0]
    assert blackboard.read("last_scheduler_decision") is None
    assert blackboard.read("last_scheduler_error")["reason"] == result.errors[0]


def test_scheduler_model_output_missing_component_is_explicit_error(monkeypatch) -> None:
    def fake_call_component_json(context, *, instruction, payload, image_paths, render_format):
        return {
            "control": "run_skill",
            "stage": "observe",
            "next_skill": "capture_views",
            "reason": "forgot component",
            "payload": {},
        }

    monkeypatch.setattr("clawvla.components.scheduler.call_component_json", fake_call_component_json)
    blackboard = Blackboard(task_instruction="place the container on the plate")
    result = choose_next_skill(
        SkillRequest(
            component="scheduler",
            skill="choose_next_skill",
            payload={"loop_mode": True, "current_stage": "observe", "allowed_skills": {"vision": ["capture_views"]}},
            stage="observe",
        ),
        SkillContext("scheduler", blackboard, SimpleNamespace(enabled=True)),
    )

    assert result.success is False
    assert result.status == "scheduler_invalid_model_output"
    assert result.output["missing_fields"] == ["next_component"]
    assert "If you forgot next_component" in result.errors[0]
    assert result.output["raw_model_output"]["next_skill"] == "capture_views"
    assert blackboard.read("last_scheduler_decision") is None


def test_build_task_plan_without_model_is_unavailable_not_template() -> None:
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write(
        "world_state",
        WorldState(
            task_instruction="place the container on the plate",
            source_candidate_id="C1",
            target_candidate_id="C2",
        ),
    )

    result = build_task_plan(
        SkillRequest(component="scheduler", skill="build_task_plan", payload={}, stage="plan"),
        SkillContext("scheduler", blackboard),
    )

    assert result.success is False
    assert result.status == "task_plan_unavailable"
    assert "Template task plans are disabled" in result.errors[0]
    assert blackboard.read("task_plan") is None
    assert blackboard.read("last_scheduler_error")["reason"] == result.errors[0]


def test_agent_loop_blocks_motion_before_preflight_passes() -> None:
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("current_subgoal", Subgoal("S1", "approach", source_candidate_id="C1"))
    blackboard.write("world_state", WorldState(task_instruction="place the container on the plate"))
    loop = AgentLoop.__new__(AgentLoop)
    loop.runtime = SimpleNamespace(blackboard=blackboard)

    error = loop._skill_prerequisite_error("motion", "build_motion_goal")

    assert error == "missing_preflight_report_before_execute"


def test_agent_loop_requires_explicit_emit_action_horizon() -> None:
    class ComponentsStub:
        def names(self):
            return ["motion"]

    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("current_subgoal", Subgoal("S1", "grasp", source_candidate_id="C1"))
    blackboard.write("world_state", WorldState(task_instruction="place the container on the plate"))
    blackboard.write("preflight_report", SimpleNamespace(allowed=True, status="preflight_passed", metadata={}))
    blackboard.write("motion_plan", {"status": "image_grounded_motion_plan_built", "metadata": {"subgoal_id": "S1"}})
    loop = AgentLoop.__new__(AgentLoop)
    loop.policy = PhasePolicy()
    loop.runtime = SimpleNamespace(blackboard=blackboard, components=ComponentsStub())

    missing = LoopDecision(control="run_skill", stage="execute", next_component="motion", next_skill="emit_action_chunk")
    too_short = LoopDecision(
        control="run_skill",
        stage="execute",
        next_component="motion",
        next_skill="emit_action_chunk",
        payload={"horizon": 9},
    )
    too_long = LoopDecision(
        control="run_skill",
        stage="execute",
        next_component="motion",
        next_skill="emit_action_chunk",
        payload={"horizon": 51},
    )
    minimum = LoopDecision(
        control="run_skill",
        stage="execute",
        next_component="motion",
        next_skill="emit_action_chunk",
        payload={"horizon": 10},
    )
    normal = LoopDecision(
        control="run_skill",
        stage="execute",
        next_component="motion",
        next_skill="emit_action_chunk",
        payload={"horizon": 50},
    )

    assert loop._validate_run_skill_decision(missing) == "missing_horizon_before_emit_action_chunk"
    assert (
        loop._validate_run_skill_decision(too_short)
        == "horizon_out_of_range_before_emit_action_chunk:9:expected_10_to_50"
    )
    assert loop._validate_run_skill_decision(too_long) == "horizon_out_of_range_before_emit_action_chunk:51:expected_10_to_50"
    assert loop._validate_run_skill_decision(minimum) is None
    assert loop._validate_run_skill_decision(normal) is None


def test_agent_loop_rejects_run_skill_stage_jump() -> None:
    class ComponentsStub:
        def names(self):
            return ["motion"]

    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("stage", "preflight")
    blackboard.write("observation", SimpleNamespace(observation_id="obs_ready"))
    blackboard.write("preflight_report", SimpleNamespace(allowed=True, status="preflight_passed", metadata={"observation_id": "obs_ready"}))
    loop = AgentLoop.__new__(AgentLoop)
    loop.policy = PhasePolicy()
    loop.runtime = SimpleNamespace(blackboard=blackboard, components=ComponentsStub())

    decision = LoopDecision(control="run_skill", stage="execute", next_component="motion", next_skill="execute_action")

    assert loop._validate_run_skill_decision(decision) == "run_skill_stage_must_equal_current_stage:execute!=preflight"


def test_agent_loop_routes_verify_continue_execute_without_more_verify_skills() -> None:
    class ComponentsStub:
        def names(self):
            return ["vision", "motion", "scheduler"]

    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("observation", SimpleNamespace(observation_id="obs_ready"))
    blackboard.write("preflight_report", SimpleNamespace(allowed=True, status="preflight_passed", metadata={"observation_id": "obs_ready"}))
    blackboard.write(
        "last_verification_report",
        VerificationReport(
            success=False,
            partial_progress=True,
            failure_type="not_done",
            metadata={"next_action": "continue_execute"},
        ),
    )
    loop = AgentLoop.__new__(AgentLoop)
    loop.policy = PhasePolicy()
    loop.runtime = SimpleNamespace(blackboard=blackboard, components=ComponentsStub())

    blocked = LoopDecision(control="run_skill", stage="verify", next_component="vision", next_skill="capture_verify_views")
    allowed_repair = LoopDecision(
        control="run_skill",
        stage="verify",
        next_component="scheduler",
        next_skill="repair_stage_transition",
        payload={"target_stage": "preflight", "reason": "verification requested continuing the current subgoal"},
    )

    assert (
        loop._validate_run_skill_decision(blocked)
        == "verify_report_requires_next_action:continue_execute:not_vision.capture_verify_views"
    )
    assert loop._state_gated_allowed_skills("verify", {"vision": ["perceive_scene"]}) == {"scheduler": ["repair_stage_transition"]}
    assert loop._validate_run_skill_decision(allowed_repair) is None


def test_agent_loop_blocks_verify_continue_execute_when_preflight_is_stale() -> None:
    class ComponentsStub:
        def names(self):
            return ["scheduler"]

    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("observation", SimpleNamespace(observation_id="obs_new"))
    blackboard.write("preflight_report", SimpleNamespace(allowed=True, status="preflight_passed", metadata={"observation_id": "obs_old"}))
    blackboard.write(
        "last_verification_report",
        VerificationReport(
            success=False,
            partial_progress=True,
            failure_type="not_done",
            metadata={"next_action": "continue_execute"},
        ),
    )
    loop = AgentLoop.__new__(AgentLoop)
    loop.policy = PhasePolicy()
    loop.runtime = SimpleNamespace(blackboard=blackboard, components=ComponentsStub())

    explicit_stage_error = loop._validate_advance_stage("verify", LoopDecision(control="advance_stage", stage="preflight"))
    repair = LoopDecision(
        control="run_skill",
        stage="verify",
        next_component="scheduler",
        next_skill="repair_stage_transition",
        payload={"target_stage": "preflight", "reason": "verification requested another execution attempt"},
    )

    assert explicit_stage_error == "advance_stage_must_not_set_destination_stage:preflight"
    assert loop._validate_run_skill_decision(repair) is None


def test_agent_loop_blocks_motion_when_preflight_is_stale_without_auto_routing() -> None:
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("observation", SimpleNamespace(observation_id="obs_new"))
    blackboard.write("current_subgoal", Subgoal("S1", "approach", source_candidate_id="C1"))
    blackboard.write("world_state", WorldState(task_instruction="place the container on the plate"))
    blackboard.write("preflight_report", SimpleNamespace(allowed=True, status="preflight_passed", metadata={"observation_id": "obs_old"}))
    loop = AgentLoop.__new__(AgentLoop)
    loop.runtime = SimpleNamespace(blackboard=blackboard)

    error = loop._skill_prerequisite_error("motion", "build_motion_goal")

    assert error == "stale_preflight_report:obs_old->obs_new"


def test_agent_loop_runtime_state_reports_preflight_readiness() -> None:
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("observation", SimpleNamespace(observation_id="obs_new"))
    blackboard.write("preflight_report", SimpleNamespace(allowed=True, status="preflight_passed", metadata={"observation_id": "obs_old"}))
    loop = AgentLoop.__new__(AgentLoop)
    loop.runtime = SimpleNamespace(blackboard=blackboard)

    summary = loop._runtime_state_summary()

    assert summary["preflight_ready"] is False
    assert summary["preflight_error"] == "stale_preflight_report:obs_old->obs_new"


def test_agent_loop_runtime_state_reports_visual_freshness_after_stale_preflight() -> None:
    candidates = [SceneCandidate(candidate_id="C1", label="container", visibility="yes")]
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("observation", SimpleNamespace(observation_id="obs_new"))
    blackboard.write(
        "perception",
        PerceptionResult(observation_id="obs_new", candidates=candidates, source_candidate_id="C1", target_candidate_id="C2"),
    )
    blackboard.write(
        "world_state",
        WorldState(
            task_instruction="place the container on the plate",
            candidates=candidates,
            source_candidate_id="C1",
            target_candidate_id="C2",
            metadata={"observation_id": "obs_new"},
        ),
    )
    blackboard.write(
        "preflight_report",
        SimpleNamespace(allowed=False, status="preflight_failed", errors=["stale_perception", "stale_world_state"]),
    )
    loop = AgentLoop.__new__(AgentLoop)
    loop.runtime = SimpleNamespace(blackboard=blackboard)

    summary = loop._runtime_state_summary()

    assert summary["perception_observation_id"] == "obs_new"
    assert summary["world_state_observation_id"] == "obs_new"
    assert summary["visual_state_fresh_for_current_observation"] is True
    assert summary["stale_visual_state_unresolved"] is False
    assert summary["perception_source_candidate_id"] == "C1"
    assert summary["perception_target_candidate_id"] == "C2"
    assert summary["world_state_source_candidate_id"] == "C1"
    assert summary["world_state_target_candidate_id"] == "C2"
    assert summary["world_state_ready"] is True
    assert summary["world_state_ready_error"] is None
    assert summary["observe_complete"] is True


def test_phase_policy_allows_preflight_observation_refresh_only_in_preflight() -> None:
    policy = PhasePolicy()

    assert "refresh_preflight_observation" in policy.allowed_for_stage("preflight")["vision"]
    assert "refresh_preflight_observation" not in policy.allowed_for_stage("observe")["vision"]


def test_agent_loop_validates_preflight_observation_refresh_error() -> None:
    class ComponentsStub:
        def names(self):
            return ["vision", "scheduler"]

    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write(
        "preflight_report",
        SimpleNamespace(allowed=False, status="preflight_failed", errors=["stale_perception", "stale_world_state"]),
    )
    loop = AgentLoop.__new__(AgentLoop)
    loop.policy = PhasePolicy()
    loop.runtime = SimpleNamespace(blackboard=blackboard, components=ComponentsStub())

    allowed = LoopDecision(
        control="run_skill",
        stage="preflight",
        next_component="vision",
        next_skill="refresh_preflight_observation",
    )
    wrong_stage = LoopDecision(
        control="run_skill",
        stage="observe",
        next_component="vision",
        next_skill="refresh_preflight_observation",
    )
    wrong_repair = LoopDecision(
        control="run_skill",
        stage="preflight",
        next_component="scheduler",
        next_skill="repair_stage_transition",
        payload={"target_stage": "observe", "reason": "stale preflight visual state"},
    )

    assert loop._validate_run_skill_decision(allowed) is None
    assert (
        loop._validate_run_skill_decision(wrong_stage)
        == "skill_not_allowed:observe.vision.refresh_preflight_observation"
    )
    assert loop._validate_run_skill_decision(wrong_repair) == "repair_stage_transition_target_not_allowed:observe:allowed_[]"


def test_agent_loop_requires_preflight_action_after_visual_refresh() -> None:
    class ComponentsStub:
        def names(self):
            return ["vision", "safety"]

    candidates = [SceneCandidate(candidate_id="C1", label="container", visibility="yes")]
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("stage", "preflight")
    blackboard.write("observation", SimpleNamespace(observation_id="obs_new"))
    blackboard.write(
        "perception",
        PerceptionResult(observation_id="obs_new", candidates=candidates, source_candidate_id="C1", target_candidate_id="C2"),
    )
    blackboard.write(
        "world_state",
        WorldState(
            task_instruction="place the container on the plate",
            candidates=candidates,
            source_candidate_id="C1",
            target_candidate_id="C2",
            metadata={"observation_id": "obs_new"},
        ),
    )
    blackboard.write(
        "preflight_report",
        SimpleNamespace(allowed=False, status="preflight_failed", errors=["stale_perception", "stale_world_state"]),
    )
    loop = AgentLoop.__new__(AgentLoop)
    loop.policy = PhasePolicy()
    loop.runtime = SimpleNamespace(blackboard=blackboard, components=ComponentsStub())

    refresh = LoopDecision(
        control="run_skill",
        stage="preflight",
        next_component="vision",
        next_skill="refresh_preflight_observation",
    )
    required = loop._runtime_state_summary()["next_required_decision"]

    assert required["next_component"] == "safety"
    assert required["next_skill"] == "preflight_action"
    assert loop._validate_run_skill_decision(refresh) == "preflight_visual_state_already_refreshed_run_preflight_action"


def test_verify_requires_capture_verify_views_before_verifier() -> None:
    class ComponentsStub:
        def names(self):
            return ["vision", "verifier", "scheduler"]

    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("stage", "verify")
    blackboard.write("current_subgoal", Subgoal("S1", "grasp", source_candidate_id="C1"))
    blackboard.write(
        "execution_report",
        {"status": "action_executed", "observation": {"observation_id": "obs_after"}, "success": False},
    )
    loop = AgentLoop.__new__(AgentLoop)
    loop.policy = PhasePolicy()
    loop.runtime = SimpleNamespace(blackboard=blackboard, components=ComponentsStub())

    required = loop._runtime_state_summary()["next_required_decision"]
    verifier = LoopDecision(control="run_skill", stage="verify", next_component="verifier", next_skill="verify_progress")
    capture = LoopDecision(control="run_skill", stage="verify", next_component="vision", next_skill="capture_verify_views")

    assert required["next_component"] == "vision"
    assert required["next_skill"] == "capture_verify_views"
    assert loop._validate_run_skill_decision(verifier) == "missing_fresh_verify_observation_before_verify_progress"
    assert loop._validate_run_skill_decision(capture) is None


def test_verify_progress_uses_fresh_verify_images(tmp_path) -> None:
    class ComponentsStub:
        def names(self):
            return ["vision", "verifier", "scheduler"]

    image_path = tmp_path / "verify_head.png"
    image_path.write_bytes(b"not-an-image-but-valid-path-for-payload")
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("stage", "verify")
    blackboard.write("current_subgoal", Subgoal("S1", "grasp", source_candidate_id="C1"))
    blackboard.write(
        "execution_report",
        {"status": "action_executed", "observation": {"observation_id": "obs_after"}, "success": False},
    )
    blackboard.write(
        "verify_observation",
        ObservationBundle(
            observation_id="obs_verify",
            camera_views={"head_camera": CameraView(name="head_camera", rgb_path=str(image_path))},
            metadata={"verify_active": True, "source_execution_observation_id": "obs_after"},
        ),
    )
    loop = AgentLoop.__new__(AgentLoop)
    loop.policy = PhasePolicy()
    loop.runtime = SimpleNamespace(blackboard=blackboard, components=ComponentsStub())

    required = loop._runtime_state_summary()["next_required_decision"]
    verifier = LoopDecision(control="run_skill", stage="verify", next_component="verifier", next_skill="verify_progress")
    capture = LoopDecision(control="run_skill", stage="verify", next_component="vision", next_skill="capture_verify_views")

    assert required["next_component"] == "verifier"
    assert required["next_skill"] == "verify_progress"
    assert loop._prepare_payload(verifier)["image_paths"] == [str(image_path)]
    assert loop._validate_run_skill_decision(verifier) is None
    assert loop._validate_run_skill_decision(capture) == "verify_observation_already_captured_run_verify_progress"


def test_verifier_rejects_text_only_verification() -> None:
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("current_subgoal", Subgoal("S1", "grasp", source_candidate_id="C1"))
    blackboard.write(
        "execution_report",
        {"status": "action_executed", "observation": {"observation_id": "obs_after"}, "success": False},
    )
    model_runtime = SimpleNamespace(enabled=True, generate_text=lambda **kwargs: "{}")

    result = verify_progress(
        SkillRequest(component="verifier", skill="verify_progress", payload={"use_model": True, "image_paths": []}),
        SkillContext("verifier", blackboard, model_runtime),
    )

    assert result.success is False
    assert result.status == "verification_unavailable"
    assert result.errors == ["missing_verify_images"]


def test_verifier_model_payload_judges_current_subgoal_not_full_task(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_call_component_json(context, *, instruction, payload, image_paths, render_format):
        captured["instruction"] = instruction
        captured["payload"] = payload
        captured["image_paths"] = image_paths
        return {
            "subgoal_success": True,
            "task_success": False,
            "partial_progress": True,
            "failure_type": "none",
            "progress_score": 1.0,
            "should_reobserve": False,
            "next_action": "advance_subgoal",
            "notes": ["source is held by the gripper"],
        }

    monkeypatch.setattr("clawvla.components.verifier.call_component_json", fake_call_component_json)
    image_path = tmp_path / "verify_head.png"
    image_path.write_bytes(b"x")
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write(
        "current_subgoal",
        Subgoal("S1", "grasp", source_candidate_id="C1", completion_criteria={"grasped": "C1"}),
    )
    blackboard.write(
        "execution_report",
        {"status": "action_executed", "observation": {"observation_id": "obs_after"}, "success": False},
    )
    model_runtime = SimpleNamespace(enabled=True, generate_text=lambda **kwargs: "{}")

    result = verify_progress(
        SkillRequest(
            component="verifier",
            skill="verify_progress",
            payload={"use_model": True, "image_paths": [str(image_path)]},
        ),
        SkillContext("verifier", blackboard, model_runtime),
    )

    contract = captured["payload"]["subgoal_success_contract"]
    assert result.success is True
    assert result.status == "subgoal_verified_success"
    assert captured["payload"]["execution_report"]["full_task_success"] is False
    assert contract["subgoal_type"] == "grasp"
    assert "held by the gripper" in contract["success_condition"]
    assert "placing on the target" in contract["not_required"]
    assert "Do not judge the whole task" in captured["instruction"]
    assert captured["image_paths"] == [str(image_path)]


def test_verifier_canonicalizes_not_done_to_continue_execute(monkeypatch, tmp_path) -> None:
    def fake_call_component_json(context, *, instruction, payload, image_paths, render_format):
        return {
            "subgoal_success": False,
            "task_success": False,
            "partial_progress": True,
            "failure_type": "not_done",
            "progress_score": 0.4,
            "should_reobserve": False,
            "next_action": "recover",
            "notes": ["the gripper is closer but the object is not held"],
        }

    monkeypatch.setattr("clawvla.components.verifier.call_component_json", fake_call_component_json)
    image_path = tmp_path / "verify_head.png"
    image_path.write_bytes(b"x")
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write(
        "current_subgoal",
        Subgoal("S1", "grasp", instruction="grasp the container", source_candidate_id="C1"),
    )
    blackboard.write("execution_report", {"status": "action_executed", "success": False})

    result = verify_progress(
        SkillRequest(
            component="verifier",
            skill="verify_progress",
            payload={"use_model": True, "image_paths": [str(image_path)]},
        ),
        SkillContext("verifier", blackboard, SimpleNamespace(enabled=True)),
    )

    report = blackboard.read("last_verification_report")
    assert result.success is True
    assert result.status == "subgoal_verification_failed"
    assert report.failure_type == "not_done"
    assert report.metadata["raw_next_action"] == "recover"
    assert report.metadata["next_action"] == "continue_execute"


def test_verifier_canonicalizes_execution_failed_to_recover(monkeypatch, tmp_path) -> None:
    def fake_call_component_json(context, *, instruction, payload, image_paths, render_format):
        return {
            "subgoal_success": False,
            "task_success": False,
            "partial_progress": False,
            "failure_type": "execution_failed",
            "progress_score": 0.0,
            "should_reobserve": False,
            "next_action": "continue_execute",
            "notes": ["the wrong object was moved"],
        }

    monkeypatch.setattr("clawvla.components.verifier.call_component_json", fake_call_component_json)
    image_path = tmp_path / "verify_head.png"
    image_path.write_bytes(b"x")
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write(
        "current_subgoal",
        Subgoal("S1", "grasp", instruction="grasp the container", source_candidate_id="C1"),
    )
    blackboard.write("execution_report", {"status": "action_executed", "success": False})

    verify_progress(
        SkillRequest(
            component="verifier",
            skill="verify_progress",
            payload={"use_model": True, "image_paths": [str(image_path)]},
        ),
        SkillContext("verifier", blackboard, SimpleNamespace(enabled=True)),
    )

    report = blackboard.read("last_verification_report")
    assert report.failure_type == "execution_failed"
    assert report.metadata["raw_next_action"] == "continue_execute"
    assert report.metadata["next_action"] == "recover"


def test_recovery_applies_model_subgoal_patch(monkeypatch) -> None:
    def fake_call_component_json(context, *, instruction, payload, image_paths, render_format):
        return {
            "recoverable": True,
            "failure_diagnosis": "the first grasp missed the container",
            "patch_type": "replace_current_subgoal",
            "next_stage": "preflight",
            "repaired_subgoal": {
                "subgoal_id": "S1",
                "type": "grasp",
                "instruction": "regrasp the container from the side",
                "source_candidate_id": "C1",
                "target_candidate_id": None,
                "status": "pending",
                "completion_criteria": {
                    "natural_language": "the container is visibly held securely by the gripper"
                },
            },
            "notes": ["try a side grasp"],
        }

    monkeypatch.setattr("clawvla.components.recovery.call_component_json", fake_call_component_json)
    subgoal = Subgoal(
        "S1",
        "grasp",
        instruction="grasp the container",
        source_candidate_id="C1",
        completion_criteria={"natural_language": "the container is held"},
        status="running",
    )
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("current_subgoal", subgoal)
    blackboard.write("task_plan", TaskPlan(subgoals=[subgoal], current_subgoal_id="S1"))
    blackboard.write("preflight_report", SimpleNamespace(allowed=True, status="preflight_passed"))
    blackboard.write(
        "last_verification_report",
        VerificationReport(success=False, failure_type="execution_failed", metadata={"next_action": "recover"}),
    )
    blackboard.write("execution_report", {"status": "action_executed", "success": False})

    decide_result = decide_recovery(
        SkillRequest(component="recovery", skill="decide_recovery", payload={"use_model": True}, stage="recover"),
        SkillContext("recovery", blackboard, SimpleNamespace(enabled=True)),
    )
    retry_result = build_retry_request(
        SkillRequest(component="recovery", skill="build_retry_request", payload={}, stage="recover"),
        SkillContext("recovery", blackboard),
    )

    patched = blackboard.read("current_subgoal")
    retry_request = blackboard.read("last_retry_request")
    assert decide_result.success is True
    assert decide_result.status == "recovery_patch_decided"
    assert retry_result.success is True
    assert retry_request["stage"] == "preflight"
    assert retry_request["patch_type"] == "replace_current_subgoal"
    assert patched.instruction == "regrasp the container from the side"
    assert patched.metadata["recovery_patch"]["failure_diagnosis"] == "the first grasp missed the container"
    assert blackboard.read("preflight_report") is None


def test_recovery_rejects_not_done_verification() -> None:
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write(
        "last_verification_report",
        VerificationReport(success=False, failure_type="not_done", metadata={"next_action": "continue_execute"}),
    )
    blackboard.write("execution_report", {"status": "action_executed", "success": False})

    result = decide_recovery(
        SkillRequest(component="recovery", skill="decide_recovery", payload={"use_model": True}, stage="recover"),
        SkillContext("recovery", blackboard, SimpleNamespace(enabled=True)),
    )

    assert result.success is False
    assert result.status == "recovery_unavailable"
    assert result.errors == ["recovery_requires_verification_next_action_recover:continue_execute"]


def test_verify_observation_is_cleared_after_verify_stage_transition(tmp_path) -> None:
    image_path = tmp_path / "verify_head.png"
    image_path.write_bytes(b"x")
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write(
        "verify_observation",
        ObservationBundle(
            observation_id="obs_verify",
            camera_views={"head_camera": CameraView(name="head_camera", rgb_path=str(image_path))},
            metadata={"verify_active": True, "source_execution_observation_id": "obs_after"},
        ),
    )
    loop = AgentLoop.__new__(AgentLoop)
    loop.policy = PhasePolicy()
    loop.runtime = SimpleNamespace(blackboard=blackboard)
    decision = LoopDecision(
        control="run_skill",
        stage="verify",
        next_component="scheduler",
        next_skill="repair_stage_transition",
    )

    loop._post_skill_update(decision, SkillResult(success=True, status="repair_stage_transition_requested"))

    assert blackboard.read("verify_observation") is None
    cleared = blackboard.read("last_cleared_verify_observation")
    assert cleared["reason"] == "verify_repair_stage_transition_completed"
    assert cleared["image_paths"] == [str(image_path)]


def test_recovery_uses_archived_verify_images_after_verify_observation_cleared(tmp_path) -> None:
    image_path = tmp_path / "verify_head.png"
    image_path.write_bytes(b"x")
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write(
        "last_cleared_verify_observation",
        {"reason": "verify_repair_stage_transition_completed", "image_paths": [str(image_path)]},
    )
    loop = AgentLoop.__new__(AgentLoop)
    loop.runtime = SimpleNamespace(blackboard=blackboard)
    decision = LoopDecision(control="run_skill", stage="recover", next_component="recovery", next_skill="decide_recovery")

    payload = loop._prepare_payload(decision)

    assert payload["use_model"] is True
    assert payload["image_paths"] == [str(image_path)]


def test_reobserve_transition_clears_stale_visual_state_and_forces_capture() -> None:
    class ComponentsStub:
        def names(self):
            return ["vision", "state"]

    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("stage", "verify")
    blackboard.write("observation", SimpleNamespace(observation_id="obs_old"))
    blackboard.write("perception", SimpleNamespace(observation_id="obs_old", source_candidate_id="C1", target_candidate_id="C2"))
    blackboard.write(
        "world_state",
        WorldState(
            task_instruction="place the container on the plate",
            source_candidate_id="C1",
            target_candidate_id="C2",
            metadata={"observation_id": "obs_old"},
        ),
    )
    blackboard.write("grounding_overlay", {"observation_id": "obs_old"})
    blackboard.write(
        "last_verification_report",
        VerificationReport(success=False, should_reobserve=True, metadata={"next_action": "reobserve"}),
    )

    result = repair_stage_transition(
        SkillRequest(
            component="scheduler",
            skill="repair_stage_transition",
            payload={"target_stage": "observe", "reason": "verification_next_action:reobserve"},
            stage="verify",
        ),
        SkillContext("scheduler", blackboard),
    )
    loop = AgentLoop.__new__(AgentLoop)
    loop.policy = PhasePolicy()
    loop.runtime = SimpleNamespace(blackboard=blackboard, components=ComponentsStub())
    required = loop._runtime_state_summary()["next_required_decision"]

    assert result.success is True
    assert blackboard.read("stage") == "observe"
    assert blackboard.read("observation") is None
    assert blackboard.read("perception") is None
    assert blackboard.read("world_state") is None
    assert blackboard.read("grounding_overlay") is None
    assert blackboard.read("last_reobserve_request")["previous_observation_id"] == "obs_old"
    assert required["next_component"] == "vision"
    assert required["next_skill"] == "capture_views"


def test_recovery_retry_to_preflight_keeps_visual_state_and_requests_preflight() -> None:
    class ComponentsStub:
        def names(self):
            return ["safety", "scheduler"]

    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("stage", "recover")
    blackboard.write("observation", SimpleNamespace(observation_id="obs_after"))
    blackboard.write(
        "world_state",
        WorldState(
            task_instruction="place the container on the plate",
            source_candidate_id="C1",
            target_candidate_id="C2",
            metadata={"observation_id": "obs_after"},
        ),
    )
    blackboard.write(
        "current_subgoal",
        Subgoal("S1", "grasp", instruction="regrasp the container from the side", source_candidate_id="C1"),
    )
    blackboard.write(
        "last_retry_request",
        {"stage": "preflight", "patch_type": "replace_current_subgoal", "reason": "missed grasp"},
    )

    result = repair_stage_transition(
        SkillRequest(
            component="scheduler",
            skill="repair_stage_transition",
            payload={"target_stage": "preflight", "reason": "recovery retry request"},
            stage="recover",
        ),
        SkillContext("scheduler", blackboard),
    )
    loop = AgentLoop.__new__(AgentLoop)
    loop.policy = PhasePolicy()
    loop.runtime = SimpleNamespace(blackboard=blackboard, components=ComponentsStub())
    required = loop._runtime_state_summary()["next_required_decision"]

    assert result.success is True
    assert blackboard.read("stage") == "preflight"
    assert blackboard.read("observation").observation_id == "obs_after"
    assert blackboard.read("world_state").metadata["observation_id"] == "obs_after"
    assert required["next_component"] == "safety"
    assert required["next_skill"] == "preflight_action"


def test_agent_loop_does_not_overwrite_advance_subgoal_stage_transition() -> None:
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("stage", "preflight")
    loop = AgentLoop.__new__(AgentLoop)
    loop.runtime = SimpleNamespace(blackboard=blackboard)

    loop._apply_stage(
        LoopDecision(control="run_skill", stage="verify", next_component="scheduler", next_skill="advance_subgoal")
    )

    assert blackboard.read("stage") == "preflight"


def test_recover_requires_explicit_repair_stage_transition() -> None:
    class ComponentsStub:
        def names(self):
            return ["scheduler"]

    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("last_retry_request", {"stage": "preflight", "component": "safety", "skill": "preflight_action"})
    blackboard.write("stage", "recover")
    loop = AgentLoop.__new__(AgentLoop)
    loop.policy = PhasePolicy()
    loop.runtime = SimpleNamespace(blackboard=blackboard, components=ComponentsStub())

    default_error = loop._validate_advance_stage("recover", LoopDecision(control="advance_stage"))
    repair = LoopDecision(
        control="run_skill",
        stage="recover",
        next_component="scheduler",
        next_skill="repair_stage_transition",
        payload={"target_stage": "preflight", "reason": "retry request requires another preflight"},
    )

    assert loop.policy.next_stage("recover") == "recover"
    assert default_error == "advance_stage_noop:recover"
    assert loop._validate_run_skill_decision(repair) is None


def test_agent_loop_allows_advance_subgoal_only_after_successful_verify_report() -> None:
    class ComponentsStub:
        def names(self):
            return ["scheduler", "vision"]

    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("task_plan", TaskPlan(subgoals=[Subgoal("S1", "grasp", source_candidate_id="C1")]))
    blackboard.write("current_subgoal", Subgoal("S1", "grasp", source_candidate_id="C1"))
    blackboard.write(
        "last_verification_report",
        VerificationReport(success=True, metadata={"next_action": "advance_subgoal"}),
    )
    loop = AgentLoop.__new__(AgentLoop)
    loop.policy = PhasePolicy()
    loop.runtime = SimpleNamespace(blackboard=blackboard, components=ComponentsStub())

    advance = LoopDecision(control="run_skill", stage="verify", next_component="scheduler", next_skill="advance_subgoal")
    extra_vision = LoopDecision(control="run_skill", stage="verify", next_component="vision", next_skill="capture_verify_views")

    assert loop._state_gated_allowed_skills("verify", {"scheduler": ["advance_subgoal"], "vision": ["perceive_scene"]}) == {
        "scheduler": ["advance_subgoal"]
    }
    assert loop._validate_run_skill_decision(advance) is None
    assert (
        loop._validate_run_skill_decision(extra_vision)
        == "verify_report_requires_next_action:advance_subgoal:not_vision.capture_verify_views"
    )


def test_agent_loop_finishes_after_advance_subgoal_completes_task_plan() -> None:
    class ComponentsStub:
        def names(self):
            return ["scheduler", "vision", "verifier"]

    task_plan = TaskPlan(
        subgoals=[Subgoal("S1", "grasp", source_candidate_id="C1", status="succeeded")],
        current_subgoal_id=None,
        status="succeeded",
    )
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("stage", "verify")
    blackboard.write("task_plan", task_plan)
    blackboard.write("current_subgoal", None)
    blackboard.write(
        "last_verification_report",
        VerificationReport(success=True, metadata={"next_action": "advance_subgoal", "current_subgoal_id": "S1"}),
    )
    loop = AgentLoop.__new__(AgentLoop)
    loop.policy = PhasePolicy()
    loop.runtime = SimpleNamespace(blackboard=blackboard, components=ComponentsStub())

    required = loop._runtime_state_summary()["next_required_decision"]

    assert required["control"] == "finish_run"
    assert required["reason"] == "task_plan_complete"
    assert loop._state_gated_allowed_skills("verify", {"scheduler": ["advance_subgoal"]}) == {}


def test_agent_loop_blocks_repeated_decision_after_limit() -> None:
    loop = AgentLoop.__new__(AgentLoop)
    loop.config = AgentLoopConfig(allow_same_decision_repeats=2)
    repeat_state = {"last_key": None, "count": 0}
    decision = LoopDecision(control="run_skill", next_component="vision", next_skill="perceive_scene")

    assert loop._check_repeat(decision, repeat_state) is None
    assert loop._check_repeat(decision, repeat_state) is None
    error = loop._check_repeat(decision, repeat_state)

    assert error == "repeated_decision_limit_exceeded:vision.perceive_scene:3>2"


def test_agent_loop_returns_skill_failure_to_scheduler_without_exiting() -> None:
    class ComponentsStub:
        def names(self):
            return ["scheduler", "vision"]

    class RuntimeStub:
        def __init__(self) -> None:
            self.blackboard = Blackboard(task_instruction="place the container on the plate")
            self.components = ComponentsStub()
            self.scheduler_calls = 0

        def run_skill(self, component: str, skill: str, payload: dict | None = None, **kwargs):
            if component == "scheduler" and skill == "choose_next_skill":
                self.scheduler_calls += 1
                if self.scheduler_calls == 1:
                    return SkillResult(
                        success=True,
                        status="next_skill_chosen_by_model",
                        output={
                            "loop_decision": {
                                "control": "run_skill",
                                "stage": "observe",
                                "next_component": "vision",
                                "next_skill": "capture_views",
                            }
                        },
                    )
                return SkillResult(
                    success=True,
                    status="next_skill_chosen_by_model",
                    output={"loop_decision": {"control": "finish_run", "reason": "saw previous skill failure"}},
                )
            if component == "vision" and skill == "capture_views":
                return SkillResult(success=False, status="observation_unavailable", errors=["rendering_device_missing"])
            raise AssertionError(f"unexpected skill {component}.{skill}")

    runtime = RuntimeStub()
    loop = AgentLoop(runtime, config=AgentLoopConfig(max_steps=3))

    result = loop.run()

    assert result.status == "finished"
    assert runtime.scheduler_calls == 2
    assert result.steps[0].status == "observation_unavailable"
    assert result.steps[0].result["errors"] == ["rendering_device_missing"]


def test_execute_action_blocks_stale_action_chunk(tmp_path) -> None:
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("observation", ObservationBundle(observation_id="obs_test"))
    blackboard.write("current_subgoal", Subgoal("S1", "approach", source_candidate_id="C1"))
    blackboard.write(
        "action_chunk",
        ActionChunk(
            action_type="qpos",
            commands=[[0.0] * 14],
            metadata={"subgoal_id": "S1", "observation_id": "obs_test", "stale": True},
        ),
    )
    blackboard.write("env_adapter", SimpleNamespace(execute_action=lambda action_chunk: {"status": "should_not_execute"}))

    result = execute_action(SkillRequest(component="motion", skill="execute_action"), SkillContext("motion", blackboard))

    assert result.success is False
    assert result.status == "action_chunk_validation_failed"
    assert result.errors == ["stale_action_chunk"]


def _preflight_blackboard(tmp_path, *, subgoal_type: str, include_target: bool) -> Blackboard:
    from PIL import Image

    artifact_dir = tmp_path / "artifact"
    image_dir = artifact_dir / "images"
    image_dir.mkdir(parents=True)
    camera_views = {}
    for camera_name in ("head_camera", "front_camera", "left_camera", "right_camera"):
        image_path = image_dir / f"{camera_name}_rgb.png"
        Image.new("RGB", (960, 540), color=(20, 30, 40)).save(image_path)
        camera_views[camera_name] = CameraView(name=camera_name, rgb_path=str(image_path))

    summary_path = artifact_dir / "raw_observation_summary.json"
    summary_path.write_text(json.dumps({"joint_action_vector": [0.0] * 14}), encoding="utf-8")
    observation = ObservationBundle(
        observation_id="obs_test",
        camera_views=camera_views,
        robot_arms={
            "left": RobotArmState("left", joint_positions=[0.0] * 6, gripper_value=0.0),
            "right": RobotArmState("right", joint_positions=[0.0] * 6, gripper_value=0.0),
        },
        raw={"summary_ref": str(summary_path)},
    )
    candidates = [
        SceneCandidate(
            candidate_id="C1",
            label="container",
            visibility="yes",
        )
    ]
    if include_target:
        candidates.append(
            SceneCandidate(
                candidate_id="C2",
                label="plate",
                visibility="yes",
            )
        )
    perception = PerceptionResult(
        observation_id="obs_test",
        candidates=candidates,
        source_candidate_id="C1",
        target_candidate_id="C2" if include_target else None,
    )
    world_state = WorldState(
        task_instruction="place the container on the plate",
        candidates=candidates,
        robot_arms=observation.robot_arms,
        source_candidate_id="C1",
        target_candidate_id="C2" if include_target else None,
        metadata={"observation_id": "obs_test"},
    )
    subgoal = Subgoal(
        "S1",
        subgoal_type,
        source_candidate_id="C1",
        target_candidate_id="C2" if include_target else None,
    )
    blackboard = Blackboard(task_instruction="place the container on the plate")
    blackboard.write("observation", observation)
    blackboard.write("perception", perception)
    blackboard.write("world_state", world_state)
    blackboard.write("task_plan", TaskPlan(subgoals=[subgoal], current_subgoal_id="S1"))
    blackboard.write("current_subgoal", subgoal)
    blackboard.write("env_adapter", SimpleNamespace(session=SimpleNamespace(task_env=object()), last_observation=observation))
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    blackboard.write(
        "action_backend",
        SimpleNamespace(
            config={
                "enabled": True,
                "pretrained_path": str(policy_dir),
                "openpi_runtime": {"mode": "worker", "host": "127.0.0.1", "port": 8765},
            }
        ),
    )
    return blackboard


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
