from __future__ import annotations

import pytest

from clawvla.rl.openrlhf_runtime_patches import (
    _align_batched_kwargs_by_modality,
    _neutral_training_padding,
)


def test_alignment_padding_survives_openrlhf_experience_split() -> None:
    torch = pytest.importorskip("torch")
    experience_module = pytest.importorskip("openrlhf.trainer.ppo_utils.experience")
    Experience = experience_module.Experience
    split_experience_batch = experience_module.split_experience_batch

    experience = Experience(
        sequences=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        action_mask=torch.ones(1, 2, dtype=torch.bool),
        action_log_probs=torch.zeros(1, 2),
        base_action_log_probs=torch.zeros(1, 2),
        rollout_log_probs=torch.zeros(1, 2),
        values=torch.zeros(1, 2),
        returns=torch.ones(1, 2),
        advantages=torch.ones(1, 2),
        kl=torch.zeros(1, 2),
        rewards=torch.ones(1),
        scores=torch.ones(1),
        response_length=torch.tensor([2]),
        truncated=torch.tensor([False]),
        total_length=torch.tensor([3]),
        images=[None],
        mm_train_inputs=[None],
        info={"reward": torch.ones(1)},
    )

    padded = _neutral_training_padding(experience)
    split = split_experience_batch(padded)

    assert len(split) == 1
    assert padded.action_mask.sum().item() == 0
    assert isinstance(padded.info["clawvla_alignment_padding"], torch.Tensor)
    assert tuple(padded.info["clawvla_alignment_padding"].shape) == (1,)
    assert split[0].info["clawvla_alignment_padding"].item() == 1


def test_exact_failed_batch_four_text_537_multimodal_survives_split() -> None:
    torch = pytest.importorskip("torch")
    experience_module = pytest.importorskip("openrlhf.trainer.ppo_utils.experience")
    Experience = experience_module.Experience
    split_experience_batch = experience_module.split_experience_batch

    def make_experience(*, image: bool, index: int):
        return Experience(
            sequences=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            action_mask=torch.ones(1, 2, dtype=torch.bool),
            action_log_probs=torch.zeros(1, 2),
            base_action_log_probs=torch.zeros(1, 2),
            rollout_log_probs=torch.zeros(1, 2),
            values=torch.zeros(1, 2),
            returns=torch.ones(1, 2),
            advantages=torch.ones(1, 2),
            kl=torch.zeros(1, 2),
            rewards=torch.ones(1),
            scores=torch.ones(1),
            response_length=torch.tensor([2]),
            truncated=torch.tensor([False]),
            total_length=torch.tensor([3]),
            images=[[f"frame_{index}.png"]] if image else [None],
            mm_train_inputs=[{"pixel_values": [[float(index)]]}] if image else [None],
            info={"reward": torch.ones(1)},
        )

    original = [
        *[make_experience(image=False, index=index) for index in range(4)],
        *[make_experience(image=True, index=index) for index in range(537)],
    ]
    aligned, stats = _align_batched_kwargs_by_modality(
        {"experience": original}, effective_actors=4
    )
    experiences = aligned["experience"]

    assert stats == {
        "dp": 4,
        "text": 4,
        "multimodal": 537,
        "padding": 3,
        "local_steps": 136,
    }
    assert len(experiences) == 544
    assert all("clawvla_alignment_padding" in experience.info for experience in experiences)
    assert sum(
        experience.info["clawvla_alignment_padding"].item() == 0
        for experience in experiences
    ) == 541
    aligned_padding = [
        experience
        for experience in experiences
        if experience.info.get("clawvla_alignment_padding", torch.zeros(1)).item() == 1
    ]
    assert len(aligned_padding) == 3
    assert all(
        tuple(item.info["clawvla_alignment_padding"].shape) == (1,)
        for item in aligned_padding
    )
    split = [item for experience in experiences for item in split_experience_batch(experience)]
    padding = [
        item
        for item in split
        if item.info.get("clawvla_alignment_padding", torch.zeros(1)).item() == 1
    ]
    assert len(split) == 544
    assert len(padding) == 3
    assert all(item.action_mask.sum().item() == 0 for item in padding)
    assert all(item.info["clawvla_alignment_padding"].ndim == 0 for item in padding)


def test_exact_nccl_timeout_batch_has_identical_info_schema_per_dp_column() -> None:
    torch = pytest.importorskip("torch")
    experience_module = pytest.importorskip("openrlhf.trainer.ppo_utils.experience")
    Experience = experience_module.Experience

    def make_experience(*, image: bool, index: int):
        return Experience(
            sequences=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            action_mask=torch.ones(1, 2, dtype=torch.bool),
            action_log_probs=torch.zeros(1, 2),
            base_action_log_probs=torch.zeros(1, 2),
            rollout_log_probs=torch.zeros(1, 2),
            values=torch.zeros(1, 2),
            returns=torch.ones(1, 2),
            advantages=torch.ones(1, 2),
            kl=torch.zeros(1, 2),
            rewards=torch.ones(1),
            scores=torch.ones(1),
            response_length=torch.tensor([2]),
            truncated=torch.tensor([False]),
            total_length=torch.tensor([3]),
            images=[[f"frame_{index}.png"]] if image else [None],
            mm_train_inputs=[{"pixel_values": [[float(index)]]}] if image else [None],
            info={"reward": torch.ones(1)},
        )

    original = [
        *[make_experience(image=False, index=index) for index in range(4)],
        *[make_experience(image=True, index=index) for index in range(306)],
    ]
    aligned, stats = _align_batched_kwargs_by_modality(
        {"experience": original}, effective_actors=4
    )
    experiences = aligned["experience"]
    rank_chunks = [experiences[rank * 78 : (rank + 1) * 78] for rank in range(4)]

    assert stats == {
        "dp": 4,
        "text": 4,
        "multimodal": 306,
        "padding": 2,
        "local_steps": 78,
    }
    assert [len(chunk) for chunk in rank_chunks] == [78, 78, 78, 78]
    for local_step in range(78):
        schemas = [tuple(sorted(rank_chunks[rank][local_step].info)) for rank in range(4)]
        assert len(set(schemas)) == 1
    assert [
        rank_chunks[rank][-1].info["clawvla_alignment_padding"].item()
        for rank in range(4)
    ] == [0, 0, 1, 1]
