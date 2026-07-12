from __future__ import annotations

import argparse
from pathlib import Path
import threading
import time

import pytest

from clawvla.scripts import repair_robotwin_subtask_annotations as repair


INVALID = repair.DEFAULT_INVALID_LABELS


def _segment(
    *,
    index: int = 0,
    arm: str | None = "right",
    raw: str | None = "arm_tag",
    polished: str | None = None,
) -> dict:
    primitive = {"left": [], "right": [], "raw_arms": []}
    if arm is not None:
        primitive[arm] = [{"action": {"arm_tag": arm}}]
        primitive["raw_arms"] = [arm]
    return {
        "segment_id": f"demo_ep0000_seg{index:03d}",
        "task_name": "demo",
        "episode_index": 0,
        "segment_index": index,
        "frame_start": index * 10,
        "frame_end_exclusive": index * 10 + 10,
        "num_saved_frames": 10,
        "raw_canonical_instruction": raw,
        "canonical_instruction": raw,
        "polished_instruction": polished,
        "source_code": "self.move(...) prescribed arm",
        "source_context": "demo context",
        "primitive_summary": primitive,
    }


def _good_item(index: int = 0, arm: str = "right") -> dict:
    return {
        "segment_index": index,
        "subgoal_type": "lift",
        "instruction": f"Lift the medium block with the {arm} arm.",
        "completion_criteria": "The medium block is clear of the table.",
        "paraphrases": [
            f"Raise the medium block using the {arm} gripper.",
            f"Use the {arm} arm to lift the medium block.",
            f"Pick up the medium block with the {arm} hand.",
        ],
    }


def test_scope_classification_is_explicit() -> None:
    placeholder = _segment(raw="arm_tag")
    coarse = _segment(raw="Move above the block")
    missing = _segment(raw=None)
    polished = _segment(raw="Move above the block", polished="Approach the block with the right arm.")

    assert repair.classify_segment(placeholder, invalid_labels=INVALID) == "raw_only"
    assert repair._selected_by_scope(placeholder, "broken", INVALID)
    assert not repair._selected_by_scope(coarse, "broken", INVALID)
    assert repair._selected_by_scope(coarse, "all-unpolished", INVALID)
    assert repair._selected_by_scope(missing, "broken", INVALID)
    assert not repair._selected_by_scope(polished, "all-unpolished", INVALID)


@pytest.mark.parametrize(
    "items,error_fragment",
    [
        ([_good_item(), _good_item()], "duplicate"),
        ([{**_good_item(), "segment_index": "0"}], "must be an integer"),
        ([{**_good_item(), "segment_index": True}], "must be an integer"),
        ([{**_good_item(), "segment_index": 1}], "missing=[0], extra=[1]"),
    ],
)
def test_model_output_indices_are_exact(items: list[dict], error_fragment: str) -> None:
    with pytest.raises(ValueError, match=error_fragment.replace("[", r"\[").replace("]", r"\]")):
        repair._validate_model_outputs(items, [0])


def test_arm_validation_distinguishes_arm_from_target_side() -> None:
    right_segment = _segment(arm="right")
    good = _good_item(arm="right")
    good["instruction"] = "Place the block at the left target with the right arm."
    assert repair.validate_repair(
        right_segment, good, invalid_labels=INVALID, expected_paraphrases=3
    )["valid"]

    wrong = _good_item(arm="left")
    validation = repair.validate_repair(
        right_segment, wrong, invalid_labels=INVALID, expected_paraphrases=3
    )
    assert not validation["valid"]
    assert "instruction_mentions_wrong_left_arm" in validation["errors"]


def test_single_arm_held_object_continuation_may_omit_arm() -> None:
    item = _good_item(arm="right")
    item["instruction"] = "Move the held brown bottle upward."
    item["paraphrases"] = [
        "Raise the held brown bottle.",
        "Move the grasped bottle upward.",
        "Lift the brown bottle from the table.",
    ]
    validation = repair.validate_repair(
        _segment(arm="right"), item, invalid_labels=INVALID, expected_paraphrases=3
    )
    assert validation["valid"]


def test_handover_may_mention_active_and_context_arm() -> None:
    item = _good_item(arm="right")
    item["instruction"] = "Use the right arm to grasp the microphone from the left arm."
    item["paraphrases"] = [
        "Take the microphone with the right gripper from the left hand.",
        "Use the right hand to receive the microphone from the left arm.",
        "Grasp the microphone from the left gripper using the right arm.",
    ]
    validation = repair.validate_repair(
        _segment(arm="right"), item, invalid_labels=INVALID, expected_paraphrases=3
    )
    assert validation["valid"]


def test_paraphrase_context_arm_does_not_reject_valid_canonical_instruction() -> None:
    item = _good_item(arm="right")
    item["instruction"] = "Open the right gripper to release the bottle."
    item["paraphrases"] = [
        "Release the bottle from the right gripper.",
        "Open the right gripper and transfer the bottle left.",
        "Let the left gripper take the bottle from the right.",
    ]
    validation = repair.validate_repair(
        _segment(arm="right"), item, invalid_labels=INVALID, expected_paraphrases=3
    )
    assert validation["valid"]


def test_ambiguous_source_arm_is_rejected() -> None:
    validation = repair.validate_repair(
        _segment(arm=None), _good_item(), invalid_labels=INVALID, expected_paraphrases=3
    )
    assert not validation["valid"]
    assert "ambiguous_source_arm" in validation["errors"]


def test_resume_only_skips_same_model_and_source(tmp_path: Path) -> None:
    segment = _segment()
    payload = {"task_name": "demo", "episode_index": 0, "segments": [segment]}
    record_id = repair._record_id("demo", 0, 0)
    base_record = {
        "record_id": record_id,
        "status": "accepted",
        "model": "gpt-5.6-sol",
        "source_fingerprint": repair._fingerprint(segment),
    }
    common = {
        "scope": "broken",
        "invalid_labels": INVALID,
        "task_names": set(),
        "episode_indices": set(),
        "retry_rejected": False,
        "max_episodes": None,
        "max_targets": None,
        "batch_size": 8,
    }
    jobs, plan = repair.plan_jobs(
        [(tmp_path / "segments/demo/episode0.json", payload)],
        existing_records={record_id: base_record},
        expected_model="gpt-5.6-sol",
        **common,
    )
    assert jobs == []
    assert plan["skipped_existing_records"] == 1

    jobs, plan = repair.plan_jobs(
        [(tmp_path / "segments/demo/episode0.json", payload)],
        existing_records={record_id: {**base_record, "model": "gpt-5.5"}},
        expected_model="gpt-5.6-sol",
        **common,
    )
    assert len(jobs) == 1
    assert plan["stale_existing_records"] == 1


def test_revalidate_existing_rejected_without_api() -> None:
    segment = _segment(arm="right")
    item = _good_item(arm="right")
    item["instruction"] = "Move the held bottle upward."
    item["paraphrases"] = [
        "Raise the held bottle.",
        "Move the grasped bottle upward.",
        "Lift the bottle from the table.",
    ]
    record_id = repair._record_id("demo", 0, 0)
    record = {
        "record_id": record_id,
        "status": "rejected",
        "model": "gpt-5.6-sol",
        "source_fingerprint": repair._fingerprint(segment),
        "repair": item,
        "validation": {"valid": False, "errors": ["old_strict_arm_rule"], "warnings": []},
    }
    promotions = repair._revalidate_existing_rejected(
        {record_id: record},
        [(Path("segments/demo/episode0.json"), {"task_name": "demo", "episode_index": 0, "segments": [segment]})],
        expected_model="gpt-5.6-sol",
        invalid_labels=INVALID,
        expected_paraphrases=3,
    )
    assert len(promotions) == 1
    assert promotions[0]["status"] == "accepted"
    assert promotions[0]["local_revalidation"]["api_request_sent"] is False


def test_repair_chunk_passes_episode_context_without_network(monkeypatch, tmp_path: Path) -> None:
    segment = _segment()
    payload = {
        "task_name": "demo",
        "episode_index": 0,
        "instruction": "Rank the blocks by size.",
        "episode_info": {"info": {"block1": "large", "block2": "medium", "block3": "small"}},
        "segments": [segment],
    }
    job = repair.EpisodeJob(
        path=tmp_path / "segments/demo/episode0.json",
        payload=payload,
        targets=(segment,),
    )
    captured: dict = {}

    def fake_request(**kwargs):
        captured.update(kwargs["payload"])
        return {"segments": [_good_item()], "_raw_text_preview": "mocked"}

    monkeypatch.setattr(repair, "_request_segment_polish", fake_request)
    monkeypatch.setattr(repair, "_build_segment_visual_inputs", lambda *args: ([], {"enabled": False}))
    args = argparse.Namespace(
        images=False,
        allow_missing_images=False,
        image_camera="head_camera",
        image_samples="mid",
        max_images=24,
        image_size=384,
        image_quality=70,
        paraphrases=3,
        temperature=0.35,
        max_tokens=5000,
        api_retries=1,
        config=str(tmp_path / "config.json"),
    )
    records = repair.repair_chunk(
        args=args,
        config={"model": "gpt-5.6-sol"},
        model="gpt-5.6-sol",
        dataset_root=tmp_path,
        job=job,
        chunk=(segment,),
        chunk_index=1,
        invalid_labels=INVALID,
    )
    assert records[0]["status"] == "accepted"
    assert captured["episode_info"] == payload["episode_info"]
    assert captured["repair_target_indices"] == [0]
    assert captured["episode_sequence"][0]["active_arms"] == ["right"]
    assert captured["segments"][0]["previous_segment"] is None


def test_episode_workers_run_concurrently_but_keep_each_episode_sequential(monkeypatch, tmp_path: Path) -> None:
    jobs = []
    for episode_index in range(4):
        segments = []
        for segment_index in range(2):
            segment = _segment(index=segment_index)
            segment["episode_index"] = episode_index
            segments.append(segment)
        jobs.append(
            repair.EpisodeJob(
                path=tmp_path / f"segments/demo/episode{episode_index}.json",
                payload={"task_name": "demo", "episode_index": episode_index, "segments": segments},
                targets=tuple(segments),
            )
        )

    state_lock = threading.Lock()
    active = 0
    max_active = 0
    active_episodes: set[int] = set()
    chunk_order: dict[int, list[int]] = {index: [] for index in range(4)}
    written = []

    def fake_repair_chunk(**kwargs):
        nonlocal active, max_active
        episode_index = kwargs["job"].episode_index
        with state_lock:
            assert episode_index not in active_episodes
            active_episodes.add(episode_index)
            active += 1
            max_active = max(max_active, active)
            chunk_order[episode_index].append(kwargs["chunk_index"])
        time.sleep(0.03)
        with state_lock:
            active -= 1
            active_episodes.remove(episode_index)
        return [{"status": "accepted", "episode_index": episode_index}]

    monkeypatch.setattr(repair, "repair_chunk", fake_repair_chunk)
    monkeypatch.setattr(repair, "_append_jsonl", lambda path, records: written.extend(records))
    args = argparse.Namespace(batch_size=1, workers=3, fail_fast=False, sleep_seconds=0.0)
    counts = repair._run_episode_jobs(
        args=args,
        config={},
        model="gpt-5.6-sol",
        config_path=tmp_path / "config.json",
        dataset_root=tmp_path,
        jobs=jobs,
        invalid_labels=INVALID,
        output_path=tmp_path / "repairs.jsonl",
    )

    assert counts == {"accepted": 8}
    assert len(written) == 8
    assert max_active >= 2
    assert all(order == [1, 2] for order in chunk_order.values())
