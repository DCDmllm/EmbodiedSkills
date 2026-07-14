from clawvla.scripts.build_robotwin_planner_sft import _mix_rows, _to_messages_row


def test_grounding_sharegpt_rows_are_normalized_and_masked_by_caller() -> None:
    row = _to_messages_row(
        {
            "conversations": [
                {"from": "human", "value": "Locate the red block."},
                {"from": "gpt", "value": '{"bbox": [1, 2, 3, 4]}'},
            ],
            "images": ["scene.png"],
        }
    )

    assert row is not None
    assert row["messages"][0] == {"role": "user", "content": "Locate the red block."}
    assert row["messages"][1]["role"] == "assistant"
    assert row["images"] == ["scene.png"]


def test_planner_grounding_mix_keeps_requested_ratio() -> None:
    planner = [{"metadata": {"sample_type": "planner_subgoals"}} for _ in range(6)]
    grounding = [{"metadata": {"sample_type": "grounding"}} for _ in range(2)]

    mixed = _mix_rows(planner, grounding, planner_ratio=0.6, seed=42)

    assert len(mixed) == 10
    assert sum(row["metadata"]["sample_type"] == "planner_subgoals" for row in mixed) == 6
    assert sum(row["metadata"]["sample_type"] == "grounding" for row in mixed) == 4
