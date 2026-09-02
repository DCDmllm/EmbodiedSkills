# OpenPI subtask training

EmbodiedSkills keeps the benchmark data preparation and subtask loader in this
repository while leaving OpenPI as an external dependency. Use an upstream
OpenPI checkout and make these two small integration changes:

1. Add the following fields to `openpi.training.config.DataConfig`:

```python
episode_split_manifest: str | None = None
episode_split_name: Literal["train", "val"] | None = None
robotwin_subtask_repair_ledger: str | None = None
robotwin_subtask_strict_prompts: bool = True
robotwin_subtask_prompt_variants: str | None = None
robotwin_subtask_prompt_variant_probability: float = 0.0
```

2. In OpenPI's single-repository dataset factory, detect a local root with
   `segments/` and `raw/` and instantiate
   `clawvla.training.RoboTwinSubtaskDataset`. The exact hook is shown in
   [`data_loader_hook.py`](data_loader_hook.py).

The loader emits the ordinary Aloha/OpenPI fields:

```text
observation.state
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
action
prompt
```

Use `prompt_from_task=False`, `action_sequence_keys=("action",)`, and a
32-step action horizon. The training recipe in
[`training_recipe.py`](training_recipe.py) is intentionally a function rather
than a machine-specific registry entry. Import it from the OpenPI config file,
append the returned `TrainConfig` to `_CONFIGS`, then run the standard OpenPI
normalization-statistics and training commands.

This is the only VLA training route retained here. Direct scripts that replay a
hard-coded list of subtasks into a VLA are deliberately excluded.
