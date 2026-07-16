# RoboCerebra Handoff

This is the compact handoff for RoboCerebra subgoal-level VLA data and pi0.5/OpenPI experiments in EmbodiedSkills.

## Scope

Use the LeRobot dataset for VLA training:

```text
lerobot/robocerebra_unified
```

Do not use the raw RoboCerebra 137GB HDF5 dump for the default VLA path. The raw metadata is useful for planning/subgoal language inspection, but the trainable VLA pipeline uses LeRobot parquet and mp4 shards.

## Dataset Interpretation

The LeRobot version is already segmented at short-horizon skill level. Treat each `episode_index` as one VLA subgoal episode.

Important fields:

- `episode_index`
- `task_index`
- `task_text`
- `dataset_from_index`
- `dataset_to_index`
- `num_frames`
- front video path
- wrist video path
- raw action shape `[7]`
- raw state shape `[8]`

Current full-data index path:

```text
outputs/robocerebra_lerobot_full_index.jsonl
```

Local full LeRobot dataset path used during development:

```text
/mnt/raid1/mjh/datasets/robocerebra_lerobot_unified
```

Verified full-data scale:

- episodes: `6660`
- frames: `571116`
- fps: `20`
- local parquet/mp4 footprint: about `1.5GB`
- action rows align with episode frame ranges
- front/wrist video decode probes passed with PyAV/libdav1d

## VLA Schema

The pi0.5/OpenPI batch schema is:

- prompt: LeRobot `task_text`
- `image.base_0_rgb`: front image
- `image.left_wrist_0_rgb`: wrist image
- `image.right_wrist_0_rgb`: zero image, mask false
- raw state: 8D, padded to model dim 32
- raw action: 7D, padded to model action dim 32
- action horizon: 32
- `action_mask`: valid action chunk steps

Use RoboCerebra-specific norm stats:

```text
outputs/openpi_assets/robocerebra_unified_full/norm_stats.json
```

Do not reuse RoboTwin norm stats.

Full norm stats integrity:

- valid frames: `571116`
- state NaN/Inf: `0 / 0`
- action NaN/Inf: `0 / 0`
- no near-zero std action dimensions
- raw gripper action is dim 6 and is binary-like `0/1`

## Key Scripts

Dataset and metadata:

```text
scripts/inspect_robocerebra_metadata.py
scripts/check_robocerebra_alignment.py
scripts/download_robocerebra_lerobot_full.py
scripts/build_robocerebra_lerobot_full_index.py
scripts/compute_robocerebra_norm_stats.py
scripts/export_robocerebra_lerobot_vla.py
scripts/export_robocerebra_subgoals.py
scripts/export_robocerebra_visual_sample.py
```

pi0.5/OpenPI:

```text
scripts/openpi_robocerebra_config.py
scripts/pi05_robocerebra_inference_server.py
scripts/robocerebra_full_task_comparison.py
scripts/train_pi05_robocerebra_lora_minimal.py
scripts/train_pi05_robocerebra_lora_random.py
scripts/train_pi05_robocerebra_lora_full_multigpu.py
```

## Important Training Fix

The first full-data pi0.5 LoRA run exposed a schema bug:

- RoboCerebra actions are raw 7D.
- pi0.5 action dim is 32.
- The old trainer padded action dims `7:32` with zeros and then averaged loss across all 32 dimensions.
- That gave the real 7D action only `7/32 = 21.875%` of the unweighted action-dimension loss, and the gripper only `1/32 = 3.125%`.

The full-data trainer now computes per-dimension flow loss and applies:

- horizon `action_mask`
- real 7D action weights
- padded dims weight `0`
- default raw action weights `1,1,1,1,1,1,4`

This makes the gripper dimension materially visible to LoRA training.

## Graspfix 500-Step Result

Checkpoint:

```text
outputs/pi05_robocerebra_lora_full_6gpu_graspfix_500step/lora_params.pkl
```

First GT subgoal:

```text
Pick up cream cheese from coffee table
```

Seed 7 chunk sweep, max steps 300:

| execute_chunk_len | min EEF-to-cream | gripper contact | lift | cream displacement |
|---:|---:|---|---:|---:|
| 16 | 0.0477 | true | 0.0000 | 0.0522 |
| 8 | 0.0530 | true | 0.0000 | 0.0529 |
| 4 | 0.0221 | false | 0.0000 | 0.0521 |
| 1 | 0.0129 | true | 0.0000 | 0.0552 |

Chunk length 1 is best by proximity/contact, but it is slow because it calls pi0.5 every env step. Chunk length 16 is the practical multi-seed setting and still shows contact/displacement.

Chunk 16 multi-seed result:

| seed | closest object | min EEF-to-cream | gripper contact | lift | cream displacement |
|---:|---|---:|---|---:|---:|
| 7 | cream_cheese_1 | 0.0477 | true | 0.0000 | 0.0522 |
| 8 | cream_cheese_1 | 0.0396 | true | 0.0000 | 0.0534 |
| 9 | cream_cheese_1 | 0.0416 | false | 0.0000 | 0.0521 |

Interpretation:

- The model is no longer only approaching.
- Contact appears in seeds 7/8 with chunk 16.
- The target object moves by about 5cm in all three seeds.
- The model still does not reliably grasp/lift or complete the pick subgoal.

This passes the gate for staged continuation training, but rollout should keep focusing on first-subgoal contact, grasp, and lift rather than full-task success.

## Current Long Run

Started as a night run:

```text
tmux session: robocerebra_graspfix_cont3k
output dir: outputs/pi05_robocerebra_lora_full_6gpu_graspfix_cont3k_from500
init: outputs/pi05_robocerebra_lora_full_6gpu_graspfix_500step/lora_params.pkl
```

Config:

- 6 GPUs
- global batch size 6
- 3000 continuation steps
- save/eval every 500 steps
- phase-balanced sampler
- raw action weights `1,1,1,1,1,1,4`
- padded action weight `0`

Initial live check:

- step 2 loss: `0.3798`
- step 3 loss: `0.1902`
- step 4 loss: `0.6102`
- max GPU memory: about `36.21GB`
- no NaN at startup
- steady step time: about `36.5s`

Expected runtime for 3000 continuation steps is roughly 31 hours plus eval/checkpoint overhead.

## Next Gate

After checkpoints are available, evaluate first GT subgoal before doing full-task tests:

1. `lora_params_step_000500.pkl`
2. `lora_params_step_001000.pkl`
3. final `lora_params.pkl`

Use `execute_chunk_len=16` for routine seed 7/8/9 comparison, and optionally recheck chunk 1 on seed 7 when the model looks promising.

Metrics to compare:

- success
- completed subgoals
- min EEF-to-cream distance
- closest object
- gripper contact
- first close distance
- cream displacement
- lift height
- NaN/Inf
- rollout video
