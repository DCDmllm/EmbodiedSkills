# RoboCerebra

Utilities and notes for using RoboCerebra as subgoal-level VLA data in EmbodiedSkills.

The intended path is the LeRobot dataset, not the raw RoboCerebra HDF5 dump:

```text
lerobot/robocerebra_unified
```

Each LeRobot episode is treated as one short-horizon subgoal episode:

- prompt: LeRobot task text
- state: raw 8D RoboCerebra state
- action: raw 7D RoboCerebra action
- images: front and wrist videos
- boundary: episode frame range from LeRobot metadata

See [handoff.md](handoff.md) for the current dataset, pi0.5/OpenPI training, rollout, and next-step status.

Small sample records are in [samples/robocerebra_subgoals_sample.jsonl](samples/robocerebra_subgoals_sample.jsonl). Large datasets, decoded videos, rollout logs, checkpoints, and local `outputs/` artifacts are intentionally not tracked.
