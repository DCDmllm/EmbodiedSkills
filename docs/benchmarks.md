# Benchmark adapters

## RoboTwin 2.0

`RoboTwinAdapter` owns simulator setup, four-view observation normalization,
14D joint state extraction, qpos/EE action execution, and `check_success()`.
Clean and randomized expert trajectories can be collected with the same
script; the task config and seed are stored in every manifest.

## RMBench

`RmbenchAdapter` has a separate runtime identity while reusing the compatible
RoboTwin session and observation/action contract. Its official
`language_annotation.json` supplies executor segment text and durations. The
RMBench builder validates that segment durations exactly cover every HDF5 and
creates an episode-level train/validation split.

## LIBERO

`LiberoAdapter` normalizes agent and wrist images, the 8D policy state, and 7D
relative end-effector actions. It queries LIBERO's own `check_success()` and
does not reuse RoboTwin completion logic. Standard LIBERO/OpenPI datasets can be
used directly; no RoboTwin HDF5 conversion is applied.
