# Training contracts

The release contains two training paths and no RL path.

## pi0.5 subtask SFT

Each episode manifest points to one successful expert HDF5 and lists ordered
segments with `frame_start`, `frame_end_exclusive`, and one accepted natural
language subtask. Merged subtasks retain the complete ordered list of source
segments so their frame ranges remain auditable.

`RoboTwinSubtaskDataset` samples every eligible frame. The state and three RGB
views come from that frame. The action label contains at most 32 expert actions
from the same segment and repeats its last action when fewer than 32 remain.
The prompt is only the current subtask. It never includes the full episode plan
and never crosses into the next segment.

Train/validation membership is episode-level and read from a manifest. Optional
planner prompt variants can add language noise, but a SHA-256 link to the
canonical prompt prevents stale variants from being used.

## Qwen AgentLoop SFT

The trajectory collector replays successful expert subtasks through the real
AgentLoop with deterministic teacher decisions. It records model input at each
scheduler, vision, verifier, and recovery call after production history
compaction. A row is admitted only when its image placeholders, image files,
stage legality, prompt rendering, and context length pass validation.

Normal plan generation and execution examples come from successful expert
episodes. Engineering sets may inject explicit controller failures, stale
observations, or invalid scheduling choices to exercise runtime rejection and
recovery. They do not fabricate physical object disturbances.

The subset builder retains every plan-generation row. Remaining skill rows are
sampled deterministically by task, decision family, and history depth, then
split into exact-row train and validation JSONL files compatible with
LLaMA-Factory's ShareGPT multimodal format.

The supplied Qwen recipe uses FlashAttention 2, DeepSpeed ZeRO-3, LoRA rank 64,
65,536-token context, evaluation every 50 optimizer steps, and W&B reporting.
Hardware paths, API keys, and generated datasets are not committed.
