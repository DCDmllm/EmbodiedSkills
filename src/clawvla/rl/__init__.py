"""Agent RL scaffolding for ClawVLA.

The RL package is intentionally split from the runtime package dependencies:
the trainer can run in a verl Python environment while rollout workers run the
existing ClawVLA/RoboTwin stack in its own environment.
"""

from .config import RLConfig, load_rl_config
from .trajectory import EpisodeRecord, PolicyCallTrace, RewardRecord, TrajectoryWriter

__all__ = [
    "EpisodeRecord",
    "PolicyCallTrace",
    "RLConfig",
    "RewardRecord",
    "TrajectoryWriter",
    "load_rl_config",
]
