"""Agent RL scaffolding for ClawVLA.

The active trainer backend is OpenRLHF. The old VERL implementation remains
available under :mod:`clawvla.rl.legacy_verl` for reproduction/debugging while
rollout workers run the existing ClawVLA/RoboTwin stack in its own environment.
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
