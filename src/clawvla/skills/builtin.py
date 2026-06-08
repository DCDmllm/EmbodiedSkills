from __future__ import annotations

from .base import SkillRegistry
from ..components.motion import register_motion_skills
from ..components.recovery import register_recovery_skills
from ..components.safety import register_safety_skills
from ..components.scheduler import register_scheduler_skills
from ..components.state import register_state_skills
from ..components.verifier import register_verifier_skills
from ..components.vision import register_vision_skills


def register_builtin_skills(registry: SkillRegistry) -> None:
    register_vision_skills(registry)
    register_state_skills(registry)
    register_scheduler_skills(registry)
    register_safety_skills(registry)
    register_motion_skills(registry)
    register_verifier_skills(registry)
    register_recovery_skills(registry)
