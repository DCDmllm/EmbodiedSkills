from __future__ import annotations

import json

import numpy as np

from clawvla.config import load_config
from clawvla.envs import RoboTwinAdapter
from clawvla.schema import ActionChunk


class FakeRobotWinTask:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def take_action(self, action, action_type="qpos"):
        self.calls.append({"action": list(action), "action_type": action_type})

    def get_obs(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        return {
            "observation": {"front_camera": {"rgb": image}},
            "endpose": {},
            "joint_action": {},
        }

    def check_success(self):
        return False


def main() -> None:
    adapter = RoboTwinAdapter(load_config("configs/robotwin_default.json").robotwin)
    fake_env = FakeRobotWinTask()
    adapter.bind_task_env(fake_env)
    report = adapter.execute_action(
        ActionChunk(
            action_type="qpos",
            commands=[[0.0, 0.1, 0.2]],
            control_horizon=1,
            metadata={"artifact_prefix": "execute_smoke"},
        )
    )
    print(json.dumps({"report": report, "calls": fake_env.calls}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
