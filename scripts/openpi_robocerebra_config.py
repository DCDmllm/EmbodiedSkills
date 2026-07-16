"""OpenPI data config for RoboCerebra-LeRobot.

This file is meant to be imported from the external OpenPI training tree, e.g.
with PYTHONPATH containing both this repository and OpenPI's ``src`` directory.
It defines the dataset-side transforms only; it does not register itself in
OpenPI's global config list.
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np
from typing_extensions import override

from openpi import transforms
from openpi.models import model as _model
import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.weight_loaders as weight_loaders


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboCerebraInputs(transforms.DataTransformFn):
    """Map RoboCerebra observations to pi0/pi0.5's canonical image slots."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RoboCerebraOutputs(transforms.DataTransformFn):
    """Trim padded pi0.5 actions back to RoboCerebra's raw 7D action space."""

    raw_action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.raw_action_dim])}


@dataclasses.dataclass(frozen=True)
class RoboCerebraLeRobotDataConfig(_config.DataConfigFactory):
    """LeRobot config for ``lerobot/robocerebra_unified``.

    Native LeRobot fields:
    - ``observation.images.image``: front RGB video frame.
    - ``observation.images.wrist_image``: wrist RGB video frame.
    - ``observation.state``: 8D single-arm state.
    - ``action``: 7D raw action.
    - ``task_index``: maps through ``tasks.parquet`` to the subgoal prompt.
    """

    repo_id: str = "lerobot/robocerebra_unified"
    raw_action_dim: int = 7

    @override
    def create(self, assets_dirs, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        repack_transform = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        "observation/wrist_image": "observation.images.wrist_image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = transforms.Group(
            inputs=[RoboCerebraInputs(model_type=model_config.model_type)],
            outputs=[RoboCerebraOutputs(raw_action_dim=self.raw_action_dim)],
        )
        model_transforms = _config.ModelTransformFactory()(model_config)

        base_config = self.create_base_config(assets_dirs, model_config)
        return dataclasses.replace(
            base_config,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            prompt_from_task=True,
        )


def make_pi05_robocerebra_lora_config(
    *,
    repo_id: str = "lerobot/robocerebra_unified",
    assets_base_dir: str = "outputs/openpi_assets",
    checkpoint_base_dir: str = "outputs/openpi_checkpoints",
    batch_size: int = 8,
    num_train_steps: int = 100,
) -> _config.TrainConfig:
    """Construct a pi0.5 LoRA TrainConfig initialized from official pi0.5 base."""

    model = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_horizon=32,
        action_dim=32,
    )
    return _config.TrainConfig(
        name="pi05_base_robocerebra_lora",
        exp_name="robocerebra_overfit",
        model=model,
        data=RoboCerebraLeRobotDataConfig(
            repo_id=repo_id,
            assets=_config.AssetsConfig(assets_dir=assets_base_dir, asset_id="robocerebra_unified"),
        ),
        freeze_filter=model.get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        assets_base_dir=assets_base_dir,
        checkpoint_base_dir=checkpoint_base_dir,
        batch_size=batch_size,
        num_workers=0,
        num_train_steps=num_train_steps,
        log_interval=1,
        save_interval=max(num_train_steps, 1),
        overwrite=True,
        wandb_enabled=False,
        fsdp_devices=1,
    )
