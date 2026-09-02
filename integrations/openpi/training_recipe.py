"""Generic OpenPI recipe for frame-aligned subtask SFT.

This file is imported from an OpenPI checkout, where the private module names
below are available. It contains no local paths and is not imported by the
EmbodiedSkills runtime.
"""


def make_train_config(
    *,
    name: str,
    dataset_root: str,
    split_manifest: str,
    base_checkpoint: str,
    assets_dir: str,
    checkpoint_dir: str,
    prompt_variants: str | None = None,
    batch_size: int = 64,
    fsdp_devices: int = 8,
    num_train_steps: int = 30_000,
):
    from openpi.models import pi0_config
    import openpi.transforms as transforms
    from openpi.training import config
    from openpi.training import optimizer
    from openpi.training import weight_loaders

    return config.TrainConfig(
        name=name,
        project_name="embodiedskills_pi05_sft",
        exp_name=name,
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32, max_token_len=256),
        data=config.LeRobotAlohaDataConfig(
            repo_id=dataset_root,
            assets=config.AssetsConfig(assets_dir=assets_dir, asset_id="embodiedskills_pi05_subtasks"),
            repack_transforms=transforms.Group(
                inputs=[
                    transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=config.DataConfig(
                prompt_from_task=False,
                robotwin_subtask_strict_prompts=True,
                robotwin_subtask_prompt_variants=prompt_variants,
                robotwin_subtask_prompt_variant_probability=0.55 if prompt_variants else 0.0,
                episode_split_manifest=split_manifest,
                episode_split_name="train",
            ),
            adapt_to_pi=True,
            use_delta_joint_actions=True,
            action_sequence_keys=("action",),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(base_checkpoint),
        assets_base_dir=assets_dir,
        checkpoint_base_dir=checkpoint_dir,
        batch_size=batch_size,
        num_workers=8,
        num_train_steps=num_train_steps,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1.5e-5,
            decay_steps=num_train_steps,
            decay_lr=1.5e-6,
        ),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        fsdp_devices=fsdp_devices,
        log_interval=100,
        save_interval=1_000,
        keep_period=5_000,
        seed=42,
        overwrite=False,
        resume=False,
        wandb_enabled=True,
    )
