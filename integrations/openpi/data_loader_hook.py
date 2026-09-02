from clawvla.training import RoboTwinSubtaskDataset


def maybe_create_embodiedskills_dataset(repo_id, data_config, action_horizon):
    """Return an EmbodiedSkills dataset or None for OpenPI's normal loader."""
    import pathlib

    local_root = pathlib.Path(repo_id)
    if not ((local_root / "segments").is_dir() and (local_root / "raw").is_dir()):
        return None
    return RoboTwinSubtaskDataset(
        local_root,
        action_horizon,
        repair_ledger=data_config.robotwin_subtask_repair_ledger,
        strict_prompts=data_config.robotwin_subtask_strict_prompts,
        episode_split_manifest=data_config.episode_split_manifest,
        episode_split_name=data_config.episode_split_name,
        prompt_variant_manifest=data_config.robotwin_subtask_prompt_variants,
        prompt_variant_probability=data_config.robotwin_subtask_prompt_variant_probability,
    )
