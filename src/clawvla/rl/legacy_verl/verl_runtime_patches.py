from __future__ import annotations

import functools
import sys
from typing import Any


_FORCE_NESTED_TENSOR_KEYS = {
    "prompts",
    "responses",
    "response_mask",
    "rollout_log_probs",
    "rm_scores",
    "loss_mask",
    "input_ids",
    "position_ids",
}


def apply_verl_runtime_patches() -> None:
    """Patch local veRL 0.8 gaps needed by ClawVLA agent-loop training."""
    _patch_list_of_dict_to_tensordict()
    _patch_maybe_fix_3d_position_ids()
    _patch_fsdp_qwen3vl_position_ids()
    _patch_qwen3vl_text_model_forward()
    _patch_training_worker_position_ids_from_module()
    if _cuda_devices_visible():
        _patch_logprob_position_id_fix()


def _cuda_devices_visible() -> bool:
    import os

    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    return bool(value and value.strip())


def _patch_list_of_dict_to_tensordict() -> None:
    tu = sys.modules.get("verl.utils.tensordict_utils")
    if tu is None:
        return

    import torch
    from tensordict import TensorDict
    from tensordict.tensorclass import NonTensorStack

    if getattr(tu.list_of_dict_to_tensordict, "_clawvla_patched", False):
        return

    def patched_list_of_dict_to_tensordict(list_of_dicts: list[dict[str, Any]]) -> TensorDict:
        if not list_of_dicts:
            raise AssertionError("Must provide at least one dictionary.")

        keys = list_of_dicts[0].keys()
        dict_of_lists = {key: [item[key] for item in list_of_dicts] for key in keys}
        batch_size = len(list_of_dicts)
        final_data = {}

        for key, values in dict_of_lists.items():
            if values and all(isinstance(item, torch.Tensor) for item in values):
                if key in _FORCE_NESTED_TENSOR_KEYS:
                    nested = torch.nested.as_nested_tensor(values, layout=torch.jagged)
                    final_data[key] = nested
                elif all(item.shape == values[0].shape for item in values):
                    final_data[key] = torch.stack(values)
                else:
                    final_data[key] = torch.nested.as_nested_tensor(values, layout=torch.jagged)
            else:
                final_data[key] = NonTensorStack(*values)

        return TensorDict(final_data, batch_size=[batch_size])

    patched_list_of_dict_to_tensordict._clawvla_patched = True
    tu.list_of_dict_to_tensordict = patched_list_of_dict_to_tensordict

    main_ppo_sync = sys.modules.get("verl.trainer.main_ppo_sync")
    if main_ppo_sync is not None:
        main_ppo_sync.list_of_dict_to_tensordict = patched_list_of_dict_to_tensordict


def _patch_maybe_fix_3d_position_ids() -> None:
    tu = sys.modules.get("verl.utils.tensordict_utils")
    if tu is None:
        return

    original_fix = tu.maybe_fix_3d_position_ids
    if not getattr(original_fix, "_clawvla_patched", False):

        @functools.wraps(original_fix)
        def patched_maybe_fix_3d_position_ids(data):
            _normalize_qwen3vl_position_ids(data)
            return original_fix(data)

        patched_maybe_fix_3d_position_ids._clawvla_patched = True
        patched_maybe_fix_3d_position_ids._clawvla_original = original_fix
        tu.maybe_fix_3d_position_ids = patched_maybe_fix_3d_position_ids

    engine_base = sys.modules.get("verl.workers.engine.base")
    if engine_base is not None:
        engine_base.maybe_fix_3d_position_ids = tu.maybe_fix_3d_position_ids


def _patch_logprob_position_id_fix() -> None:
    tu = sys.modules.get("verl.utils.tensordict_utils")
    engine_workers = sys.modules.get("verl.workers.engine_workers")
    if tu is None or engine_workers is None:
        return

    cls = getattr(engine_workers, "ActorRolloutRefWorker", None)
    training_cls = getattr(engine_workers, "TrainingWorker", None)
    if cls is None or training_cls is None:
        return
    if getattr(cls, "_clawvla_logprob_position_patch", False):
        return

    maybe_fix_3d_position_ids = tu.maybe_fix_3d_position_ids
    original_compute_log_prob = cls.compute_log_prob
    original_compute_ref_log_prob = cls.compute_ref_log_prob
    original_update_actor = cls.update_actor

    @functools.wraps(original_compute_log_prob)
    def compute_log_prob(self, data, *args, **kwargs):
        _ensure_qwen3vl_position_runtime_patches()
        _maybe_fix_3d_position_ids_for_logprob(data, maybe_fix_3d_position_ids)
        return original_compute_log_prob(self, data, *args, **kwargs)

    @functools.wraps(original_compute_ref_log_prob)
    def compute_ref_log_prob(self, data, *args, **kwargs):
        _ensure_qwen3vl_position_runtime_patches()
        _maybe_fix_3d_position_ids_for_logprob(data, maybe_fix_3d_position_ids)
        return original_compute_ref_log_prob(self, data, *args, **kwargs)

    @functools.wraps(original_update_actor)
    def update_actor(self, data, *args, **kwargs):
        _ensure_qwen3vl_position_runtime_patches()
        _maybe_fix_3d_position_ids_for_logprob(data, maybe_fix_3d_position_ids)
        return original_update_actor(self, data, *args, **kwargs)

    cls.compute_log_prob = compute_log_prob
    cls.compute_ref_log_prob = compute_ref_log_prob
    cls.update_actor = update_actor
    cls._clawvla_logprob_position_patch = True

    _patch_training_worker_position_ids(training_cls)


def _patch_training_worker_position_ids_from_module() -> None:
    engine_workers = sys.modules.get("verl.workers.engine_workers")
    if engine_workers is None:
        return
    training_cls = getattr(engine_workers, "TrainingWorker", None)
    if training_cls is not None:
        _patch_training_worker_position_ids(training_cls)


def _ensure_qwen3vl_position_runtime_patches() -> None:
    _patch_fsdp_qwen3vl_position_ids(force_import=True)
    _patch_qwen3vl_text_model_forward(force_import=True)


def _patch_training_worker_position_ids(training_cls: Any) -> None:
    if getattr(training_cls, "_clawvla_position_patch", False):
        return

    original_infer_batch = training_cls.infer_batch
    original_train_batch = training_cls.train_batch

    @functools.wraps(original_infer_batch)
    def infer_batch(self, data, *args, **kwargs):
        _ensure_qwen3vl_position_runtime_patches()
        _normalize_qwen3vl_position_ids(data)
        return original_infer_batch(self, data, *args, **kwargs)

    @functools.wraps(original_train_batch)
    def train_batch(self, data, *args, **kwargs):
        _ensure_qwen3vl_position_runtime_patches()
        _normalize_qwen3vl_position_ids(data)
        return original_train_batch(self, data, *args, **kwargs)

    training_cls.infer_batch = infer_batch
    training_cls.train_batch = train_batch
    training_cls._clawvla_position_patch = True


def _patch_fsdp_qwen3vl_position_ids(*, force_import: bool = False) -> None:
    transformer_impl = sys.modules.get("verl.workers.engine.fsdp.transformer_impl")
    if transformer_impl is None and force_import:
        import importlib

        transformer_impl = importlib.import_module("verl.workers.engine.fsdp.transformer_impl")
    if transformer_impl is None:
        return

    cls = getattr(transformer_impl, "FSDPEngineWithLMHead", None)
    if cls is None or getattr(cls, "_clawvla_qwen3vl_position_patch", False):
        return

    original_prepare_model_inputs = cls.prepare_model_inputs

    @functools.wraps(original_prepare_model_inputs)
    def prepare_model_inputs(self, micro_batch, *args, **kwargs):
        _normalize_qwen3vl_position_ids(micro_batch)
        model_inputs, output_args = original_prepare_model_inputs(self, micro_batch, *args, **kwargs)
        _normalize_qwen3vl_model_position_ids(model_inputs)
        return model_inputs, output_args

    cls.prepare_model_inputs = prepare_model_inputs
    cls._clawvla_qwen3vl_position_patch = True


def _patch_qwen3vl_text_model_forward(*, force_import: bool = False) -> None:
    module = sys.modules.get("transformers.models.qwen3_vl.modeling_qwen3_vl")
    if module is None and force_import:
        import importlib

        module = importlib.import_module("transformers.models.qwen3_vl.modeling_qwen3_vl")
    if module is None:
        return

    cls = getattr(module, "Qwen3VLTextModel", None)
    rotary_cls = getattr(module, "Qwen3VLTextRotaryEmbedding", None)
    if cls is not None and not getattr(cls, "_clawvla_position_patch", False):
        original_forward = cls.forward

        @functools.wraps(original_forward)
        def forward(self, *args, **kwargs):
            args = list(args)
            if len(args) >= 3 and args[2] is not None:
                args[2] = _normalize_qwen3vl_forward_position_ids(args[2])
            if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                kwargs["position_ids"] = _normalize_qwen3vl_forward_position_ids(kwargs["position_ids"])
            return original_forward(self, *args, **kwargs)

        cls.forward = forward
        cls._clawvla_position_patch = True

    if rotary_cls is not None and not getattr(rotary_cls, "_clawvla_position_patch", False):
        original_rotary_forward = rotary_cls.forward

        @functools.wraps(original_rotary_forward)
        def rotary_forward(self, x, position_ids):
            position_ids = _normalize_qwen3vl_rotary_position_ids(position_ids)
            return original_rotary_forward(self, x, position_ids)

        rotary_cls.forward = rotary_forward
        rotary_cls._clawvla_position_patch = True


def _maybe_fix_3d_position_ids_for_logprob(data: Any, fix_fn: Any) -> None:
    position_ids = data.get("position_ids") if hasattr(data, "get") else None
    if position_ids is None or not getattr(position_ids, "is_nested", False) or position_ids.dim() != 3:
        return
    _normalize_qwen3vl_position_ids(data)
    fix_fn(data)


def _normalize_qwen3vl_position_ids(data: Any) -> None:
    position_ids = data.get("position_ids") if hasattr(data, "get") else None
    if position_ids is None:
        return

    import torch

    if not getattr(position_ids, "is_nested", False):
        if not isinstance(position_ids, torch.Tensor):
            return
        pieces = _dense_qwen3vl_position_pieces(position_ids)
        if pieces is None:
            return
        data["position_ids"] = torch.nested.as_nested_tensor(pieces, layout=torch.jagged)
        data["position_ids"]._ragged_idx = 2
        return

    if position_ids.dim() != 3:
        return

    values = position_ids.values()
    if values.dim() != 2 or values.shape[-1] not in (3, 4) or values.shape[0] in (3, 4):
        return

    offsets = position_ids.offsets()
    if offsets.numel() < 2 or int(offsets[-1].item()) != int(values.shape[0]):
        return

    pieces = []
    offset_values = offsets.detach().cpu().tolist()
    for start, end in zip(offset_values, offset_values[1:]):
        pieces.append(values[int(start) : int(end), :].transpose(0, 1).contiguous())
    data["position_ids"] = torch.nested.as_nested_tensor(pieces, layout=torch.jagged)
    data["position_ids"]._ragged_idx = 2


def _dense_qwen3vl_position_pieces(position_ids: Any) -> list[Any] | None:
    if position_ids.dim() == 2 and position_ids.shape[-1] in (3, 4):
        return [position_ids.transpose(0, 1).contiguous()]
    if position_ids.dim() != 3:
        return None

    if position_ids.shape[1] in (3, 4):
        return [sample.contiguous() for sample in position_ids.unbind(0)]
    if position_ids.shape[-1] in (3, 4):
        return [sample.transpose(0, 1).contiguous() for sample in position_ids.unbind(0)]
    return None


def _normalize_qwen3vl_model_position_ids(model_inputs: dict[str, Any]) -> None:
    position_ids = model_inputs.get("position_ids")
    if position_ids is None:
        return
    model_inputs["position_ids"] = _normalize_qwen3vl_forward_position_ids(position_ids)


def _normalize_qwen3vl_forward_position_ids(position_ids: Any) -> Any:
    if position_ids.dim() == 2:
        if position_ids.shape[0] in (3, 4):
            return position_ids.unsqueeze(1).contiguous()
        if position_ids.shape[-1] in (3, 4):
            return position_ids.transpose(0, 1).unsqueeze(1).contiguous()
        return position_ids

    if position_ids.dim() != 3:
        return position_ids

    diagonal = _qwen3vl_axis_vector_diagonal(position_ids)
    if diagonal is not None:
        return diagonal

    if position_ids.shape[0] in (3, 4):
        return position_ids

    if position_ids.shape[-1] in (3, 4):
        if position_ids.shape[1] == 1:
            return position_ids.permute(2, 1, 0).contiguous()
        return position_ids.movedim(-1, 0).contiguous()

    if position_ids.shape[1] in (3, 4):
        if position_ids.shape[-1] == 1:
            return position_ids.permute(1, 2, 0).contiguous()
        return position_ids.transpose(0, 1).contiguous()
    return position_ids


def _normalize_qwen3vl_rotary_position_ids(position_ids: Any) -> Any:
    position_ids = _normalize_qwen3vl_forward_position_ids(position_ids)
    if position_ids.dim() == 3 and position_ids.shape[0] == 4:
        return position_ids[1:].contiguous()
    return position_ids


def _qwen3vl_axis_vector_diagonal(position_ids: Any) -> Any | None:
    if position_ids.dim() != 3:
        return None
    axis_count = int(position_ids.shape[0])
    vector_count = int(position_ids.shape[-1])
    if axis_count not in (3, 4) or vector_count not in (3, 4) or int(position_ids.shape[1]) <= vector_count:
        return None

    if axis_count == 4 and vector_count == 4:
        columns = (0, 1, 2, 3)
    elif axis_count == 3 and vector_count == 4:
        columns = (1, 2, 3)
    elif axis_count == 3 and vector_count == 3:
        columns = (0, 1, 2)
    else:
        return None

    import torch

    rows = [position_ids[row, :, column] for row, column in enumerate(columns)]
    return torch.stack(rows, dim=0).unsqueeze(1).contiguous()
