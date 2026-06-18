from __future__ import annotations

import builtins
import importlib.machinery
import os
import sys
import types


_ORIGINAL_IMPORT = builtins.__import__
_PATCHING = False


def _install_flash_attn_padding_fallback() -> None:
    if "flash_attn.bert_padding" in sys.modules:
        return
    if os.environ.get("CLAWVLA_OPENRLHF_FLASH_ATTN_FALLBACK") != "1":
        return

    import torch

    flash_attn_module = types.ModuleType("flash_attn")
    flash_attn_module.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None, is_package=True)
    flash_attn_module.__path__ = []
    bert_padding_module = types.ModuleType("flash_attn.bert_padding")
    bert_padding_module.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", loader=None)
    utils_module = types.ModuleType("flash_attn.utils")
    utils_module.__spec__ = importlib.machinery.ModuleSpec("flash_attn.utils", loader=None, is_package=True)
    distributed_module = types.ModuleType("flash_attn.utils.distributed")
    distributed_module.__spec__ = importlib.machinery.ModuleSpec("flash_attn.utils.distributed", loader=None)

    def index_first_axis(input_tensor, indices):
        return input_tensor[indices]

    def rearrange(input_tensor, pattern, **kwargs):
        if pattern == "b s ... -> (b s) ...":
            return input_tensor.reshape(input_tensor.shape[0] * input_tensor.shape[1], *input_tensor.shape[2:])
        raise NotImplementedError(f"flash_attn fallback only supports OpenRLHF padding pattern: {pattern}")

    def unpad_input(input_tensor, attention_mask):
        indices = torch.nonzero(attention_mask.reshape(-1), as_tuple=False).reshape(-1)
        unpadded = input_tensor.reshape(-1, *input_tensor.shape[2:])[indices]
        seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        max_seqlen = int(seqlens.max().item()) if seqlens.numel() else 0
        return unpadded, indices, cu_seqlens, max_seqlen, None

    def pad_input(input_tensor, indices, batch, seqlen):
        output = input_tensor.new_zeros((batch * seqlen, *input_tensor.shape[1:]))
        output[indices] = input_tensor
        return output.reshape(batch, seqlen, *input_tensor.shape[1:])

    def all_gather(input_tensor, group=None):
        if group is None:
            return input_tensor
        output = [torch.empty_like(input_tensor) for _ in range(torch.distributed.get_world_size(group=group))]
        torch.distributed.all_gather(output, input_tensor, group=group)
        return torch.cat(output, dim=0)

    bert_padding_module.index_first_axis = index_first_axis
    bert_padding_module.pad_input = pad_input
    bert_padding_module.rearrange = rearrange
    bert_padding_module.unpad_input = unpad_input
    distributed_module.all_gather = all_gather
    utils_module.distributed = distributed_module
    flash_attn_module.bert_padding = bert_padding_module
    flash_attn_module.utils = utils_module

    sys.modules["flash_attn"] = flash_attn_module
    sys.modules["flash_attn.bert_padding"] = bert_padding_module
    sys.modules["flash_attn.utils"] = utils_module
    sys.modules["flash_attn.utils.distributed"] = distributed_module


def _install_transformers_tokenizer_compat() -> None:
    if os.environ.get("CLAWVLA_OPENRLHF_TOKENIZER_COMPAT") != "1":
        return

    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        @property
        def all_special_tokens_extended(self):
            return list(self.all_special_tokens)

        PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended

    try:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    except Exception:
        return

    if not hasattr(Qwen3VLTextConfig, "tie_word_embeddings"):
        Qwen3VLTextConfig.tie_word_embeddings = False


def _apply_loaded_verl_patches() -> None:
    global _PATCHING
    if _PATCHING:
        return
    _PATCHING = True
    try:
        from clawvla.rl.verl_runtime_patches import apply_verl_runtime_patches

        apply_verl_runtime_patches()
    except Exception as exc:
        if os.environ.get("CLAWVLA_VERL_RUNTIME_PATCHES_STRICT") == "1":
            raise
        print(f"clawvla verl runtime patch unavailable: {type(exc).__name__}: {exc}")
    finally:
        _PATCHING = False


def _apply_loaded_openrlhf_patches() -> None:
    global _PATCHING
    if _PATCHING:
        return
    _PATCHING = True
    try:
        from clawvla.rl.openrlhf_runtime_patches import apply_openrlhf_runtime_patches

        apply_openrlhf_runtime_patches()
    except Exception as exc:
        if os.environ.get("CLAWVLA_OPENRLHF_RUNTIME_PATCHES_STRICT") == "1":
            raise
        print(f"clawvla openrlhf runtime patch unavailable: {type(exc).__name__}: {exc}")
    finally:
        _PATCHING = False


def _clawvla_import_hook(name, globals=None, locals=None, fromlist=(), level=0):
    module = _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
    if (
        level == 0
        and os.environ.get("CLAWVLA_ENABLE_VERL_RUNTIME_PATCHES") == "1"
        and (name == "verl" or name.startswith("verl."))
    ):
        _apply_loaded_verl_patches()
    if (
        level == 0
        and os.environ.get("CLAWVLA_ENABLE_OPENRLHF_RUNTIME_PATCHES") == "1"
        and (name == "openrlhf" or name.startswith("openrlhf."))
    ):
        _apply_loaded_openrlhf_patches()
    return module


if (
    os.environ.get("CLAWVLA_ENABLE_VERL_RUNTIME_PATCHES") == "1"
    or os.environ.get("CLAWVLA_ENABLE_OPENRLHF_RUNTIME_PATCHES") == "1"
):
    _install_flash_attn_padding_fallback()
    _install_transformers_tokenizer_compat()
    builtins.__import__ = _clawvla_import_hook
    if os.environ.get("CLAWVLA_ENABLE_VERL_RUNTIME_PATCHES") == "1" and any(
        name == "verl" or name.startswith("verl.") for name in sys.modules
    ):
        _apply_loaded_verl_patches()
    if os.environ.get("CLAWVLA_ENABLE_OPENRLHF_RUNTIME_PATCHES") == "1" and any(
        name == "openrlhf" or name.startswith("openrlhf.") for name in sys.modules
    ):
        _apply_loaded_openrlhf_patches()
