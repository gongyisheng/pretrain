"""Shared helpers for attention-related tests.

Used by ``tests/fast/layers/test_attention.py`` plus the model-level tests
(``tests/fast/model/test_gpt2.py``, ``test_qwen3.py``) so they parametrize
identically over ``attn_implementation`` × mask kind.
"""
import pytest
import torch

from src.utils.masking_utils import (
    build_causal_attention_mask,
    build_intra_doc_attention_mask,
)


# Attention backends to exercise. Device comes from the conftest ``device``
# fixture (--device=cpu|cuda, default cuda if available else cpu); one device
# per session. flex_attention is CUDA-only — skipped at runtime when device == "cpu".
ATTN_IMPLEMENTATION = ["sdpa", "flex_attention"]

# Mask shapes the trainer builds.
#   "causal":    pure causal (used during generation / no intra-doc config)
#   "intra_doc": doc-causal (qwen3_57m default — blocks attention across doc boundaries)
MASK_KIND = ["causal", "intra_doc"]


def skip_if_unsupported(impl: str, device: str) -> None:
    if impl == "flex_attention" and device == "cpu":
        pytest.skip("flex_attention requires CUDA")


def make_attn_mask(kind: str, impl: str, position_ids: torch.Tensor, dtype: torch.dtype):
    """Build the mask the kernel consumes plus a matching reference-side dense mask.

    The eager references (``mha_ref`` / ``gqa_ref``) take either ``None``
    (when the kernel does pure-causal) or the dense intra-doc mask — never a
    ``BlockMask``. Returning both lets callers feed the right thing to each side.
    """
    B, S = position_ids.shape
    if kind == "causal":
        kernel_mask = build_causal_attention_mask(B, S, position_ids.device, attn_implementation=impl)
        ref_mask = None
    elif kind == "intra_doc":
        kernel_mask = build_intra_doc_attention_mask(position_ids, position_ids.device, dtype, attn_implementation=impl)
        ref_mask = build_intra_doc_attention_mask(position_ids, position_ids.device, dtype, attn_implementation="sdpa")
    else:
        raise AssertionError(f"unknown mask kind: {kind!r}")
    return kernel_mask, ref_mask
