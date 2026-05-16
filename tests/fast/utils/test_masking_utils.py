import pytest
import torch

from src.utils.masking_utils import (
    build_causal_attention_mask,
    build_intra_doc_attention_mask,
    build_position_ids,
)
from tests.fast.helpers import ATTN_IMPLEMENTATION, skip_if_unsupported


EOT = 0  # token ID used as end-of-text in these tests


# ==================== build_position_ids ====================

def test_build_position_ids_single_doc():
    x = torch.tensor([[1, 2, 3, 4]])
    pos = build_position_ids(x, EOT)
    assert pos.tolist() == [[0, 1, 2, 3]]


def test_build_position_ids_resets_after_eot():
    # EOT at index 2 → next token starts at 0
    x = torch.tensor([[1, 2, EOT, 3, 4]])
    pos = build_position_ids(x, EOT)
    assert pos.tolist() == [[0, 1, 2, 0, 1]]


def test_build_position_ids_eot_belongs_to_its_doc():
    # EOT itself is position 2 of doc 0, not position 0 of doc 1
    x = torch.tensor([[1, 2, EOT, 3]])
    pos = build_position_ids(x, EOT)
    assert pos[0, 2].item() == 2   # EOT is the 3rd token of doc 0
    assert pos[0, 3].item() == 0   # next token starts doc 1 at position 0


def test_build_position_ids_multiple_docs():
    x = torch.tensor([[1, EOT, 2, EOT, 3]])
    pos = build_position_ids(x, EOT)
    assert pos.tolist() == [[0, 1, 0, 1, 0]]


def test_build_position_ids_batch():
    x = torch.tensor([
        [1, EOT, 2, 3],
        [4, 5, 6, EOT],
    ])
    pos = build_position_ids(x, EOT)
    assert pos[0].tolist() == [0, 1, 0, 1]
    assert pos[1].tolist() == [0, 1, 2, 3]


def test_build_position_ids_shape_and_dtype():
    x = torch.tensor([[1, 2, 3]])
    pos = build_position_ids(x, EOT)
    assert pos.shape == x.shape
    assert pos.dtype == torch.long


# ---- packing=False tests ----

def test_build_position_ids_unpacked_single_doc():
    # doc=[1,2,EOT] padded to len 6: real=0,1,2; padding=-1
    x = torch.tensor([[1, 2, EOT, EOT, EOT, EOT]])
    pos = build_position_ids(x, EOT, packing=False)
    assert pos.tolist() == [[0, 1, 2, -1, -1, -1]]


def test_build_position_ids_unpacked_no_eot():
    # truncated doc, no EOT — all positions are real
    x = torch.tensor([[1, 2, 3, 4]])
    pos = build_position_ids(x, EOT, packing=False)
    assert pos.tolist() == [[0, 1, 2, 3]]


def test_build_position_ids_unpacked_eot_belongs_to_doc():
    # EOT at position 2 is real (part of the document), not padding
    x = torch.tensor([[1, 2, EOT, EOT]])
    pos = build_position_ids(x, EOT, packing=False)
    assert pos[0, 2].item() == 2    # EOT itself is the last real position
    assert pos[0, 3].item() == -1   # token after EOT is padding


def test_build_position_ids_unpacked_batch():
    # Two sequences with different padding lengths
    x = torch.tensor([
        [1, 2, EOT, EOT, EOT],   # doc ends at pos 2
        [3, 4, 5, EOT, EOT],     # doc ends at pos 3
    ])
    pos = build_position_ids(x, EOT, packing=False)
    assert pos[0].tolist() == [0, 1, 2, -1, -1]
    assert pos[1].tolist() == [0, 1, 2, 3, -1]


def test_build_position_ids_unpacked_shape_and_dtype():
    x = torch.tensor([[1, 2, EOT, EOT]])
    pos = build_position_ids(x, EOT, packing=False)
    assert pos.shape == x.shape
    assert pos.dtype == torch.long


def test_build_position_ids_unpacked_all_real():
    # No EOT at all — entire sequence is real content
    x = torch.tensor([[5, 6, 7, 8, 9]])
    pos = build_position_ids(x, EOT, packing=False)
    assert pos.tolist() == [[0, 1, 2, 3, 4]]


# ==================== build_intra_doc_attention_mask: sdpa (dense additive) ====================
# These pin down the dense additive-mask representation produced for the SDPA
# backend. Same semantics as the flex_attention path; that parity is checked
# separately via flex_attention output equivalence.

def _sdpa_intra_doc(pos, dtype=torch.float32):
    return build_intra_doc_attention_mask(pos, pos.device, dtype, attn_implementation="sdpa")


def test_intra_doc_sdpa_shape():
    pos = torch.tensor([[0, 1, 2, 3]])  # (1, 4)
    assert _sdpa_intra_doc(pos).shape == (1, 1, 4, 4)


def test_intra_doc_sdpa_batch_shape():
    pos = torch.tensor([[0, 1, 2], [0, 1, 2]])  # (2, 3)
    assert _sdpa_intra_doc(pos).shape == (2, 1, 3, 3)


def test_intra_doc_sdpa_blocks_future():
    pos = torch.tensor([[0, 1, 2, 3]])
    m = _sdpa_intra_doc(pos)[0, 0]  # (4, 4)
    for i in range(4):
        for j in range(i + 1, 4):
            assert m[i, j].item() == float('-inf'), f"Expected -inf at ({i},{j})"


def test_intra_doc_sdpa_allows_same_doc_causal():
    pos = torch.tensor([[0, 1, 2, 3]])  # single doc
    m = _sdpa_intra_doc(pos)[0, 0]
    for i in range(4):
        for j in range(i + 1):
            assert m[i, j].item() == 0.0, f"Expected 0.0 at ({i},{j})"


def test_intra_doc_sdpa_blocks_cross_doc():
    # position_ids [0,1,0,1]: doc0=[pos0,pos1], doc1=[pos2,pos3]
    pos = torch.tensor([[0, 1, 0, 1]])
    m = _sdpa_intra_doc(pos)[0, 0]
    assert m[2, 0].item() == float('-inf')
    assert m[2, 1].item() == float('-inf')
    assert m[2, 2].item() == 0.0
    assert m[3, 2].item() == 0.0   # same doc, causal


def test_intra_doc_sdpa_three_docs():
    # position_ids = [0,1,2, 0,1,2,3, 0,1] → three documents
    pos = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 0, 1]])
    m = _sdpa_intra_doc(pos)[0, 0]
    # doc1 (rows 3-6) must not attend to doc0 (cols 0-2)
    assert m[3, 0].item() == float('-inf')
    assert m[3, 1].item() == float('-inf')
    assert m[3, 2].item() == float('-inf')
    assert m[3, 3].item() == 0.0
    # doc2 (rows 7-8) must not attend to doc0 or doc1
    assert m[7, 0].item() == float('-inf')
    assert m[7, 3].item() == float('-inf')
    assert m[7, 7].item() == 0.0
    assert m[8, 7].item() == 0.0


def test_intra_doc_sdpa_batch_independent():
    # Each batch item has its own document structure
    pos = torch.tensor([
        [0, 1, 0, 1],   # two docs: [0,1], [2,3]
        [0, 1, 2, 3],   # single doc
    ])
    mask = _sdpa_intra_doc(pos)
    # batch 0: cross-doc blocked
    assert mask[0, 0, 2, 0].item() == float('-inf')
    assert mask[0, 0, 2, 1].item() == float('-inf')
    # batch 1: all same doc, only future blocked
    assert mask[1, 0, 2, 0].item() == 0.0
    assert mask[1, 0, 2, 1].item() == 0.0
    assert mask[1, 0, 1, 2].item() == float('-inf')


def test_intra_doc_sdpa_dtype_preserved():
    pos = torch.tensor([[0, 1, 2]])
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        mask = build_intra_doc_attention_mask(pos, pos.device, dtype, attn_implementation="sdpa")
        assert mask.dtype == dtype


# ==================== build_causal_attention_mask: sdpa (None sentinel) ====================
# sdpa's causal path uses F.scaled_dot_product_attention(..., is_causal=True),
# which doesn't need an additive mask — the builder returns None as a signal.

def test_causal_sdpa_returns_none():
    """sdpa causal mask is None so SDPA dispatches to flash with is_causal=True."""
    assert build_causal_attention_mask(2, 8, torch.device("cpu"), attn_implementation="sdpa") is None


# ==================== flex_attention parity (BlockMask) ====================
# FlexAttention requires CUDA. We can't inspect a BlockMask's values directly,
# so verify the kernel-level behavior: feeding the flex BlockMask to
# flex_attention produces the same output as feeding sdpa's representation
# (dense for intra_doc, is_causal=True for causal) to F.scaled_dot_product_attention.

def _maybe_skip_flex(device):
    skip_if_unsupported("flex_attention", device)


def test_intra_doc_flex_matches_sdpa(device):
    _maybe_skip_flex(device)
    import torch.nn.functional as F
    from torch.nn.attention.flex_attention import flex_attention

    torch.manual_seed(0)
    pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    B, S, H, D = 1, 8, 2, 16
    q = torch.randn(B, H, S, D, dtype=torch.float32)
    k = torch.randn(B, H, S, D, dtype=torch.float32)
    v = torch.randn(B, H, S, D, dtype=torch.float32)

    dense_mask = build_intra_doc_attention_mask(pos, pos.device, q.dtype, attn_implementation="sdpa")
    out_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=dense_mask, is_causal=False)

    block_mask = build_intra_doc_attention_mask(pos, pos.device, q.dtype, attn_implementation="flex_attention")
    out_flex = flex_attention(q, k, v, block_mask=block_mask)

    assert torch.allclose(out_sdpa, out_flex, atol=1e-5)


def test_intra_doc_flex_three_docs_matches_sdpa(device):
    _maybe_skip_flex(device)
    import torch.nn.functional as F
    from torch.nn.attention.flex_attention import flex_attention

    torch.manual_seed(0)
    pos = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 0, 1]])
    B, S, H, D = 1, 9, 2, 16
    q = torch.randn(B, H, S, D, dtype=torch.float32)
    k = torch.randn(B, H, S, D, dtype=torch.float32)
    v = torch.randn(B, H, S, D, dtype=torch.float32)

    dense = build_intra_doc_attention_mask(pos, pos.device, q.dtype, attn_implementation="sdpa")
    out_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=dense, is_causal=False)
    flex_mask = build_intra_doc_attention_mask(pos, pos.device, q.dtype, attn_implementation="flex_attention")
    out_flex = flex_attention(q, k, v, block_mask=flex_mask)

    assert torch.allclose(out_sdpa, out_flex, atol=1e-5)


def test_causal_flex_matches_sdpa_is_causal(device):
    """flex causal-only BlockMask gives the same output as sdpa + is_causal=True."""
    _maybe_skip_flex(device)
    import torch.nn.functional as F
    from torch.nn.attention.flex_attention import flex_attention

    torch.manual_seed(0)
    B, S, H, D = 1, 8, 2, 16
    q = torch.randn(B, H, S, D, dtype=torch.float32)
    k = torch.randn(B, H, S, D, dtype=torch.float32)
    v = torch.randn(B, H, S, D, dtype=torch.float32)

    # sdpa: None → is_causal=True path
    assert build_causal_attention_mask(B, S, q.device, attn_implementation="sdpa") is None
    out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    block_mask = build_causal_attention_mask(B, S, q.device, attn_implementation="flex_attention")
    out_flex = flex_attention(q, k, v, block_mask=block_mask)

    assert torch.allclose(out_sdpa, out_flex, atol=1e-5)


# ==================== Misc ====================

@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_build_intra_doc_attention_mask_runs(impl, device):
    """Smoke test: both backends accept the standard call signature."""
    skip_if_unsupported(impl, device)
    pos = torch.tensor([[0, 1, 0, 1]])
    mask = build_intra_doc_attention_mask(pos, pos.device, torch.float32, attn_implementation=impl)
    assert mask is not None


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_build_causal_attention_mask_runs(impl, device):
    skip_if_unsupported(impl, device)
    mask = build_causal_attention_mask(2, 8, torch.device(device), attn_implementation=impl)
    if impl == "sdpa":
        assert mask is None  # sentinel for is_causal=True
    else:
        assert mask is not None


def test_build_intra_doc_attention_mask_rejects_unknown_impl():
    pos = torch.tensor([[0, 1, 2]])
    with pytest.raises(ValueError, match="unknown attn_implementation"):
        build_intra_doc_attention_mask(pos, pos.device, torch.float32, attn_implementation="nope")


def test_build_causal_attention_mask_rejects_unknown_impl():
    with pytest.raises(ValueError, match="unknown attn_implementation"):
        build_causal_attention_mask(1, 8, torch.device("cpu"), attn_implementation="nope")
