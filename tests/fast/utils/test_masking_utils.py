import torch
from src.utils.masking_utils import build_causal_mask, build_position_ids


EOT = 0  # token ID used as end-of-text in these tests


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


def test_build_causal_mask_shape():
    pos = torch.tensor([[0, 1, 2, 3]])  # (1, 4)
    mask = build_causal_mask(pos, device=pos.device, dtype=torch.float32)
    assert mask.shape == (1, 1, 4, 4)


def test_build_causal_mask_batch_shape():
    pos = torch.tensor([[0, 1, 2], [0, 1, 2]])  # (2, 3)
    mask = build_causal_mask(pos, device=pos.device, dtype=torch.float32)
    assert mask.shape == (2, 1, 3, 3)


def test_build_causal_mask_blocks_future():
    pos = torch.tensor([[0, 1, 2, 3]])
    mask = build_causal_mask(pos, device=pos.device, dtype=torch.float32)
    m = mask[0, 0]  # (4, 4)
    for i in range(4):
        for j in range(i + 1, 4):
            assert m[i, j].item() == float('-inf'), f"Expected -inf at ({i},{j})"


def test_build_causal_mask_allows_same_doc_causal():
    pos = torch.tensor([[0, 1, 2, 3]])  # single doc
    mask = build_causal_mask(pos, device=pos.device, dtype=torch.float32)
    m = mask[0, 0]
    for i in range(4):
        for j in range(i + 1):
            assert m[i, j].item() == 0.0, f"Expected 0.0 at ({i},{j})"


def test_build_causal_mask_blocks_cross_doc():
    # position_ids [0,1,0,1]: doc0=[pos0,pos1], doc1=[pos2,pos3]
    pos = torch.tensor([[0, 1, 0, 1]])
    mask = build_causal_mask(pos, device=pos.device, dtype=torch.float32)
    m = mask[0, 0]
    assert m[2, 0].item() == float('-inf')
    assert m[2, 1].item() == float('-inf')
    assert m[2, 2].item() == 0.0
    assert m[3, 2].item() == 0.0   # same doc, causal


def test_build_causal_mask_three_docs():
    # position_ids = [0,1,2, 0,1,2,3, 0,1] → three documents
    pos = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 0, 1]])
    mask = build_causal_mask(pos, device=pos.device, dtype=torch.float32)
    m = mask[0, 0]
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


def test_build_causal_mask_batch_independent():
    # Each batch item has its own document structure
    pos = torch.tensor([
        [0, 1, 0, 1],   # two docs: [0,1], [2,3]
        [0, 1, 2, 3],   # single doc
    ])
    mask = build_causal_mask(pos, device=pos.device, dtype=torch.float32)
    # batch 0: cross-doc blocked
    assert mask[0, 0, 2, 0].item() == float('-inf')
    assert mask[0, 0, 2, 1].item() == float('-inf')
    # batch 1: all same doc, only future blocked
    assert mask[1, 0, 2, 0].item() == 0.0
    assert mask[1, 0, 2, 1].item() == 0.0
    assert mask[1, 0, 1, 2].item() == float('-inf')


def test_build_causal_mask_dtype_preserved():
    pos = torch.tensor([[0, 1, 2]])
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        mask = build_causal_mask(pos, device=pos.device, dtype=dtype)
        assert mask.dtype == dtype
