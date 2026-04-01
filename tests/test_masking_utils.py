import torch
from src.utils.masking_utils import build_causal_mask


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
