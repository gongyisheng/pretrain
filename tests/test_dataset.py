import pytest
import numpy as np
import os
import tempfile
import torch
from src.data.dataset import PretrainDataset


@pytest.fixture
def tmp_bin_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.bin")
        tokens = np.arange(1024, dtype=np.uint16)
        tokens.tofile(path)
        yield path


def test_dataset_length(tmp_bin_file):
    ds = PretrainDataset(tmp_bin_file, seq_len=128)
    assert len(ds) == 1024 // 128 - 1  # 7


def test_dataset_getitem_shape(tmp_bin_file):
    ds = PretrainDataset(tmp_bin_file, seq_len=128)
    tokens, position_ids = ds[0]
    assert tokens.shape == (129,)   # seq_len + 1
    assert tokens.dtype == torch.long
    assert position_ids.shape == (128,)
    assert position_ids.dtype == torch.long


def test_dataset_getitem_position_ids(tmp_bin_file):
    # No EOT tokens in the data (use an out-of-range eot_token_id)
    # → position_ids should be monotonically increasing
    ds = PretrainDataset(tmp_bin_file, seq_len=128, eot_token_id=9999)
    _, position_ids = ds[0]
    assert position_ids.tolist() == list(range(128))


# ---- packing=False tests ----

EOT = 1  # arbitrary token used as EOT in these tests
PAD = 0  # pad_token_id == eot_token_id in practice, but use 0 here for clarity


@pytest.fixture
def packed_bin_file():
    """Binary file with three documents: [2,3,EOT], [4,5,6,EOT], [7,EOT]."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "docs.bin")
        tokens = np.array([2, 3, EOT, 4, 5, 6, EOT, 7, EOT], dtype=np.uint16)
        tokens.tofile(path)
        yield path


def test_single_doc_length(packed_bin_file):
    ds = PretrainDataset(packed_bin_file, seq_len=8, packing=False, eot_token_id=EOT, pad_token_id=PAD)
    assert len(ds) == 3


def test_single_doc_getitem_returns_two_tuple(packed_bin_file):
    ds = PretrainDataset(packed_bin_file, seq_len=8, packing=False, eot_token_id=EOT, pad_token_id=PAD)
    result = ds[0]
    assert len(result) == 2


def test_single_doc_shapes(packed_bin_file):
    ds = PretrainDataset(packed_bin_file, seq_len=8, packing=False, eot_token_id=EOT, pad_token_id=PAD)
    tokens, loss_mask = ds[0]
    assert tokens.shape == (9,)      # seq_len + 1
    assert loss_mask.shape == (8,)   # seq_len
    assert loss_mask.dtype == torch.bool


def test_single_doc_content(packed_bin_file):
    ds = PretrainDataset(packed_bin_file, seq_len=8, packing=False, eot_token_id=EOT, pad_token_id=PAD)
    # doc0 = [2, 3, EOT] → tokens=[2,3,EOT,PAD,...], loss_mask=[T,T,F,F,F,F,F,F]
    tokens, loss_mask = ds[0]
    assert tokens[0].item() == 2
    assert tokens[1].item() == 3
    assert tokens[2].item() == EOT
    assert tokens[3].item() == PAD
    assert loss_mask[:2].all()
    assert not loss_mask[2:].any()


def test_single_doc_eot_in_loss(packed_bin_file):
    """EOT prediction must be included in the loss (loss_mask covers it)."""
    ds = PretrainDataset(packed_bin_file, seq_len=8, packing=False, eot_token_id=EOT, pad_token_id=PAD)
    # doc1 = [4, 5, 6, EOT] → tokens=[4,5,6,EOT,PAD,...], y[2]=tokens[3]=EOT, loss_mask=[T,T,T,F,...]
    tokens, loss_mask = ds[1]
    assert tokens[3].item() == EOT   # y[2] = tokens[3]
    assert loss_mask[2].item() is True
    assert not loss_mask[3].item()


def test_single_doc_truncation():
    """Documents longer than seq_len+1 are truncated; loss_mask is all True."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "long.bin")
        # doc = [1,2,3,4,5,6,7,8,9,EOT], seq_len=4
        tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, EOT], dtype=np.uint16)
        tokens.tofile(path)
        ds = PretrainDataset(path, seq_len=4, packing=False, eot_token_id=EOT, pad_token_id=PAD)
        tokens_out, loss_mask = ds[0]
        assert tokens_out.shape == (5,)   # seq_len + 1
        assert loss_mask.all()            # no padding when truncated
