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
    x, y = ds[0]
    assert x.shape == (128,)
    assert y.shape == (128,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_dataset_getitem_shift(tmp_bin_file):
    ds = PretrainDataset(tmp_bin_file, seq_len=128)
    x, y = ds[0]
    assert torch.equal(y[:-1], x[1:])


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


def test_single_doc_getitem_returns_three_tuple(packed_bin_file):
    ds = PretrainDataset(packed_bin_file, seq_len=8, packing=False, eot_token_id=EOT, pad_token_id=PAD)
    result = ds[0]
    assert len(result) == 3


def test_single_doc_shapes(packed_bin_file):
    ds = PretrainDataset(packed_bin_file, seq_len=8, packing=False, eot_token_id=EOT, pad_token_id=PAD)
    x, y, loss_mask = ds[0]
    assert x.shape == (8,)
    assert y.shape == (8,)
    assert loss_mask.shape == (8,)
    assert loss_mask.dtype == torch.bool


def test_single_doc_content(packed_bin_file):
    ds = PretrainDataset(packed_bin_file, seq_len=8, packing=False, eot_token_id=EOT, pad_token_id=PAD)
    # doc0 = [2, 3, EOT] → x=[2,3,PAD,...], y=[3,EOT,PAD,...], loss_mask=[T,T,F,F,F,F,F,F]
    x, y, loss_mask = ds[0]
    assert x[0].item() == 2
    assert x[1].item() == 3
    assert x[2].item() == PAD
    assert y[0].item() == 3
    assert y[1].item() == EOT   # EOT prediction included in loss
    assert y[2].item() == PAD
    assert loss_mask[:2].all()
    assert not loss_mask[2:].any()


def test_single_doc_eot_in_loss(packed_bin_file):
    """EOT prediction must be included in the loss (loss_mask covers it)."""
    ds = PretrainDataset(packed_bin_file, seq_len=8, packing=False, eot_token_id=EOT, pad_token_id=PAD)
    # doc1 = [4, 5, 6, EOT] → y=[5,6,EOT,PAD,...], loss_mask=[T,T,T,F,...]
    _, y, loss_mask = ds[1]
    assert y[2].item() == EOT
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
        x, y, loss_mask = ds[0]
        assert x.shape == (4,)
        assert loss_mask.all()   # no padding when truncated
        assert torch.equal(y[:-1], x[1:])  # still a valid shift
