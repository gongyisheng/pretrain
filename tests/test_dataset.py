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
    input_ids, position_ids = ds[0]
    assert input_ids.shape == (129,)     # seq_len + 1
    assert input_ids.dtype == torch.long
    assert position_ids.shape == (128,)  # seq_len
    assert position_ids.dtype == torch.long


def test_dataset_getitem_position_ids(tmp_bin_file):
    # No EOT tokens in the data (use an out-of-range eot_token_id)
    # → position_ids should be monotonically increasing
    ds = PretrainDataset(tmp_bin_file, seq_len=128, eot_token_id=9999)
    _, position_ids = ds[0]
    assert position_ids.tolist() == list(range(128))


# ---- packing=False tests ----

EOT = 1  # arbitrary token used as EOT in these tests
PAD = 1  # pad_token_id == eot_token_id (same token)


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
    input_ids, position_ids = ds[0]
    assert input_ids.shape == (9,)      # seq_len + 1
    assert position_ids.shape == (8,)   # seq_len
    assert position_ids.dtype == torch.long


def test_single_doc_content(packed_bin_file):
    ds = PretrainDataset(packed_bin_file, seq_len=8, packing=False, eot_token_id=EOT, pad_token_id=PAD)
    # doc0 = [2, 3, EOT] → input_ids=[2,3,EOT,PAD,...], position_ids=[0,1,2,-1,...]
    input_ids, position_ids = ds[0]
    assert input_ids[0].item() == 2
    assert input_ids[1].item() == 3
    assert input_ids[2].item() == EOT
    assert input_ids[3].item() == PAD
    # position_ids: real positions for content, -1 for padding
    assert position_ids[:3].tolist() == [0, 1, 2]
    assert (position_ids[3:] == -1).all()


def test_single_doc_eot_in_loss(packed_bin_file):
    """EOT prediction must be included in the loss (position_ids[2] >= 0)."""
    ds = PretrainDataset(packed_bin_file, seq_len=8, packing=False, eot_token_id=EOT, pad_token_id=PAD)
    # doc1 = [4, 5, 6, EOT] → position_ids=[0,1,2,3,-1,-1,-1,-1]
    _, position_ids = ds[1]
    assert position_ids[3].item() == 3   # EOT is the 4th real token (position 3)
    assert position_ids[4].item() == -1  # padding starts here


def test_single_doc_truncation():
    """Documents longer than seq_len+1 are truncated; all position_ids >= 0."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "long.bin")
        # doc = [2,3,4,5,6,7,8,9,EOT], seq_len=4
        tokens = np.array([2, 3, 4, 5, 6, 7, 8, 9, EOT], dtype=np.uint16)
        tokens.tofile(path)
        ds = PretrainDataset(path, seq_len=4, packing=False, eot_token_id=EOT, pad_token_id=PAD)
        input_ids, position_ids = ds[0]
        assert input_ids.shape == (5,)   # seq_len + 1
        assert (position_ids >= 0).all() # no padding when truncated
        assert position_ids.tolist() == [0, 1, 2, 3]
