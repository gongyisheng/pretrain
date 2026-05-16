"""Unit tests for PretrainDataset.

Tests are in-memory: np.memmap is mocked so tests register pre-built ndarrays
keyed by a synthetic "path" rather than writing real .bin files.
"""

import numpy as np
import pytest
import torch

from src.data.dataset import PretrainDataset


@pytest.fixture
def mock_memmap(monkeypatch):
    """Replace np.memmap with a dict-backed lookup.

    Tests insert `storage[fake_path] = np.array(...)`; PretrainDataset(fake_path, ...)
    receives that array instead of reading from disk.
    """
    storage: dict[str, np.ndarray] = {}

    def fake_memmap(path, dtype, mode):
        return storage[path]

    monkeypatch.setattr(np, "memmap", fake_memmap)
    return storage


# --- packing=True (default) ---


def test_dataset_length(mock_memmap):
    mock_memmap["/x.bin"] = np.arange(1024, dtype=np.uint16)
    ds = PretrainDataset("/x.bin", seq_len=128, vocab_size=1024)
    assert len(ds) == 1024 // 128 - 1  # 7


def test_dataset_getitem_shape(mock_memmap):
    mock_memmap["/x.bin"] = np.arange(1024, dtype=np.uint16)
    ds = PretrainDataset("/x.bin", seq_len=128, vocab_size=1024)
    input_ids, position_ids = ds[0]
    assert input_ids.shape == (129,)  # seq_len + 1
    assert input_ids.dtype == torch.long
    assert position_ids.shape == (128,)  # seq_len
    assert position_ids.dtype == torch.long


def test_dataset_getitem_position_ids(mock_memmap):
    """Without EOT tokens in the data, position_ids should be sequential."""
    mock_memmap["/x.bin"] = np.arange(1024, dtype=np.uint16)
    ds = PretrainDataset("/x.bin", seq_len=128, vocab_size=1024, eot_token_id=9999)
    _, position_ids = ds[0]
    assert position_ids.tolist() == list(range(128))


# --- packing=False ---

EOT = 1
PAD = 1  # pad_token_id == eot_token_id (same token)


@pytest.fixture
def packed_docs(mock_memmap):
    """Three documents: [2,3,EOT], [4,5,6,EOT], [7,EOT]."""
    mock_memmap["/docs.bin"] = np.array(
        [2, 3, EOT, 4, 5, 6, EOT, 7, EOT], dtype=np.uint16
    )
    return "/docs.bin"


def test_single_doc_length(packed_docs):
    ds = PretrainDataset(
        packed_docs,
        seq_len=8,
        vocab_size=256,
        packing=False,
        eot_token_id=EOT,
        pad_token_id=PAD,
    )
    assert len(ds) == 3


def test_single_doc_getitem_returns_two_tuple(packed_docs):
    ds = PretrainDataset(
        packed_docs,
        seq_len=8,
        vocab_size=256,
        packing=False,
        eot_token_id=EOT,
        pad_token_id=PAD,
    )
    result = ds[0]
    assert len(result) == 2


def test_single_doc_shapes(packed_docs):
    ds = PretrainDataset(
        packed_docs,
        seq_len=8,
        vocab_size=256,
        packing=False,
        eot_token_id=EOT,
        pad_token_id=PAD,
    )
    input_ids, position_ids = ds[0]
    assert input_ids.shape == (9,)  # seq_len + 1
    assert position_ids.shape == (8,)  # seq_len
    assert position_ids.dtype == torch.long


def test_single_doc_content(packed_docs):
    ds = PretrainDataset(
        packed_docs,
        seq_len=8,
        vocab_size=256,
        packing=False,
        eot_token_id=EOT,
        pad_token_id=PAD,
    )
    # doc0 = [2, 3, EOT] → input_ids=[2,3,EOT,PAD,...], position_ids=[0,1,2,-1,...]
    input_ids, position_ids = ds[0]
    assert input_ids[0].item() == 2
    assert input_ids[1].item() == 3
    assert input_ids[2].item() == EOT
    assert input_ids[3].item() == PAD
    assert position_ids[:3].tolist() == [0, 1, 2]
    assert (position_ids[3:] == -1).all()


def test_single_doc_eot_in_loss(packed_docs):
    """EOT prediction must be included in the loss (position_ids[2] >= 0)."""
    ds = PretrainDataset(
        packed_docs,
        seq_len=8,
        vocab_size=256,
        packing=False,
        eot_token_id=EOT,
        pad_token_id=PAD,
    )
    # doc1 = [4, 5, 6, EOT] → position_ids=[0,1,2,3,-1,-1,-1,-1]
    _, position_ids = ds[1]
    assert position_ids[3].item() == 3  # EOT is the 4th real token (position 3)
    assert position_ids[4].item() == -1  # padding starts here


def test_single_doc_truncation(mock_memmap):
    """Documents longer than seq_len+1 are truncated; all position_ids >= 0."""
    # doc = [2,3,4,5,6,7,8,9,EOT], seq_len=4
    mock_memmap["/long.bin"] = np.array([2, 3, 4, 5, 6, 7, 8, 9, EOT], dtype=np.uint16)
    ds = PretrainDataset(
        "/long.bin",
        seq_len=4,
        vocab_size=256,
        packing=False,
        eot_token_id=EOT,
        pad_token_id=PAD,
    )
    input_ids, position_ids = ds[0]
    assert input_ids.shape == (5,)
    assert (position_ids >= 0).all()
    assert position_ids.tolist() == [0, 1, 2, 3]
