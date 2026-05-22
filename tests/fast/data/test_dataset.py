"""Unit tests for PretrainDataset and SFTDataset.

Tests are in-memory: np.memmap is mocked so tests register pre-built ndarrays
keyed by a synthetic "path" rather than writing real .bin files.
"""

import numpy as np
import pytest
import torch

from src.data.dataset import PretrainDataset


@pytest.fixture
def mock_memmap(monkeypatch):
    storage: dict[str, np.ndarray] = {}

    def fake_memmap(path, dtype, mode):
        return storage[path]

    monkeypatch.setattr(np, "memmap", fake_memmap)
    return storage


# --- packing=True (default) ---


def test_dataset_length(mock_memmap):
    mock_memmap["/x.bin"] = np.arange(1024, dtype=np.uint16)
    ds = PretrainDataset("/x.bin", seq_len=128, vocab_size=1024)
    assert len(ds) == 1024 // 128 - 1


def test_dataset_getitem_shape(mock_memmap):
    mock_memmap["/x.bin"] = np.arange(1024, dtype=np.uint16)
    ds = PretrainDataset("/x.bin", seq_len=128, vocab_size=1024)
    input_ids, position_ids, labels = ds[0]
    assert input_ids.shape == (129,)
    assert input_ids.dtype == torch.long
    assert position_ids.shape == (128,)
    assert position_ids.dtype == torch.long
    assert labels.shape == (128,)
    assert labels.dtype == torch.long


def test_dataset_getitem_position_ids(mock_memmap):
    mock_memmap["/x.bin"] = np.arange(1024, dtype=np.uint16)
    ds = PretrainDataset("/x.bin", seq_len=128, vocab_size=1024, eot_token_id=9999)
    _, position_ids, _ = ds[0]
    assert position_ids.tolist() == list(range(128))


def test_dataset_packed_labels_match_shifted_inputs(mock_memmap):
    """packing=True: labels == input_ids[1:] with no -100s."""
    mock_memmap["/x.bin"] = np.arange(1024, dtype=np.uint16)
    ds = PretrainDataset("/x.bin", seq_len=128, vocab_size=1024, eot_token_id=9999)
    input_ids, _, labels = ds[0]
    assert torch.equal(labels, input_ids[1:])
    assert (labels != -100).all()


# --- packing=False ---

EOT = 1
PAD = 1


@pytest.fixture
def packed_docs(mock_memmap):
    mock_memmap["/docs.bin"] = np.array(
        [2, 3, EOT, 4, 5, 6, EOT, 7, EOT], dtype=np.uint16
    )
    return "/docs.bin"


def _unpacked_ds(path):
    return PretrainDataset(
        path,
        seq_len=8,
        vocab_size=256,
        packing=False,
        eot_token_id=EOT,
        pad_token_id=PAD,
    )


def test_single_doc_length(packed_docs):
    assert len(_unpacked_ds(packed_docs)) == 3


def test_single_doc_getitem_returns_three_tuple(packed_docs):
    result = _unpacked_ds(packed_docs)[0]
    assert len(result) == 3


def test_single_doc_shapes(packed_docs):
    input_ids, position_ids, labels = _unpacked_ds(packed_docs)[0]
    assert input_ids.shape == (9,)
    assert position_ids.shape == (8,)
    assert labels.shape == (8,)


def test_single_doc_content(packed_docs):
    input_ids, position_ids, _ = _unpacked_ds(packed_docs)[0]
    assert input_ids[0].item() == 2
    assert input_ids[1].item() == 3
    assert input_ids[2].item() == EOT
    assert input_ids[3].item() == PAD
    assert position_ids[:3].tolist() == [0, 1, 2]
    assert (position_ids[3:] == -1).all()


def test_single_doc_unpacked_labels_minus_100_at_padding(packed_docs):
    """packing=False: labels has -100 wherever the corresponding x is padding (position_ids < 0)."""
    # doc0 = [2, 3, EOT]
    # input_ids = [2, 3, EOT, PAD, PAD, PAD, PAD, PAD, PAD]
    # x         = input_ids[:-1] = [2, 3, EOT, PAD, PAD, PAD, PAD, PAD]
    # position_ids = [0, 1, 2, -1, -1, -1, -1, -1]
    # x[0..2] are real (position_ids[0..2] = [0,1,2]),
    # so labels[0..2] = input_ids[1..3] = [3, EOT, PAD].
    # x[3..7] are padding, so labels[3..7] = -100.
    input_ids, position_ids, labels = _unpacked_ds(packed_docs)[0]
    assert labels[:3].tolist() == [3, EOT, PAD]
    assert (labels[3:] == -100).all()


def test_single_doc_eot_in_loss(packed_docs):
    """EOT prediction (predicting EOT from previous token) must be included in the loss."""
    # doc1 = [4, 5, 6, EOT] → x = [4, 5, 6, EOT, PAD, PAD, PAD, PAD]
    # labels = [5, 6, EOT, PAD, -100, -100, -100, -100]
    input_ids, position_ids, labels = _unpacked_ds(packed_docs)[1]
    assert labels[2].item() == EOT
    assert labels[3].item() == PAD
    assert (labels[4:] == -100).all()


def test_single_doc_truncation(mock_memmap):
    mock_memmap["/long.bin"] = np.array([2, 3, 4, 5, 6, 7, 8, 9, EOT], dtype=np.uint16)
    ds = PretrainDataset(
        "/long.bin",
        seq_len=4,
        vocab_size=256,
        packing=False,
        eot_token_id=EOT,
        pad_token_id=PAD,
    )
    input_ids, position_ids, labels = ds[0]
    assert input_ids.shape == (5,)
    assert (position_ids >= 0).all()
    assert position_ids.tolist() == [0, 1, 2, 3]
    assert (labels != -100).all()
