"""Unit tests for PretrainDataset and SFTDataset.

Tests are in-memory: np.memmap is mocked so tests register pre-built ndarrays
keyed by a synthetic "path" rather than writing real .bin files.
"""

import numpy as np
import pytest
import torch

from src.data.dataset import PretrainDataset, SFTDataset


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
    # All three are pre-shifted by the dataset: shape (seq_len,) each.
    assert input_ids.shape == (128,)
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


def test_dataset_packed_shift_against_disk(mock_memmap):
    """packing=True: input_ids == data[:seq_len], labels == data[1:seq_len+1], no -100s."""
    data = np.arange(1024, dtype=np.uint16)
    mock_memmap["/x.bin"] = data
    ds = PretrainDataset("/x.bin", seq_len=128, vocab_size=1024, eot_token_id=9999)
    input_ids, _, labels = ds[0]
    assert torch.equal(input_ids, torch.from_numpy(data[:128].astype(np.int64)))
    assert torch.equal(labels, torch.from_numpy(data[1:129].astype(np.int64)))
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
    # All three pre-shifted: shape (seq_len,) each.
    assert input_ids.shape == (8,)
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
    """packing=False: labels has -100 wherever input_ids is padding (position_ids < 0)."""
    # doc0 = [2, 3, EOT]
    # raw chunk    = [2, 3, EOT, PAD, PAD, PAD, PAD, PAD, PAD]  (seq_len + 1 = 9 tokens)
    # input_ids    = chunk[:-1] = [2, 3, EOT, PAD, PAD, PAD, PAD, PAD]
    # position_ids = [0, 1, 2, -1, -1, -1, -1, -1]
    # labels (raw) = chunk[1:]  = [3, EOT, PAD, PAD, PAD, PAD, PAD, PAD]
    # then labels[position_ids < 0] = -100  →  [3, EOT, PAD, -100, -100, -100, -100, -100]
    input_ids, position_ids, labels = _unpacked_ds(packed_docs)[0]
    assert labels[:3].tolist() == [3, EOT, PAD]
    assert (labels[3:] == -100).all()


def test_single_doc_eot_in_loss(packed_docs):
    """EOT prediction (predicting EOT from previous token) must be included in the loss."""
    # doc1 = [4, 5, 6, EOT]
    # input_ids = [4, 5, 6, EOT, PAD, PAD, PAD, PAD]
    # labels    = [5, 6, EOT, PAD, -100, -100, -100, -100]
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
    # Pre-shifted: input_ids is seq_len long, not seq_len+1.
    assert input_ids.shape == (4,)
    assert (position_ids >= 0).all()
    assert position_ids.tolist() == [0, 1, 2, 3]
    assert (labels != -100).all()


# --- SFTDataset ---


@pytest.fixture
def sft_bin(mock_memmap):
    # Three samples, each [a, op, b, =, c] = 5 tokens.
    # Sample 0: [3, 100, 5, 104, 8]
    # Sample 1: [10, 100, 20, 104, 30]
    # Sample 2: [50, 100, 40, 104, 90]
    flat = np.array(
        [3, 100, 5, 104, 8, 10, 100, 20, 104, 30, 50, 100, 40, 104, 90],
        dtype=np.uint16,
    )
    mock_memmap["/sft.bin"] = flat
    return "/sft.bin"


def _sft_ds(path, question_len=4, answer_len=1):
    return SFTDataset(
        bin_path=path,
        vocab_size=128,
        question_len=question_len,
        answer_len=answer_len,
    )


def test_sft_dataset_length(sft_bin):
    ds = _sft_ds(sft_bin)
    assert len(ds) == 3


def test_sft_dataset_shapes(sft_bin):
    ds = _sft_ds(sft_bin)
    input_ids, position_ids, labels = ds[0]
    # Pre-shifted: all three are (stride - 1,) = (4,).
    assert input_ids.shape == (4,)
    assert position_ids.shape == (4,)
    assert labels.shape == (4,)
    assert input_ids.dtype == torch.long
    assert position_ids.dtype == torch.long
    assert labels.dtype == torch.long


def test_sft_dataset_input_ids_match_disk(sft_bin):
    """input_ids is the on-disk sample with the answer token dropped (model input only)."""
    ds = _sft_ds(sft_bin)
    input_ids, _, _ = ds[1]
    # Sample 1 on disk: [10, 100, 20, 104, 30]; input_ids drops the final answer token.
    assert input_ids.tolist() == [10, 100, 20, 104]


def test_sft_dataset_position_ids_sequential(sft_bin):
    ds = _sft_ds(sft_bin)
    _, position_ids, _ = ds[0]
    assert position_ids.tolist() == [0, 1, 2, 3]


def test_sft_dataset_labels_mask_layout(sft_bin):
    """Labels: -100 for the first (question_len-1) positions; answer tokens for the last answer_len."""
    # stride=5, question_len=4, answer_len=1 → 4 label positions.
    # labels[0..2] = -100  (first Lq-1=3 positions masked)
    # labels[3]   = 8      (last La=1 positions hold the answer)
    ds = _sft_ds(sft_bin, question_len=4, answer_len=1)
    _, _, labels = ds[0]
    assert labels[:3].tolist() == [-100, -100, -100]
    assert labels[-1].item() == 8


def test_sft_dataset_labels_multi_token_answer(mock_memmap):
    """Generalization check: answer_len=2 sets the last two labels to the answer tokens.

    On disk each sample is 6 tokens; pre-shifted input_ids drops the very last token.
    The model's input still contains the first answer token (so it can predict the second).
    """
    mock_memmap["/sft2.bin"] = np.array(
        [1, 2, 100, 104, 5, 6, 11, 12, 100, 104, 15, 16],
        dtype=np.uint16,
    )
    ds = SFTDataset(bin_path="/sft2.bin", vocab_size=128, question_len=4, answer_len=2)
    assert len(ds) == 2
    input_ids, position_ids, labels = ds[1]
    # Sample 1 on disk = [11, 12, 100, 104, 15, 16]; input_ids drops the last (16).
    assert input_ids.tolist() == [11, 12, 100, 104, 15]
    assert position_ids.tolist() == [0, 1, 2, 3, 4]
    assert labels[:3].tolist() == [-100, -100, -100]
    assert labels[-2:].tolist() == [15, 16]


def test_sft_dataset_raises_on_misaligned_file(mock_memmap):
    """ValueError if the .bin file length isn't a multiple of stride."""
    mock_memmap["/bad.bin"] = np.arange(13, dtype=np.uint16)  # 13 % 5 != 0
    with pytest.raises(ValueError, match="not a multiple of stride"):
        SFTDataset(bin_path="/bad.bin", vocab_size=128, question_len=4, answer_len=1)


def test_sft_dataset_packing_true_raises(sft_bin):
    """packing=True is not yet implemented; constructor must raise."""
    with pytest.raises(NotImplementedError, match="packing"):
        SFTDataset(
            bin_path=sft_bin,
            vocab_size=128,
            question_len=4,
            answer_len=1,
            packing=True,
        )
