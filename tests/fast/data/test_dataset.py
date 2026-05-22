"""Unit tests for PretrainDataset and SFTDataset.

Tests are in-memory: np.memmap is mocked so tests register pre-built ndarrays
keyed by a synthetic "path" rather than writing real .bin files.
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
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
#
# SFTDataset is parquet-backed. Tests write small parquet files to tmp_path
# and verify the per-row (question_ids, answer_ids) → (input_ids, position_ids,
# labels) assembly + masking.


SFT_EOT = 99
SFT_PAD = 99


def _write_sft_parquet(path, rows):
    """rows = [(q_ids, a_ids), ...]"""
    table = pa.table(
        {
            "question_ids": [list(q) for q, _ in rows],
            "answer_ids": [list(a) for _, a in rows],
        }
    )
    pq.write_table(table, path)


@pytest.fixture
def sft_parquet(tmp_path):
    rows = [
        ([3, 100, 5, 104], [8]),
        ([10, 100, 20, 104], [30]),
        ([50, 100, 40, 104], [90]),
    ]
    path = tmp_path / "sft.parquet"
    _write_sft_parquet(path, rows)
    return str(path)


def _sft_ds(path, seq_len=6):
    return SFTDataset(
        parquet_path=path,
        seq_len=seq_len,
        eot_token_id=SFT_EOT,
        pad_token_id=SFT_PAD,
    )


def test_sft_dataset_length(sft_parquet):
    ds = _sft_ds(sft_parquet)
    assert len(ds) == 3


def test_sft_dataset_shapes(sft_parquet):
    ds = _sft_ds(sft_parquet)
    input_ids, position_ids, labels = ds[0]
    assert input_ids.shape == (6,)
    assert position_ids.shape == (6,)
    assert labels.shape == (6,)
    assert input_ids.dtype == torch.long
    assert position_ids.dtype == torch.long
    assert labels.dtype == torch.long


def test_sft_dataset_input_ids_assembled(sft_parquet):
    """input_ids = [q..., a..., EOT, PAD...] then shifted left by 1."""
    ds = _sft_ds(sft_parquet, seq_len=6)
    # Row 1: q=[10, 100, 20, 104], a=[30] → sample=[10, 100, 20, 104, 30, EOT]
    # Padded to seq_len + 1 = 7: [10, 100, 20, 104, 30, EOT, PAD]
    # input_ids = chunk[:-1] = [10, 100, 20, 104, 30, EOT]
    input_ids, _, _ = ds[1]
    assert input_ids.tolist() == [10, 100, 20, 104, 30, SFT_EOT]


def test_sft_dataset_labels_mask_question_supervise_answer(sft_parquet):
    """Labels: -100 for the first (q_len - 1) positions; answer + EOT supervised."""
    # Row 0: q_len=4, a_len=1, sample=[3, 100, 5, 104, 8, EOT], n_content=6.
    # chunk = [3, 100, 5, 104, 8, EOT, PAD]
    # input_ids = [3, 100, 5, 104, 8, EOT]
    # Supervised label indices: [q_len-1, q_len+a_len) = [3, 5).
    #   labels[3] = chunk[4] = 8        (answer prediction)
    #   labels[4] = chunk[5] = EOT      (stop prediction)
    # All others (labels[0..2] and labels[5]) = -100.
    ds = _sft_ds(sft_parquet, seq_len=6)
    _, _, labels = ds[0]
    assert labels[:3].tolist() == [-100, -100, -100]
    assert labels[3].item() == 8
    assert labels[4].item() == SFT_EOT
    assert labels[5].item() == -100


def test_sft_dataset_labels_multi_token_answer(tmp_path):
    """Generalization: multi-token answer supervises every answer position + EOT."""
    rows = [([1, 2, 3], [10, 20])]  # q_len=3, a_len=2
    path = tmp_path / "multi.parquet"
    _write_sft_parquet(path, rows)
    ds = SFTDataset(
        parquet_path=str(path),
        seq_len=6,
        eot_token_id=SFT_EOT,
        pad_token_id=SFT_PAD,
    )
    input_ids, _, labels = ds[0]
    # sample = [1, 2, 3, 10, 20, EOT], padded to 7: [..., PAD]
    # input_ids = [1, 2, 3, 10, 20, EOT]
    # Supervised range: [q_len-1, q_len+a_len) = [2, 5).
    #   labels[2] = chunk[3] = 10
    #   labels[3] = chunk[4] = 20
    #   labels[4] = chunk[5] = EOT
    assert input_ids.tolist() == [1, 2, 3, 10, 20, SFT_EOT]
    assert labels[:2].tolist() == [-100, -100]
    assert labels[2:5].tolist() == [10, 20, SFT_EOT]
    assert labels[5].item() == -100


def test_sft_dataset_variable_length_per_sample(tmp_path):
    """Different rows can have different q_len / a_len; masking adapts."""
    rows = [
        ([1, 2], [3]),  # q_len=2, a_len=1
        ([10, 20, 30], [40, 50]),  # q_len=3, a_len=2
    ]
    path = tmp_path / "var.parquet"
    _write_sft_parquet(path, rows)
    ds = SFTDataset(
        parquet_path=str(path), seq_len=8, eot_token_id=SFT_EOT, pad_token_id=SFT_PAD
    )
    # Row 0: sample=[1, 2, 3, EOT], padded to 9. input_ids = [1, 2, 3, EOT, PAD, PAD, PAD, PAD].
    # Supervised labels[q_len-1 : q_len+a_len) = labels[1:3] = chunk[2:4] = [3, EOT].
    _, _, labels0 = ds[0]
    assert labels0[0].item() == -100
    assert labels0[1:3].tolist() == [3, SFT_EOT]
    assert (labels0[3:] == -100).all()

    # Row 1: sample=[10, 20, 30, 40, 50, EOT], padded to 9.
    # Supervised labels[2:5] = chunk[3:6] = [40, 50, EOT].
    _, _, labels1 = ds[1]
    assert labels1[:2].tolist() == [-100, -100]
    assert labels1[2:5].tolist() == [40, 50, SFT_EOT]
    assert (labels1[5:] == -100).all()


# --- SFTDataset packing=True ---
#
# Packing concatenates all samples into a flat stream and slices every
# seq_len tokens — same single-slice pattern as PretrainDataset(packing=True),
# with a parallel target-mask array to set -100 on question-position labels.


def test_sft_packing_flat_stream_length(tmp_path):
    """n_sequences = total tokens // seq_len - 1, like PretrainDataset packing."""
    # 3 samples × (2 q + 1 a + 1 EOT) = 12 flat tokens. seq_len=4 → n = 12//4 - 1 = 2.
    rows = [([1, 2], [3]), ([5, 6], [7]), ([9, 10], [11])]
    path = tmp_path / "pack_len.parquet"
    _write_sft_parquet(path, rows)
    ds = SFTDataset(
        parquet_path=str(path),
        seq_len=4,
        eot_token_id=SFT_EOT,
        pad_token_id=SFT_PAD,
        packing=True,
    )
    assert len(ds) == 2


def test_sft_packing_masks_question_positions(tmp_path):
    """In a packed chunk, labels at positions predicting question tokens are -100."""
    # Flat stream: [1, 2, 3, EOT, 5, 6, 7, EOT, 9, 10, 11, EOT].
    # is_target:   [F, F, T, T,   F, F, T, T,   F, F, T,  T]
    # seq_len=8 → chunk 0 spans flat[0:9] = [1, 2, 3, EOT, 5, 6, 7, EOT, 9].
    # input_ids = chunk[:-1] = [1, 2, 3, EOT, 5, 6, 7, EOT]  (length 8)
    # labels    = chunk[1:]  = [2, 3, EOT, 5, 6, 7, EOT, 9]
    # target_mask[1:9] = [F, T, T, F, F, T, T, F]
    # After mask: labels = [-100, 3, EOT, -100, -100, 7, EOT, -100]
    rows = [([1, 2], [3]), ([5, 6], [7]), ([9, 10], [11])]
    path = tmp_path / "pack_mask.parquet"
    _write_sft_parquet(path, rows)
    ds = SFTDataset(
        parquet_path=str(path),
        seq_len=8,
        eot_token_id=SFT_EOT,
        pad_token_id=SFT_PAD,
        packing=True,
    )
    input_ids, position_ids, labels = ds[0]
    assert input_ids.tolist() == [1, 2, 3, SFT_EOT, 5, 6, 7, SFT_EOT]
    assert labels.tolist() == [-100, 3, SFT_EOT, -100, -100, 7, SFT_EOT, -100]
    # Intra-doc position_ids reset after each EOT inside the chunk.
    assert position_ids.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]


def test_sft_packing_sample_can_span_chunk_boundary(tmp_path):
    """A long sample straddling chunk boundaries works — masking is per-token,
    not per-sample. Behavior matches PretrainDataset(packing=True) where
    documents may also span chunks."""
    # One big sample: q=[1,2,3,4,5], a=[6,7,8], + EOT → 9 flat tokens.
    # seq_len=4 → n = 9//4 - 1 = 1, chunk 0 = flat[0:5] = [1, 2, 3, 4, 5].
    # is_target[0:5] = [F, F, F, F, F]  (all question tokens — answer starts at flat[5]=6)
    # labels = [2, 3, 4, 5], target_mask[1:5] = [F, F, F, F]
    # → all labels become -100.
    rows = [([1, 2, 3, 4, 5], [6, 7, 8])]
    path = tmp_path / "pack_span.parquet"
    _write_sft_parquet(path, rows)
    ds = SFTDataset(
        parquet_path=str(path),
        seq_len=4,
        eot_token_id=SFT_EOT,
        pad_token_id=SFT_PAD,
        packing=True,
    )
    assert len(ds) == 1
    input_ids, _, labels = ds[0]
    assert input_ids.tolist() == [1, 2, 3, 4]
    assert (labels == -100).all()
