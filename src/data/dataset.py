from abc import ABC, abstractmethod

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from src.utils.masking_utils import build_position_ids


class _AbstractDataset(Dataset, ABC):
    """Abstract base for token datasets used by the trainer.

    Concrete subclasses return ``(input_ids, position_ids, labels)`` from
    ``__getitem__``, all tensors of shape ``(seq_len,)``:

    - ``input_ids``: model input, with the next-token shift already applied.
    - ``position_ids``: intra-document positions; negative values mark padding.
    - ``labels``: next-token targets, with ``-100`` at positions to skip
      (consumed by ``F.cross_entropy(..., ignore_index=-100)``).

    Storage formats and constructor arguments are subclass-specific.
    """

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx): ...


class PretrainDataset(_AbstractDataset):
    """Memory-mapped .bin dataset for pretraining.

    Two modes:

    - ``packing=True`` (default): the on-disk stream is sliced every ``seq_len``
      tokens, so a single returned chunk may span multiple documents joined by
      ``eot_token_id``. ``labels`` has no ``-100``s — every position contributes
      to the loss.
    - ``packing=False``: each returned sample is one document (delimited by
      ``eot_token_id`` in the stream), padded with ``pad_token_id`` to
      ``seq_len + 1`` tokens. ``labels`` is set to ``-100`` at padding
      positions so the loss skips them.
    """

    def __init__(
        self,
        bin_path: str,
        seq_len: int,
        vocab_size: int,
        packing: bool = True,
        eot_token_id: int = 0,
        pad_token_id: int = 0,
    ):
        self.seq_len = seq_len
        self.eot_token_id = eot_token_id
        self.pad_token_id = pad_token_id
        self.packing = packing
        dtype = np.uint32 if vocab_size > 65535 else np.uint16
        self.data = np.memmap(bin_path, dtype=dtype, mode="r")

        if packing:
            self.n_sequences = len(self.data) // seq_len - 1
        else:
            eot_positions = np.where(self.data == eot_token_id)[0]
            if len(eot_positions) == 0:
                starts = np.array([0], dtype=np.int64)
                ends = np.array([len(self.data)], dtype=np.int64)
            else:
                starts = np.concatenate([[0], eot_positions[:-1] + 1]).astype(np.int64)
                ends = (eot_positions + 1).astype(np.int64)
            lengths = ends - starts
            valid = lengths > 1
            self._doc_starts = starts[valid]
            self._doc_ends = ends[valid]

    def __len__(self):
        if self.packing:
            return self.n_sequences
        return len(self._doc_starts)

    def __getitem__(self, idx):
        if self.packing:
            start = idx * self.seq_len
            chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
            chunk_t = torch.from_numpy(chunk)
            input_ids = chunk_t[:-1]
            labels = chunk_t[1:].clone()
            position_ids = build_position_ids(
                input_ids.unsqueeze(0), self.eot_token_id, packing=True
            ).squeeze(0)
            return input_ids, position_ids, labels

        doc_start = int(self._doc_starts[idx])
        doc_end = int(self._doc_ends[idx])
        doc_len = doc_end - doc_start
        n_content = min(doc_len, self.seq_len + 1)
        doc = self.data[doc_start : doc_start + n_content].astype(np.int64)

        chunk_arr = np.full(self.seq_len + 1, self.pad_token_id, dtype=np.int64)
        chunk_arr[:n_content] = doc[:n_content]
        chunk_t = torch.from_numpy(chunk_arr)
        input_ids = chunk_t[:-1]
        labels = chunk_t[1:].clone()

        position_ids = build_position_ids(
            input_ids.unsqueeze(0), self.eot_token_id, packing=False
        ).squeeze(0)

        # Wherever input_ids[i] is padding (position_ids[i] < 0), the loss
        # should be skipped → set labels[i] = -100 via the cross-entropy
        # ignore_index convention.
        labels[position_ids < 0] = -100
        return input_ids, position_ids, labels


class SFTDataset(_AbstractDataset):
    """Parquet-backed dataset for supervised fine-tuning.

    Reads a tokenized Parquet file with two list-of-int columns produced by
    ``experiments/grokking/tokenize_data.py``:

    - ``question_ids``: tokens of the question (prompt). Length varies per row.
    - ``answer_ids``:   tokens of the answer. Length varies per row.

    Per-sample, the dataset constructs the full sequence
    ``[q_ids..., a_ids..., EOT]``, pads it with ``pad_token_id`` to
    ``seq_len + 1`` tokens, and applies the next-token shift. Labels carry
    ``-100`` everywhere except at the positions that predict the answer tokens
    and the EOT itself, so the loss is computed only on the answer (and the
    stop-token prediction).
    """

    def __init__(
        self,
        parquet_path: str,
        seq_len: int,
        eot_token_id: int = 0,
        pad_token_id: int = 0,
    ):
        self.seq_len = seq_len
        self.eot_token_id = eot_token_id
        self.pad_token_id = pad_token_id
        table = pq.read_table(parquet_path)
        self._question_ids = table.column("question_ids").to_pylist()
        self._answer_ids = table.column("answer_ids").to_pylist()

    def __len__(self):
        return len(self._question_ids)

    def __getitem__(self, idx):
        q = self._question_ids[idx]
        a = self._answer_ids[idx]
        q_len = len(q)
        a_len = len(a)
        # Full on-the-wire sample: [q..., a..., EOT].
        sample = q + a + [self.eot_token_id]
        n_content = min(len(sample), self.seq_len + 1)

        chunk_arr = np.full(self.seq_len + 1, self.pad_token_id, dtype=np.int64)
        chunk_arr[:n_content] = sample[:n_content]
        chunk_t = torch.from_numpy(chunk_arr)
        input_ids = chunk_t[:-1]

        # Labels: -100 everywhere except the answer-prediction range.
        # Supervised label indices: [q_len - 1, q_len + a_len) = predicting
        # each answer token plus the EOT.
        labels = torch.full((self.seq_len,), -100, dtype=torch.long)
        sup_lo = q_len - 1
        sup_hi = min(q_len + a_len, self.seq_len)
        if 0 <= sup_lo < sup_hi:
            labels[sup_lo:sup_hi] = chunk_t[sup_lo + 1 : sup_hi + 1]

        position_ids = build_position_ids(
            input_ids.unsqueeze(0), self.eot_token_id, packing=False
        ).squeeze(0)
        return input_ids, position_ids, labels
