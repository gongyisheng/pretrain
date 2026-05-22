from abc import ABC, abstractmethod

import numpy as np
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
    """Memory-mapped .bin dataset for supervised fine-tuning.

    Reads two parallel flat streams produced by
    ``experiments/grokking/tokenize_data.py``:

    - ``<bin_path>``                   — tokens, ``uint16``/``uint32``.
      Format: ``[q0..., a0..., EOT, q1..., a1..., EOT, ...]``.
    - ``<bin_path stem>_targets.bin``  — parallel uint8 mask.
      1 at positions whose token is an answer or EOT (a supervised target);
      0 at question positions.

    Two modes (mirror ``PretrainDataset``):

    - ``packing=True``: slice every ``seq_len`` tokens. The targets mask is
      sliced in lockstep and used to set ``-100`` on label positions whose
      target token isn't an answer/EOT. Single one-line ``__getitem__``.
    - ``packing=False``: each returned sample is one inter-EOT document,
      padded to ``seq_len + 1``. Labels for question positions become ``-100``
      via the same targets mask; padding positions also become ``-100``.
    """

    def __init__(
        self,
        bin_path: str,
        seq_len: int,
        vocab_size: int,
        packing: bool = False,
        eot_token_id: int = 0,
        pad_token_id: int = 0,
    ):
        self.seq_len = seq_len
        self.eot_token_id = eot_token_id
        self.pad_token_id = pad_token_id
        self.packing = packing
        dtype = np.uint32 if vocab_size > 65535 else np.uint16
        self.data = np.memmap(bin_path, dtype=dtype, mode="r")
        targets_path = bin_path.removesuffix(".bin") + "_targets.bin"
        self.targets = np.memmap(targets_path, dtype=np.uint8, mode="r")
        if len(self.data) != len(self.targets):
            raise ValueError(
                f"SFTDataset: token stream ({len(self.data)}) and targets mask "
                f"({len(self.targets)}) have different lengths"
            )

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
            tgt = self.targets[start : start + self.seq_len + 1]
            chunk_t = torch.from_numpy(chunk)
            target_mask = torch.from_numpy(tgt.astype(np.bool_))
            input_ids = chunk_t[:-1]
            labels = chunk_t[1:].clone()
            # labels[i] predicts chunk[i+1]; supervise iff target_mask[i+1].
            labels[~target_mask[1:]] = -100
            position_ids = build_position_ids(
                input_ids.unsqueeze(0), self.eot_token_id, packing=True
            ).squeeze(0)
            return input_ids, position_ids, labels

        # packing=False: one inter-EOT document → one padded sample.
        doc_start = int(self._doc_starts[idx])
        doc_end = int(self._doc_ends[idx])
        n_content = min(doc_end - doc_start, self.seq_len + 1)
        doc = self.data[doc_start : doc_start + n_content].astype(np.int64)
        tgt = self.targets[doc_start : doc_start + n_content]

        chunk_arr = np.full(self.seq_len + 1, self.pad_token_id, dtype=np.int64)
        chunk_arr[:n_content] = doc[:n_content]
        tgt_arr = np.zeros(self.seq_len + 1, dtype=np.bool_)
        tgt_arr[:n_content] = tgt.astype(np.bool_)
        chunk_t = torch.from_numpy(chunk_arr)
        target_mask = torch.from_numpy(tgt_arr)
        input_ids = chunk_t[:-1]
        labels = chunk_t[1:].clone()
        # Mask positions whose target token isn't an answer/EOT (questions + padding).
        labels[~target_mask[1:]] = -100
        position_ids = build_position_ids(
            input_ids.unsqueeze(0), self.eot_token_id, packing=False
        ).squeeze(0)
        return input_ids, position_ids, labels
