import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.masking_utils import build_position_ids


class PretrainDataset(Dataset):
    """Memory-mapped dataset for pretraining.

    Returns (input_ids, position_ids, labels).

    - ``input_ids``: shape (seq_len + 1,). The model input is ``input_ids[:-1]``.
    - ``position_ids``: shape (seq_len,). Intra-document positions of the input
      tokens. In packing=False mode, negative entries mark padding.
    - ``labels``: shape (seq_len,). Next-token targets aligned with the model
      input. Positions whose corresponding input token is padding (i.e.
      ``position_ids[i] < 0``) are set to -100 so the loss skips them via the
      cross-entropy ``ignore_index=-100`` convention. In packing=True mode no
      -100s appear because there's no padding.
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
        self.packing = packing
        self.pad_token_id = pad_token_id
        self.eot_token_id = eot_token_id
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
            input_ids = torch.from_numpy(chunk)
            position_ids = build_position_ids(
                input_ids[:-1].unsqueeze(0), self.eot_token_id, packing=True
            ).squeeze(0)
            labels = input_ids[1:].clone()
            return input_ids, position_ids, labels

        doc_start = int(self._doc_starts[idx])
        doc_end = int(self._doc_ends[idx])
        doc_len = doc_end - doc_start
        n_content = min(doc_len, self.seq_len + 1)
        doc = self.data[doc_start : doc_start + n_content].astype(np.int64)

        input_ids_arr = np.full(self.seq_len + 1, self.pad_token_id, dtype=np.int64)
        input_ids_arr[:n_content] = doc[:n_content]
        input_ids = torch.from_numpy(input_ids_arr)

        position_ids = build_position_ids(
            input_ids[:-1].unsqueeze(0), self.eot_token_id, packing=False
        ).squeeze(0)

        # labels are the next-token targets for the input tokens (x = input_ids[:-1]).
        # Wherever x is padding (position_ids[i] < 0), the loss should be skipped
        # → set labels[i] = -100 via the cross-entropy ignore_index convention.
        labels = input_ids[1:].clone()
        labels[position_ids < 0] = -100
        return input_ids, position_ids, labels


class SFTDataset(Dataset):
    """Memory-mapped dataset for supervised fine-tuning.

    Each on-disk sample is ``question_len + answer_len`` tokens laid out as
    ``[question_tokens..., answer_tokens...]``. The dataset assembles the full
    sequence in code and constructs the ``labels`` tensor with the HF -100
    ignore-index convention: only the last ``answer_len`` label positions
    carry the answer tokens; all earlier positions are -100 and skipped by the
    loss.

    Returns ``(input_ids, position_ids, labels)`` with shapes
    ``(stride,), (stride - 1,), (stride - 1,)`` where ``stride = question_len +
    answer_len``.
    """

    def __init__(
        self,
        bin_path: str,
        vocab_size: int,
        question_len: int,
        answer_len: int,
    ):
        self.question_len = question_len
        self.answer_len = answer_len
        self.stride = question_len + answer_len
        dtype = np.uint32 if vocab_size > 65535 else np.uint16
        self.data = np.memmap(bin_path, dtype=dtype, mode="r")
        if len(self.data) % self.stride != 0:
            raise ValueError(
                f"SFTDataset: bin file length {len(self.data)} is not a multiple "
                f"of stride {self.stride} (question_len={question_len} + "
                f"answer_len={answer_len})"
            )
        self.n_samples = len(self.data) // self.stride

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.data[start : start + self.stride].astype(np.int64)
        input_ids = torch.from_numpy(chunk)
        position_ids = torch.arange(self.stride - 1, dtype=torch.long)
        labels = torch.full((self.stride - 1,), -100, dtype=torch.long)
        # Last `answer_len` label positions hold the answer tokens.
        labels[-self.answer_len :] = input_ids[-self.answer_len :]
        return input_ids, position_ids, labels
