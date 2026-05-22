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
