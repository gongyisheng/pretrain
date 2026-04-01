import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.masking_utils import build_position_ids


class PretrainDataset(Dataset):
    """Memory-mapped dataset for pretraining.

    Both modes return (input_ids, position_ids):

    packing=True (default): input_ids has seq_len+1 tokens from a flat packed
    stream. position_ids[i] is the intra-document position of input token i
    (resets to 0 after each EOT), used to build block-causal masks. All
    position_ids are >= 0.

    packing=False: single-document mode. input_ids is one document padded to
    seq_len+1 tokens. position_ids are sequential (0, 1, 2, ...) for real
    content tokens and -1 for padding tokens (those after the document-ending
    EOT). Callers derive the loss mask via ``position_ids >= 0``.

    In both modes the caller splits input_ids into model input and next-token
    targets via next_token_targets() in training/loss.py.
    """

    def __init__(
        self,
        bin_path: str,
        seq_len: int,
        packing: bool = True,
        eot_token_id: int = 0,
        pad_token_id: int = 0,
    ):
        self.seq_len = seq_len
        self.packing = packing
        self.pad_token_id = pad_token_id
        self.eot_token_id = eot_token_id
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")

        if packing:
            self.n_sequences = len(self.data) // seq_len - 1
        else:
            eot_positions = np.where(self.data == eot_token_id)[0]
            if len(eot_positions) == 0:
                # No EOT tokens — treat entire file as a single document
                starts = np.array([0], dtype=np.int64)
                ends = np.array([len(self.data)], dtype=np.int64)
            else:
                starts = np.concatenate([[0], eot_positions[:-1] + 1]).astype(np.int64)
                ends = (eot_positions + 1).astype(np.int64)
            # Drop degenerate docs (length <= 1 can't form even one (x, y) pair)
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
            # position_ids correspond to the input tokens (all but last)
            position_ids = build_position_ids(
                input_ids[:-1].unsqueeze(0), self.eot_token_id, packing=True
            ).squeeze(0)
            return input_ids, position_ids

        # Single-document mode
        doc_start = int(self._doc_starts[idx])
        doc_end = int(self._doc_ends[idx])
        doc_len = doc_end - doc_start

        # Take at most seq_len+1 tokens so the caller can form seq_len (x, y) pairs
        n_content = min(doc_len, self.seq_len + 1)
        doc = self.data[doc_start : doc_start + n_content].astype(np.int64)

        input_ids_arr = np.full(self.seq_len + 1, self.pad_token_id, dtype=np.int64)
        input_ids_arr[:n_content] = doc[:n_content]
        input_ids = torch.from_numpy(input_ids_arr)

        # position_ids for the input tokens: -1 marks padding positions
        position_ids = build_position_ids(
            input_ids[:-1].unsqueeze(0), self.eot_token_id, packing=False
        ).squeeze(0)

        return input_ids, position_ids
