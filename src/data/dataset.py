import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.masking_utils import build_position_ids


class PretrainDataset(Dataset):
    """Memory-mapped dataset for pretraining.

    packing=True (default): returns fixed-length chunks from a flat token stream
    (multiple documents packed per sequence). Returns (tokens, position_ids) where
    tokens has length seq_len+1 and position_ids[i] is the intra-document position
    of token i (resets to 0 after each EOT), used to build block-causal masks.

    packing=False: single-document mode — one document per sample, padded to
    seq_len+1 tokens. Returns (tokens, loss_mask) where loss_mask (length seq_len)
    is True for valid (non-pad) positions. The EOT token at the end of each
    document IS included in the loss.

    In both modes the caller is responsible for splitting tokens into model input
    and next-token targets via next_token_targets() in training/loss.py.

    Note: pad_token_id fills token positions beyond the document boundary.
    Because EOT doubles as pad (no dedicated pad token), loss_mask is tracked
    explicitly so EOT predictions inside content are always included in the loss.
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
            tokens = torch.from_numpy(chunk)
            # position_ids correspond to the input (all but last token)
            position_ids = build_position_ids(tokens[:-1].unsqueeze(0), self.eot_token_id).squeeze(0)
            return tokens, position_ids

        # Single-document mode
        doc_start = int(self._doc_starts[idx])
        doc_end = int(self._doc_ends[idx])
        doc_len = doc_end - doc_start

        # Take at most seq_len+1 tokens so the caller can form seq_len (x, y) pairs
        n_content = min(doc_len, self.seq_len + 1)
        doc = self.data[doc_start : doc_start + n_content].astype(np.int64)

        tokens_arr = np.full(self.seq_len + 1, self.pad_token_id, dtype=np.int64)
        tokens_arr[:n_content] = doc[:n_content]

        # loss_mask covers positions where y = tokens[1:] is a real target token
        n_valid = n_content - 1
        loss_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        loss_mask[:n_valid] = True

        return torch.from_numpy(tokens_arr), loss_mask
