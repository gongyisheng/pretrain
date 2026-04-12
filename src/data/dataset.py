import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """Memory-mapped dataset that reads fixed-length chunks from a .bin file.

    Each sample returns (x, y) where y = x shifted by 1 token (next-token prediction).
    """

    def __init__(self, bin_path: str, seq_len: int, vocab_size: int = 65535):
        self.seq_len = seq_len
        dtype = np.uint32 if vocab_size > 65535 else np.uint16
        self.data = np.memmap(bin_path, dtype=dtype, mode="r")
        self.n_sequences = len(self.data) // seq_len - 1

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y
