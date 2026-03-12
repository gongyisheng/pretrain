import pytest
import numpy as np
import os
import tempfile
import torch
from src.data.dataset import PretrainDataset


@pytest.fixture
def tmp_bin_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.bin")
        tokens = np.arange(1024, dtype=np.uint16)
        tokens.tofile(path)
        yield path


def test_dataset_length(tmp_bin_file):
    ds = PretrainDataset(tmp_bin_file, seq_len=128)
    assert len(ds) == 1024 // 128 - 1  # 7


def test_dataset_getitem_shape(tmp_bin_file):
    ds = PretrainDataset(tmp_bin_file, seq_len=128)
    x, y = ds[0]
    assert x.shape == (128,)
    assert y.shape == (128,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_dataset_getitem_shift(tmp_bin_file):
    ds = PretrainDataset(tmp_bin_file, seq_len=128)
    x, y = ds[0]
    assert torch.equal(y[:-1], x[1:])
