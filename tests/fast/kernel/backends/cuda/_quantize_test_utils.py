"""CUDA quantizer test helpers."""

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch


_LARGE_OFFSET = 1 << 31
_MINIMUM_FREE_BYTES = 6 * (1 << 30)
_LOCK_PATH = "/tmp/pretrain-quantize-large-offset.lock"
LARGE_OFFSET_CASES = (
    ((2, 32), (_LARGE_OFFSET - 32, 1)),
    ((2, 32), (_LARGE_OFFSET - 31, 1)),
    ((2, 32), (_LARGE_OFFSET, 1)),
    ((3, 32), (1 << 30, 1)),
    ((2, 1, 32), (_LARGE_OFFSET, 32, 1)),
)


@contextmanager
def large_offset_source(
    shape: tuple[int, ...], strides: tuple[int, ...]
) -> Iterator[torch.Tensor]:
    """Yield one view at or beyond the int32 address boundary.

    Set PRETRAIN_RUN_LARGE_OFFSET_TESTS=1 and use -n 0 to run these 4 GiB cases.
    """
    if os.environ.get("PRETRAIN_RUN_LARGE_OFFSET_TESTS") != "1":
        pytest.skip("set PRETRAIN_RUN_LARGE_OFFSET_TESTS=1 for 4 GiB offset coverage")
    with open(_LOCK_PATH, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        torch.cuda.empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < _MINIMUM_FREE_BYTES:
            pytest.skip("insufficient free CUDA memory for 4 GiB offset coverage")
        storage = torch.empty(_LARGE_OFFSET + 32, dtype=torch.bfloat16, device="cuda")
        source = storage.as_strided(shape, strides)
        try:
            yield source
        finally:
            del source
            del storage
            torch.cuda.empty_cache()
