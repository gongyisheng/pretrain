import pytest
import torch


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def cuda_capability_at_least(capability: tuple[int, int]) -> bool:
    return (
        torch.cuda.is_available() and torch.cuda.get_device_capability() >= capability
    )


cuda_sm89_or_newer = pytest.mark.skipif(
    not cuda_capability_at_least((8, 9)), reason="CUDA SM89 or newer required"
)

cuda_sm100_or_newer = pytest.mark.skipif(
    not cuda_capability_at_least((10, 0)), reason="CUDA SM100 or newer required"
)
