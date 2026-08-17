import pytest
import torch

from src.kernel.ops.gemm import int8_scaled_grouped_mm, grouped_mm

E, M, K, N = 4, 32, 64, 48


def _offs():
    return torch.arange(1, E + 1, dtype=torch.int32) * (M // E)


def _scaled_args(**overrides):
    args = {
        "aq": torch.zeros(M, K, dtype=torch.int8),
        "bq": torch.zeros(E, K, N, dtype=torch.int8),
        "sa": torch.ones(M, 1),
        "sb": torch.ones(E, 1, N),
        "offs": _offs(),
        "out_dtype": torch.bfloat16,
        "block_size": 0,
    }
    args.update(overrides)
    return args


def test_non_contiguous_offs_is_rejected():
    wide = torch.zeros(E * 2, dtype=torch.int32)
    wide[::2] = _offs()

    with pytest.raises(ValueError, match="offs must be contiguous"):
        int8_scaled_grouped_mm(**_scaled_args(offs=wide[::2]))


def test_non_contiguous_scale_is_rejected():
    wide = torch.ones(M, 2)

    with pytest.raises(ValueError, match="scales must be contiguous"):
        int8_scaled_grouped_mm(**_scaled_args(sa=wide[:, :1]))


def test_wrong_scale_block_count_is_rejected():
    with pytest.raises(ValueError, match="scale block count"):
        int8_scaled_grouped_mm(**_scaled_args(sa=torch.ones(M, 4), block_size=32))


def test_contraction_mismatch_is_rejected():
    with pytest.raises(ValueError, match="contraction mismatch"):
        int8_scaled_grouped_mm(
            **_scaled_args(bq=torch.zeros(E, K + 32, N, dtype=torch.int8))
        )


def test_grouped_mm_rejects_non_contiguous_offs():
    wide = torch.zeros(E * 2, dtype=torch.int32)
    wide[::2] = _offs()

    with pytest.raises(ValueError, match="offs must be contiguous"):
        grouped_mm(torch.zeros(M, K), torch.zeros(E, K, N), wide[::2])


def test_valid_arguments_reach_the_kernel():
    out = int8_scaled_grouped_mm(**_scaled_args())

    assert out.shape == (M, N)
