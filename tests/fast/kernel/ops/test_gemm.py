import pytest
import torch

from src.kernel.ops.gemm import (
    fp8_scaled_grouped_mm,
    fp8_scaled_mm,
    grouped_mm,
    int8_scaled_grouped_mm,
    int8_scaled_mm,
    mxfp8_scaled_grouped_mm,
    mxfp8_scaled_mm,
)

E, M, K, N = 4, 32, 64, 48

SCALED_MM_OPS = [int8_scaled_mm, fp8_scaled_mm, mxfp8_scaled_mm]
SCALED_GROUPED_MM_OPS = [
    int8_scaled_grouped_mm,
    fp8_scaled_grouped_mm,
    mxfp8_scaled_grouped_mm,
]
OP_IDS = ["int8", "fp8", "mxfp8"]


def _offs(groups=E, stride=M // E):
    return torch.arange(1, groups + 1, dtype=torch.int32) * stride


def _non_contiguous_offs():
    wide = torch.zeros(E * 2, dtype=torch.int32)
    wide[::2] = _offs()
    return wide[::2]


def _mm_args(**overrides):
    args = {
        "aq": torch.zeros(M, K, dtype=torch.int8),
        "bq": torch.zeros(K, N, dtype=torch.int8),
        "sa": torch.ones(M, 1),
        "sb": torch.ones(1, N),
        "out_dtype": torch.bfloat16,
        "block_size": 0,
    }
    args.update(overrides)
    return args


def _ragged_m_args(**overrides):
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


def _ragged_n_args(**overrides):
    args = {
        "aq": torch.zeros(E, M, K, dtype=torch.int8),
        "bq": torch.zeros(K, N, dtype=torch.int8),
        "sa": torch.ones(E, M, 1),
        "sb": torch.ones(1, N),
        "offs": _offs(stride=N // E),
        "out_dtype": torch.bfloat16,
        "block_size": 0,
    }
    args.update(overrides)
    return args


def _ragged_k_args(**overrides):
    # 2D x 2D: the contraction is ragged, so sa/sb carry one block per group
    # rather than one sized against the dense K count.
    args = {
        "aq": torch.zeros(M, K, dtype=torch.int8),
        "bq": torch.zeros(K, N, dtype=torch.int8),
        "sa": torch.ones(M, E),
        "sb": torch.ones(E, N),
        "offs": _offs(stride=K // E),
        "out_dtype": torch.bfloat16,
        "block_size": 0,
    }
    args.update(overrides)
    return args


SCALED_MM_ERROR_CASES = [
    pytest.param(
        _mm_args(bq=torch.zeros(K + 32, N, dtype=torch.int8)),
        "contraction mismatch",
        id="contraction",
    ),
    pytest.param(
        _mm_args(sa=torch.ones(M, 4), sb=torch.ones(2, N), block_size=32),
        "sa block count",
        id="sa_blocks",
    ),
    pytest.param(
        _mm_args(sa=torch.ones(M, 2), sb=torch.ones(4, N), block_size=32),
        "sb block count",
        id="sb_blocks",
    ),
]

SCALED_GROUPED_MM_ERROR_CASES = [
    pytest.param(
        _ragged_m_args(offs=_non_contiguous_offs()),
        "offs must be contiguous",
        id="offs_non_contiguous",
    ),
    pytest.param(
        _ragged_m_args(bq=torch.zeros(E, K + 32, N, dtype=torch.int8)),
        "contraction mismatch",
        id="contraction",
    ),
    pytest.param(
        _ragged_m_args(sa=torch.ones(M, 4), block_size=32),
        "sa block count",
        id="ragged_m_sa_blocks",
    ),
    pytest.param(
        _ragged_m_args(sb=torch.ones(E, 1, N + 1)),
        "sb block count",
        id="ragged_m_sb_shape",
    ),
    pytest.param(
        _ragged_n_args(sa=torch.ones(E, M, 2)),
        "sa block count",
        id="ragged_n_sa_shape",
    ),
    pytest.param(
        _ragged_n_args(sb=torch.ones(1, N + 1)),
        "sb block count",
        id="ragged_n_sb_shape",
    ),
    pytest.param(
        _ragged_k_args(sa=torch.ones(M + 1, E)),
        "sa rows",
        id="ragged_k_sa_rows",
    ),
    pytest.param(
        _ragged_k_args(sb=torch.ones(E, N + 1)),
        "sb cols",
        id="ragged_k_sb_cols",
    ),
    pytest.param(
        _ragged_k_args(sa=torch.ones(M, E + 1)),
        "sa block count",
        id="ragged_k_scale_pair",
    ),
    pytest.param(
        # block_size 0 means one scale per group, so blocks must equal offs groups.
        _ragged_k_args(sa=torch.ones(M, E + 1), sb=torch.ones(E + 1, N)),
        "offs groups",
        id="ragged_k_offs_groups",
    ),
]

GROUPED_MM_ERROR_CASES = [
    pytest.param(
        (torch.zeros(M, K), torch.zeros(E, K, N), _non_contiguous_offs()),
        "offs must be contiguous",
        id="offs_non_contiguous",
    ),
    pytest.param(
        (torch.zeros(M, K), torch.zeros(E, K + 32, N), _offs()),
        "contraction mismatch",
        id="contraction",
    ),
]


@pytest.mark.parametrize(("args", "match"), GROUPED_MM_ERROR_CASES)
def test_grouped_mm_raise_error(args, match):
    with pytest.raises(ValueError, match=match):
        grouped_mm(*args)


@pytest.mark.parametrize("op", SCALED_MM_OPS, ids=OP_IDS)
@pytest.mark.parametrize(("args", "match"), SCALED_MM_ERROR_CASES)
def test_scaled_mm_raise_error(op, args, match):
    with pytest.raises(ValueError, match=match):
        op(**args)


@pytest.mark.parametrize("op", SCALED_GROUPED_MM_OPS, ids=OP_IDS)
@pytest.mark.parametrize(("args", "match"), SCALED_GROUPED_MM_ERROR_CASES)
def test_scaled_grouped_mm_raise_error(op, args, match):
    with pytest.raises(ValueError, match=match):
        op(**args)
