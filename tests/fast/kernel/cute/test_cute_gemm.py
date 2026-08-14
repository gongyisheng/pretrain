"""CuTe DSL mxfp8 kernels (sm120). Numerics are checked against the same
oracle the Triton backend uses, so the two cannot silently diverge."""

import pytest
import torch

from src.quant.quantize import quantize_operand
from tests.fast.kernel._refs import scaled_gemm_ref

MX = {"granularity": "blockwise", "block_size": 32, "scale_dtype": "fp8_e8m0"}

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (12, 0),
    reason="CuTe mxfp8 kernels are sm120-only",
)


def _operands(M, K, N, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    aq, sa = quantize_operand(a, -1, "fp8_e4m3", MX)
    bq, sb = quantize_operand(b, -2, "fp8_e4m3", MX)
    return aq, bq, sa, sb


def _operands_spread_scales(M, K, N, seed=0):
    """Operands whose per-scale-block magnitudes span 2^-8..2^7.

    A's magnitude varies per (row, K-block) and B's per (K-block, column), so both
    axes of *both* scale-factor fragment mappings carry signal. Uniform magnitudes
    would leave SFB's N axis untested, which is the half of the mapping that does not
    mirror SFA -- it reads a different thread coordinate and regroups its modes.
    """
    torch.manual_seed(seed)
    n_blocks = K // 32
    exp = torch.randint(-8, 8, (M, n_blocks), device="cuda").float()
    a = torch.randn(M, n_blocks, 32, device="cuda") * torch.exp2(exp)[:, :, None]
    exp = torch.randint(-8, 8, (n_blocks, N), device="cuda").float()
    b = torch.randn(n_blocks, 32, N, device="cuda") * torch.exp2(exp)[:, None, :]
    aq, sa = quantize_operand(a.reshape(M, K).bfloat16(), -1, "fp8_e4m3", MX)
    bq, sb = quantize_operand(b.reshape(K, N).bfloat16(), -2, "fp8_e4m3", MX)
    return aq, bq, sa, sb


@pytest.mark.parametrize("shape", [(128, 128, 128), (256, 128, 256)])
def test_scaled_gemm_mxfp8_matches_oracle(shape):
    from src.kernel.cute.gemm import scaled_gemm_mxfp8

    M, K, N = shape
    aq, bq, sa, sb = _operands(M, K, N)
    got = scaled_gemm_mxfp8(aq, bq, sa, sb, torch.float32)
    want = scaled_gemm_ref(aq, bq, sa, sb, 32)

    assert got.shape == (M, N)
    assert got.dtype == torch.float32
    # relative Frobenius norm, matching the bar the Triton mxfp8 kernel is held to
    # in tests/fast/kernel/triton/test_triton_gemm.py::test_scaled_gemm_mx_matches_oracle.
    # Elementwise tolerances give spurious failures on block-quantized outputs.
    assert (got - want).norm() / want.norm() < 0.02


@pytest.mark.parametrize(
    "shape",
    [
        (256, 512, 128),
        (128, 256, 256),
        (384, 128, 128),
        (192, 160, 128),  # short K tile, and a partial M tile (192 = 128 + 64)
        (129, 128, 130),  # partial M and N tiles
    ],
)
def test_scaled_gemm_mxfp8_adds_no_error_over_dequantizing(shape):
    """Applying the e8m0 scales inside the MMA must be as exact as dequantizing first.

    The last two shapes are predicated: one leaves a short K tile, the other partial M
    and N tiles. Predication is part of the scale feed -- each scale tensor is masked
    against its own extent, since a ragged M or N makes the scale loads partial too --
    so it belongs under the tight bound rather than only under the 2e-2 oracle checks,
    where per-block magnitudes barely differ and a misfed scale has somewhere to hide.

    The bound is fp32-rounding tight (measured 1.0e-7..2.9e-7) on purpose. It sits an
    order of magnitude above the Triton test's 2e-8 because these operands span 2^15
    of per-block magnitude, and summing terms of such different size costs fp32
    precision -- that spread is the point, since it denies a misplaced scale anywhere
    to hide.

    What the tight bound buys over the 2e-2 oracle comparison in
    `test_scaled_gemm_mxfp8_matches_oracle` is precision, not placement. Verified by
    mutation: rounding the result path through bf16 -- standing in for scales or
    accumulation applied at less than fp32 -- passes every 2e-2 assertion in this file
    and fails every shape here. Placement is caught either way, because an e8m0
    scale is a power of two, so landing one on the wrong row, column or k-tile is at
    least a factor-2 error and never sits inside the quantization noise.

    This is the assertion that pins the instruction's arithmetic, and the one the
    pipelining work -- which rewrites which k-tile's scales are resident when the MMA
    fires -- must keep green. It is the CuTe counterpart of the Triton backend's
    `test_scaled_gemm_mxfp8_adds_no_error_over_dequantizing`.
    """
    from src.kernel.cute.gemm import scaled_gemm_mxfp8

    M, K, N = shape
    aq, bq, sa, sb = _operands_spread_scales(M, K, N)
    got = scaled_gemm_mxfp8(aq, bq, sa, sb, torch.float32)
    oracle = scaled_gemm_ref(aq, bq, sa, sb, 32)
    rel = (got - oracle).norm() / oracle.norm()
    assert rel < 1e-6, rel


def test_compile_cache_reused():
    """M is dynamic in the compiled kernel, so a batch-size change must not recompile."""
    from src.kernel.cute import gemm as cute_gemm

    aq, bq, sa, sb = _operands(128, 128, 128)
    cute_gemm.scaled_gemm_mxfp8(aq, bq, sa, sb, torch.float32)
    n_after_first = len(cute_gemm._COMPILED)
    # cache key mirrors scaled_gemm_mxfp8's: (K, N, a_dtype, b_dtype, out_dtype)
    key = (128, 128, aq.dtype, bq.dtype, torch.float32)
    compiled_after_first = cute_gemm._COMPILED[key]

    aq2, bq2, sa2, sb2 = _operands(256, 128, 128, seed=1)  # different M, same K/N
    got = cute_gemm.scaled_gemm_mxfp8(aq2, bq2, sa2, sb2, torch.float32)

    # without this, dropping the store into _COMPILED would leave both checks below
    # passing while every call silently re-JITs -- the exact cost the cache prevents
    assert n_after_first >= 1, "the first call must have cached a compiled kernel"
    assert len(cute_gemm._COMPILED) == n_after_first, "M must not trigger a recompile"
    # a mutation that stores into _COMPILED but never looks up would keep the length
    # check above green while re-JITting on every call -- catch that by requiring the
    # second call to have left the *same* compiled object in place, not an equal one
    assert cute_gemm._COMPILED[key] is compiled_after_first, (
        "the second call must reuse the cached compiled kernel, not recompile it"
    )
    # a cached kernel with M baked static would satisfy the count check above while
    # returning garbage here, so assert the reused kernel is actually correct
    want = scaled_gemm_ref(aq2, bq2, sa2, sb2, 32)
    assert (got - want).norm() / want.norm() < 0.02


def _operands_fmt(M, K, N, a_fmt, b_fmt, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    aq, sa = quantize_operand(a, -1, a_fmt, MX)
    bq, sb = quantize_operand(b, -2, b_fmt, MX)
    return aq, bq, sa, sb


@pytest.mark.parametrize(
    "a_fmt,b_fmt",
    [("fp8_e4m3", "fp8_e4m3"), ("fp8_e4m3", "fp8_e5m2"), ("fp8_e5m2", "fp8_e4m3")],
)
def test_scaled_gemm_mxfp8_dtype_combinations(a_fmt, b_fmt):
    from src.kernel.cute.gemm import scaled_gemm_mxfp8

    aq, bq, sa, sb = _operands_fmt(128, 128, 128, a_fmt, b_fmt)
    got = scaled_gemm_mxfp8(aq, bq, sa, sb, torch.float32)
    want = scaled_gemm_ref(aq, bq, sa, sb, 32)
    assert (got - want).norm() / want.norm() < 0.02


@pytest.mark.parametrize(
    "shape,out_dtype",
    [
        ((128, 128, 128), torch.float32),
        ((128, 128, 128), torch.float16),
        ((128, 128, 128), torch.bfloat16),
        # crossing the two axes: the store converts and predicates in the same pass,
        # and it has to mask per element, since an odd N splits the C fragment's
        # column pair -- on an aligned shape that mask is always true
        ((129, 160, 130), torch.bfloat16),
    ],
)
def test_scaled_gemm_mxfp8_preserves_out_dtype(shape, out_dtype):
    from src.kernel.cute.gemm import scaled_gemm_mxfp8

    aq, bq, sa, sb = _operands(*shape)
    got = scaled_gemm_mxfp8(aq, bq, sa, sb, out_dtype)
    want = scaled_gemm_ref(aq, bq, sa, sb, 32)

    assert got.dtype == out_dtype
    # fp16/bf16 out costs precision on top of the mxfp8 quantization, so this bar is
    # looser than the fp32 case above
    assert (got.float() - want.float()).norm() / want.float().norm() < 0.05


@pytest.mark.parametrize(
    "shape",
    [
        (64, 64, 64),
        (192, 160, 128),
        (129, 128, 130),
        # N=33 is the only shape here with M aligned, and the only one narrow enough to
        # drive B's copy down to one byte per thread -- a different thread layout
        (256, 128, 33),
    ],
)
def test_scaled_gemm_mxfp8_ragged_shapes(shape):
    from src.kernel.cute.gemm import scaled_gemm_mxfp8

    M, K, N = shape
    aq, bq, sa, sb = _operands(M, K, N)
    got = scaled_gemm_mxfp8(aq, bq, sa, sb, torch.float32)
    want = scaled_gemm_ref(aq, bq, sa, sb, 32)

    assert got.shape == (M, N)
    assert (got - want).norm() / want.norm() < 0.02


def _to_blocked(scale_2d):
    """Rearrange an (rows, n_blocks) e8m0 scale into cuBLASLt's SWIZZLE_32_4_4 layout."""
    rows, cols = scale_2d.shape
    n_row_tiles, n_col_tiles = (rows + 127) // 128, (cols + 3) // 4
    padded_rows, padded_cols = n_row_tiles * 128, n_col_tiles * 4
    padded = scale_2d
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols), device=scale_2d.device, dtype=scale_2d.dtype
        )
        padded[:rows, :cols] = scale_2d
    blocks = padded.view(n_row_tiles, 128, n_col_tiles, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


@pytest.mark.parametrize("shape", [(256, 512, 128), (2048, 512, 512), (128, 256, 256)])
def test_scaled_gemm_mxfp8_agrees_with_scaled_mm(shape):
    """Cross-check against the vendor kernel that runs the same MMA instruction.

    cuBLASLt's sm120 block-scaled kernel and ours issue the identical m16n8k32
    block-scaled MMA, so the only thing that can differ is the feed path: which
    operand element meets which scale in the tensor core. That makes this the
    designated feed-path oracle, and the reason it is worth having on top of the
    `_refs` comparison -- `_refs` proves the arithmetic, this proves the plumbing
    against an independent implementation of the same plumbing.

    Shapes are tile-aligned because cuBLASLt requires it; the ragged cases stay on
    the `_refs` oracle.
    """
    from src.kernel.cute.gemm import scaled_gemm_mxfp8

    M, K, N = shape
    aq, bq, sa, sb = _operands(M, K, N)

    got = scaled_gemm_mxfp8(aq, bq, sa, sb, torch.bfloat16)

    want = torch.nn.functional.scaled_mm(
        aq,
        bq.t().contiguous().t(),
        _to_blocked(sa.to(torch.float8_e8m0fnu)),
        torch.nn.functional.ScalingType.BlockWise1x32,
        _to_blocked(sb.t().contiguous().to(torch.float8_e8m0fnu)),
        torch.nn.functional.ScalingType.BlockWise1x32,
        swizzle_a=torch.nn.functional.SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=torch.nn.functional.SwizzleType.SWIZZLE_32_4_4,
        output_dtype=torch.bfloat16,
    )

    # same instruction, same operands, same scales -- only accumulation order can
    # differ, so this bar is far tighter than the 0.02 oracle bar
    assert (got.float() - want.float()).norm() / want.float().norm() < 1e-3


def test_bench_cute_gemm_importable():
    import importlib

    m = importlib.import_module("benchmarks.gemm.bench_cute_gemm")
    assert hasattr(m, "main")
