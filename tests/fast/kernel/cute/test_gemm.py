"""SM120 CuTe MXFP8 GEMM tests against shared and vendor oracles.

The kernel keeps B K-contiguous as `(N, K)`; reference calls use a free transpose
to bridge their `(K, N)` convention.
"""

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
    """`(aq, bq, sa, sb)` in the kernel's convention: A `(M, K)`, B `(N, K)`."""
    torch.manual_seed(seed)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    aq, sa = quantize_operand(a, -1, "fp8_e4m3", MX)
    bq, sb = quantize_operand(b, -1, "fp8_e4m3", MX)
    return aq, bq, sa, sb


def _operands_spread_scales(M, K, N, seed=0):
    """Vary every scale-fragment axis so misplaced SFA/SFB values are observable."""
    torch.manual_seed(seed)
    n_blocks = K // 32
    exp = torch.randint(-8, 8, (M, n_blocks), device="cuda").float()
    a = torch.randn(M, n_blocks, 32, device="cuda") * torch.exp2(exp)[:, :, None]
    exp = torch.randint(-8, 8, (N, n_blocks), device="cuda").float()
    b = torch.randn(N, n_blocks, 32, device="cuda") * torch.exp2(exp)[:, :, None]
    aq, sa = quantize_operand(a.reshape(M, K).bfloat16(), -1, "fp8_e4m3", MX)
    bq, sb = quantize_operand(b.reshape(N, K).bfloat16(), -1, "fp8_e4m3", MX)
    return aq, bq, sa, sb


@pytest.mark.parametrize("shape", [(128, 128, 128), (256, 128, 256)])
def test_scaled_gemm_mxfp8_matches_oracle(shape):
    from src.kernel.cute.gemm import scaled_gemm_mxfp8

    M, K, N = shape
    aq, bq, sa, sb = _operands(M, K, N)
    got = scaled_gemm_mxfp8(aq, bq, sa, sb, torch.float32)
    want = scaled_gemm_ref(aq, bq.t(), sa, sb.t(), 32)

    assert got.shape == (M, N)
    assert got.dtype == torch.float32
    # A relative norm avoids unstable elementwise bounds on quantized outputs.
    assert (got - want).norm() / want.norm() < 0.02


@pytest.mark.parametrize(
    "shape", [(256, 128, 256), (257, 128, 256), (256, 128, 257), (257, 128, 257)]
)
def test_scaled_gemm_mxfp8_epilogue_tile_boundaries(shape):
    """Cover interior, M-edge, N-edge, and combined-edge epilogue paths."""
    from src.kernel.cute.gemm import scaled_gemm_mxfp8

    M, K, N = shape
    aq, bq, sa, sb = _operands(M, K, N)
    got = scaled_gemm_mxfp8(aq, bq, sa, sb, torch.float32)
    want = scaled_gemm_ref(aq, bq.t(), sa, sb.t(), 32)

    assert got.shape == (M, N)
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
    """Require in-MMA scaling to match fp32 dequantization, including ragged tiles.

    Widely varying block scales expose feed errors; the tight bound isolates scaling
    and accumulation precision from the looser end-to-end quantization tolerance.
    """
    from src.kernel.cute.gemm import scaled_gemm_mxfp8

    M, K, N = shape
    aq, bq, sa, sb = _operands_spread_scales(M, K, N)
    got = scaled_gemm_mxfp8(aq, bq, sa, sb, torch.float32)
    oracle = scaled_gemm_ref(aq, bq.t(), sa, sb.t(), 32)
    rel = (got - oracle).norm() / oracle.norm()
    assert rel < 1e-6, rel


def test_compile_cache_reused():
    """M is dynamic in the compiled kernel, so a batch-size change must not recompile."""
    from src.kernel.cute import gemm as cute_gemm

    aq, bq, sa, sb = _operands(128, 128, 128)
    cute_gemm.scaled_gemm_mxfp8(aq, bq, sa, sb, torch.float32)
    n_after_first = len(cute_gemm._COMPILED)
    key = (128, 128, aq.dtype, bq.dtype, torch.float32)
    compiled_after_first = cute_gemm._COMPILED[key]

    aq2, bq2, sa2, sb2 = _operands(256, 128, 128, seed=1)  # different M, same K/N
    got = cute_gemm.scaled_gemm_mxfp8(aq2, bq2, sa2, sb2, torch.float32)

    assert n_after_first >= 1, "the first call must have cached a compiled kernel"
    assert len(cute_gemm._COMPILED) == n_after_first, "M must not trigger a recompile"
    # Identity catches replacement with a freshly compiled artifact.
    assert cute_gemm._COMPILED[key] is compiled_after_first, (
        "the second call must reuse the cached compiled kernel, not recompile it"
    )
    # Correctness confirms M is truly dynamic in the reused artifact.
    want = scaled_gemm_ref(aq2, bq2.t(), sa2, sb2.t(), 32)
    assert (got - want).norm() / want.norm() < 0.02


def _operands_fmt(M, K, N, a_fmt, b_fmt, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    aq, sa = quantize_operand(a, -1, a_fmt, MX)
    bq, sb = quantize_operand(b, -1, b_fmt, MX)
    return aq, bq, sa, sb


@pytest.mark.parametrize(
    "a_fmt,b_fmt",
    [("fp8_e4m3", "fp8_e4m3"), ("fp8_e4m3", "fp8_e5m2"), ("fp8_e5m2", "fp8_e4m3")],
)
def test_scaled_gemm_mxfp8_dtype_combinations(a_fmt, b_fmt):
    from src.kernel.cute.gemm import scaled_gemm_mxfp8

    aq, bq, sa, sb = _operands_fmt(128, 128, 128, a_fmt, b_fmt)
    got = scaled_gemm_mxfp8(aq, bq, sa, sb, torch.float32)
    want = scaled_gemm_ref(aq, bq.t(), sa, sb.t(), 32)
    assert (got - want).norm() / want.norm() < 0.02


@pytest.mark.parametrize(
    "shape,out_dtype",
    [
        ((128, 128, 128), torch.float32),
        ((128, 128, 128), torch.float16),
        ((128, 128, 128), torch.bfloat16),
        # Exercise dtype conversion and edge predication together.
        ((129, 160, 130), torch.bfloat16),
    ],
)
def test_scaled_gemm_mxfp8_preserves_out_dtype(shape, out_dtype):
    from src.kernel.cute.gemm import scaled_gemm_mxfp8

    aq, bq, sa, sb = _operands(*shape)
    got = scaled_gemm_mxfp8(aq, bq, sa, sb, out_dtype)
    want = scaled_gemm_ref(aq, bq.t(), sa, sb.t(), 32)

    assert got.dtype == out_dtype
    # Output conversion requires a looser tolerance than fp32.
    assert (got.float() - want.float()).norm() / want.float().norm() < 0.05


@pytest.mark.parametrize(
    "shape",
    [
        (64, 64, 64),
        (192, 160, 128),
        (129, 128, 130),
        # Isolate B's N-edge predication with a mostly empty scale tile.
        (256, 128, 33),
    ],
)
def test_scaled_gemm_mxfp8_ragged_shapes(shape):
    from src.kernel.cute.gemm import scaled_gemm_mxfp8

    M, K, N = shape
    aq, bq, sa, sb = _operands(M, K, N)
    got = scaled_gemm_mxfp8(aq, bq, sa, sb, torch.float32)
    want = scaled_gemm_ref(aq, bq.t(), sa, sb.t(), 32)

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
    """Use cuBLASLt as an independent scale-feed oracle for aligned shapes.

    Both kernels use the same MMA, so disagreement points to operand/scale plumbing;
    ragged shapes remain on the shared oracle because cuBLASLt requires alignment.
    """
    from src.kernel.cute.gemm import scaled_gemm_mxfp8

    M, K, N = shape
    aq, bq, sa, sb = _operands(M, K, N)

    got = scaled_gemm_mxfp8(aq, bq, sa, sb, torch.bfloat16)

    # cuBLASLt consumes free transposed views of the kernel's K-contiguous B/SB.
    want = torch.nn.functional.scaled_mm(
        aq,
        bq.t(),
        _to_blocked(sa.to(torch.float8_e8m0fnu)),
        torch.nn.functional.ScalingType.BlockWise1x32,
        _to_blocked(sb.to(torch.float8_e8m0fnu)),
        torch.nn.functional.ScalingType.BlockWise1x32,
        swizzle_a=torch.nn.functional.SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=torch.nn.functional.SwizzleType.SWIZZLE_32_4_4,
        output_dtype=torch.bfloat16,
    )

    # Only accumulation order may differ, allowing a tighter tolerance.
    assert (got.float() - want.float()).norm() / want.float().norm() < 1e-3


def test_bench_cute_gemm_importable():
    import importlib

    m = importlib.import_module("benchmarks.gemm.bench_cute_gemm")
    assert hasattr(m, "main")


@pytest.mark.parametrize("shape", [(128, 160, 128), (256, 96, 128), (128, 224, 256)])
def test_scaled_gemm_mxfp8_partial_scale_group(shape):
    """Require padding when a four-byte scale copy reaches a partial K tile."""
    from src.kernel.cute.gemm import scaled_gemm_mxfp8

    M, K, N = shape
    assert (K // 32) % 4 != 0, "this test is pointless on a K whose blocks divide 4"

    aq, bq, sa, sb = _operands(M, K, N)
    got = scaled_gemm_mxfp8(aq, bq, sa, sb, torch.float32)
    want = scaled_gemm_ref(aq, bq.t(), sa, sb.t(), 32)

    assert got.shape == (M, N)
    assert (got - want).norm() / want.norm() < 0.02
