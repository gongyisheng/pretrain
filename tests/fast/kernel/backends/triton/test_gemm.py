"""Parity of the Triton grouped GEMM prototype vs torch._grouped_mm (bf16),
plus the scaled (quantized) GEMM Triton kernel vs a fake-quantize oracle."""

import importlib

import pytest
import torch

from src.kernel.backends.triton import gemm
import src.kernel.backends.triton.gemm as gemm_mod
from src.kernel.ops.gemm import (
    grouped_gemm as _public_grouped_gemm,
    scaled_gemm as _public_scaled_gemm,
    scaled_grouped_gemm as _public_scaled_grouped_gemm,
)
from src.kernel.selector import KernelSelectionError
from src.quant.quantize import quantize_operand
from src.kernel.backends.reference.gemm import _dequant_a, _dequant_b


def grouped_gemm_ref(a, b, offs, bias=None):
    return _public_grouped_gemm(a, b, offs, bias=bias, backend="reference")


def scaled_gemm_ref(aq, bq, sa, sb, block_size, bias=None):
    return _public_scaled_gemm(
        aq,
        bq,
        sa,
        sb,
        torch.float32,
        block_size,
        bias=bias,
        backend="reference",
    )


def scaled_grouped_gemm_ref(
    aq,
    bq,
    sa,
    sb,
    offs,
    block_size,
    bias=None,
):
    return _public_scaled_grouped_gemm(
        aq,
        bq,
        sa,
        sb,
        offs,
        torch.float32,
        block_size,
        bias=bias,
        backend="reference",
    )


def triton_grouped_gemm(a, b, offs, bias=None):
    return _public_grouped_gemm(a, b, offs, bias=bias, backend="triton")


def triton_scaled_gemm(
    aq,
    bq,
    sa,
    sb,
    out_dtype,
    block_size,
    bias=None,
    scale_dtype=None,
):
    return _public_scaled_gemm(
        aq,
        bq,
        sa,
        sb,
        out_dtype,
        block_size,
        bias=bias,
        scale_dtype=scale_dtype,
        backend="triton",
    )


def triton_scaled_grouped_gemm(
    aq,
    bq,
    sa,
    sb,
    offs,
    out_dtype,
    block_size,
    bias=None,
    scale_dtype=None,
):
    return _public_scaled_grouped_gemm(
        aq,
        bq,
        sa,
        sb,
        offs,
        out_dtype,
        block_size,
        bias=bias,
        scale_dtype=scale_dtype,
        backend="triton",
    )


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton GEMM kernels are CUDA only"
)

E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2
FMT = {E4M3: "fp8_e4m3", E5M2: "fp8_e5m2"}
INT_FMT = {127: "int8", 63: "int7", 31: "int6", 15: "int5", 7: "int4"}


def _scaling(gran, bs=0, scale_dtype=None):
    return {
        "granularity": gran,
        "block_shape": (1, bs) if bs else (0, 0),
        "scale_dtype": scale_dtype,
    }


def _run(a, b, gran, bs, a_fmt, b_fmt, scale_dtype=None):
    # the kernel takes the 0-sentinel width; the oracle's pad math takes a count
    scaling = _scaling(gran, bs, scale_dtype)
    aq, sa = quantize_operand(a, -1, a_fmt, scaling)
    bq, sb = quantize_operand(b, -2, b_fmt, scaling)
    out = triton_scaled_gemm(aq, bq, sa, sb, torch.float32, bs)
    oracle = scaled_gemm_ref(aq, bq, sa, sb, bs or a.shape[1])
    return out, oracle


def _make(counts, K, N, seed=0):
    """Build (a (R,K), b=w.mT (E,K,N) transposed view, offs (E,) int32) on CUDA/bf16."""
    torch.manual_seed(seed)
    E, R = len(counts), sum(counts)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    b = (torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16) * 0.1).mT
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    return a, b, offs


# The three ragged layouts, named by which dim `offs` partitions. Counts sum to a
# multiple of 8 so every operand satisfies torch's 16-byte stride alignment; the
# empty group exercises the degenerate case.
LAYOUTS = ("ragged_m", "ragged_k", "ragged_n")
_LAYOUT_COUNTS = [64, 0, 130, 46]  # sums to 240


def _make_layout(layout, counts=_LAYOUT_COUNTS, M=32, K=64, N=48, seed=0):
    """Build (a, b, offs) for one ragged layout, in torch._grouped_mm's convention."""
    torch.manual_seed(seed)
    G, R = len(counts), sum(counts)
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.1

    if layout == "ragged_m":  # (R,K) x (G,K,N) -> (R,N)
        return rand(R, K), rand(G, N, K).mT, offs
    if layout == "ragged_k":  # (M,R) x (R,N) -> (G,M,N)
        return rand(R, M).mT, rand(R, N), offs
    if layout == "ragged_n":  # (G,M,K) x (K,R) -> (M,R)
        return rand(G, M, K), rand(K, R), offs
    raise ValueError(layout)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_all_ragged_layouts_match_torch(layout):
    a, b, offs = _make_layout(layout)
    ref = torch._grouped_mm(a, b, offs=offs)
    got = triton_grouped_gemm(a, b, offs)
    assert got.shape == ref.shape
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_output_stride_matches_torch():
    """A last dim that is not 8-element aligned is where torch's padding shows."""
    a, b, offs = _make_layout("ragged_m", N=51)
    assert (
        triton_grouped_gemm(a, b, offs).stride()
        == torch._grouped_mm(a, b, offs=offs).stride()
    )


@pytest.mark.parametrize("layout", ("ragged_m", "ragged_k"))
def test_bias_matches_reference(layout):
    a, b, offs = _make_layout(layout)
    G, N = offs.shape[0], 48
    bias = torch.randn(G, N, device="cuda", dtype=torch.bfloat16)
    got = triton_grouped_gemm(a, b, offs, bias=bias)
    ref = grouped_gemm_ref(a, b, offs, bias=bias)
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_bias_rejected_for_ragged_n():
    """Ragged N partitions the columns, so a (G,N) row-broadcast bias is meaningless."""
    a, b, offs = _make_layout("ragged_n")
    bias = torch.zeros(offs.shape[0], b.shape[1], device="cuda", dtype=torch.bfloat16)
    with pytest.raises((ValueError, AssertionError, NotImplementedError)):
        triton_grouped_gemm(a, b, offs, bias=bias)


def test_parity_balanced_groups():
    a, b, offs = _make([128, 128, 128, 128], K=64, N=48)
    got = triton_grouped_gemm(a, b, offs)
    ref = grouped_gemm_ref(a, b, offs)
    assert got.shape == ref.shape
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_parity_empty_and_uneven_groups():
    # includes an empty group (0 rows) and uneven sizes
    a, b, offs = _make([5, 0, 130, 41, 1], K=64, N=48)
    got = triton_grouped_gemm(a, b, offs)
    torch.testing.assert_close(
        got.float(), grouped_gemm_ref(a, b, offs).float(), rtol=2e-2, atol=2e-2
    )


def test_parity_single_group():
    a, b, offs = _make([200], K=192, N=64)
    got = triton_grouped_gemm(a, b, offs)
    torch.testing.assert_close(
        got.float(), grouped_gemm_ref(a, b, offs).float(), rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize(
    "K,N",
    [
        (64, 384),  # e512_k48 gate_up
        (192, 64),  # e512_k48 down
        (512, 384),  # e64_k6 gate_up
        (192, 512),  # e64_k6 down
    ],
)
def test_parity_config_shapes(K, N):
    # skewed load across 16 experts, summing to a realistic row count
    counts = [300, 5, 120, 0, 44, 210, 7, 90, 33, 150, 1, 260, 12, 80, 40, 400]
    a, b, offs = _make(counts, K=K, N=N, seed=1)
    got = triton_grouped_gemm(a, b, offs)
    torch.testing.assert_close(
        got.float(), grouped_gemm_ref(a, b, offs).float(), rtol=2e-2, atol=2e-2
    )


def test_wgrad_parity_uneven_and_empty():
    torch.manual_seed(0)
    counts = [5, 0, 130, 41, 1]
    R, K, N = sum(counts), 64, 48
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    grad_c = torch.randn(R, N, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    ref = torch._grouped_mm(a.mT, grad_c, offs=offs)
    got = triton_grouped_gemm(a.mT, grad_c, offs)
    assert got.shape == (len(counts), K, N)
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "K,N",
    [
        (64, 384),  # e512_k48 gate_up
        (192, 64),  # e512_k48 down
        (512, 384),  # e64_k6 gate_up
        (192, 512),  # e64_k6 down
    ],
)
def test_wgrad_parity_config_shapes(K, N):
    torch.manual_seed(1)
    counts = [300, 5, 120, 0, 44, 210, 7, 90, 33, 150, 1, 260, 12, 80, 40, 400]
    R = sum(counts)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    # scale grad_c down so the K=512 accumulator (~sqrt(K) in magnitude) stays in
    # a range where bf16's ULP is below atol; at unit variance the ULP (~0.125)
    # exceeds atol=2e-2 and a handful of near-zero elements trip element-wise.
    grad_c = torch.randn(R, N, device="cuda", dtype=torch.bfloat16) * 0.1
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    ref = torch._grouped_mm(a.mT, grad_c, offs=offs)
    got = triton_grouped_gemm(a.mT, grad_c, offs)
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "gran,bs",
    [
        ("tensorwise", 0),
        ("rowwise", 0),
        # 16 is the finest tiling width; BLOCK_K floors at 32, so it exercises the
        # scale-block mask that keeps a 32-wide tile inside a 16-wide block.
        ("blockwise", 16),
        ("blockwise", 32),
        ("blockwise", 64),
        ("blockwise", 128),
    ],
)
def test_fp8_e4m3_matches_oracle(gran, bs):
    torch.manual_seed(0)
    a = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 96, device="cuda", dtype=torch.bfloat16)
    out, oracle = _run(a, b, gran, bs, FMT[E4M3], FMT[E4M3])
    assert out.shape == (128, 96) and out.dtype == torch.float32
    assert (out - oracle).norm() / oracle.norm() < 0.02


def test_fp8_e5m2_times_e5m2_ok():
    # cuBLAS rejects e5m2 x e5m2; the Triton kernel accepts it.
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    out, oracle = _run(a, b, "rowwise", 0, FMT[E5M2], FMT[E5M2])
    assert torch.isfinite(out).all()
    assert (out - oracle).norm() / oracle.norm() < 0.05


def test_fp8_mixed_e5m2_e4m3():
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    out, oracle = _run(a, b, "blockwise", 32, FMT[E5M2], FMT[E4M3])
    assert (out - oracle).norm() / oracle.norm() < 0.05


def test_pow2_e8m0_blockwise_matches_oracle():
    # power-of-two (E8M0 / MX-style) blockwise scale through the real kernel.
    torch.manual_seed(0)
    a = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    out, oracle = _run(
        a, b, "blockwise", 32, FMT[E4M3], FMT[E4M3], scale_dtype="fp8_e8m0"
    )
    assert (out - oracle).norm() / oracle.norm() < 0.02


@pytest.mark.parametrize("qmax", [127, 63, 31, 15, 7])
def test_int_matches_oracle(qmax):
    torch.manual_seed(0)
    a = torch.randn(96, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    out, oracle = _run(a, b, "rowwise", 0, INT_FMT[qmax], INT_FMT[qmax])
    assert (out - oracle).norm() / oracle.norm() < 0.02


@pytest.mark.parametrize(
    "gran,bs", [("tensorwise", 0), ("rowwise", 0), ("blockwise", 32)]
)
def test_bias_matches_oracle(gran, bs):
    """Fused (N,) bias, broadcast over rows.

    Blockwise is the discriminating case: the bias belongs on the final
    accumulator, so an implementation that folded it into the per-scale-block
    partial would add it once per block (8x here) instead of once.
    """
    torch.manual_seed(0)
    a = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 96, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(96, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, FMT[E4M3], _scaling(gran, bs))
    bq, sb = quantize_operand(b, -2, FMT[E4M3], _scaling(gran, bs))
    out = triton_scaled_gemm(aq, bq, sa, sb, torch.float32, bs, bias=bias)
    oracle = scaled_gemm_ref(aq, bq, sa, sb, bs or a.shape[1]) + bias.float()
    assert (out - oracle).norm() / oracle.norm() < 0.02


def test_bias_none_unchanged():
    """bias=None must be byte-identical to omitting it -- the HAS_BIAS=False path."""
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, FMT[E4M3], _scaling("rowwise"))
    bq, sb = quantize_operand(b, -2, FMT[E4M3], _scaling("rowwise"))
    torch.testing.assert_close(
        triton_scaled_gemm(aq, bq, sa, sb, torch.float32, 0, bias=None),
        triton_scaled_gemm(aq, bq, sa, sb, torch.float32, 0),
        rtol=0,
        atol=0,
    )


def test_nonmultiple_shapes_mask_correctly():
    torch.manual_seed(0)
    a = torch.randn(70, 100, device="cuda", dtype=torch.bfloat16)  # M,K odd
    b = torch.randn(100, 130, device="cuda", dtype=torch.bfloat16)  # N odd, K%32!=0
    out, oracle = _run(a, b, "blockwise", 32, FMT[E4M3], FMT[E4M3])
    assert out.shape == (70, 130)
    assert (out - oracle).norm() / oracle.norm() < 0.03


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_output_dtype_respected(out_dtype):
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, FMT[E4M3], _scaling("rowwise"))
    bq, sb = quantize_operand(b, -2, FMT[E4M3], _scaling("rowwise"))
    out = triton_scaled_gemm(
        aq, bq, sa, sb, out_dtype, 0
    )  # rowwise -> one block over K
    assert out.dtype == out_dtype


def test_bench_scaled_gemm_importable():
    m = importlib.import_module("benchmarks.gemm.bench_scaled_gemm")
    assert hasattr(m, "main")


# ---------------------------------------------------------------------------
# Scaled GEMM: the mxfp8 path (tl.dot_scaled / QMMA.SF)
# ---------------------------------------------------------------------------

MX = _scaling("blockwise", 32, "fp8_e8m0")


@pytest.mark.parametrize("a_dtype,b_dtype", [(E4M3, E4M3), (E4M3, E5M2), (E5M2, E4M3)])
@pytest.mark.parametrize("shape", [(128, 256, 96), (64, 64, 64), (192, 160, 128)])
def test_scaled_gemm_mx_matches_oracle(a_dtype, b_dtype, shape):
    """The mxfp8 path hands the e8m0 scales to the MMA instead of the epilogue.

    Which means the scales never touch the accumulator this kernel builds, so a wrong
    scale layout -- the operand is (M,K/32) but B's is read transposed out of a (K/32,N)
    buffer -- produces plausible finite garbage rather than an error. Shapes that are
    not multiples of the tile cover the masked tail, where a masked scale rides along
    with a zeroed operand.
    """
    M, K, N = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, FMT[a_dtype], MX)
    bq, sb = quantize_operand(b, -2, FMT[b_dtype], MX)
    out = triton_scaled_gemm(aq, bq, sa, sb, torch.float32, 32, scale_dtype="fp8_e8m0")
    oracle = scaled_gemm_ref(aq, bq, sa, sb, 32)
    assert (out - oracle).norm() / oracle.norm() < 0.02


def test_scaled_gemm_mx_agrees_with_epilogue_path():
    """Same inputs, same answer, whichever kernel runs them.

    `scale_dtype` only picks an implementation; it must not change the result. This is
    what would catch the MMA applying scales per 32-wide group where the other kernel
    applies them per scale block, had those two ever disagreed.
    """
    torch.manual_seed(0)
    a = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, FMT[E4M3], MX)
    bq, sb = quantize_operand(b, -2, FMT[E4M3], MX)
    mx = triton_scaled_gemm(aq, bq, sa, sb, torch.float32, 32, scale_dtype="fp8_e8m0")
    epilogue = triton_scaled_gemm(aq, bq, sa, sb, torch.float32, 32)
    assert (mx - epilogue).norm() / epilogue.norm() < 1e-6


def test_scaled_gemm_mx_bias_lands_once():
    """The bias rides the epilogue past the scales, as on the other kernel."""
    torch.manual_seed(0)
    a = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 96, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(96, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, FMT[E4M3], MX)
    bq, sb = quantize_operand(b, -2, FMT[E4M3], MX)
    out = triton_scaled_gemm(
        aq, bq, sa, sb, torch.float32, 32, bias=bias, scale_dtype="fp8_e8m0"
    )
    oracle = scaled_gemm_ref(aq, bq, sa, sb, 32) + bias.float()
    assert (out - oracle).norm() / oracle.norm() < 0.02


@pytest.mark.parametrize("shape", [(256, 512, 192), (128, 256, 96), (192, 160, 128)])
def test_scaled_gemm_mxfp8_adds_no_error_over_dequantizing(shape):
    """Applying the e8m0 scales inside the MMA must be as exact as dequantizing first.

    The bound is fp32-rounding tight (measured 2e-8..4e-8) on purpose. The oracle
    comparison in `test_scaled_gemm_mx_matches_oracle` allows 2e-2, which is ~5e5x
    looser than reality: it would still pass if `QMMA.SF` were applying the scales at
    reduced precision. It would not hide a misaligned scale operand, though: an e8m0
    scale is a power of two, so landing one on the wrong row, column or k-tile is at
    least a factor-2 error and never sits inside the quantization noise -- that part
    the loose oracle bound already catches on its own. This is the assertion that
    pins the instruction's arithmetic.
    """
    M, K, N = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, FMT[E4M3], MX)
    bq, sb = quantize_operand(b, -2, FMT[E4M3], MX)
    got = triton_scaled_gemm(aq, bq, sa, sb, torch.float32, 32, scale_dtype="fp8_e8m0")
    oracle = scaled_gemm_ref(aq, bq, sa, sb, 32)
    rel = (got - oracle).norm() / oracle.norm()
    assert rel < 1e-6, rel


def test_scaled_gemm_mxfp8_precision_band_against_unquantized():
    """mxfp8 must land at e4m3 precision against an unquantized reference.

    Both bounds carry weight. The upper one catches scales that are wrong by a power
    of two, a mismatched format string (e5m2 read as e4m3), or scales dropped
    altogether -- all of which leave the output finite. The lower one catches a
    reference that is itself quantized, which would make the test vacuous.

    Note what this does *not* pin: the error is dominated by the e4m3 mantissa, so
    coarser granularities measure the same ~3.7e-2 on well-conditioned data (verified
    up to a 2^+-30 spread of per-block exponents). Finer scale blocks buy range, not
    accuracy, in the relative-norm sense -- so there is deliberately no assertion here
    that mxfp8 beats rowwise or tensorwise, because it does not.
    """
    torch.manual_seed(0)
    a = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(512, 192, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, FMT[E4M3], MX)
    bq, sb = quantize_operand(b, -2, FMT[E4M3], MX)
    got = triton_scaled_gemm(aq, bq, sa, sb, torch.float32, 32, scale_dtype="fp8_e8m0")
    truth = a.float() @ b.float()
    rel = ((got - truth).norm() / truth.norm()).item()
    assert 2e-2 < rel < 6e-2, rel


@pytest.mark.parametrize(
    "fmt,bs,scale_dtype",
    [
        ("int8", 32, "fp8_e8m0"),  # tl.dot_scaled has no integer format
        (FMT[E4M3], 16, "fp8_e8m0"),  # 16 clears the power-of-two->=16 guard but is
        # not a multiple of the MMA's fixed 32-wide scale vector
        (FMT[E4M3], 32, None),  # fp32 scales are not e8m0 exponents
    ],
)
def test_scaled_gemm_mx_declines_unsupported_combinations(fmt, bs, scale_dtype):
    """Only fp8 x (any nonzero multiple of 32) x e8m0 may take the MMA path; the rest
    must fall back to epilogue scaling.

    Each of these would fail differently if the dispatch were loosened: int8 is
    rejected outright by tl.dot_scaled ("Invalid float format"), a 16-wide block is
    not a multiple of the instruction's fixed 32-wide scale vector so it cannot be
    reached by replication, and fp32 scales are not exponent bytes at all.
    """
    torch.manual_seed(0)
    a = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 96, device="cuda", dtype=torch.bfloat16)
    scaling = _scaling("blockwise", bs, scale_dtype)
    aq, sa = quantize_operand(a, -1, fmt, scaling)
    bq, sb = quantize_operand(b, -2, fmt, scaling)
    out = triton_scaled_gemm(aq, bq, sa, sb, torch.float32, bs, scale_dtype=scale_dtype)
    oracle = scaled_gemm_ref(aq, bq, sa, sb, bs)
    assert (out - oracle).norm() / oracle.norm() < 0.02


def rand(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)


@pytest.mark.parametrize("bs", [32, 64, 128])
def test_e8m0_routes_to_the_mxfp8_kernel_at_any_multiple_of_32(monkeypatch, bs):
    """A scale width wider than 32 is still the block-scaled MMA's job, via replication."""
    launched = []
    real_wrap = gemm_mod.wrap_triton

    def spy(kernel):
        # repr(kernel) is just the Autotuner object's address in this Triton version;
        # the kernel's name lives on the wrapped fn.
        launched.append(kernel.fn.__name__)
        return real_wrap(kernel)

    monkeypatch.setattr(gemm_mod, "wrap_triton", spy)

    a, b = rand(256, 512), rand(512, 128)
    scaling = _scaling("blockwise", bs, "fp8_e8m0")
    aq, sa = quantize_operand(a, -1, FMT[E4M3], scaling)
    bq, sb = quantize_operand(b, -2, FMT[E4M3], scaling)
    triton_scaled_gemm(aq, bq, sa, sb, torch.float32, bs, scale_dtype="fp8_e8m0")
    assert any("mxfp8" in name for name in launched), launched


@pytest.mark.parametrize("bs", [32, 64, 128])
def test_mxfp8_kernel_matches_epilogue_at_any_multiple_of_32(bs):
    """The e8m0 fast path must agree with epilogue scaling at every legal width."""
    a, b = rand(256, 512), rand(512, 128)
    scaling = _scaling("blockwise", bs, "fp8_e8m0")
    aq, sa = quantize_operand(a, -1, FMT[E4M3], scaling)
    bq, sb = quantize_operand(b, -2, FMT[E4M3], scaling)
    mx = triton_scaled_gemm(aq, bq, sa, sb, torch.float32, bs, scale_dtype="fp8_e8m0")
    epilogue = triton_scaled_gemm(aq, bq, sa, sb, torch.float32, bs)
    torch.testing.assert_close(mx, epilogue, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("bs", [32, 64, 128])
def test_mxfp8_kernel_matches_epilogue_when_k_is_not_a_multiple_of_32(bs):
    """K not a multiple of 32 leaves sa/sb with a trailing partial scale block --
    n_scale_blocks is ceil(K/32), one more than K // 32. The bs=32 case is the one that
    pins the regression: block_size == _MXFP8_BLOCK_SIZE is supposed to be a bit-exact
    no-op, so if this fails only at bs=32 the bug is in the block count, not the
    replication.
    """
    torch.manual_seed(0)
    a, b = rand(64, 163), rand(163, 96)
    scaling = _scaling("blockwise", bs, "fp8_e8m0")
    aq, sa = quantize_operand(a, -1, FMT[E4M3], scaling)
    bq, sb = quantize_operand(b, -2, FMT[E4M3], scaling)
    mx = triton_scaled_gemm(aq, bq, sa, sb, torch.float32, bs, scale_dtype="fp8_e8m0")
    epilogue = triton_scaled_gemm(aq, bq, sa, sb, torch.float32, bs)
    torch.testing.assert_close(mx, epilogue, rtol=2e-2, atol=2e-2)


def test_mxfp8_kernel_declines_block_size_zero():
    """block_size 0 is the "one scale block spans all of K" sentinel, not a multiple of
    32 in any meaningful sense -- but 0 % 32 == 0 in Python, so an unguarded condition
    would route it into the replication branch, where rep_k=0 collapses sa/sb to a
    zero-width tensor and the kernel launch reads out of bounds.
    """
    torch.manual_seed(0)
    a, b = rand(64, 163), rand(163, 96)
    scaling = _scaling("tensorwise", 0, "fp8_e8m0")
    aq, sa = quantize_operand(a, -1, FMT[E4M3], scaling)
    bq, sb = quantize_operand(b, -2, FMT[E4M3], scaling)
    out = triton_scaled_gemm(aq, bq, sa, sb, torch.float32, 0, scale_dtype="fp8_e8m0")
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Scaled GEMM: input validation
# ---------------------------------------------------------------------------

_VM, _VK, _VN, _VBS = 64, 256, 96, 128  # K / block_size -> 2 scale blocks


def _valid_scaled_args(bs=_VBS):
    """Consistent (aq, bq, sa, sb, bias), for a case below to break one way."""
    torch.manual_seed(0)
    a = torch.randn(_VM, _VK, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(_VK, _VN, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, FMT[E4M3], _scaling("blockwise", bs))
    bq, sb = quantize_operand(b, -2, FMT[E4M3], _scaling("blockwise", bs))
    return aq, bq, sa, sb, torch.randn(_VN, device="cuda", dtype=torch.bfloat16)


# Each case breaks exactly one invariant and leaves the rest consistent, so the named
# check is the one that has to fire. Untouched, all of them index out of bounds: the
# kernel takes M/K from aq, N from bq and the block count from sa, and trusts every
# other operand to agree.
_MALFORMED = {
    "contraction": lambda aq, bq, sa, sb, bias: (aq, bq[: _VK // 2], sa, sb, bias),
    "sa_rows": lambda aq, bq, sa, sb, bias: (aq, bq, sa[: _VM // 2], sb, bias),
    "sa_1d": lambda aq, bq, sa, sb, bias: (aq, bq, sa[:, 0], sb, bias),
    "sb_cols": lambda aq, bq, sa, sb, bias: (aq, bq, sa, sb[:, : _VN // 2], bias),
    "sb_blocks": lambda aq, bq, sa, sb, bias: (aq, bq, sa, sb[:1], bias),
    "sb_1d": lambda aq, bq, sa, sb, bias: (aq, bq, sa, sb[0], bias),
    # sa and sb agree with each other but not with block_size: the block loop then
    # covers 1 x 128 of K instead of 256, dropping the tail with no error at all
    "scale_blocks_vs_block_size": lambda aq, bq, sa, sb, bias: (
        aq,
        bq,
        sa[:, :1],
        sb[:1],
        bias,
    ),
    "bias_len": lambda aq, bq, sa, sb, bias: (aq, bq, sa, sb, bias[: _VN // 2]),
    "bias_2d": lambda aq, bq, sa, sb, bias: (aq, bq, sa, sb, bias[None]),
    "cross_family_dtype": lambda aq, bq, sa, sb, bias: (
        aq,
        bq.to(torch.int8),
        sa,
        sb,
        bias,
    ),
}
# no device case: Triton's own pointer check already raises ValueError for an operand
# on another device, whether that is the CPU or a second GPU


@pytest.mark.parametrize("case", sorted(_MALFORMED))
def test_scaled_gemm_rejects_malformed_inputs(case):
    """Reject on the host rather than reading out of bounds in the kernel.

    Left unchecked these do not raise: a short sb or a wrong-length bias returns
    finite garbage and a bad contraction returns nan, so a training run keeps going
    on numbers that mean nothing.
    """
    aq, bq, sa, sb, bias = _MALFORMED[case](*_valid_scaled_args())
    with pytest.raises((KernelSelectionError, ValueError, AssertionError)):
        triton_scaled_gemm(aq, bq, sa, sb, torch.float32, _VBS, bias=bias)


@pytest.mark.parametrize("bs", [8, 96])
def test_scaled_gemm_rejects_unusable_block_size(bs):
    """block_size is the scale-block width that the autotuned BLOCK_K tiles, so a
    width that is not a power of two leaves every block ending on a masked partial
    tile, and one under 16 cannot fill even the narrowest tile fp8/int8 tl.dot
    accepts. It comes straight from the scaling config, and unchecked it fails deep
    inside Triton's compiler pointing at a line unrelated to the cause.
    """
    a = torch.randn(_VM, _VK, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(_VK, _VN, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, FMT[E4M3], _scaling("blockwise", bs))
    bq, sb = quantize_operand(b, -2, FMT[E4M3], _scaling("blockwise", bs))
    with pytest.raises((KernelSelectionError, ValueError)):
        triton_scaled_gemm(aq, bq, sa, sb, torch.float32, bs)


def test_scaled_gemm_accepts_non_pow2_block_size_spanning_k():
    """Guards the check above against over-rejecting: a block_size >= K collapses to
    a single scale block, which no BLOCK_K ever has to tile, so it needs no power of
    two.
    """
    a = torch.randn(_VM, _VK, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(_VK, _VN, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, FMT[E4M3], _scaling("blockwise", _VK + 44))
    bq, sb = quantize_operand(b, -2, FMT[E4M3], _scaling("blockwise", _VK + 44))
    assert sa.shape[1] == 1
    assert torch.isfinite(
        triton_scaled_gemm(aq, bq, sa, sb, torch.float32, _VK + 44)
    ).all()


# ---------------------------------------------------------------------------
# Scaled grouped GEMM: fp8/int8 ragged expert matmul
# ---------------------------------------------------------------------------


SCALED_LAYOUTS = ("ragged_m", "ragged_k", "ragged_n")
_SCALED_COUNTS = [64, 0, 130, 46]  # sums to 240; the empty group is deliberate
# The ragged-contraction layout dequants the same codes as its oracle and only
# reorders the fp32 accumulation, so it must agree far more tightly than the
# forward layouts, whose oracle pads scale blocks independently of the tiling.
_SCALED_TOL = {"ragged_m": 0.02, "ragged_k": 1e-5, "ragged_n": 0.02}


def _make_scaled_layout(
    layout, counts, gran, bs, fmt, M=32, K=64, N=48, seed=0, scale_dtype=None
):
    """Quantized operands + scales for one ragged layout of the scaled grouped GEMM.

    Scales follow the kernel's invariant: each mirrors its operand's axis order with
    the contraction axis replaced by the scale-block axis. `bs` 0 (row/tensorwise)
    means one block per contraction segment.
    """
    torch.manual_seed(seed)
    E, R = len(counts), sum(counts)
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    scaling = _scaling(gran, bs, scale_dtype)

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.1

    if layout == "ragged_m":  # (R,K) x (E,K,N) -> (R,N)
        aq, sa = quantize_operand(rand(R, K), -1, fmt, scaling)
        b = torch.stack([rand(K, N) for _ in range(E)])
        bq, sb = quantize_operand(b, -2, fmt, scaling)
        return aq, bq, sa, sb, offs, bs

    if layout == "ragged_k":  # (M,R) x (R,N) -> (E,M,N)
        aq, sa = quantize_operand(
            rand(R, M), -2, fmt, scaling, offs=offs, ragged_dim=-2
        )
        bq, sb = quantize_operand(
            rand(R, N), -2, fmt, scaling, offs=offs, ragged_dim=-2
        )
        return aq.mT, bq, sa.mT, sb, offs, bs

    if layout == "ragged_n":  # (E,M,K) x (K,R) -> (M,R)
        a = torch.stack([rand(M, K) for _ in range(E)])
        aq, sa = quantize_operand(a, -1, fmt, scaling)
        bqT, sbT = quantize_operand(
            rand(R, K), -1, fmt, scaling, offs=offs, ragged_dim=-2
        )
        return aq, bqT.mT, sa, sbT.mT, offs, bs

    raise ValueError(layout)


def _expected_shape(layout, counts, M=32, N=48):
    E, R = len(counts), sum(counts)
    return {"ragged_m": (R, N), "ragged_k": (E, M, N), "ragged_n": (M, R)}[layout]


def test_scaled_grouped_gemm_oracle_agrees_with_dequant():
    # The oracle walks its own per-group loop; this pins it against an explicit
    # dequant matmul so a bug in its block/pad bookkeeping cannot pass unnoticed.
    counts = [128, 128]
    aq, bq, sa, sb, offs, kbs = _make_scaled_layout(
        "ragged_m", counts, "rowwise", 0, "fp8_e4m3"
    )
    oracle = scaled_grouped_gemm_ref(aq, bq, sa, sb, offs, kbs)
    K = aq.shape[1]
    a_deq = _dequant_a(aq, sa, kbs or K)
    # per-expert B (E,K,N), each slice dequantized against its own (nkb,N) scale
    b_deq = torch.stack(
        [_dequant_b(bq[g], sb[g], kbs or K) for g in range(bq.shape[0])]
    )
    dequant = torch.cat([a_deq[:128] @ b_deq[0], a_deq[128:] @ b_deq[1]])
    assert oracle.shape == (256, 48)
    assert (oracle - dequant).norm() / dequant.norm() < 0.02


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "int8"])
@pytest.mark.parametrize(
    "gran,bs",
    # 48 is deliberately not a multiple of any BLOCK_K: the kernel masks the
    # reduction to the scale block's end rather than clamping BLOCK_K to it, so
    # correctness must not depend on which config autotune picks.
    [
        ("rowwise", 0),
        ("tensorwise", 0),
        ("blockwise", 16),
        ("blockwise", 32),
        ("blockwise", 48),
    ],
)
@pytest.mark.parametrize("layout", SCALED_LAYOUTS)
def test_scaled_grouped_layouts_match_oracle(layout, gran, bs, fmt):
    aq, bq, sa, sb, offs, kbs = _make_scaled_layout(
        layout, _SCALED_COUNTS, gran, bs, fmt
    )
    got = triton_scaled_grouped_gemm(aq, bq, sa, sb, offs, torch.float32, kbs)
    ref = scaled_grouped_gemm_ref(aq, bq, sa, sb, offs, kbs)
    assert got.shape == _expected_shape(layout, _SCALED_COUNTS)
    rel = (got.float() - ref).norm() / ref.norm().clamp_min(1e-12)
    assert rel < _SCALED_TOL[layout], rel


@pytest.mark.parametrize("layout", SCALED_LAYOUTS)
def test_scaled_grouped_mxfp8_layouts_match_oracle(layout):
    """The mxfp8 grouped path, for each ragged layout.

    ragged_k is the discriminating one: its scale blocks re-tile inside every group, so
    the kernel carries a cursor across the group loop instead of indexing k // 32
    globally. Getting that wrong shifts every group after the first onto the previous
    group's scales, which stays finite and roughly the right magnitude.
    """
    aq, bq, sa, sb, offs, kbs = _make_scaled_layout(
        layout, _SCALED_COUNTS, "blockwise", 32, "fp8_e4m3", scale_dtype="fp8_e8m0"
    )
    got = triton_scaled_grouped_gemm(
        aq, bq, sa, sb, offs, torch.float32, kbs, scale_dtype="fp8_e8m0"
    )
    ref = scaled_grouped_gemm_ref(aq, bq, sa, sb, offs, kbs)
    assert got.shape == _expected_shape(layout, _SCALED_COUNTS)
    rel = (got.float() - ref).norm() / ref.norm().clamp_min(1e-12)
    assert rel < _SCALED_TOL[layout], rel


@pytest.mark.parametrize("bs", [32, 64, 128])
def test_scaled_grouped_mxfp8_ragged_m_matches_oracle_when_k_is_not_a_multiple_of_32(
    bs,
):
    """Same regression as the dense-gemm K=163 case, for the ragged-M grouped layout --
    K here is shared by every group (not ragged), so it is the layout the generalized
    condition applies to. A dropped trailing scale column here does not just lose
    accuracy: the kernel's own `scale_block_end = tl.cdiv(K, 32)` still expects it, so
    reading past a host-side tensor narrowed one column short produces NaN, not just a
    large relative error.
    """
    aq, bq, sa, sb, offs, kbs = _make_scaled_layout(
        "ragged_m",
        _SCALED_COUNTS,
        "blockwise",
        bs,
        "fp8_e4m3",
        K=163,
        scale_dtype="fp8_e8m0",
    )
    got = triton_scaled_grouped_gemm(
        aq, bq, sa, sb, offs, torch.float32, kbs, scale_dtype="fp8_e8m0"
    )
    ref = scaled_grouped_gemm_ref(aq, bq, sa, sb, offs, kbs)
    assert torch.isfinite(got).all(), got
    rel = (got.float() - ref).norm() / ref.norm().clamp_min(1e-12)
    assert rel < _SCALED_TOL["ragged_m"], rel


@pytest.mark.parametrize("layout", SCALED_LAYOUTS)
def test_scaled_grouped_mxfp8_agrees_with_epilogue_path(layout):
    """`scale_dtype` selects a kernel, not a result."""
    args = _make_scaled_layout(
        layout,
        [5, 0, 130, 41, 1],
        "blockwise",
        32,
        "fp8_e4m3",
        seed=1,
        scale_dtype="fp8_e8m0",
    )
    aq, bq, sa, sb, offs, kbs = args
    mx = triton_scaled_grouped_gemm(
        aq, bq, sa, sb, offs, torch.float32, kbs, scale_dtype="fp8_e8m0"
    )
    epilogue = triton_scaled_grouped_gemm(aq, bq, sa, sb, offs, torch.float32, kbs)
    rel = (mx - epilogue).norm() / epilogue.norm().clamp_min(1e-12)
    assert rel < 1e-5, rel


@pytest.mark.parametrize("layout", SCALED_LAYOUTS)
def test_scaled_grouped_mxfp8_adds_no_error_over_dequantizing(layout):
    """As for the dense kernel: fp32-rounding tight, not quantization-noise loose.

    Per-layout tolerances here are 1e-5 (ragged_k) to 2e-2 (ragged_m/n), so this is
    what would catch the group cursor drifting a scale block -- which stays well inside
    2e-2 while being plainly wrong.
    """
    aq, bq, sa, sb, offs, kbs = _make_scaled_layout(
        layout, _SCALED_COUNTS, "blockwise", 32, "fp8_e4m3", scale_dtype="fp8_e8m0"
    )
    got = triton_scaled_grouped_gemm(
        aq, bq, sa, sb, offs, torch.float32, kbs, scale_dtype="fp8_e8m0"
    )
    oracle = scaled_grouped_gemm_ref(aq, bq, sa, sb, offs, kbs)
    rel = (got.float() - oracle).norm() / oracle.norm().clamp_min(1e-12)
    assert rel < 1e-6, rel


def test_scaled_grouped_mxfp8_kernel_declines_block_size_zero():
    """As for the dense kernel's block_size-0 regression: 0 % 32 == 0 in Python, so an
    unguarded condition would route the "one scale block spans the whole segment"
    sentinel into the replication branch, collapsing rep_k to 0 and handing the kernel
    a zero-width scale tensor -- a CUDA illegal memory access hit during development.
    """
    aq, bq, sa, sb, offs, kbs = _make_scaled_layout(
        "ragged_m", _SCALED_COUNTS, "tensorwise", 0, "fp8_e4m3", scale_dtype="fp8_e8m0"
    )
    got = triton_scaled_grouped_gemm(
        aq, bq, sa, sb, offs, torch.float32, kbs, scale_dtype="fp8_e8m0"
    )
    assert got.shape == _expected_shape("ragged_m", _SCALED_COUNTS)
    assert torch.isfinite(got).all()


def test_scaled_grouped_mxfp8_declines_int8():
    """tl.dot_scaled has no integer format, so int8 must stay on the other kernel."""
    aq, bq, sa, sb, offs, kbs = _make_scaled_layout(
        "ragged_m", _SCALED_COUNTS, "blockwise", 32, "int8", scale_dtype="fp8_e8m0"
    )
    got = triton_scaled_grouped_gemm(
        aq, bq, sa, sb, offs, torch.float32, kbs, scale_dtype="fp8_e8m0"
    )
    ref = scaled_grouped_gemm_ref(aq, bq, sa, sb, offs, kbs)
    rel = (got.float() - ref).norm() / ref.norm().clamp_min(1e-12)
    assert rel < _SCALED_TOL["ragged_m"], rel


@pytest.mark.parametrize("layout", SCALED_LAYOUTS)
def test_scaled_grouped_layouts_uneven_groups(layout):
    counts = [5, 0, 130, 41, 1]
    aq, bq, sa, sb, offs, kbs = _make_scaled_layout(
        layout, counts, "blockwise", 32, "fp8_e4m3", seed=1
    )
    got = triton_scaled_grouped_gemm(aq, bq, sa, sb, offs, torch.float32, kbs)
    ref = scaled_grouped_gemm_ref(aq, bq, sa, sb, offs, kbs)
    rel = (got.float() - ref).norm() / ref.norm().clamp_min(1e-12)
    assert rel < _SCALED_TOL[layout], rel


def test_scaled_grouped_ragged_n_shape_and_group_isolation():
    # (E,M,K) x (K,R) -> (M,R): offs partitions b's columns, so each group's column
    # block must come from its own expert slice of a
    counts = [17, 0, 40, 7]
    aq, bq, sa, sb, offs, kbs = _make_scaled_layout(
        "ragged_n", counts, "rowwise", 0, "fp8_e4m3", M=32, K=64
    )
    got = triton_scaled_grouped_gemm(aq, bq, sa, sb, offs, torch.float32, kbs)
    ref = scaled_grouped_gemm_ref(aq, bq, sa, sb, offs, kbs)
    assert got.shape == (32, sum(counts))
    rel = (got.float() - ref).norm() / ref.norm().clamp_min(1e-12)
    assert rel < 0.02, rel


def test_scaled_grouped_ragged_k_empty_group_is_zero():
    # an expert with no tokens owns no blockwise scale blocks; its (M,N) slice of the
    # (E,M,N) output must still be zeroed rather than left uninitialized
    counts = [8, 0, 8]
    aq, bq, sa, sb, offs, kbs = _make_scaled_layout(
        "ragged_k", counts, "blockwise", 32, "fp8_e4m3"
    )
    out = triton_scaled_grouped_gemm(aq, bq, sa, sb, offs, torch.float32, kbs)
    assert (out[1] == 0).all()


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("layout", SCALED_LAYOUTS)
def test_scaled_grouped_layouts_out_dtype(layout, out_dtype):
    aq, bq, sa, sb, offs, kbs = _make_scaled_layout(
        layout, _SCALED_COUNTS, "rowwise", 0, "fp8_e4m3", seed=3
    )
    out = triton_scaled_grouped_gemm(aq, bq, sa, sb, offs, out_dtype, kbs)
    assert out.dtype == out_dtype


@pytest.mark.parametrize(
    "gran,bs", [("tensorwise", 0), ("rowwise", 0), ("blockwise", 32)]
)
@pytest.mark.parametrize("layout", ("ragged_m", "ragged_k"))
def test_scaled_grouped_bias_matches_oracle(layout, gran, bs):
    """Fused (G,N) bias, broadcast over the output's row dim.

    Blockwise is the discriminating case: the bias is unscaled and belongs on the
    final accumulator, so folding it into a per-scale-block partial would both
    multiply it by that block's scales and add it once per block.
    """
    aq, bq, sa, sb, offs, kbs = _make_scaled_layout(
        layout, _SCALED_COUNTS, gran, bs, "fp8_e4m3"
    )
    bias = torch.randn(offs.shape[0], 48, device="cuda", dtype=torch.bfloat16)
    got = triton_scaled_grouped_gemm(
        aq, bq, sa, sb, offs, torch.float32, kbs, bias=bias
    )
    ref = scaled_grouped_gemm_ref(aq, bq, sa, sb, offs, kbs, bias=bias)
    rel = (got.float() - ref).norm() / ref.norm()
    assert rel < _SCALED_TOL[layout], rel


def test_scaled_grouped_bias_none_unchanged():
    """bias=None must be byte-identical to omitting it -- the HAS_BIAS=False path."""
    aq, bq, sa, sb, offs, kbs = _make_scaled_layout(
        "ragged_m", _SCALED_COUNTS, "rowwise", 0, "fp8_e4m3"
    )
    torch.testing.assert_close(
        triton_scaled_grouped_gemm(aq, bq, sa, sb, offs, torch.float32, kbs, bias=None),
        triton_scaled_grouped_gemm(aq, bq, sa, sb, offs, torch.float32, kbs),
        rtol=0,
        atol=0,
    )


def test_scaled_grouped_bias_rejected_for_ragged_n():
    """Ragged N partitions the columns, so a (G,N) row-broadcast bias is meaningless."""
    aq, bq, sa, sb, offs, kbs = _make_scaled_layout(
        "ragged_n", _SCALED_COUNTS, "rowwise", 0, "fp8_e4m3"
    )
    bias = torch.zeros(offs.shape[0], bq.shape[1], device="cuda", dtype=torch.bfloat16)
    with pytest.raises((KernelSelectionError, NotImplementedError)):
        triton_scaled_grouped_gemm(aq, bq, sa, sb, offs, torch.float32, kbs, bias=bias)


def test_bench_scaled_grouped_gemm_importable():
    m = importlib.import_module("benchmarks.gemm.bench_scaled_grouped_gemm")
    assert hasattr(m, "main")


def test_quantize_operand_2d_compiles_fullgraph():
    scaling = {
        "granularity": "blockwise",
        "block_shape": (32, 32),
        "scale_dtype": "fp32",
    }
    x = rand(256, 512)
    fn = torch.compile(
        lambda: quantize_operand(x, -1, FMT[E4M3], scaling), fullgraph=True
    )
    codes, scale = fn()
    assert scale.shape == (256, 16)


def _prune_scaled_configs(configs, named_args):
    scale_block = named_args["SCALE_BLOCK_SIZE"]
    return gemm._early_prune_scaled_configs(
        configs,
        named_args,
        scale_block,
        False,
        None,
        False,
    )


def _prune_scaled_grouped_configs(configs, named_args):
    scale_block = named_args["SCALE_BLOCK_SIZE"]
    return gemm._early_prune_scaled_grouped_configs(
        configs,
        named_args,
        36,
        True,
        False,
        scale_block,
        False,
        None,
        False,
    )


@pytest.mark.parametrize(
    "prune,configs",
    [
        (_prune_scaled_configs, gemm._SCALED_CONFIGS),
        (_prune_scaled_grouped_configs, gemm._SCALED_GROUPED_CONFIGS),
    ],
    ids=["scaled", "scaled_grouped"],
)
def test_early_prune_caps_block_k_at_the_scale_block(prune, configs):
    """A BLOCK_K wider than one scale block only ever runs masked-off lanes, so the
    pruner drops it -- but must never prune the list empty, which Triton reports as
    an unrelated autotune failure. The floor is 32, the narrowest fp8/int8 tl.dot.
    Each scaled kernel owns a pruner, so this runs against both.
    """
    for scale_block in (16, 32, 128, 4096):
        kept = prune(list(configs), {"SCALE_BLOCK_SIZE": scale_block})
        assert kept, f"pruned empty at scale_block={scale_block}"
        assert all(c.kwargs["BLOCK_K"] <= max(32, scale_block) for c in kept)
        # only the overshooting configs go, never a legal one
        assert {id(c) for c in kept} == {
            id(c) for c in configs if c.kwargs["BLOCK_K"] <= max(32, scale_block)
        }

    # 0 means one block spanning the whole contraction segment, which leaves BLOCK_K
    # free -- capping it at the floor would hide every wide config. Only
    # scaled_grouped_gemm passes it, its contraction being ragged with no single
    # width to resolve it to; both pruners handle it, so both are held to it.
    assert prune(list(configs), {"SCALE_BLOCK_SIZE": 0}) == list(configs)


def test_scaled_grouped_eligibility_rejects_malformed_scale_layout():
    aq = torch.empty((4, 8), device="cuda", dtype=torch.int8)
    bq = torch.empty((2, 8, 6), device="cuda", dtype=torch.int8)
    sa = torch.empty((4, 2), device="cuda")
    sb = torch.empty((2, 2, 5), device="cuda")
    offs = torch.tensor([2, 4], device="cuda", dtype=torch.int32)
    support = gemm.scaled_grouped_gemm_eligibility(
        (aq, bq, sa, sb, offs, torch.float32, 4, None, None), {}
    )
    assert not support.supported
    assert support.reason == "invalid scale layout"
