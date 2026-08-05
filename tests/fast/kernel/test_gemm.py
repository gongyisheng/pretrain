"""Parity of the Triton grouped GEMM prototype vs torch._grouped_mm (bf16),
plus the scaled (quantized) GEMM Triton kernel vs a fake-quantize oracle."""

import importlib

import pytest
import torch

from src.kernel import gemm
from src.kernel.gemm import (
    _grouped_gemm,
    grouped_gemm,
    scaled_gemm,
    scaled_grouped_gemm,
)
from src.quant.quantize import (
    effective_block_size,
    kernel_block_size,
    quantize_operand,
    ragged_scale_blocks,
)
from tests.fast.kernel._refs import (
    _dequant_a,
    _dequant_b_grouped,
    grouped_gemm_ref,
    grouped_gemm_wgrad_ref,
    scaled_gemm_ref,
    scaled_grouped_gemm_ref,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton GEMM kernels are CUDA only"
)

E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2
FMT = {E4M3: "fp8_e4m3", E5M2: "fp8_e5m2"}
INT_FMT = {127: "int8", 63: "int7", 31: "int6", 15: "int5", 7: "int4"}


def _fp8_kw(dt, scale_dtype=None):
    return dict(fmt=FMT[dt], scale_dtype=scale_dtype)


def _int_kw(qmax):
    return dict(fmt=INT_FMT[qmax])


def _run(a, b, gran, bs, a_kw, b_kw):
    # the kernel takes the 0-sentinel width; the oracle's pad math takes a count
    aq, sa = quantize_operand(a, -1, gran, bs, **a_kw)
    bq, sb = quantize_operand(b, 0, gran, bs, **b_kw)
    out = scaled_gemm(aq, bq, sa, sb, torch.float32, kernel_block_size(gran, bs))
    oracle = scaled_gemm_ref(aq, bq, sa, sb, effective_block_size(gran, bs, a.shape[1]))
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
    got = grouped_gemm(a, b, offs)
    assert got.shape == ref.shape
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_all_ragged_layouts_grads_match_torch(layout):
    """Every layout's dgrad/wgrad lands in another layout of the same set, so the
    set is only differentiable once all three are implemented."""
    a0, b0, offs = _make_layout(layout)

    def run(fn):
        a = a0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        out = fn(a, b, offs)
        (out * out).sum().backward()
        return out, a.grad, b.grad

    ref = run(lambda a, b, o: torch._grouped_mm(a, b, offs=o))
    got = run(lambda a, b, o: _grouped_gemm(a, b, o))
    for g, r in zip(got, ref):
        torch.testing.assert_close(g.float(), r.float(), rtol=2e-2, atol=2e-2)


def test_output_stride_matches_torch():
    """A last dim that is not 8-element aligned is where torch's padding shows."""
    a, b, offs = _make_layout("ragged_m", N=51)
    assert (
        grouped_gemm(a, b, offs).stride() == torch._grouped_mm(a, b, offs=offs).stride()
    )


def _bias_ref(a, b, offs, bias, layout):
    """torch._grouped_mm plus a (G,N) bias broadcast over the output's row dim.

    torch rejects bias outright, so the oracle adds it separately: per output row
    for ragged M (each row belongs to one group), per group slice for ragged K.
    """
    out = torch._grouped_mm(a, b, offs=offs).float()
    if layout == "ragged_k":
        return out + bias.float()[:, None, :]
    rows = torch.arange(out.shape[0], device=out.device)
    return out + bias.float()[torch.searchsorted(offs, rows, right=True)]


@pytest.mark.parametrize("layout", ("ragged_m", "ragged_k"))
def test_bias_matches_reference(layout):
    a, b, offs = _make_layout(layout)
    G, N = offs.shape[0], 48
    bias = torch.randn(G, N, device="cuda", dtype=torch.bfloat16)
    got = grouped_gemm(a, b, offs, bias=bias)
    ref = _bias_ref(a, b, offs, bias, layout)
    torch.testing.assert_close(got.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("layout", ("ragged_m", "ragged_k"))
def test_bias_grads_match_reference(layout):
    a0, b0, offs = _make_layout(layout)
    G, N = offs.shape[0], 48
    bias0 = torch.randn(G, N, device="cuda", dtype=torch.bfloat16)

    def run(fn):
        a, b = a0.clone().requires_grad_(True), b0.clone().requires_grad_(True)
        bias = bias0.clone().requires_grad_(True)
        out = fn(a, b, offs, bias)
        (out * out).sum().backward()
        return out, a.grad, b.grad, bias.grad

    ref = run(lambda a, b, o, bias: _bias_ref(a, b, o, bias, layout))
    got = run(lambda a, b, o, bias: _grouped_gemm(a, b, o, bias))
    for g, r in zip(got, ref):
        torch.testing.assert_close(g.float(), r.float(), rtol=2e-2, atol=2e-2)


def test_bias_rejected_for_ragged_n():
    """Ragged N partitions the columns, so a (G,N) row-broadcast bias is meaningless."""
    a, b, offs = _make_layout("ragged_n")
    bias = torch.zeros(offs.shape[0], b.shape[1], device="cuda", dtype=torch.bfloat16)
    with pytest.raises((ValueError, AssertionError, NotImplementedError)):
        grouped_gemm(a, b, offs, bias=bias)


def test_parity_balanced_groups():
    a, b, offs = _make([128, 128, 128, 128], K=64, N=48)
    got = grouped_gemm(a, b, offs)
    ref = grouped_gemm_ref(a, b, offs)
    assert got.shape == ref.shape
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_parity_empty_and_uneven_groups():
    # includes an empty group (0 rows) and uneven sizes
    a, b, offs = _make([5, 0, 130, 41, 1], K=64, N=48)
    got = grouped_gemm(a, b, offs)
    torch.testing.assert_close(
        got.float(), grouped_gemm_ref(a, b, offs).float(), rtol=2e-2, atol=2e-2
    )


def test_parity_single_group():
    a, b, offs = _make([200], K=192, N=64)
    got = grouped_gemm(a, b, offs)
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
    got = grouped_gemm(a, b, offs)
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
    ref = grouped_gemm_wgrad_ref(a, grad_c, offs)
    got = grouped_gemm(a.mT, grad_c, offs)
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
    ref = grouped_gemm_wgrad_ref(a, grad_c, offs)
    got = grouped_gemm(a.mT, grad_c, offs)
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_grouped_gemm_fwd_bwd_matches_torch():
    torch.manual_seed(0)
    counts = [64, 0, 130, 41]
    R, K, N = sum(counts), 64, 48
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a0 = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(len(counts), N, K, device="cuda", dtype=torch.bfloat16) * 0.1

    def run(fn):
        a = a0.clone().requires_grad_(True)
        wv = w.clone().requires_grad_(True)
        c = fn(a, wv.mT, offs)
        c.backward(torch.ones_like(c))
        return c, a.grad, wv.grad

    c_ref, ga_ref, gw_ref = run(lambda a, b, o: torch._grouped_mm(a, b, offs=o))
    c_got, ga_got, gw_got = run(lambda a, b, o: _grouped_gemm(a, b, o))
    torch.testing.assert_close(c_got.float(), c_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(ga_got.float(), ga_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(gw_got.float(), gw_ref.float(), rtol=2e-2, atol=2e-2)


def test_grouped_gemm_compiles_fullgraph_and_matches_eager():
    torch.manual_seed(0)
    counts = [64, 0, 130, 41]
    R, K, N = sum(counts), 64, 48
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a0 = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(len(counts), N, K, device="cuda", dtype=torch.bfloat16) * 0.1

    def fn(a, b, o):
        return _grouped_gemm(a, b, o)

    def run(f):
        a = a0.clone().requires_grad_(True)
        wv = w.clone().requires_grad_(True)
        c = f(a, wv.mT, offs)
        c.sum().backward()
        return c, a.grad, wv.grad

    eager = run(fn)
    # fullgraph=True raises if the op graph-breaks
    compiled_fn = torch.compile(fn, fullgraph=True)
    comp = run(compiled_fn)
    for e, c in zip(eager, comp):
        torch.testing.assert_close(c.float(), e.float(), rtol=2e-2, atol=2e-2)


def test_dispatch_impl_selection_and_parity():
    torch.manual_seed(0)
    counts = [64, 0, 130, 41]
    R, K, N = sum(counts), 64, 48
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(len(counts), N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = w.mT
    ref = torch._grouped_mm(a, b, offs=offs)
    for impl in ("auto", "triton", "torch"):
        got = gemm.grouped_gemm(a, b, offs, impl)
        torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)
    # torch path is exactly torch._grouped_mm
    assert torch.equal(gemm.grouped_gemm(a, b, offs, "torch"), ref)
    # invalid impl rejected
    with pytest.raises((ValueError, AssertionError)):
        gemm.grouped_gemm(a, b, offs, "nonsense")


def test_grouped_gemm_auto_compiles_fullgraph():
    torch.manual_seed(0)
    counts = [64, 0, 130, 41]
    R, K, N = sum(counts), 64, 48
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(len(counts), N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    fn = torch.compile(lambda a, b, o: grouped_gemm(a, b, o, "auto"), fullgraph=True)
    out = fn(a, w.mT, offs)
    torch.testing.assert_close(
        out.float(),
        torch._grouped_mm(a, w.mT, offs=offs).float(),
        rtol=2e-2,
        atol=2e-2,
    )


def test_grouped_gemm_auto_selects_triton_on_this_arch():
    major = torch.cuda.get_device_capability()[0]
    if major in (9, 10):
        pytest.skip("fused torch path available; auto uses torch here")
    counts = [64, 0, 130, 41]
    R, K, N = sum(counts), 64, 48
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    b = (torch.randn(len(counts), N, K, device="cuda", dtype=torch.bfloat16) * 0.1).mT
    # auto must take the Triton path here → bitwise-identical to forcing triton,
    # and (almost certainly) NOT bitwise-equal to the torch fallback.
    auto = gemm.grouped_gemm(a, b, offs, "auto")
    assert torch.equal(auto, gemm.grouped_gemm(a, b, offs, "triton"))


def test_grouped_gemm_auto_uses_torch_for_non_bf16():
    counts = [64, 130]
    R, K, N = sum(counts), 64, 48
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a = torch.randn(R, K, device="cuda", dtype=torch.float32)
    b = torch.randn(len(counts), N, K, device="cuda", dtype=torch.float32).mT
    assert torch.equal(
        gemm.grouped_gemm(a, b, offs, "auto"), torch._grouped_mm(a, b, offs=offs)
    )


def test_dispatch_triton_rejects_non_bf16():
    counts = [64, 130]
    R, K, N = sum(counts), 64, 48
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a = torch.randn(R, K, device="cuda", dtype=torch.float32)  # not bf16
    w = torch.randn(len(counts), N, K, device="cuda", dtype=torch.float32) * 0.1
    b = w.mT
    with pytest.raises(ValueError):
        gemm.grouped_gemm(a, b, offs, "triton")


@pytest.mark.parametrize(
    "gran,bs",
    [
        ("tensorwise", 0),
        ("rowwise", 0),
        ("blockwise", 32),
        ("blockwise", 64),
        ("blockwise", 128),
    ],
)
def test_fp8_e4m3_matches_oracle(gran, bs):
    torch.manual_seed(0)
    a = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 96, device="cuda", dtype=torch.bfloat16)
    out, oracle = _run(a, b, gran, bs, _fp8_kw(E4M3), _fp8_kw(E4M3))
    assert out.shape == (128, 96) and out.dtype == torch.float32
    assert (out - oracle).norm() / oracle.norm() < 0.02


def test_fp8_e5m2_times_e5m2_ok():
    # cuBLAS rejects e5m2 x e5m2; the Triton kernel accepts it.
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    out, oracle = _run(a, b, "rowwise", 0, _fp8_kw(E5M2), _fp8_kw(E5M2))
    assert torch.isfinite(out).all()
    assert (out - oracle).norm() / oracle.norm() < 0.05


def test_fp8_mixed_e5m2_e4m3():
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    out, oracle = _run(a, b, "blockwise", 32, _fp8_kw(E5M2), _fp8_kw(E4M3))
    assert (out - oracle).norm() / oracle.norm() < 0.05


def test_pow2_e8m0_blockwise_matches_oracle():
    # power-of-two (E8M0 / MX-style) blockwise scale through the real kernel.
    torch.manual_seed(0)
    a = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    kw = _fp8_kw(E4M3, scale_dtype="fp8_e8m0")
    out, oracle = _run(a, b, "blockwise", 32, kw, kw)
    assert (out - oracle).norm() / oracle.norm() < 0.02


@pytest.mark.parametrize("qmax", [127, 63, 31, 15, 7])
def test_int_matches_oracle(qmax):
    torch.manual_seed(0)
    a = torch.randn(96, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    out, oracle = _run(a, b, "rowwise", 0, _int_kw(qmax), _int_kw(qmax))
    assert (out - oracle).norm() / oracle.norm() < 0.02


def test_nonmultiple_shapes_mask_correctly():
    torch.manual_seed(0)
    a = torch.randn(70, 100, device="cuda", dtype=torch.bfloat16)  # M,K odd
    b = torch.randn(100, 130, device="cuda", dtype=torch.bfloat16)  # N odd, K%32!=0
    out, oracle = _run(a, b, "blockwise", 32, _fp8_kw(E4M3), _fp8_kw(E4M3))
    assert out.shape == (70, 130)
    assert (out - oracle).norm() / oracle.norm() < 0.03


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_output_dtype_respected(out_dtype):
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, "rowwise", 0, **_fp8_kw(E4M3))
    bq, sb = quantize_operand(b, 0, "rowwise", 0, **_fp8_kw(E4M3))
    out = scaled_gemm(aq, bq, sa, sb, out_dtype, 0)  # rowwise -> one block over K
    assert out.dtype == out_dtype


def test_bench_scaled_gemm_importable():
    m = importlib.import_module("benchmarks.gemm.bench_scaled_gemm")
    assert hasattr(m, "main")


# ---------------------------------------------------------------------------
# Scaled grouped GEMM: fp8/int8 ragged expert matmul
# ---------------------------------------------------------------------------


SCALED_LAYOUTS = ("ragged_m", "ragged_k", "ragged_n")
_SCALED_COUNTS = [64, 0, 130, 46]  # sums to 240; the empty group is deliberate
# The ragged-contraction layout dequants the same codes as its oracle and only
# reorders the fp32 accumulation, so it must agree far more tightly than the
# forward layouts, whose oracle pads scale blocks independently of the tiling.
_SCALED_TOL = {"ragged_m": 0.02, "ragged_k": 1e-5, "ragged_n": 0.02}


def _make_scaled_layout(layout, counts, gran, bs, fmt, M=32, K=64, N=48, seed=0):
    """Quantized operands + scales for one ragged layout of the scaled grouped GEMM.

    Scales follow the kernel's invariant: each mirrors its operand's axis order with
    the contraction axis replaced by the scale-block axis. Returns the kernel's
    block_size too, where 0 means one block per contraction segment.
    """
    torch.manual_seed(seed)
    E, R = len(counts), sum(counts)
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    kernel_bs = bs if gran == "blockwise" else 0
    kw = dict(fmt=fmt)

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.1

    if layout == "ragged_m":  # (R,K) x (E,K,N) -> (R,N)
        aq, sa = quantize_operand(rand(R, K), -1, gran, bs, **kw)
        per = [quantize_operand(rand(K, N), 0, gran, bs, **kw) for _ in range(E)]
        bq = torch.stack([q for q, _ in per])
        sb = torch.stack([s for _, s in per])
        return aq, bq, sa, sb, offs, kernel_bs

    if layout == "ragged_k":  # (M,R) x (R,N) -> (E,M,N)
        blocks = ragged_scale_blocks(offs, R, gran, bs)
        aq, sa = quantize_operand(rand(R, M), 0, gran, bs, ragged=blocks, **kw)
        bq, sb = quantize_operand(rand(R, N), 0, gran, bs, ragged=blocks, **kw)
        return aq.mT, bq, sa.mT, sb, offs, kernel_bs

    if layout == "ragged_n":  # (E,M,K) x (K,R) -> (M,R)
        per = [quantize_operand(rand(M, K), -1, gran, bs, **kw) for _ in range(E)]
        aq = torch.stack([q for q, _ in per])
        sa = torch.stack([s for _, s in per])
        # b's ragged axis is its columns. A per-column scale cannot pool amax across
        # groups by construction, so only tensorwise needs `ragged` -- reached here
        # through the existing row-ragged path on the transpose.
        blocks = ragged_scale_blocks(offs, R, gran, bs)
        bqT, sbT = quantize_operand(rand(R, K), -1, gran, bs, ragged=blocks, **kw)
        return aq, bqT.mT, sa, sbT.mT, offs, kernel_bs

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
    b_deq = _dequant_b_grouped(bq, sb, kbs or K)
    dequant = torch.cat([a_deq[:128] @ b_deq[0], a_deq[128:] @ b_deq[1]])
    assert oracle.shape == (256, 48)
    assert (oracle - dequant).norm() / dequant.norm() < 0.02


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "int8"])
@pytest.mark.parametrize(
    "gran,bs",
    # 48 is deliberately not a multiple of any BLOCK_K: the kernel masks the
    # reduction to the scale block's end rather than clamping BLOCK_K to it, so
    # correctness must not depend on which config autotune picks.
    [("rowwise", 0), ("tensorwise", 0), ("blockwise", 32), ("blockwise", 48)],
)
@pytest.mark.parametrize("layout", SCALED_LAYOUTS)
def test_scaled_grouped_layouts_match_oracle(layout, gran, bs, fmt):
    aq, bq, sa, sb, offs, kbs = _make_scaled_layout(
        layout, _SCALED_COUNTS, gran, bs, fmt
    )
    got = scaled_grouped_gemm(aq, bq, sa, sb, offs, torch.float32, kbs)
    ref = scaled_grouped_gemm_ref(aq, bq, sa, sb, offs, kbs)
    assert got.shape == _expected_shape(layout, _SCALED_COUNTS)
    rel = (got.float() - ref).norm() / ref.norm().clamp_min(1e-12)
    assert rel < _SCALED_TOL[layout], rel


@pytest.mark.parametrize("layout", SCALED_LAYOUTS)
def test_scaled_grouped_layouts_uneven_groups(layout):
    counts = [5, 0, 130, 41, 1]
    aq, bq, sa, sb, offs, kbs = _make_scaled_layout(
        layout, counts, "blockwise", 32, "fp8_e4m3", seed=1
    )
    got = scaled_grouped_gemm(aq, bq, sa, sb, offs, torch.float32, kbs)
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
    got = scaled_grouped_gemm(aq, bq, sa, sb, offs, torch.float32, kbs)
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
    out = scaled_grouped_gemm(aq, bq, sa, sb, offs, torch.float32, kbs)
    assert (out[1] == 0).all()


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("layout", SCALED_LAYOUTS)
def test_scaled_grouped_layouts_out_dtype(layout, out_dtype):
    aq, bq, sa, sb, offs, kbs = _make_scaled_layout(
        layout, _SCALED_COUNTS, "rowwise", 0, "fp8_e4m3", seed=3
    )
    out = scaled_grouped_gemm(aq, bq, sa, sb, offs, out_dtype, kbs)
    assert out.dtype == out_dtype


@pytest.mark.parametrize("layout", SCALED_LAYOUTS)
def test_scaled_grouped_layouts_compile_fullgraph(layout):
    aq, bq, sa, sb, offs, kbs = _make_scaled_layout(
        layout, _SCALED_COUNTS, "blockwise", 32, "fp8_e4m3", seed=4
    )
    fn = torch.compile(
        lambda: scaled_grouped_gemm(aq, bq, sa, sb, offs, torch.bfloat16, kbs),
        fullgraph=True,
    )
    assert torch.isfinite(fn()).all()


def test_bench_scaled_grouped_gemm_importable():
    m = importlib.import_module("benchmarks.gemm.bench_scaled_grouped_gemm")
    assert hasattr(m, "main")


def test_compiles_fullgraph():
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, "blockwise", 32, **_fp8_kw(E4M3))
    bq, sb = quantize_operand(b, 0, "blockwise", 32, **_fp8_kw(E4M3))
    fn = torch.compile(
        lambda: scaled_gemm(aq, bq, sa, sb, torch.bfloat16, 32), fullgraph=True
    )
    assert torch.isfinite(fn()).all()
