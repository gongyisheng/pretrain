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
    scaled_grouped_gemm_wgrad,
)
from src.quant.quantize import (
    effective_block_size,
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
    scaled_grouped_gemm_wgrad_ref,
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
    ebs = effective_block_size(gran, bs, a.shape[1])
    aq, sa = quantize_operand(a, -1, gran, bs, **a_kw)
    bq, sb = quantize_operand(b, 0, gran, bs, **b_kw)
    out = scaled_gemm(aq, bq, sa, sb, torch.float32, ebs)
    oracle = scaled_gemm_ref(aq, bq, sa, sb, ebs)
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
    out = scaled_gemm(aq, bq, sa, sb, out_dtype, 128)
    assert out.dtype == out_dtype


def test_bench_scaled_gemm_importable():
    m = importlib.import_module("benchmarks.gemm.bench_scaled_gemm")
    assert hasattr(m, "main")


# ---------------------------------------------------------------------------
# Scaled grouped GEMM: fp8/int8 ragged expert matmul
# ---------------------------------------------------------------------------


def _quant_grouped(a, b, counts, gran, bs, a_kw, b_kw):
    """Quantize A (R,K) and per-expert B (E,K,N); return aq,bq,sa,sb,offs,ebs."""
    ebs = effective_block_size(gran, bs, a.shape[1])
    aq, sa = quantize_operand(a, -1, gran, bs, **a_kw)
    bq_list, sb_list = [], []
    for g in range(b.shape[0]):
        bqg, sbg = quantize_operand(b[g], 0, gran, bs, **b_kw)
        bq_list.append(bqg)
        sb_list.append(sbg)
    bq = torch.stack(bq_list)  # (E, K, N)
    sb = torch.stack(sb_list)  # (E, nkb, N)
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    return aq, bq, sa, sb, offs, ebs


def _make_grouped_ab(counts, K, N, seed=0):
    torch.manual_seed(seed)
    R, E = sum(counts), len(counts)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(E, K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    return a, b


def test_scaled_grouped_gemm_oracle_native_agrees_with_dequant():
    # The oracle uses torch._scaled_grouped_mm when applicable (per-row fp8 on
    # sm90/sm100) and dequants otherwise. Whichever path it takes, it must match
    # an explicit dequant matmul. On sm120 the native op raises and the oracle
    # already is the dequant path -> the two agree exactly.
    counts = [128, 128]
    a, b = _make_grouped_ab(counts, K=64, N=48)
    aq, bq, sa, sb, offs, ebs = _quant_grouped(
        a, b, counts, "rowwise", 0, _fp8_kw(E4M3), _fp8_kw(E4M3)
    )
    oracle = scaled_grouped_gemm_ref(aq, bq, sa, sb, offs, ebs)
    a_deq, b_deq = _dequant_a(aq, sa, ebs), _dequant_b_grouped(bq, sb, ebs)
    dequant = torch.cat([a_deq[:128] @ b_deq[0], a_deq[128:] @ b_deq[1]])
    assert oracle.shape == (256, 48)
    assert (oracle - dequant).norm() / dequant.norm() < 0.02


@pytest.mark.parametrize(
    "gran,bs",
    [("rowwise", 0), ("tensorwise", 0), ("blockwise", 32)],
)
def test_scaled_grouped_gemm_fp8_matches_oracle(gran, bs):
    counts = [128, 128, 128, 128]
    a, b = _make_grouped_ab(counts, K=64, N=48)
    aq, bq, sa, sb, offs, ebs = _quant_grouped(
        a, b, counts, gran, bs, _fp8_kw(E4M3), _fp8_kw(E4M3)
    )
    out = scaled_grouped_gemm(aq, bq, sa, sb, offs, torch.float32, ebs)
    oracle = scaled_grouped_gemm_ref(aq, bq, sa, sb, offs, ebs)
    assert out.shape == (512, 48) and out.dtype == torch.float32
    assert (out - oracle).norm() / oracle.norm() < 0.02


def test_scaled_grouped_gemm_empty_and_uneven_groups():
    counts = [5, 0, 130, 41, 1]
    a, b = _make_grouped_ab(counts, K=64, N=48, seed=1)
    aq, bq, sa, sb, offs, ebs = _quant_grouped(
        a, b, counts, "rowwise", 0, _fp8_kw(E4M3), _fp8_kw(E4M3)
    )
    out = scaled_grouped_gemm(aq, bq, sa, sb, offs, torch.float32, ebs)
    oracle = scaled_grouped_gemm_ref(aq, bq, sa, sb, offs, ebs)
    assert (out - oracle).norm() / oracle.norm() < 0.02


@pytest.mark.parametrize("qmax", [127, 31, 7])
def test_scaled_grouped_gemm_int_matches_oracle(qmax):
    counts = [64, 0, 130, 41]
    a, b = _make_grouped_ab(counts, K=128, N=64, seed=2)
    aq, bq, sa, sb, offs, ebs = _quant_grouped(
        a, b, counts, "rowwise", 0, _int_kw(qmax), _int_kw(qmax)
    )
    out = scaled_grouped_gemm(aq, bq, sa, sb, offs, torch.float32, ebs)
    oracle = scaled_grouped_gemm_ref(aq, bq, sa, sb, offs, ebs)
    assert (out - oracle).norm() / oracle.norm() < 0.02


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_scaled_grouped_gemm_output_dtype_respected(out_dtype):
    counts = [64, 130]
    a, b = _make_grouped_ab(counts, K=64, N=48, seed=3)
    aq, bq, sa, sb, offs, ebs = _quant_grouped(
        a, b, counts, "rowwise", 0, _fp8_kw(E4M3), _fp8_kw(E4M3)
    )
    out = scaled_grouped_gemm(aq, bq, sa, sb, offs, out_dtype, ebs)
    assert out.dtype == out_dtype


def test_scaled_grouped_gemm_compiles_fullgraph():
    counts = [64, 0, 130, 41]
    a, b = _make_grouped_ab(counts, K=64, N=48, seed=4)
    aq, bq, sa, sb, offs, ebs = _quant_grouped(
        a, b, counts, "blockwise", 32, _fp8_kw(E4M3), _fp8_kw(E4M3)
    )
    fn = torch.compile(
        lambda: scaled_grouped_gemm(aq, bq, sa, sb, offs, torch.bfloat16, ebs),
        fullgraph=True,
    )
    assert torch.isfinite(fn()).all()


def test_bench_scaled_grouped_gemm_importable():
    m = importlib.import_module("benchmarks.gemm.bench_scaled_grouped_gemm")
    assert hasattr(m, "main")


def _quant_wgrad(a, g, fmt, gran="rowwise", bs=0, offs=None):
    """Quantize X (R,K) and gY (R,N) along the ragged token axis (dim 0) for wgrad.

    Scale blocks restart at each expert, so both operands share one mapping.
    """
    blocks = ragged_scale_blocks(offs, a.shape[0], gran, bs)
    aq, sa = quantize_operand(a, 0, gran, bs, fmt=fmt, ragged=blocks)
    gq, sg = quantize_operand(g, 0, gran, bs, fmt=fmt, ragged=blocks)
    kernel_bs = bs if gran == "blockwise" else 0
    return aq, gq, sa, sg, kernel_bs


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "int8"])
@pytest.mark.parametrize(
    "gran,bs",
    [
        ("rowwise", 0),
        ("tensorwise", 0),
        ("blockwise", 32),
        ("blockwise", 48),
        ("blockwise", 128),
    ],
)
@pytest.mark.parametrize(
    "counts,K,N",
    [
        ([128, 128, 128, 128], 64, 48),  # scale blocks aligned to expert boundaries
        ([5, 0, 130, 41, 1], 64, 48),  # empty group, short trailing blocks
        ([300, 5, 120, 0, 44, 210], 512, 384),  # many blocks per expert
    ],
)
def test_scaled_grouped_gemm_wgrad_matches_oracle(counts, K, N, gran, bs, fmt):
    torch.manual_seed(0)
    R = sum(counts)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(R, N, device="cuda", dtype=torch.bfloat16) * 0.1
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    aq, gq, sa, sg, kbs = _quant_wgrad(a, g, fmt, gran, bs, offs)
    got = scaled_grouped_gemm_wgrad(aq, gq, sa, sg, offs, torch.float32, kbs)
    ref = scaled_grouped_gemm_wgrad_ref(aq, gq, sa, sg, offs, kbs)
    assert got.shape == (len(counts), K, N)
    rel = (got.float() - ref).norm() / ref.norm().clamp_min(1e-12)
    assert rel < 1e-5, rel


def test_scaled_grouped_gemm_wgrad_empty_group_is_zero():
    counts = [8, 0, 8]
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
    aq, gq, sa, sg, kbs = _quant_wgrad(a, g, "fp8_e4m3", "blockwise", 32, offs)
    out = scaled_grouped_gemm_wgrad(aq, gq, sa, sg, offs, torch.float32, kbs)
    # an expert with no tokens owns no scale blocks; its slice must still be zeroed
    assert (out[1] == 0).all()


def test_scaled_grouped_gemm_wgrad_blockwise_compiles_fullgraph():
    torch.manual_seed(0)
    counts = [64, 0, 130, 41]
    R = sum(counts)
    a = torch.randn(R, 64, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(R, 48, device="cuda", dtype=torch.bfloat16) * 0.1
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    aq, gq, sa, sg, kbs = _quant_wgrad(a, g, "fp8_e4m3", "blockwise", 32, offs)
    fn = torch.compile(
        lambda: scaled_grouped_gemm_wgrad(aq, gq, sa, sg, offs, torch.bfloat16, kbs),
        fullgraph=True,
    )
    assert torch.isfinite(fn()).all()


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
