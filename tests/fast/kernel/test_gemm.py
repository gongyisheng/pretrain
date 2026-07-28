"""Parity of the Triton grouped GEMM prototype vs torch._grouped_mm (bf16),
plus the scaled (quantized) GEMM Triton kernel vs a fake-quantize oracle."""

import importlib

import pytest
import torch

from src.kernel import gemm
from src.kernel.gemm import (
    _grouped_gemm,
    _grouped_gemm_wgrad,
    grouped_gemm,
    scaled_gemm,
    scaled_grouped_gemm,
)
from src.quant.quantize import effective_block_size, quantize_operand
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
    got = _grouped_gemm_wgrad(a, grad_c, offs)
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
    got = _grouped_gemm_wgrad(a, grad_c, offs)
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
