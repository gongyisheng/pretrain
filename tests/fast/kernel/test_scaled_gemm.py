import pytest
import torch

from src.quant.scale import quantize_operand, fake_quantize_operand

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="scaled_gemm is a CUDA Triton kernel"
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
    from src.kernel.gemm import scaled_gemm
    from src.quant.scale import effective_block_size

    ebs = effective_block_size(gran, bs, a.shape[1])
    aq, sa = quantize_operand(a, -1, gran, bs, **a_kw)
    bq, sb = quantize_operand(b, 0, gran, bs, **b_kw)
    out = scaled_gemm(aq, bq, sa, sb, torch.float32, ebs)
    oracle = (
        fake_quantize_operand(a, -1, gran, bs, **a_kw).float()
        @ fake_quantize_operand(b, 0, gran, bs, **b_kw).float()
    )
    return out, oracle


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
    from src.kernel.gemm import scaled_gemm

    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, "rowwise", 0, **_fp8_kw(E4M3))
    bq, sb = quantize_operand(b, 0, "rowwise", 0, **_fp8_kw(E4M3))
    out = scaled_gemm(aq, bq, sa, sb, out_dtype, 128)
    assert out.dtype == out_dtype


def test_bench_scaled_gemm_importable():
    import importlib

    m = importlib.import_module("benchmarks.gemm.bench_scaled_gemm")
    assert hasattr(m, "main")


def test_compiles_fullgraph():
    from src.kernel.gemm import scaled_gemm

    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, "blockwise", 32, **_fp8_kw(E4M3))
    bq, sb = quantize_operand(b, 0, "blockwise", 32, **_fp8_kw(E4M3))
    fn = torch.compile(
        lambda: scaled_gemm(aq, bq, sa, sb, torch.bfloat16, 32), fullgraph=True
    )
    assert torch.isfinite(fn()).all()
