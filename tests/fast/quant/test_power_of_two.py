import pytest
import torch

from src.quant.scale import quantize_operand, fake_quantize_operand

E4M3 = torch.float8_e4m3fn
KW = dict(fmt="fp8_e4m3", scale_dtype="fp8_e8m0")


def test_pow2_scale_block32_shape_and_exactness():
    torch.manual_seed(0)
    x = torch.randn(4, 64)
    xq, s = quantize_operand(x, -1, "blockwise", 32, **KW)
    assert xq.dtype == E4M3 and s.shape == (4, 2)
    log2s = torch.log2(s[s > 0])
    assert torch.allclose(log2s, log2s.round())


def test_pow2_fake_quant_reasonable_error():
    torch.manual_seed(0)
    x = torch.randn(8, 128)
    fq = fake_quantize_operand(x, -1, "blockwise", 32, **KW)
    assert (fq - x).norm() / x.norm() < 0.15


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_pow2_scaled_gemm_matches_oracle():
    from src.kernel.gemm import scaled_gemm

    torch.manual_seed(0)
    a = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, "blockwise", 32, **KW)
    bq, sb = quantize_operand(b, 0, "blockwise", 32, **KW)
    out = scaled_gemm(aq, bq, sa, sb, torch.bfloat16, 32)
    oracle = (
        fake_quantize_operand(a, -1, "blockwise", 32, **KW).float()
        @ fake_quantize_operand(b, 0, "blockwise", 32, **KW).float()
    )
    assert (out.float() - oracle).norm() / oracle.norm() < 0.02
