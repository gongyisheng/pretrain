import torch
import pytest
from src.quant.utils import str_to_min_subnormal, str_to_qmax
from src.quant.quantize import operand_quotient, fake_quantize_operand


def test_min_subnormal_values():
    assert str_to_min_subnormal("fp8_e4m3") == 2**-9
    assert str_to_min_subnormal("fp8_e5m2") == 2**-16
    assert str_to_min_subnormal("int8") == 0.5
    assert str_to_min_subnormal("int4") == 0.5


def test_min_subnormal_rejects_e8m0():
    with pytest.raises(KeyError):
        str_to_min_subnormal("fp8_e8m0")


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp8_e5m2"])
def test_operand_quotient_dequant_matches_fake_quant(fmt):
    torch.manual_seed(0)
    x = torch.randn(64, 128) * 3.0
    q_abs, deq, xf = operand_quotient(x, -1, "tensorwise", 0, fmt)
    ref = fake_quantize_operand(x, -1, "tensorwise", 0, fmt).float()
    assert torch.allclose(deq, ref, atol=0, rtol=0)
    # amax element maps to exactly qmax under amax scaling
    assert q_abs.max().item() == pytest.approx(str_to_qmax(fmt), rel=1e-4)
    assert xf.shape == x.shape and deq.shape == x.shape


def test_operand_quotient_rowwise_matches_fake_quant():
    torch.manual_seed(1)
    x = torch.randn(32, 96)
    _, deq, _ = operand_quotient(x, -1, "rowwise", 0, "fp8_e4m3")
    ref = fake_quantize_operand(x, -1, "rowwise", 0, "fp8_e4m3").float()
    assert torch.allclose(deq, ref, atol=1e-6)
