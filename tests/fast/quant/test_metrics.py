import torch
import pytest
from src.quant.utils import str_to_min_subnormal, str_to_qmax
from src.quant.quantize import operand_quotient, fake_quantize_operand
from src.quant.metrics import compute_operand_metrics


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


def test_metrics_passthrough_returns_none():
    x = torch.randn(8, 8)
    assert compute_operand_metrics(x, "bf16", "tensorwise", 0, -1) is None


def test_metrics_underflow_on_outlier_inflated_scale():
    # one huge outlier inflates tensorwise scale so the small bulk underflows.
    x = torch.zeros(1, 256)
    x[0, 0] = 1000.0  # outlier
    x[0, 1:] = 1e-3  # bulk, well below (1000/448)*2^-9 for e4m3
    m = compute_operand_metrics(x, "fp8_e4m3", "tensorwise", 0, -1)
    assert m["underflow_rate"].item() > 0.9  # nearly all bulk flushed
    assert m["range_ratio"].item() > 100.0  # amax/median huge


def test_metrics_clip_zero_under_amax_scaling():
    torch.manual_seed(0)
    x = torch.randn(32, 64)
    m = compute_operand_metrics(x, "fp8_e4m3", "tensorwise", 0, -1)
    # amax maps exactly to qmax; nothing strictly exceeds it.
    assert m["clip_rate"].item() == 0.0


def test_metrics_sqnr_finite_and_positive_for_smooth_tensor():
    torch.manual_seed(0)
    x = torch.randn(64, 64)
    m = compute_operand_metrics(x, "fp8_e4m3", "tensorwise", 0, -1)
    assert torch.isfinite(m["sqnr"]).all()
    assert m["sqnr"].item() > 10.0  # e4m3 on N(0,1) is well above 10 dB


def test_metrics_all_zero_dim_float32():
    x = torch.randn(8, 8)
    m = compute_operand_metrics(x, "fp8_e5m2", "tensorwise", 0, -1)
    for k in ("underflow_rate", "clip_rate", "range_ratio", "sqnr"):
        assert m[k].ndim == 0 and m[k].dtype == torch.float32
