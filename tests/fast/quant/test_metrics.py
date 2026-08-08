import torch
import pytest

from src.quant.quantize import dequantize_operand, quantize_operand
from src.quant.utils import str_to_qmax
from src.utils.metric_utils import compute_quantization_metrics

METRICS = ("sqnr", "underflow_rate", "clip_rate")
SPREAD_METRICS = ("sqnr_min", "underflow_rate_max", "clip_rate_max")


# --- compute_quantization_metrics: a pure function of (source, dequantized, codes) ---


def _codes(source, dequantized):
    """Stand-in codes for the hand-built (source, dequantized) pairs below: the
    metrics only read which entries are zero and which sit at qmax."""
    return dequantized


def test_identical_tensors_have_infinite_sqnr_and_no_loss():
    x = torch.randn(16, 32)
    m = compute_quantization_metrics(x, x, _codes(x, x), 1e9)
    assert m["sqnr"].item() > 200.0  # error clamped to EPS, not zero
    assert m["underflow_rate"].item() == 0.0
    assert m["clip_rate"].item() == 0.0


def test_underflow_counts_nonzero_source_whose_codes_are_zero():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    deq = torch.tensor([[1.0, 0.0, 0.0, 4.0]])
    m = compute_quantization_metrics(x, deq, _codes(x, deq), 1e9)
    assert m["underflow_rate"].item() == 0.5


def test_underflow_ignores_zeros_already_in_the_source():
    x = torch.tensor([[0.0, 0.0, 1.0, 2.0]])
    m = compute_quantization_metrics(x, x, _codes(x, x), 1e9)["underflow_rate"]
    assert m.item() == 0.0


def test_clip_rate_counts_codes_sitting_at_qmax():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    codes = torch.tensor([[7.0, -7.0, 3.0, 4.0]])
    m = compute_quantization_metrics(x, x, codes, 7.0)
    assert m["clip_rate"].item() == 0.5  # both tails count


def test_sqnr_matches_hand_computed_ratio():
    x = torch.tensor([[3.0, 4.0]])  # norm 5
    deq = torch.tensor([[3.0, 3.0]])  # error norm 1
    expected = 20.0 * torch.log10(torch.tensor(5.0))
    m = compute_quantization_metrics(x, deq, _codes(x, deq), 1e9)
    assert m["sqnr"].item() == pytest.approx(expected.item(), rel=1e-5)


def test_metrics_are_zero_dim_float32_and_device_local():
    x = torch.randn(8, 8, dtype=torch.bfloat16)
    m = compute_quantization_metrics(x, x, _codes(x, x), 1e9)
    assert set(m) == set(METRICS)
    for value in m.values():
        assert value.ndim == 0 and value.dtype == torch.float32


def test_spread_keys_appear_only_for_grouped_operands():
    x = torch.randn(8, 8)
    assert set(compute_quantization_metrics(x, x, _codes(x, x), 1e9)) == set(METRICS)
    offs = torch.tensor([4, 8], dtype=torch.int32)
    grouped = compute_quantization_metrics(x, x, _codes(x, x), 1e9, offs=offs)
    assert set(grouped) == set(METRICS) | set(SPREAD_METRICS)


def test_one_bad_expert_shows_in_the_spread_but_not_the_mean():
    """The mean averages a cold expert's noise away; sqnr_min is what catches it."""
    x = torch.ones(4, 4)
    deq = x.clone()
    deq[2:] = 0.5  # expert 1 reconstructs badly, expert 0 exactly
    offs = torch.tensor([2, 4], dtype=torch.int32)
    m = compute_quantization_metrics(x, deq, _codes(x, deq), 1e9, offs=offs)
    assert m["sqnr_min"].item() < m["sqnr"].item()


def test_spread_excludes_empty_experts():
    x = torch.randn(6, 4)
    offs = torch.tensor([6, 6], dtype=torch.int32)  # expert 1 got no tokens
    m = compute_quantization_metrics(x, x, _codes(x, x), 1e9, offs=offs)
    assert torch.isfinite(m["sqnr_min"])  # not the +inf sentinel for the empty one
    assert m["underflow_rate_max"].item() == 0.0


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp8_e5m2", "int8"])
def test_sqnr_positive_and_finite_over_a_real_round_trip(fmt):
    torch.manual_seed(0)
    x = torch.randn(64, 64)
    q, scale = quantize_operand(x, -1, "tensorwise", 0, fmt)
    m = compute_quantization_metrics(
        x, dequantize_operand(q, scale, -1, 0), q, str_to_qmax(fmt)
    )
    assert torch.isfinite(m["sqnr"]) and m["sqnr"].item() > 10.0


def test_outlier_inflated_scale_shows_up_as_underflow():
    # One huge outlier inflates the tensorwise scale so the small bulk underflows.
    x = torch.zeros(1, 256)
    x[0, 0] = 1000.0
    x[0, 1:] = 1e-3  # well below (1000/448) * 2^-9 for e4m3
    q, scale = quantize_operand(x, -1, "tensorwise", 0, "fp8_e4m3")
    m = compute_quantization_metrics(
        x, dequantize_operand(q, scale, -1, 0), q, str_to_qmax("fp8_e4m3")
    )
    assert m["underflow_rate"].item() > 0.9


def test_tight_scale_shows_up_as_clipping():
    # A scale from the bulk, not the amax, saturates the outliers instead.
    x = torch.full((1, 256), 1e-3)
    x[0, 0] = 1000.0
    scale = torch.full((1, 1), 1e-3 / str_to_qmax("int8"))
    codes = (x / scale).round().clamp(-127.0, 127.0)
    m = compute_quantization_metrics(x, codes * scale, codes, str_to_qmax("int8"))
    assert m["clip_rate"].item() > 0.0
