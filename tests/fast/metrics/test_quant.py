import math

import torch
import pytest

from src.quant.quantize import (
    dequantize_operand,
    quantize_operand,
)
from src.metrics.functional import compute_quantization_metrics
from src.metrics.quant import (
    QuantizationStats,
    accumulate_quantization_sums,
    record_operand,
    set_quantization_monitoring_status,
)
from src.quant.rotation import HadamardRotation
from src.quant.utils import is_fp4

_TENSORWISE = {
    "granularity": "tensorwise",
    "enable_global_scale": False,
    "block_shape": (0, 0),
    "scale_dtype": torch.float32,
}

METRICS = ("sqnr", "underflow_rate")
OPERAND_SHAPES = [(6, 2), (3, 2, 2)]


# --- compute_quantization_metrics: a pure function of (source, dequantized, codes) ---


def _metrics(source, dequantized, codes):
    """The one-shot form these tests were written against: fold one operand, then
    read it. Production splits the two so a window can span many micro-batches."""
    sums = accumulate_quantization_sums(source, codes, dequantized)
    stats = QuantizationStats("operand", source.device)
    set_quantization_monitoring_status(True)
    try:
        record_operand(stats, torch.stack(sums, dim=-1))
    finally:
        set_quantization_monitoring_status(False)
    return {
        key.removesuffix("/operand"): value
        for key, value in compute_quantization_metrics(stats).items()
    }


def _codes(source, dequantized):
    """Stand-in codes for the hand-built (source, dequantized) pairs below: the
    metrics only read which entries are zero."""
    return dequantized


def test_identical_tensors_have_infinite_sqnr_and_no_loss():
    x = torch.randn(16, 32)
    m = _metrics(x, x, _codes(x, x))
    assert m["sqnr"] > 200.0  # error clamped to EPS, not zero
    assert m["underflow_rate"] == 0.0


def test_underflow_counts_nonzero_source_whose_codes_are_zero():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    deq = torch.tensor([[1.0, 0.0, 0.0, 4.0]])
    m = _metrics(x, deq, _codes(x, deq))
    assert m["underflow_rate"] == 0.5


def test_underflow_ignores_zeros_already_in_the_source():
    x = torch.tensor([[0.0, 0.0, 1.0, 2.0]])
    m = _metrics(x, x, _codes(x, x))["underflow_rate"]
    assert m == 0.0


def test_accumulate_quantization_sums_counts_packed_fp4_underflow():
    source = torch.ones(1, 16)
    codes = torch.full((1, 8), 0x11, dtype=torch.uint8)
    codes[0, 0] = 0x10
    sums = accumulate_quantization_sums(source, codes, source, contract_dim=-1)
    assert sums[2].item() == 1.0


def test_underflow_rate_is_not_diluted_by_zeros_already_in_the_source():
    """The rate is over values that could underflow at all. An exact zero in the
    source can never round to zero, so padding with zeros must not move the rate.
    Grads carry many exact zeros (ignore_index=-100), as do relu activations."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    deq = torch.tensor([[1.0, 0.0, 0.0, 4.0]])
    plain = _metrics(x, deq, _codes(x, deq))["underflow_rate"]

    padded_x = torch.cat([x, torch.zeros(1, 12)], dim=1)
    padded_deq = torch.cat([deq, torch.zeros(1, 12)], dim=1)
    padded = _metrics(padded_x, padded_deq, _codes(padded_x, padded_deq))
    assert plain == pytest.approx(0.5)
    assert padded["underflow_rate"] == pytest.approx(0.5)


def test_sqnr_is_already_immune_to_zeros_in_the_source():
    """A ratio of two sums of squares: an exact zero contributes 0 to both, so the
    zero fraction cancels. Guards against 'fixing' sqnr along with the rates."""
    x = torch.tensor([[3.0, 4.0]])
    deq = torch.tensor([[3.0, 3.0]])
    plain = _metrics(x, deq, _codes(x, deq))["sqnr"]

    padded_x = torch.cat([x, torch.zeros(1, 6)], dim=1)
    padded_deq = torch.cat([deq, torch.zeros(1, 6)], dim=1)
    padded = _metrics(padded_x, padded_deq, _codes(padded_x, padded_deq))
    assert padded["sqnr"] == pytest.approx(plain)


@pytest.mark.parametrize("shape", OPERAND_SHAPES)
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_accumulate_quantization_sums(shape, transpose, dtype):
    source = torch.arange(1, 13, dtype=dtype).reshape(shape)
    if transpose:
        source = source.mT
    sums = accumulate_quantization_sums(source, source, source)
    expected = (
        torch.tensor([650.0]),
        torch.zeros(1),
        torch.zeros(1),
        torch.tensor([12.0]),
        torch.tensor([12.0]),
    )
    for got, want in zip(sums, expected):
        assert got.dtype == torch.float32 and got.device == source.device
        assert torch.equal(got, want)


def test_sqnr_matches_hand_computed_ratio():
    x = torch.tensor([[3.0, 4.0]])  # norm 5
    deq = torch.tensor([[3.0, 3.0]])  # error norm 1
    expected = 20.0 * torch.log10(torch.tensor(5.0))
    m = _metrics(x, deq, _codes(x, deq))
    assert m["sqnr"] == pytest.approx(expected.item(), rel=1e-5)


@pytest.mark.parametrize("shape", OPERAND_SHAPES)
@pytest.mark.parametrize("transpose", [False, True])
def test_compute_quantization_metrics(shape, transpose):
    source = torch.tensor([10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0])
    dequantized = source.clone()
    dequantized[4:6] = 0
    source = source.reshape(shape)
    dequantized = dequantized.reshape(shape)
    if transpose:
        source, dequantized = source.mT, dequantized.mT
    metrics = _metrics(source, dequantized, dequantized)
    assert set(metrics) == set(METRICS)
    assert metrics["sqnr"] == pytest.approx(10.0 * math.log10(402.0 / 2.0))
    assert metrics["underflow_rate"] == pytest.approx(2.0 / 6.0)


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp8_e5m2", "int8"])
def test_sqnr_positive_and_finite_over_a_real_round_trip(fmt):
    torch.manual_seed(0)
    x = torch.randn(64, 64)
    q, scale, _, _ = quantize_operand(x, -1, fmt, _TENSORWISE)
    m = _metrics(x, dequantize_operand(q, scale, -1, _TENSORWISE), q)
    assert math.isfinite(m["sqnr"]) and m["sqnr"] > 10.0


def test_outlier_inflated_scale_shows_up_as_underflow():
    # One huge outlier inflates the tensorwise scale so the small bulk underflows.
    x = torch.zeros(1, 256)
    x[0, 0] = 1000.0
    x[0, 1:] = 1e-3  # well below (1000/448) * 2^-9 for e4m3
    q, scale, _, _ = quantize_operand(x, -1, "fp8_e4m3", _TENSORWISE)
    m = _metrics(x, dequantize_operand(q, scale, -1, _TENSORWISE), q)
    assert m["underflow_rate"] > 0.9


def test_record_operand_rotation_aligns_underflow_with_codes():
    source = torch.ones(1, 4)
    rotation = HadamardRotation(block_size=4, random_sign=False)
    codes, scale, global_scale, quantization_stats = quantize_operand(
        source,
        -1,
        "int8",
        _TENSORWISE,
        rotation=rotation,
        return_quantization_stats=True,
    )
    stats = QuantizationStats("act/x", source.device)

    set_quantization_monitoring_status(True)
    try:
        record_operand(stats, quantization_stats)
    finally:
        set_quantization_monitoring_status(False)

    dequantized = dequantize_operand(
        codes,
        scale,
        -1,
        _TENSORWISE,
        rotation=rotation,
        global_scale=global_scale,
    )
    assert stats.err_sq.item() == pytest.approx(
        (source - dequantized).square().sum().item()
    )
    assert stats.under.item() == 0.0
    assert stats.nonzero.item() == 1.0


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("with_stats", [False, True])
def test_record_operand(enabled, with_stats):
    sums = torch.arange(5, dtype=torch.float32).reshape(1, 5).requires_grad_()
    stats = QuantizationStats("act/x", sums.device) if with_stats else None
    set_quantization_monitoring_status(enabled)
    try:
        record_operand(stats, sums)
        record_operand(stats, sums)
    finally:
        set_quantization_monitoring_status(False)
    if stats is not None:
        for name, value in zip(stats.FIELDS, sums.unbind(-1)):
            actual = getattr(stats, name)
            assert not actual.requires_grad
            assert torch.equal(
                actual, value * 2 if enabled else torch.zeros_like(value)
            )


_NVFP4 = {
    "granularity": "blockwise",
    "enable_global_scale": True,
    "block_shape": (1, 16),
    "scale_dtype": torch.float8_e4m3fn,
}


GLOBAL_SCALE_FORMATS = ["int4", "fp4_e2m1", "fp4_e2m1_4over6"]


@pytest.mark.parametrize("fmt", GLOBAL_SCALE_FORMATS)
def test_record_operand_global_scale_tracks_true_error(fmt):
    """The monitored error must use the global scale the operand was quantized with.

    Dropped, the metric dequantizes codes at the wrong magnitude entirely and reports
    the operand itself as error -- near 1.0 -- no matter how well the quantizer did.
    """
    torch.manual_seed(0)
    source = torch.randn(8, 128, device="cuda") * 1e4
    codes, scale, global_scale, quantization_stats = quantize_operand(
        source, -1, fmt, _NVFP4, return_quantization_stats=True
    )
    stats = QuantizationStats("act/x", source.device)

    set_quantization_monitoring_status(True)
    try:
        record_operand(stats, quantization_stats)
    finally:
        set_quantization_monitoring_status(False)

    dequantized = dequantize_operand(
        codes, scale, -1, _NVFP4, global_scale=global_scale
    )
    if is_fp4(fmt):
        assert codes.dtype is torch.uint8 and codes.shape == (8, 64)
    assert global_scale is not None and torch.isfinite(global_scale).all()
    assert stats.err_sq.item() == pytest.approx(
        (source - dequantized).square().sum().item(), rel=1e-6
    )
    # NVFP4 and int4 over 16-wide blocks; the ratio is ~1.0 if global_scale drops.
    assert (stats.err_sq / stats.src_sq).item() < 3.7e-2


# --- accumulation: the property the fold/read split exists to provide ---


def _holder(sites):
    """The read walks a model for accumulators; these tests build them bare."""
    return torch.nn.ModuleList(sites)


def _fold(stats, source, fmt="fp8_e4m3", offs=None, ragged_dim=None):
    """Quantize one operand and fold it in, the way `record_operand` does."""
    codes, scale, _, _ = quantize_operand(
        source, -1, fmt, _TENSORWISE, offs=offs, ragged_dim=ragged_dim
    )
    deq = dequantize_operand(
        codes, scale, -1, _TENSORWISE, offs=offs, ragged_dim=ragged_dim
    )
    sums = accumulate_quantization_sums(source, codes, deq)
    for name, value in zip(stats.FIELDS, sums):
        getattr(stats, name).add_(value)


def test_folding_chunks_equals_folding_the_whole_operand():
    """Accumulation is exact: the sums over N chunks equal the sums over their
    concatenation. Quantized once up front, so this isolates the accumulation --
    quantizing each chunk separately would pick a different tensorwise scale per
    chunk and change the error itself, which is a property of the scale, not of
    the fold.
    """
    torch.manual_seed(0)
    x = torch.randn(24, 32)
    codes, scale, _, _ = quantize_operand(x, -1, "fp8_e4m3", _TENSORWISE)
    deq = dequantize_operand(codes, scale, -1, _TENSORWISE)

    whole = accumulate_quantization_sums(x, codes, deq)
    split = [torch.zeros_like(t) for t in whole]
    for lo in range(0, 24, 8):
        chunk = slice(lo, lo + 8)
        part = accumulate_quantization_sums(x[chunk], codes[chunk], deq[chunk])
        split = [acc + p for acc, p in zip(split, part)]

    for got, expected in zip(split, whole):
        assert got.item() == pytest.approx(expected.item(), rel=1e-6)


def test_compute_quantization_metrics_pools_window_sums():
    first = torch.tensor([[100.0, 1.0, 0.0, 4.0, 4.0]])
    second = torch.tensor([[2.0, 1.01, 2.0, 12.0, 3.0]])
    stats = QuantizationStats("act/x", first.device)
    set_quantization_monitoring_status(True)
    try:
        record_operand(stats, first)
        record_operand(stats, second)
    finally:
        set_quantization_monitoring_status(False)

    metrics = compute_quantization_metrics(_holder([stats]))
    expected_sqnr = 10.0 * torch.log10(torch.tensor(102.0 / 2.01))
    assert metrics["sqnr/act/x"] == pytest.approx(expected_sqnr.item())
    assert metrics["underflow_rate/act/x"] == pytest.approx(2.0 / 7.0)
    assert set(metrics) == {"sqnr/act/x", "underflow_rate/act/x"}


def test_reset_clears_every_accumulator_field():
    stats = QuantizationStats("act/x", torch.randn(1).device)
    _fold(stats, torch.randn(8, 32))
    assert stats.numel.item() > 0
    stats.reset()
    assert all(getattr(stats, name).item() == 0 for name in stats.FIELDS)


def test_compute_reads_every_site_under_its_own_key():
    device = torch.randn(1).device
    sites = [QuantizationStats(f"act/layer{i}", device) for i in range(3)]
    for site in sites:
        _fold(site, torch.randn(8, 32))
    got = compute_quantization_metrics(_holder(sites))
    for i in range(3):
        for metric in METRICS:
            assert f"{metric}/act/layer{i}" in got
    assert all(isinstance(v, float) for v in got.values())


def test_compute_on_no_sites_is_empty():
    assert compute_quantization_metrics(_holder([])) == {}


def test_never_folded_site_produces_no_key():
    """Reset but never folded -- e.g. a dgrad/wgrad operand during a no_grad eval
    pass -- must report no key at all, not a spurious sqnr of 0.0."""
    device = torch.randn(1).device
    site = QuantizationStats("grad_out/x", device)
    got = compute_quantization_metrics(_holder([site]))
    assert got == {}
    assert "sqnr/grad_out/x" not in got


def test_mixed_folded_and_unfolded_sites_only_report_the_folded_one():
    """The production shape: one folded site alongside one that was reset for the
    window but never folded (e.g. eval's backward operands)."""
    device = torch.randn(1).device
    folded = QuantizationStats("act/x", device)
    unfolded = QuantizationStats("grad_out/x", device)
    _fold(folded, torch.randn(8, 32))
    got = compute_quantization_metrics(_holder([folded, unfolded]))
    for metric in METRICS:
        assert f"{metric}/act/x" in got
        assert f"{metric}/grad_out/x" not in got


def test_each_operand_is_folded_under_its_own_element_format():
    """The metrics need the format's qmax and resolution: a coarser format must
    show a higher underflow rate on the same tensor. Previously this rode on the
    snapshot's `fmt` field; now the format reaches the fold directly."""
    torch.manual_seed(0)
    x = torch.randn(64, 64)
    rates = {}
    for fmt in ("int4", "int8"):
        stats = QuantizationStats(f"act/{fmt}", x.device)
        _fold(stats, x, fmt=fmt)
        rates[fmt] = compute_quantization_metrics(_holder([stats]))[
            f"underflow_rate/act/{fmt}"
        ]
    assert rates["int4"] > rates["int8"]
