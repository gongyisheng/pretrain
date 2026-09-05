import torch
import pytest

from src.quant.quantize import dequantize_operand, quantize_operand
from src.metrics.functional import _quantization_metrics, compute_quantization_metrics
from src.metrics.quant import (
    QuantizationStats,
    accumulate_quantization_sums,
    record_operand,
    set_quantization_monitoring_status,
)
from src.quant.rotation import HadamardRotation

_TENSORWISE = {
    "granularity": "tensorwise",
    "block_shape": (0, 0),
    "scale_dtype": torch.float32,
}

METRICS = ("sqnr", "underflow_rate")
SPREAD_METRICS = ("sqnr_min", "underflow_rate_max")
RAGGED_DIM_CASES = [(-2, False), (-1, True)]


# --- compute_quantization_metrics: a pure function of (source, dequantized, codes) ---


def _metrics(source, dequantized, codes, offs=None):
    """The one-shot form these tests were written against: fold one operand, then
    read it. Production splits the two so a window can span many micro-batches."""
    sums = accumulate_quantization_sums(source, codes, dequantized, offs=offs)
    return _quantization_metrics(*sums, grouped=offs is not None)


def _codes(source, dequantized):
    """Stand-in codes for the hand-built (source, dequantized) pairs below: the
    metrics only read which entries are zero."""
    return dequantized


def test_identical_tensors_have_infinite_sqnr_and_no_loss():
    x = torch.randn(16, 32)
    m = _metrics(x, x, _codes(x, x))
    assert m["sqnr"].item() > 200.0  # error clamped to EPS, not zero
    assert m["underflow_rate"].item() == 0.0


def test_underflow_counts_nonzero_source_whose_codes_are_zero():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    deq = torch.tensor([[1.0, 0.0, 0.0, 4.0]])
    m = _metrics(x, deq, _codes(x, deq))
    assert m["underflow_rate"].item() == 0.5


def test_underflow_ignores_zeros_already_in_the_source():
    x = torch.tensor([[0.0, 0.0, 1.0, 2.0]])
    m = _metrics(x, x, _codes(x, x))["underflow_rate"]
    assert m.item() == 0.0


def test_underflow_rate_is_not_diluted_by_zeros_already_in_the_source():
    """The rate is over values that could underflow at all. An exact zero in the
    source can never round to zero, so padding with zeros must not move the rate.
    Grads carry many exact zeros (ignore_index=-100), as do relu activations."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    deq = torch.tensor([[1.0, 0.0, 0.0, 4.0]])
    plain = _metrics(x, deq, _codes(x, deq))["underflow_rate"].item()

    padded_x = torch.cat([x, torch.zeros(1, 12)], dim=1)
    padded_deq = torch.cat([deq, torch.zeros(1, 12)], dim=1)
    padded = _metrics(padded_x, padded_deq, _codes(padded_x, padded_deq))
    assert plain == pytest.approx(0.5)
    assert padded["underflow_rate"].item() == pytest.approx(0.5)


def test_sqnr_is_already_immune_to_zeros_in_the_source():
    """A ratio of two sums of squares: an exact zero contributes 0 to both, so the
    zero fraction cancels. Guards against 'fixing' sqnr along with the rates."""
    x = torch.tensor([[3.0, 4.0]])
    deq = torch.tensor([[3.0, 3.0]])
    plain = _metrics(x, deq, _codes(x, deq))["sqnr"].item()

    padded_x = torch.cat([x, torch.zeros(1, 6)], dim=1)
    padded_deq = torch.cat([deq, torch.zeros(1, 6)], dim=1)
    padded = _metrics(padded_x, padded_deq, _codes(padded_x, padded_deq))
    assert padded["sqnr"].item() == pytest.approx(plain)


def test_an_all_zero_expert_stays_valid():
    """numel, not nonzero, is the validity signal: an expert that received tokens
    whose values happened to be all zero must not be dropped like an empty one."""
    source = torch.zeros(3, 4, 8)  # 3 experts of stacked weights
    source[0] = torch.randn(4, 8)
    offs = torch.tensor([4, 8, 12])
    sums = accumulate_quantization_sums(source, source, source, offs=offs)
    numel, nonzero = sums[-2], sums[-1]
    assert (numel > 0).all()
    assert nonzero[1].item() == 0.0 and nonzero[2].item() == 0.0


@pytest.mark.parametrize("ragged_dim,transpose", RAGGED_DIM_CASES)
def test_accumulate_quantization_sums_ragged_dim(ragged_dim, transpose):
    source = torch.arange(1, 11, dtype=torch.float32).reshape(5, 2)
    if transpose:
        source = source.mT
    offs = torch.tensor([2, 5])
    sums = accumulate_quantization_sums(
        source,
        source,
        source,
        offs=offs,
        ragged_dim=ragged_dim,
    )
    expected = (
        torch.tensor([30.0, 355.0]),
        torch.zeros(2),
        torch.zeros(2),
        torch.tensor([4.0, 6.0]),
        torch.tensor([4.0, 6.0]),
    )
    for got, want in zip(sums, expected):
        assert torch.equal(got, want)


def test_sqnr_matches_hand_computed_ratio():
    x = torch.tensor([[3.0, 4.0]])  # norm 5
    deq = torch.tensor([[3.0, 3.0]])  # error norm 1
    expected = 20.0 * torch.log10(torch.tensor(5.0))
    m = _metrics(x, deq, _codes(x, deq))
    assert m["sqnr"].item() == pytest.approx(expected.item(), rel=1e-5)


def test_metrics_are_zero_dim_float32_and_device_local():
    x = torch.randn(8, 8, dtype=torch.bfloat16)
    m = _metrics(x, x, _codes(x, x))
    assert set(m) == set(METRICS)
    for value in m.values():
        assert value.ndim == 0 and value.dtype == torch.float32


def test_spread_keys_appear_only_for_grouped_operands():
    x = torch.randn(8, 8)
    assert set(_metrics(x, x, _codes(x, x))) == set(METRICS)
    offs = torch.tensor([4, 8], dtype=torch.int32)
    grouped = _metrics(x, x, _codes(x, x), offs=offs)
    assert set(grouped) == set(METRICS) | set(SPREAD_METRICS)


def test_one_bad_expert_shows_in_the_spread_but_not_the_mean():
    """The mean averages a cold expert's noise away; sqnr_min is what catches it."""
    x = torch.ones(4, 4)
    deq = x.clone()
    deq[2:] = 0.5  # expert 1 reconstructs badly, expert 0 exactly
    offs = torch.tensor([2, 4], dtype=torch.int32)
    m = _metrics(x, deq, _codes(x, deq), offs=offs)
    assert m["sqnr_min"].item() < m["sqnr"].item()


def test_spread_excludes_empty_experts():
    x = torch.randn(6, 4)
    offs = torch.tensor([6, 6], dtype=torch.int32)  # expert 1 got no tokens
    m = _metrics(x, x, _codes(x, x), offs=offs)
    assert torch.isfinite(m["sqnr_min"])  # not the +inf sentinel for the empty one
    assert m["underflow_rate_max"].item() == 0.0


def test_sqnr_min_ignores_an_all_zero_expert():
    """numel > 0 keeps an all-zero expert valid (test_an_all_zero_expert_stays_valid),
    but sqnr_min must still skip it: its sqnr is 0.0 with no error to report, and
    must not sink sqnr_min next to a healthy sibling near 200+ dB."""
    x = torch.zeros(2, 4, 8)
    x[0] = torch.randn(4, 8)  # expert 0 healthy, expert 1 all-zero
    offs = torch.tensor([4, 8])
    m = _metrics(x, x, _codes(x, x), offs=offs)
    assert torch.isfinite(m["sqnr_min"])
    assert m["sqnr_min"].item() > 100.0


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp8_e5m2", "int8"])
def test_sqnr_positive_and_finite_over_a_real_round_trip(fmt):
    torch.manual_seed(0)
    x = torch.randn(64, 64)
    q, scale = quantize_operand(x, -1, fmt, _TENSORWISE)
    m = _metrics(x, dequantize_operand(q, scale, -1, _TENSORWISE), q)
    assert torch.isfinite(m["sqnr"]) and m["sqnr"].item() > 10.0


def test_outlier_inflated_scale_shows_up_as_underflow():
    # One huge outlier inflates the tensorwise scale so the small bulk underflows.
    x = torch.zeros(1, 256)
    x[0, 0] = 1000.0
    x[0, 1:] = 1e-3  # well below (1000/448) * 2^-9 for e4m3
    q, scale = quantize_operand(x, -1, "fp8_e4m3", _TENSORWISE)
    m = _metrics(x, dequantize_operand(q, scale, -1, _TENSORWISE), q)
    assert m["underflow_rate"].item() > 0.9


def test_record_operand_rotation_aligns_underflow_with_codes():
    source = torch.ones(1, 4)
    rotation = HadamardRotation(block_size=4, random_sign=False)
    codes, scale = quantize_operand(source, -1, "int8", _TENSORWISE, rotation=rotation)
    stats = QuantizationStats("act/x", 1, source.device)

    set_quantization_monitoring_status(True)
    try:
        record_operand(
            stats,
            source,
            codes,
            scale,
            -1,
            _TENSORWISE,
            rotation=rotation,
        )
    finally:
        set_quantization_monitoring_status(False)

    dequantized = dequantize_operand(codes, scale, -1, _TENSORWISE, rotation=rotation)
    assert stats.err_sq.item() == pytest.approx(
        (source - dequantized).square().sum().item()
    )
    assert stats.under.item() == 0.0
    assert stats.nonzero.item() == 1.0


# --- accumulation: the property the fold/read split exists to provide ---


def _holder(sites):
    """The read walks a model for accumulators; these tests build them bare."""
    return torch.nn.ModuleList(sites)


def _fold(stats, source, fmt="fp8_e4m3", offs=None, ragged_dim=None):
    """Quantize one operand and fold it in, the way `record_operand` does."""
    codes, scale = quantize_operand(
        source, -1, fmt, _TENSORWISE, offs=offs, ragged_dim=ragged_dim
    )
    deq = dequantize_operand(
        codes, scale, -1, _TENSORWISE, offs=offs, ragged_dim=ragged_dim
    )
    sums = accumulate_quantization_sums(source, codes, deq, offs=offs)
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
    codes, scale = quantize_operand(x, -1, "fp8_e4m3", _TENSORWISE)
    deq = dequantize_operand(codes, scale, -1, _TENSORWISE)

    whole = accumulate_quantization_sums(x, codes, deq)
    split = [torch.zeros_like(t) for t in whole]
    for lo in range(0, 24, 8):
        chunk = slice(lo, lo + 8)
        part = accumulate_quantization_sums(x[chunk], codes[chunk], deq[chunk])
        split = [acc + p for acc, p in zip(split, part)]

    for got, expected in zip(split, whole):
        assert got.item() == pytest.approx(expected.item(), rel=1e-6)


def test_reading_a_window_averages_operands_not_micro_batches():
    """Two micro-batches of very different scale: the window's sqnr must come from
    the summed energies, which is not the mean of the two per-batch sqnrs."""
    torch.manual_seed(0)
    micro = [torch.randn(8, 32) * 0.01, torch.randn(8, 32) * 100.0]
    device = micro[0].device

    window = QuantizationStats("act/x", 1, device)
    for x in micro:
        _fold(window, x)
    got = compute_quantization_metrics(_holder([window]))["sqnr/act/x"]

    per_batch = []
    for x in micro:
        one = QuantizationStats("act/x", 1, device)
        _fold(one, x)
        per_batch.append(compute_quantization_metrics(_holder([one]))["sqnr/act/x"])

    assert min(per_batch) <= got <= max(per_batch)


def test_reset_clears_every_accumulator_field():
    stats = QuantizationStats("act/x", 1, torch.randn(1).device)
    _fold(stats, torch.randn(8, 32))
    assert stats.numel.item() > 0
    stats.reset()
    assert all(getattr(stats, name).item() == 0 for name in stats.FIELDS)


def test_compute_reads_every_site_under_its_own_key():
    device = torch.randn(1).device
    sites = [QuantizationStats(f"act/layer{i}", 1, device) for i in range(3)]
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
    site = QuantizationStats("grad_out/x", 1, device)
    got = compute_quantization_metrics(_holder([site]))
    assert got == {}
    assert "sqnr/grad_out/x" not in got


def test_mixed_folded_and_unfolded_sites_only_report_the_folded_one():
    """The production shape: one folded site alongside one that was reset for the
    window but never folded (e.g. eval's backward operands)."""
    device = torch.randn(1).device
    folded = QuantizationStats("act/x", 1, device)
    unfolded = QuantizationStats("grad_out/x", 1, device)
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
        stats = QuantizationStats(f"act/{fmt}", 1, x.device)
        _fold(stats, x, fmt=fmt)
        rates[fmt] = compute_quantization_metrics(_holder([stats]))[
            f"underflow_rate/act/{fmt}"
        ]
    assert rates["int4"] > rates["int8"]
