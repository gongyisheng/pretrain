import pytest
import torch

from src.quant.constants import EPS
from src.quant.quantize import dequantize_operand, quantize_operand
from src.quant.rotation import build_rotation
from src.quant.utils import is_fp4, is_int8s, str_to_dtype, str_to_qmax, str_to_qmin
from tests.fast.helper import cuda_sm89_or_newer
from tests.fast.quant.helper import (
    ALL_QUANT_FORMATS,
    ALL_SCALES,
    BLOCKWISE1D_16,
    E4M3,
    BLOCKWISE1D_16_E2M1,
    ROWWISE,
    SCALES_COARSE_TO_FINE,
    TENSORWISE,
    skip_unsupported_fmt_scale,
    roundtrip,
    scale_of,
)


CONTRACT_DIMS = [-2, -1]
GEOMETRY_CASES = [
    ((64, 128), None),
    ((70, 130), None),
    ((3, 64, 128), None),
    ((70, 130), -2),
    ((70, 130), -1),
    ((128, 128), -2),
    ((128, 128), -1),
]
DENSE_SHAPES = [shape for shape, ragged_dim in GEOMETRY_CASES if ragged_dim is None]
RAGGED_GROUPS = 5
INPUT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
INIT_METHODS = ["normal", "transposed", "spread", "zeros", "tiny"]
TEST_DEVICES = ["cpu", "cuda"]
STOCHASTIC_ROUNDING = [False, True]
COMPILED_FORMATS = [E4M3, "fp4_e2m1"]
COMPILED_SCALES = [ROWWISE, BLOCKWISE1D_16_E2M1]

ROTATION_BLOCK_SIZES = [1, 32]
ROTATION_RANDOM_SIGNS = [False, True]
ROTATION_BLOCK4_CFG = {
    "rotation_cls": "hadamard",
    "rotation_kwargs": {"block_size": 4, "random_sign": False},
}
ROTATION_BLOCK32_CFG = {
    "rotation_cls": "hadamard",
    "rotation_kwargs": {"block_size": 32, "random_sign": False},
}


def _offs(counts):
    """Return cumulative group-end offsets as int32."""
    return torch.tensor(counts).cumsum(0).to(torch.int32)


def _make(init_method, shape, dtype=torch.float32):
    """Return a deterministic operand initialized by `init_method`."""
    torch.manual_seed(0)
    if init_method == "normal":
        return torch.randn(*shape, dtype=dtype) * 10.0
    if init_method == "transposed":
        return torch.randn(*shape[:-2], shape[-1], shape[-2], dtype=dtype).mT * 10.0
    if init_method == "spread":
        x = torch.randn(*shape, dtype=dtype)
        x[..., ::2, :] *= 100.0
        return x
    if init_method == "outlier":
        # Scattered outliers expose the effect of block width on amax.
        x = torch.randn(*shape, dtype=dtype)
        x[torch.rand(*shape) < 0.05] *= 100.0
        return x
    x = torch.zeros(*shape, dtype=dtype)
    if init_method == "tiny":
        # This requires a scale below E8M0's floor, forcing scale clamping.
        x[..., 0, 0] = 2**-136
    return x


def _ragged_offs(extent):
    """Return ragged groups with empty endpoints and an empty middle group."""
    first = extent // 3
    return _offs([0, first, 0, extent - first, 0])


def _bits(codes):
    """View one-byte codes as bytes to distinguish signed zero."""
    return codes.contiguous().view(torch.uint8)


def _rng_state(x):
    return (
        torch.cuda.get_rng_state(x.device)
        if x.is_cuda
        else torch.random.get_rng_state()
    )


def _sqnr(x, deq):
    """Return the signal-to-quantization-noise ratio in dB."""
    err = (x - deq).float().norm().clamp_min(EPS)
    return (20.0 * torch.log10(x.float().norm().clamp_min(EPS) / err)).item()


# --- reference oracle ---


# Independent FP4 reference values; expected packed bytes stay explicit.
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_NEGATIVE_VALUES = tuple(-value for value in reversed(E2M1_MAGNITUDES[1:]))
E2M1_GRID = E2M1_NEGATIVE_VALUES + E2M1_MAGNITUDES


# Every value e4m3 represents from +0 to its max, ascending; 0x7F is NaN and excluded.
_E4M3_GRID = torch.arange(0, 127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()


def _ref_scale(amax, fmt, scale_dtype):
    """Round block scales to their stored dtype, including range clamping."""
    if scale_dtype is torch.float8_e8m0fnu:
        exp = torch.ceil(torch.log2(amax / str_to_qmax(fmt))).clamp(-127, 127)
        reference_scale = torch.exp2(exp).to(scale_dtype).float()
    elif scale_dtype is torch.float8_e4m3fn:
        low, high = str_to_qmin("fp8_e4m3"), str_to_qmax("fp8_e4m3")
        exact = (amax / str_to_qmax(fmt)).clamp(low, high)
        # Ceiling by lookup over every value e4m3 represents, which cross-checks the
        # implementation's byte increment without restating it.
        grid = _E4M3_GRID.to(exact.device)
        reference_scale = grid[torch.searchsorted(grid, exact.contiguous())]
    else:
        reference_scale = (amax / str_to_qmax(fmt)).clamp_min(EPS)
    return reference_scale


def _ref_divisor(x, contract_dim, fmt, scale_cfg, offs=None, ragged_dim=None):
    """Compute independent block and global divisors in the operand layout."""
    expected_global = None
    global_div = torch.ones_like(x, dtype=torch.float32)
    if scale_cfg["enable_global_scale"]:
        xf = x.float()
        if offs is None:
            amax = xf.abs().amax((-2, -1)).reshape(-1)
        else:
            starts = [0, *offs.tolist()]
            amax = torch.stack(
                [
                    xf.narrow(ragged_dim, start, stop - start).abs().amax()
                    if stop > start
                    else xf.new_zeros(())
                    for start, stop in zip(starts[:-1], starts[1:])
                ]
            )
        expected_global = (
            amax / (str_to_qmax(fmt) * float(torch.finfo(scale_cfg["scale_dtype"]).max))
        ).clamp_min(EPS)
        if offs is None:
            global_div = (
                expected_global.reshape(-1, 1, 1)
                if x.ndim == 3
                else expected_global.reshape(())
            )
        else:
            starts = [0, *offs.tolist()]
            for group, (start, stop) in enumerate(zip(starts[:-1], starts[1:])):
                global_div.narrow(ragged_dim, start, stop - start).fill_(
                    expected_global[group]
                )
    scaled = x.float() / global_div
    v = scaled if contract_dim == -1 else scaled.mT  # (..., outer, contraction)
    flat = v.reshape(-1, *v.shape[-2:])
    outer_dim = -1 if contract_dim == -2 else -2
    block_outer, block_size = scale_cfg["block_shape"]
    if scale_cfg["granularity"] == "tensorwise":
        contract_extent, outer_extent = x.shape[contract_dim], x.shape[outer_dim]
    elif scale_cfg["granularity"] == "rowwise":
        contract_extent, outer_extent = x.shape[contract_dim], 1
    else:
        contract_extent, outer_extent = block_size, block_size if block_outer > 1 else 1
    ends = None if offs is None else offs.tolist()
    div = torch.zeros_like(flat)
    outer_edges = (
        [0, *ends] if ragged_dim not in (None, contract_dim) else [0, flat.shape[-2]]
    )
    contract_edges = [0, *ends] if ragged_dim == contract_dim else [0, flat.shape[-1]]
    outer_spans = [
        (i, min(i + outer_extent, hi))
        for lo, hi in zip(outer_edges[:-1], outer_edges[1:])
        for i in range(lo, hi, outer_extent)
    ]
    contract_spans = [
        (i, min(i + contract_extent, hi))
        for lo, hi in zip(contract_edges[:-1], contract_edges[1:])
        for i in range(lo, hi, contract_extent)
    ]
    for lo, hi in outer_spans:
        for start, stop in contract_spans:
            block = flat[:, lo:hi, start:stop]
            amax = block.abs().amax((-2, -1), keepdim=True)
            reference_scale = _ref_scale(amax, fmt, scale_cfg["scale_dtype"])
            div[:, lo:hi, start:stop] = reference_scale
    div = div.reshape(v.shape)
    raw_div = div if contract_dim == -1 else div.mT

    return raw_div, expected_global, global_div


def _ref_codes(x, divisor, fmt, contract_dim):
    """Round element codes independently and pack FP4 low-nibble first."""
    qmax = str_to_qmax(fmt)
    xq = (x.float() / divisor).clamp(-qmax, qmax)
    if is_fp4(fmt):
        values = xq.new_tensor(E2M1_MAGNITUDES)
        magnitude = xq.abs()
        upper = torch.searchsorted(values, magnitude.contiguous()).clamp(max=7)
        lower = (upper - 1).clamp(min=0)
        choose_upper = (magnitude - values[lower] > values[upper] - magnitude) | (
            (magnitude - values[lower] == values[upper] - magnitude)
            & (upper.remainder(2) == 0)
        )
        nibbles = torch.where(choose_upper, upper, lower).to(torch.uint8)
        nibbles |= torch.signbit(xq).to(torch.uint8) << 3
        moved = nibbles.movedim(contract_dim, -1).contiguous()
        expected_codes = (moved[..., 0::2] | moved[..., 1::2] << 4).movedim(
            -1, contract_dim
        )
    else:
        expected_codes = (torch.round(xq) if is_int8s(fmt) else xq).to(
            str_to_dtype(fmt)
        )
    return expected_codes


def _ref_dequantize(codes, fmt, contract_dim, divisor, global_divisor):
    """Decode reference codes and restore block and global scales."""
    if is_fp4(fmt):
        moved = codes.movedim(contract_dim, -1)
        nibbles = torch.stack((moved & 0xF, moved >> 4), -1).flatten(-2)
        values = codes.new_tensor(E2M1_MAGNITUDES, dtype=torch.float32)
        expected_values = values[nibbles.long() & 0x7]
        expected_values = torch.where(
            nibbles & 0x8 != 0, -expected_values, expected_values
        ).movedim(-1, contract_dim)
    else:
        expected_values = codes.float()
    return expected_values * divisor * global_divisor


# --- quantize_operand ---

COMPILED = [False, True]
QUANTIZE_ERROR_CASES = [
    (
        {
            "x": torch.ones(8, 100),
            "contract_dim": -1,
            "fmt": "int8",
            "scale_cfg": TENSORWISE,
            "rotation": build_rotation(ROTATION_BLOCK32_CFG),
        },
        ValueError,
    ),
    (
        {
            "x": torch.ones(4, 8),
            "contract_dim": -1,
            "fmt": "int8",
            "scale_cfg": TENSORWISE,
            "offs": _offs([2, 6]),
            "ragged_dim": -1,
            "rotation": build_rotation(ROTATION_BLOCK4_CFG),
        },
        AssertionError,
    ),
    (
        {
            "x": torch.ones(2, 2),
            "contract_dim": -1,
            "fmt": E4M3,
            "scale_cfg": {
                "granularity": "rowwise",
                "block_shape": (0, 0),
                "enable_global_scale": False,
            },
        },
        KeyError,
    ),
    (
        {
            "x": torch.randn(4, 8),
            "contract_dim": -1,
            "fmt": E4M3,
            "scale_cfg": scale_of("groupwise", (1, 32)),
        },
        ValueError,
    ),
    (
        {
            "x": torch.randn(4, 8),
            "contract_dim": 0,
            "fmt": E4M3,
            "scale_cfg": TENSORWISE,
        },
        ValueError,
    ),
    (
        {
            "x": torch.randn(4, 8),
            "contract_dim": -1,
            "fmt": E4M3,
            "scale_cfg": TENSORWISE,
            "offs": _offs([2, 2]),
        },
        ValueError,
    ),
    (
        {
            "x": torch.randn(4, 8),
            "contract_dim": -1,
            "fmt": E4M3,
            "scale_cfg": TENSORWISE,
            "ragged_dim": -2,
        },
        ValueError,
    ),
    (
        {
            "x": torch.randn(4, 8),
            "contract_dim": -1,
            "fmt": E4M3,
            "scale_cfg": TENSORWISE,
            "offs": _offs([2, 2]),
            "ragged_dim": 0,
        },
        ValueError,
    ),
    (
        {
            "x": torch.randn(2, 4, 8),
            "contract_dim": -2,
            "fmt": E4M3,
            "scale_cfg": TENSORWISE,
            "offs": _offs([2, 2]),
            "ragged_dim": -2,
        },
        ValueError,
    ),
    (
        {
            "x": torch.ones(2, 16),
            "contract_dim": -1,
            "fmt": "fp4_e2m1",
            "scale_cfg": BLOCKWISE1D_16_E2M1,
            "offs": _offs([8, 6]),
            "ragged_dim": -1,
        },
        AssertionError,
    ),
    (
        {
            "x": torch.ones(2, 16),
            "contract_dim": -1,
            "fmt": "fp4_e2m1",
            "scale_cfg": BLOCKWISE1D_16_E2M1,
            "offs": _offs([7, 9]),
            "ragged_dim": -1,
        },
        AssertionError,
    ),
    (
        {
            "x": torch.ones(2, 15),
            "contract_dim": -1,
            "fmt": "fp4_e2m1",
            "scale_cfg": BLOCKWISE1D_16_E2M1,
        },
        ValueError,
    ),
    (
        {
            "x": torch.ones(2, 18),
            "contract_dim": -1,
            "fmt": "fp4_e2m1",
            "scale_cfg": BLOCKWISE1D_16_E2M1,
        },
        ValueError,
    ),
    (
        {
            "x": torch.ones(15, 2),
            "contract_dim": -2,
            "fmt": "fp4_e2m1",
            "scale_cfg": BLOCKWISE1D_16_E2M1,
        },
        ValueError,
    ),
    (
        {
            "x": torch.ones(18, 2),
            "contract_dim": -2,
            "fmt": "fp4_e2m1",
            "scale_cfg": BLOCKWISE1D_16_E2M1,
        },
        ValueError,
    ),
]


@pytest.mark.parametrize("compiled", COMPILED)
@pytest.mark.parametrize("case", QUANTIZE_ERROR_CASES)
def test_quantize_operand_raise_error(case, compiled):
    kwargs, exception = case
    if compiled and exception is not AssertionError:
        pytest.skip("only runtime tensor assertions are compiled")
    if not compiled:
        with pytest.raises(exception):
            quantize_operand(**kwargs)
        return
    previous_device = torch.get_default_device()
    torch.set_default_device("cpu")
    torch.compiler.reset()
    try:
        with pytest.raises(RuntimeError):
            torch.compile(quantize_operand, backend="eager", fullgraph=True)(**kwargs)
    finally:
        torch.compiler.reset()
        torch.set_default_device(previous_device)


@pytest.mark.parametrize("init_method", INIT_METHODS)
@pytest.mark.parametrize("dtype", INPUT_DTYPES)
@pytest.mark.parametrize("geometry", GEOMETRY_CASES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("scale_cfg", ALL_SCALES)
@pytest.mark.parametrize("fmt", ALL_QUANT_FORMATS)
def test_quantize_operand_precision(
    fmt, scale_cfg, contract_dim, geometry, dtype, init_method
):
    """Check codes, scales, and layout against an independent block-loop oracle."""
    skip_unsupported_fmt_scale(fmt, scale_cfg)
    shape, ragged_dim = geometry
    if init_method == "tiny" and dtype is not torch.float32:
        pytest.skip("tiny values are not representable in this dtype")

    x = _make(init_method, shape, dtype)
    offs = None if ragged_dim is None else _ragged_offs(shape[ragged_dim])
    if is_fp4(fmt) and (
        shape[contract_dim] % 16
        or (ragged_dim == contract_dim and bool((offs % 2).any()))
    ):
        pytest.skip("fp4_e2m1 needs a 16-aligned contraction and even ragged offsets")
    raw_div, expected_global, global_div = _ref_divisor(
        x, contract_dim, fmt, scale_cfg, offs, ragged_dim
    )

    div = raw_div * global_div
    expected = list(shape)
    block_size = (
        scale_cfg["block_shape"][1] if scale_cfg["granularity"] == "blockwise" else 0
    )
    if ragged_dim == contract_dim:
        expected[contract_dim] = (
            shape[contract_dim] // block_size + RAGGED_GROUPS
            if block_size
            else RAGGED_GROUPS
        )
    else:
        expected[contract_dim] = (
            (shape[contract_dim] + block_size - 1) // block_size if block_size else 1
        )

    source_bits = x.contiguous().view(torch.uint8).clone()
    rng = _rng_state(x)
    codes, scale, global_scale = quantize_operand(
        x, contract_dim, fmt, scale_cfg, offs=offs, ragged_dim=ragged_dim
    )

    expected_codes_shape = list(shape)
    if is_fp4(fmt):
        expected_codes_shape[contract_dim] //= 2
    assert tuple(codes.shape) == tuple(expected_codes_shape)
    assert codes.dtype is str_to_dtype(fmt)
    if (
        not is_fp4(fmt)
        and init_method == "transposed"
        and scale_cfg["granularity"] != "blockwise"
    ):
        assert codes.stride() == x.stride()
    if not is_fp4(fmt):
        assert codes.float().abs().amax() <= str_to_qmax(fmt)
    assert tuple(scale.shape) == tuple(expected)
    assert scale.dtype is scale_cfg["scale_dtype"]
    if expected_global is None:
        assert global_scale is None
    else:
        assert torch.equal(global_scale, expected_global)
        assert global_scale.dtype is torch.float32
    if ragged_dim is None and scale_cfg["granularity"] == "tensorwise":
        assert 0 in scale.stride()
    # E8M0 carries no comparison or log2 kernel, so check the decoded exponents.
    decoded = scale.float()
    assert torch.isfinite(decoded).all() and (decoded > 0).all()
    if scale_cfg["scale_dtype"] is torch.float8_e8m0fnu:
        log2_scale = torch.log2(decoded)
        assert torch.equal(log2_scale, log2_scale.round())

    expected_codes = _ref_codes(x.float() / global_div, raw_div, fmt, contract_dim)
    assert torch.equal(_bits(codes), _bits(expected_codes))

    dequantized = dequantize_operand(
        codes,
        scale,
        contract_dim,
        scale_cfg,
        offs=offs,
        ragged_dim=ragged_dim,
        global_scale=global_scale,
    )
    assert dequantized.shape == x.shape
    assert dequantized.dtype is torch.float32 and dequantized.is_contiguous()
    expected_dequantized = _ref_dequantize(
        expected_codes, fmt, contract_dim, raw_div, global_div
    )
    assert torch.equal(dequantized, expected_dequantized)

    current_rng = _rng_state(x)
    assert torch.equal(current_rng, rng)
    assert torch.equal(x.contiguous().view(torch.uint8), source_bits)
    again, _, _ = quantize_operand(
        x, contract_dim, fmt, scale_cfg, offs=offs, ragged_dim=ragged_dim
    )
    assert torch.equal(_bits(again), _bits(codes))

    if init_method == "transposed":
        contiguous = quantize_operand(
            x.contiguous(),
            contract_dim,
            fmt,
            scale_cfg,
            offs=offs,
            ragged_dim=ragged_dim,
        )
        contiguous_codes, contiguous_scale, contiguous_global = contiguous
        assert torch.equal(_bits(codes), _bits(contiguous_codes))
        assert torch.equal(_bits(scale), _bits(contiguous_scale))
        assert (global_scale is None) == (contiguous_global is None)
        if global_scale is not None:
            assert torch.equal(global_scale, contiguous_global)

    if len(shape) > 2:
        per_expert = [
            quantize_operand(expert, contract_dim, fmt, scale_cfg) for expert in x
        ]
        assert torch.equal(
            _bits(codes), _bits(torch.stack([q for q, _, _ in per_expert]))
        )
        assert torch.equal(
            _bits(scale), _bits(torch.stack([s for _, s, _ in per_expert]))
        )
        if global_scale is not None:
            assert torch.equal(global_scale, torch.cat([g for _, _, g in per_expert]))

    qmax = str_to_qmax(fmt)
    # Every block peak is in range, so code clamping is inactive.
    assert (x.abs().float() <= div * qmax * (1 + 2**-20)).all()
    floored = scale_cfg["scale_dtype"] is torch.float8_e8m0fnu and not is_int8s(fmt)
    if init_method == "tiny" and floored:
        # The clamped E8M0 scale preserves this value as an fp8 subnormal.
        assert torch.equal(codes.float() * div, x)
    if init_method == "normal":
        ratio = (x.abs().float() / div).amax().item()
        if scale_cfg["scale_dtype"] is torch.float32:
            assert ratio == pytest.approx(qmax, rel=1e-6)  # The peak maps to qmax.
        elif scale_cfg["scale_dtype"] is torch.float8_e4m3fn:
            # A scale floored at e4m3's min subnormal cannot be tight: fp8_e5m2's
            # qmax of 57344 puts its natural scale three decades under that floor,
            # so every block floors and utilisation collapses to 0.038. Normalising
            # that is the global scale's job, so only blocks inside e4m3's window
            # are held to the bound. Measured worst case over the grid: 0.967 qmax.
            unfloored = raw_div > str_to_qmin("fp8_e4m3")
            if unfloored.any():
                assert (x.abs().float() / div)[unfloored].amax().item() > qmax / 2
        else:
            assert ratio > qmax / 2  # E8M0 rounding leaves under one binade unused.


@pytest.mark.parametrize("shape", DENSE_SHAPES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("fmt", ALL_QUANT_FORMATS)
def test_quantize_operand_granularity_sqnr(fmt, contract_dim, shape):
    """Check that finer scales improve SQNR on scattered outliers.

    Integer formats must be monotonic; fp8 needs only an end-to-end gain.
    """
    if is_fp4(fmt) and shape[contract_dim] % 16:
        pytest.skip("fp4_e2m1 needs a 16-aligned contraction")
    x = _make("outlier", shape)
    sqnrs = [
        _sqnr(x, roundtrip(x, contract_dim, fmt, cfg)) for cfg in SCALES_COARSE_TO_FINE
    ]

    if is_int8s(fmt):
        assert sqnrs == sorted(sqnrs)
    # 1.1 dB leaves margin below the measured 3.33 dB minimum.
    assert sqnrs[-1] - sqnrs[1] > 1.1


NARROW_SCALE_CASES = [
    (torch.float8_e4m3fn, 0.0, str_to_qmin("fp8_e4m3")),
    (torch.float8_e4m3fn, 1e-6, str_to_qmin("fp8_e4m3")),
    (torch.float8_e4m3fn, 1e9, str_to_qmax("fp8_e4m3")),
    (torch.float8_e4m3fn, float("inf"), str_to_qmax("fp8_e4m3")),
    (torch.float8_e8m0fnu, 2**-136, 2**-127),
    (torch.float8_e8m0fnu, float("inf"), 2.0**127),
]


@pytest.mark.parametrize("device", TEST_DEVICES)
@pytest.mark.parametrize("case", NARROW_SCALE_CASES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("fmt", ALL_QUANT_FORMATS)
def test_quantize_operand_narrow_scale_clamps(contract_dim, case, device, fmt):
    """Public quantization stores both narrow scale endpoints exactly."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    scale_dtype, magnitude, expected_scale = case
    if scale_dtype is torch.float8_e8m0fnu and not fmt.startswith("fp8"):
        pytest.skip("an e8m0 scale is defined only over fp8 elements")
    scale_cfg = scale_of("blockwise", (1, 16), scale_dtype)
    x = torch.full((16, 16), magnitude, device=device)
    codes, scale, global_scale = quantize_operand(x, contract_dim, fmt, scale_cfg)
    assert global_scale is None
    assert torch.equal(scale.float(), torch.full_like(scale.float(), expected_scale))
    assert torch.isfinite(codes.float()).all()
    dequantized = dequantize_operand(codes, scale, contract_dim, scale_cfg)
    if torch.isfinite(x).all():
        assert torch.isfinite(dequantized).all()


# Eight decades of operand magnitude. Without a global scale an e4m3 block scale is
# only usable near 1e0; the whole point of the scale is that these agree.
GLOBAL_SCALE_MAGNITUDES = [1e-4, 1e-2, 1.0, 1e2, 1e4]
# Worst relMSE over magnitude x contract_dim with the nvfp4 recipe, times 4.3. The
# bounds are per format because a single loose one would let int8 regress 300x and
# still pass. Unscaled, the far magnitudes sit near 1.0, so every bound has teeth.
GLOBAL_SCALE_REL_MSE_BOUND = {
    "fp8_e4m3": 0.0028,  # measured 6.429e-04
    "fp8_e5m2": 0.011,  # measured 2.647e-03
    "fp4_e2m1": 0.045,  # measured 1.003e-02
    "int4": 0.036,  # measured 8.479e-03
    "int5": 0.0078,  # measured 1.802e-03
    "int6": 0.0018,  # measured 4.221e-04
    "int7": 0.00045,  # measured 1.040e-04
    "int8": 0.00011,  # measured 2.570e-05
}


@pytest.mark.parametrize("magnitude", GLOBAL_SCALE_MAGNITUDES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("fmt", ALL_QUANT_FORMATS)
def test_quantize_operand_global_scale_is_magnitude_invariant(
    fmt, contract_dim, magnitude
):
    """A global scale makes a narrow block scale work at any operand magnitude.

    This is the property the feature exists for, and the only one a silently
    no-op implementation fails: at 1e0 an e4m3 scale needs no help, so an
    on-versus-off comparison there shows almost nothing. Away from 1e0 the
    unscaled version collapses -- every block scale floors or saturates.
    """
    torch.manual_seed(0)
    base = torch.randn(64, 128, dtype=torch.float32)
    x = base * magnitude
    deq = roundtrip(x, contract_dim, fmt, BLOCKWISE1D_16_E2M1)
    relative_mse = (
        x.double() - deq.double()
    ).square().sum() / x.double().square().sum()
    assert relative_mse.item() < GLOBAL_SCALE_REL_MSE_BOUND[fmt]


def test_quantize_operand_global_scale_is_per_group():
    """A loud group must not degrade a quiet one -- the reason it is per group.

    Contrasted against one scale over the whole tensor, where the loud group sets g
    and drives every quiet block's scale under e4m3's floor.
    """
    torch.manual_seed(0)
    quiet = torch.randn(32, 128, dtype=torch.float32)
    loud = torch.randn(32, 128, dtype=torch.float32) * 1e6
    x = torch.cat([quiet, loud], dim=0)
    offs = _offs([32, 32])

    def quiet_error(offs, ragged_dim):
        codes, scale, g = quantize_operand(
            x, -1, "int4", BLOCKWISE1D_16_E2M1, offs=offs, ragged_dim=ragged_dim
        )
        deq = dequantize_operand(
            codes,
            scale,
            -1,
            BLOCKWISE1D_16_E2M1,
            global_scale=g,
            offs=offs,
            ragged_dim=ragged_dim,
        )
        per_row = (x - deq).square().sum(-1) / x.square().sum(-1)
        return per_row[:32].mean().item(), g

    grouped, g_grouped = quiet_error(offs=offs, ragged_dim=-2)
    pooled, g_pooled = quiet_error(None, None)
    assert g_grouped.shape == (2,) and g_pooled.shape == (1,)
    # Measured 8.3e-3 per group against 5.0e-1 pooled: a 60x gap, the quiet group
    # wiped out because the loud group's amax sets g and floors every quiet block.
    assert grouped < 1e-2
    assert pooled > 0.3


# E2M1 stores the first contraction value in the low nibble and the second in the
# high nibble. These are all exactly representable values, so the bytes expose both
# the format's code table and its packing order without a rounding oracle.
E2M1_VALUES = torch.tensor(E2M1_NEGATIVE_VALUES + (0.0, -0.0) + E2M1_MAGNITUDES[1:])
E2M1_PACKED_BYTES = torch.tensor([0xEF, 0xCD, 0xAB, 0x09, 0x18, 0x32, 0x54, 0x76])
E2M1_MIDPOINTS = torch.tensor(
    [
        0.25,
        0.75,
        1.25,
        1.75,
        2.5,
        3.5,
        5.0,
        -0.25,
        -0.75,
        -1.25,
        -1.75,
        -2.5,
        -3.5,
        -5.0,
        6.0,
        -6.0,
    ]
)
E2M1_RNE = torch.tensor(
    [
        0.0,
        1.0,
        1.0,
        2.0,
        2.0,
        4.0,
        4.0,
        0.0,
        -1.0,
        -1.0,
        -2.0,
        -2.0,
        -4.0,
        -4.0,
        6.0,
        -6.0,
    ]
)
E2M1_SCALE_CONFIGS = [BLOCKWISE1D_16, BLOCKWISE1D_16_E2M1]
E2M1_BOUNDARY_CASES = ["values", "midpoints", "below_midpoints", "above_midpoints"]


@pytest.mark.parametrize("scale_cfg", E2M1_SCALE_CONFIGS)
@pytest.mark.parametrize("boundary", E2M1_BOUNDARY_CASES)
@pytest.mark.parametrize("dtype", INPUT_DTYPES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
def test_quantize_operand_fp4_e2m1_precision(contract_dim, dtype, boundary, scale_cfg):
    """E2M1 packs values and rounds each midpoint on both contraction axes."""
    if scale_cfg["enable_global_scale"] and boundary != "values":
        pytest.skip("midpoint neighbors require an exact unit scale")
    expected_codes = None
    if boundary == "values":
        values = E2M1_VALUES.to(device=torch.get_default_device(), dtype=dtype)
        expected_values = values
        expected_codes = E2M1_PACKED_BYTES.to(device=values.device)
    elif boundary == "midpoints":
        values, expected_values = (
            E2M1_MIDPOINTS.to(device=torch.get_default_device(), dtype=dtype),
            E2M1_RNE.to(device=torch.get_default_device(), dtype=dtype),
        )
        expected_codes = values.new_tensor(
            [0x20, 0x42, 0x64, 0x86, 0xAA, 0xCC, 0xEE, 0xF7], dtype=torch.uint8
        )
    else:
        direction = -torch.inf if boundary == "below_midpoints" else torch.inf
        midpoints = E2M1_MIDPOINTS[:7].to(
            device=torch.get_default_device(), dtype=dtype
        )
        neighbors = torch.nextafter(midpoints, torch.full_like(midpoints, direction))
        values = torch.cat((neighbors, -neighbors, neighbors.new_tensor([6.0, -6.0])))
        magnitudes = (
            torch.tensor(E2M1_MAGNITUDES[:-1])
            if direction == -torch.inf
            else torch.tensor(E2M1_MAGNITUDES[1:])
        ).to(dtype)
        expected_values = torch.cat(
            (magnitudes, -magnitudes, magnitudes.new_tensor([6.0, -6.0]))
        )
    if contract_dim == -1:
        source = values.repeat(2, 1)
        expected = expected_values.repeat(2, 1)
        if expected_codes is not None:
            expected_codes = expected_codes.repeat(2, 1)
    else:
        source = values[:, None].repeat(1, 2)
        expected = expected_values[:, None].repeat(1, 2)
        if expected_codes is not None:
            expected_codes = expected_codes[:, None].repeat(1, 2)
    expected = expected.to(device=source.device, dtype=torch.float32)

    codes, scale, global_scale = quantize_operand(
        source, contract_dim, "fp4_e2m1", scale_cfg
    )
    assert codes.dtype is torch.uint8
    if expected_codes is not None:
        assert torch.equal(codes, expected_codes)
    dequantized = dequantize_operand(
        codes, scale, contract_dim, scale_cfg, global_scale=global_scale
    )
    torch.testing.assert_close(
        dequantized, expected, rtol=0, atol=1e-6 if global_scale is not None else 0
    )
    if boundary == "values":
        zero = dequantized[0, 8] if contract_dim == -1 else dequantized[8, 0]
        assert torch.signbit(zero)


@pytest.mark.parametrize("enable_sr", STOCHASTIC_ROUNDING)
@pytest.mark.parametrize("fmt", ALL_QUANT_FORMATS)
@pytest.mark.parametrize("device", TEST_DEVICES)
def test_quantize_operand_stochastic_rounding(fmt, enable_sr, device):
    """Check adjacent codes, deterministic RNE, and unbiased stochastic rounding."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(0)
    if is_int8s(fmt):
        qmax = int(str_to_qmax(fmt))
        grid = torch.arange(-qmax, qmax + 1, dtype=torch.float32)
    elif is_fp4(fmt):
        grid = torch.tensor(E2M1_GRID)
    else:
        codes = torch.arange(256, dtype=torch.uint8).view(str_to_dtype(fmt)).float()
        grid = codes[codes.isfinite()].unique()
    grid = grid.to(device)
    probe = torch.cat([grid, (grid[:-1] + grid[1:]) / 2])
    if is_fp4(fmt):
        probe = torch.cat(
            [probe, grid.new_full(((-probe.numel()) % 16,), str_to_qmax(fmt))]
        )
    probe = probe.reshape(1, -1)
    rng = _rng_state(probe)
    codes, scale, _ = quantize_operand(
        probe, -1, fmt, TENSORWISE, stochastic_rounding=enable_sr
    )
    if not enable_sr:
        current_rng = _rng_state(probe)
        assert torch.equal(current_rng, rng)
    assert scale.item() == 1.0  # The row peak is qmax.
    values = (
        dequantize_operand(codes, scale, -1, TENSORWISE)
        if is_fp4(fmt)
        else codes.float()
    )
    below = torch.searchsorted(grid, probe.flatten().contiguous())
    lower = grid[(below - 1).clamp(min=0)]
    upper = grid[below.clamp(max=grid.numel() - 1)]
    assert bool(((values.flatten() == lower) | (values.flatten() == upper)).all())

    ends_source = torch.tensor([[1.0, 0.0, -0.0, -1.0]], device=device)
    if is_fp4(fmt):
        ends_source = torch.cat([ends_source, ends_source.new_zeros(1, 12)], dim=-1)
    ends, end_scale, _ = quantize_operand(
        ends_source,
        -1,
        fmt,
        TENSORWISE,
        stochastic_rounding=enable_sr,
    )
    end_values = dequantize_operand(ends, end_scale, -1, TENSORWISE)
    assert torch.equal(end_values[..., :4], ends_source[..., :4])
    if not is_int8s(fmt):
        assert torch.signbit(end_values[0, 2])  # Preserve negative zero.

    # The leading 1 fixes the scale; 0.3 lies between codes in every format.
    x = torch.cat(
        [torch.ones(1, device=device), torch.full((9999,), 0.3, device=device)]
    ).reshape(1, -1)
    draws, scale, _ = quantize_operand(
        x, -1, fmt, TENSORWISE, stochastic_rounding=enable_sr
    )
    again, _, _ = quantize_operand(
        x, -1, fmt, TENSORWISE, stochastic_rounding=enable_sr
    )
    draw_values = dequantize_operand(draws, scale, -1, TENSORWISE)
    reached = draw_values[0, 1:].unique().numel()

    if not enable_sr:
        assert reached == 1  # RNE always picks the nearer code.
        assert torch.equal(_bits(again), _bits(draws))
        return

    assert reached == 2  # SR draws both adjacent codes.
    assert draw_values[0, 1:].mean().item() == pytest.approx(0.3, abs=0.0021)
    # Repeating all 9,999 stochastic draws is negligibly likely.
    assert not torch.equal(_bits(again), _bits(draws))


# --- dequantize_operand ---

DEQUANTIZE_ERROR_CASES = [
    (
        {
            "xq": torch.ones(3, 3, 4, dtype=torch.int8),
            "scale": torch.arange(1.0, 13.0).reshape(1, 3, 4),
            "contract_dim": 0,
            "scale_cfg": TENSORWISE,
        },
        ValueError,
    ),
    (
        {
            "xq": torch.ones(4, 4, dtype=torch.int8),
            "scale": torch.ones(4, 1),
            "contract_dim": -1,
            "scale_cfg": TENSORWISE,
            "offs": _offs([2, 2]),
        },
        ValueError,
    ),
    (
        {
            "xq": torch.ones(4, 4, dtype=torch.int8),
            "scale": torch.ones(4, 1),
            "contract_dim": -1,
            "scale_cfg": TENSORWISE,
            "ragged_dim": -2,
        },
        ValueError,
    ),
    (
        {
            "xq": torch.ones(4, 4, dtype=torch.int8),
            "scale": torch.ones(4, 1),
            "contract_dim": -1,
            "scale_cfg": TENSORWISE,
            "offs": _offs([2, 2]),
            "ragged_dim": 0,
        },
        ValueError,
    ),
    (
        {
            "xq": torch.ones(2, 4, 4, dtype=torch.int8),
            "scale": torch.ones(2, 4, 1),
            "contract_dim": -1,
            "scale_cfg": TENSORWISE,
            "offs": _offs([2, 2]),
            "ragged_dim": -2,
        },
        ValueError,
    ),
    (
        {
            "xq": torch.ones(4, 4, dtype=torch.int8),
            "scale": torch.ones(4, 1),
            "contract_dim": -1,
            "scale_cfg": {"granularity": "rowwise", "enable_global_scale": False},
        },
        KeyError,
    ),
    (
        {
            "xq": torch.ones(4, 8, dtype=torch.int8),
            "scale": torch.ones(4, 2),
            "contract_dim": -1,
            "scale_cfg": TENSORWISE,
            "offs": _offs([2, 6]),
            "ragged_dim": -1,
            "rotation": build_rotation(ROTATION_BLOCK4_CFG),
        },
        AssertionError,
    ),
]


@pytest.mark.parametrize("compiled", COMPILED)
@pytest.mark.parametrize("case", DEQUANTIZE_ERROR_CASES)
def test_dequantize_operand_raise_error(case, compiled):
    kwargs, exception = case
    if compiled and exception is not AssertionError:
        pytest.skip("only runtime tensor assertions are compiled")
    if not compiled:
        with pytest.raises(exception):
            dequantize_operand(**kwargs)
        return
    previous_device = torch.get_default_device()
    torch.set_default_device("cpu")
    torch.compiler.reset()
    try:
        with pytest.raises(RuntimeError):
            torch.compile(dequantize_operand, backend="eager", fullgraph=True)(**kwargs)
    finally:
        torch.compiler.reset()
        torch.set_default_device(previous_device)


ROTATION_LAYOUTS = ["dense", "batched", "ragged_outer", "ragged_contraction"]
ROTATION_INITS = ["normal", "outlier"]
ROTATION_SCALES = [TENSORWISE, BLOCKWISE1D_16_E2M1]


@pytest.mark.parametrize("layout", ROTATION_LAYOUTS)
@pytest.mark.parametrize("init_method", ROTATION_INITS)
@pytest.mark.parametrize("block_size", ROTATION_BLOCK_SIZES)
@pytest.mark.parametrize("random_sign", ROTATION_RANDOM_SIGNS)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("scale_cfg", ROTATION_SCALES)
def test_quantize_operand_rotation(
    layout, init_method, block_size, random_sign, contract_dim, scale_cfg
):
    shape = (256, 256) if init_method == "outlier" else (64, 128)
    if layout == "batched":
        shape = (3, *shape)
    x = _make(init_method, shape)
    ragged_dim = None
    offs = None
    if layout == "ragged_outer":
        ragged_dim = -1 if contract_dim == -2 else -2
        offs = _ragged_offs(x.shape[ragged_dim])
    elif layout == "ragged_contraction":
        ragged_dim = contract_dim
        offs = _offs([0, 32, 0, x.shape[contract_dim] - 32, 0])
    rotation = build_rotation(
        {
            "rotation_cls": "hadamard",
            "rotation_kwargs": {
                "block_size": block_size,
                "random_sign": random_sign,
                "seed": 42,
            },
        }
    )
    codes, scale, global_scale = quantize_operand(
        x,
        contract_dim,
        "int8",
        scale_cfg,
        offs=offs,
        ragged_dim=ragged_dim,
        rotation=rotation,
    )
    dequantized = dequantize_operand(
        codes,
        scale,
        contract_dim,
        scale_cfg,
        offs=offs,
        ragged_dim=ragged_dim,
        rotation=rotation,
        global_scale=global_scale,
    )
    baseline_codes, baseline_scale, baseline_global = quantize_operand(
        x,
        contract_dim,
        "int8",
        scale_cfg,
        offs=offs,
        ragged_dim=ragged_dim,
    )
    baseline = dequantize_operand(
        baseline_codes,
        baseline_scale,
        contract_dim,
        scale_cfg,
        offs=offs,
        ragged_dim=ragged_dim,
        global_scale=baseline_global,
    )
    assert _sqnr(x, dequantized) > 10.0
    assert _sqnr(x, baseline) > 10.0
    if block_size == 1 and not random_sign:
        assert torch.equal(codes, baseline_codes)
        assert torch.equal(scale, baseline_scale)
        assert torch.equal(dequantized, baseline)
        if global_scale is not None:
            assert torch.equal(global_scale, baseline_global)
    if (
        block_size == 32
        and layout == "dense"
        and init_method == "outlier"
        and scale_cfg == TENSORWISE
    ):
        assert _sqnr(x, dequantized) > _sqnr(x, baseline) + 3.0


@cuda_sm89_or_newer
@pytest.mark.parametrize("fmt", COMPILED_FORMATS)
@pytest.mark.parametrize("scale_cfg", COMPILED_SCALES)
def test_quantize_operand_compiles_fullgraph(scale_cfg, fmt):
    """Rotated quantization must have identical eager and compiled bits.

    The nvfp4 case also pins that a global scale stays a tensor: reading it back
    as a Python float would sync the device and break fullgraph here.
    """
    torch.manual_seed(0)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    rotation = build_rotation(ROTATION_BLOCK32_CFG).cuda()

    def quantize(x):
        return quantize_operand(x, -1, fmt, scale_cfg, rotation=rotation)

    eager_codes, eager_scale, eager_global = quantize(x)
    compiled_codes, compiled_scale, compiled_global = torch.compile(
        quantize, fullgraph=True
    )(x)

    assert torch.equal(compiled_codes, eager_codes)
    assert torch.equal(compiled_scale, eager_scale)
    assert (eager_global is None) == (compiled_global is None)
    if eager_global is not None:
        assert torch.equal(compiled_global, eager_global)

    def dequantize(codes, scale, global_scale):
        return dequantize_operand(
            codes,
            scale,
            -1,
            scale_cfg,
            rotation=rotation,
            global_scale=global_scale,
        )

    eager_dequantized = dequantize(eager_codes, eager_scale, eager_global)
    compiled_dequantized = torch.compile(dequantize, fullgraph=True)(
        eager_codes, eager_scale, eager_global
    )
    assert torch.equal(compiled_dequantized, eager_dequantized)


@cuda_sm89_or_newer
def test_quantize_operand_fp4_e2m1_stochastic_rounding_compiles_fullgraph():
    """Compiled stochastic rounding produces legal codes with the expected mean."""
    torch.manual_seed(0)
    source = torch.full((4096, 16), 0.3, device="cuda")
    source[:, -1] = 6.0

    def quantize(x):
        return quantize_operand(
            x,
            -1,
            "fp4_e2m1",
            BLOCKWISE1D_16,
            stochastic_rounding=True,
        )

    codes, scale, global_scale = torch.compile(quantize, fullgraph=True)(source)
    dequantized = dequantize_operand(
        codes, scale, -1, BLOCKWISE1D_16, global_scale=global_scale
    )
    assert codes.dtype is torch.uint8 and codes.shape == (4096, 8)
    assert set(dequantized[:, :-1].unique().tolist()) == {0.0, 0.5}
    assert dequantized[:, :-1].mean().item() == pytest.approx(0.3, abs=0.005)
    torch.testing.assert_close(dequantized[:, -1], source[:, -1], rtol=0, atol=0)


E2M1_SR_MAGNITUDES = [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    torch.nextafter(torch.tensor(2.0), torch.tensor(-torch.inf)).item(),
    2.0,
    torch.nextafter(torch.tensor(2.0), torch.tensor(torch.inf)).item(),
    2.5,
    3.0,
    3.5,
    torch.nextafter(torch.tensor(4.0), torch.tensor(-torch.inf)).item(),
    4.0,
    torch.nextafter(torch.tensor(4.0), torch.tensor(torch.inf)).item(),
    5.0,
    6.0,
]


E2M1_SIGNS = [1.0, -1.0]


@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("sign", E2M1_SIGNS)
@pytest.mark.parametrize("magnitude", E2M1_SR_MAGNITUDES)
def test_quantize_operand_fp4_e2m1_stochastic_rounding_precision(
    monkeypatch, magnitude, sign, contract_dim
):
    """Every 8-bit random value matches Transformer Engine's fallback."""
    random_bytes = torch.arange(256, dtype=torch.int32, device="cpu")
    source = torch.full((256, 16), 6.0, device="cpu")
    source[:, 1] = sign * magnitude
    random_bits = torch.zeros_like(source, dtype=torch.int32)
    random_bits[:, 1] = random_bytes
    if contract_dim == -2:
        source, random_bits = source.mT, random_bits.mT

    def fixed_random_bits(low, high, size, dtype, device):
        assert (low, high) == (0, 256)
        assert tuple(size) == source.shape
        assert dtype is torch.int32
        assert torch.device(device).type == "cpu"
        return random_bits

    monkeypatch.setattr(torch, "randint", fixed_random_bits)
    codes, scale, global_scale = quantize_operand(
        source,
        contract_dim,
        "fp4_e2m1",
        BLOCKWISE1D_16,
        stochastic_rounding=True,
    )
    actual = dequantize_operand(
        codes, scale, contract_dim, BLOCKWISE1D_16, global_scale=global_scale
    )
    actual = actual[:, 1] if contract_dim == -1 else actual[1, :]
    values = source[:, 1] if contract_dim == -1 else source[1, :]
    magnitude = values.abs()
    unit_random = random_bytes.float() / 256.0
    step = torch.where(magnitude >= 4.0, 2.0, torch.where(magnitude >= 2.0, 1.0, 0.5))
    dithered = torch.addcmul(magnitude, unit_random, step)
    rounded_step = torch.where(
        dithered >= 4.0, 2.0, torch.where(dithered >= 2.0, 1.0, 0.5)
    )
    rounded = (torch.floor(dithered / rounded_step) * rounded_step).clamp(max=6.0)
    expected = torch.copysign(rounded, values)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
