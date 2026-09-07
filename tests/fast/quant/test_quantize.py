import pytest
import torch

from src.quant.constants import EPS
from src.quant.quantize import _compute_scale, dequantize_operand, quantize_operand
from src.quant.rotation import build_rotation
from src.quant.utils import is_int8s, str_to_dtype, str_to_qmax, str_to_qmin
from tests.fast.helper import cuda_sm89_or_newer
from tests.fast.quant.helper import (
    ALL_QUANT_FORMATS,
    ALL_SCALES,
    BLOCKWISE1D_16_E4M3,
    BLOCKWISE2D_16_E4M3,
    E4M3,
    NVFP4_16,
    NVFP4_2D_16,
    NVFP4_SCALES,
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
]
RAGGED_GROUPS = 3
INPUT_CASES = [
    (torch.float32, "normal"),
    (torch.float32, "transposed"),
    (torch.float16, "normal"),
    (torch.bfloat16, "normal"),
    (torch.bfloat16, "spread"),
    (torch.bfloat16, "zeros"),
    (torch.float32, "tiny"),
]

ROTATION_BLOCK_SIZES = [32]
ROTATION_RANDOM_SIGNS = [False, True]
ROTATION_SEEDS = [42]
ROTATION_BLOCK1_CFG = {
    "rotation_cls": "hadamard",
    "rotation_kwargs": {"block_size": 1, "random_sign": False},
}
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


def _rel_mse(source, dequantized):
    """Relative MSE in float64.

    An fp32 sum of squares underflows to zero for a tiny operand -- (2**-136)**2 is
    below fp32's min subnormal -- which would make the ratio NaN rather than small.
    """
    src, out = source.double(), dequantized.double()
    return ((src - out).square().sum() / src.square().sum()).item()


def _ragged_offs(extent):
    """Return unaligned ragged groups with an empty middle group."""
    first = extent // RAGGED_GROUPS
    return _offs([first, 0, extent - first])


def _bits(codes):
    """View one-byte codes as bytes to distinguish signed zero."""
    return codes.contiguous().view(torch.uint8)


def _rotation_cfg(block_size, random_sign, seed):
    rotation_kwargs = {
        "block_size": block_size,
        "random_sign": random_sign,
    }
    if random_sign:
        rotation_kwargs["seed"] = seed
    return {
        "rotation_cls": "hadamard",
        "rotation_kwargs": rotation_kwargs,
    }


def _n_blocks(extent, scale_cfg, ragged=False):
    """Return dense block count or a ragged static allocation bound."""
    block = _block_size(scale_cfg)
    if ragged:
        return extent // block + RAGGED_GROUPS if block else RAGGED_GROUPS
    return (extent + block - 1) // block if block else 1


def _block_size(scale_cfg):
    return scale_cfg["block_shape"][1] if scale_cfg["granularity"] == "blockwise" else 0


def _code_grid(fmt):
    """Return every finite code in `fmt`, sorted as fp32."""
    if is_int8s(fmt):
        qmax = int(str_to_qmax(fmt))
        return torch.arange(-qmax, qmax + 1, dtype=torch.float32)
    codes = torch.arange(256, dtype=torch.uint8).view(str_to_dtype(fmt)).float()
    return codes[codes.isfinite()].unique()


def _sqnr(x, deq):
    """Return the signal-to-quantization-noise ratio in dB."""
    err = (x - deq).float().norm().clamp_min(EPS)
    return (20.0 * torch.log10(x.float().norm().clamp_min(EPS) / err)).item()


class ErrorCase:
    """A rejected call and its expected exception."""

    def __init__(self, name, match, *, exception=ValueError, **kwargs):
        self.name = name
        self.match = match
        self.exception = exception
        self.kwargs = kwargs


# --- reference oracle ---


def _block_spans(length, extent, group_ends=None):
    """Return block spans that do not cross ragged group boundaries."""
    edges = [0, length] if group_ends is None else [0, *group_ends]
    return [
        (i, min(i + extent, hi))
        for lo, hi in zip(edges[:-1], edges[1:])
        for i in range(lo, hi, extent)
    ]


def _block_extents(x, contract_dim, scale_cfg):
    """Return contraction and outer block extents for `scale_cfg`."""
    outer_dim = -1 if contract_dim == -2 else -2
    block_outer, block_size = scale_cfg["block_shape"]
    if scale_cfg["granularity"] == "tensorwise":
        return x.shape[contract_dim], x.shape[outer_dim]
    if scale_cfg["granularity"] == "rowwise":
        return x.shape[contract_dim], 1
    return block_size, block_size if block_outer > 1 else 1


# Every value e4m3 represents from +0 to its max, ascending; 0x7F is NaN and excluded.
_E4M3_GRID = torch.arange(0, 127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()


def _ref_scale(amax, fmt, scale_dtype):
    """Return the reference scale, including narrow scale-dtype rounding.

    What this oracle derives independently is how blocks are cut and expanded, not
    the scalar formula, so a narrow dtype's rounding is mirrored here: divide by an
    unrounded scale and every code drifts a ULP from what the operand really used.
    """
    if scale_dtype is torch.float8_e8m0fnu:
        exp = torch.ceil(torch.log2(amax / str_to_qmax(fmt))).clamp(-127, 127)
        return torch.exp2(exp).to(scale_dtype).float()
    if scale_dtype is torch.float8_e4m3fn:
        low, high = str_to_qmin("fp8_e4m3"), str_to_qmax("fp8_e4m3")
        exact = (amax / str_to_qmax(fmt)).clamp(low, high)
        # Ceiling by lookup over every value e4m3 represents, which cross-checks the
        # implementation's byte increment without restating it.
        grid = _E4M3_GRID.to(exact.device)
        return grid[torch.searchsorted(grid, exact.contiguous())]
    return (amax / str_to_qmax(fmt)).clamp_min(EPS)


def _ref_divisor(x, contract_dim, fmt, scale_cfg, offs=None, ragged_dim=None):
    """Expand independently computed block scales to the operand layout."""
    v = (x if contract_dim == -1 else x.mT).float()  # (..., outer, contraction)
    flat = v.reshape(-1, *v.shape[-2:])
    contract_extent, outer_extent = _block_extents(x, contract_dim, scale_cfg)
    ends = None if offs is None else offs.tolist()
    div = torch.zeros_like(flat)
    for lo, hi in _block_spans(
        flat.shape[-2],
        outer_extent,
        ends if ragged_dim not in (None, contract_dim) else None,
    ):
        for start, stop in _block_spans(
            flat.shape[-1],
            contract_extent,
            ends if ragged_dim == contract_dim else None,
        ):
            block = flat[:, lo:hi, start:stop]
            amax = block.abs().amax((-2, -1), keepdim=True)
            div[:, lo:hi, start:stop] = _ref_scale(amax, fmt, scale_cfg["scale_dtype"])
    div = div.reshape(v.shape)
    return div if contract_dim == -1 else div.mT


def _ref_codes(x, div, fmt):
    """Quantize with the oracle's expanded scales."""
    qmax = str_to_qmax(fmt)
    xq = (x.float() / div).clamp(-qmax, qmax)
    return (torch.round(xq) if is_int8s(fmt) else xq).to(str_to_dtype(fmt))


# --- quantize_operand ---

QUANTIZE_ERROR_CASES = [
    ErrorCase(
        # Scale dtype is required; it is never inferred.
        "scale_cfg_without_scale_dtype",
        "scale_dtype",
        exception=KeyError,
        x=torch.ones(2, 2),
        contract_dim=-1,
        fmt=E4M3,
        scale_cfg={
            "granularity": "rowwise",
            "block_shape": (0, 0),
            "global_scale": False,
        },
    ),
    ErrorCase(
        "unknown_granularity",
        "unknown granularity",
        x=torch.randn(4, 8),
        contract_dim=-1,
        fmt=E4M3,
        scale_cfg=scale_of("groupwise", (1, 32)),
    ),
    ErrorCase(
        # Only the final two dimensions can be tiled.
        "contract_dim_out_of_range",
        "contract_dim must be -2 or -1",
        x=torch.randn(4, 8),
        contract_dim=0,
        fmt=E4M3,
        scale_cfg=TENSORWISE,
    ),
    ErrorCase(
        "offs_without_ragged_dim",
        "together",
        x=torch.randn(4, 8),
        contract_dim=-1,
        fmt=E4M3,
        scale_cfg=TENSORWISE,
        offs=_offs([2, 2]),
    ),
    ErrorCase(
        "ragged_dim_without_offs",
        "together",
        x=torch.randn(4, 8),
        contract_dim=-1,
        fmt=E4M3,
        scale_cfg=TENSORWISE,
        ragged_dim=-2,
    ),
    ErrorCase(
        "ragged_dim_out_of_range",
        "ragged_dim must be -2 or -1",
        x=torch.randn(4, 8),
        contract_dim=-1,
        fmt=E4M3,
        scale_cfg=TENSORWISE,
        offs=_offs([2, 2]),
        ragged_dim=0,
    ),
    ErrorCase(
        "ragged_3d_operand",
        "2D operand",
        x=torch.randn(2, 4, 8),
        contract_dim=-2,
        fmt=E4M3,
        scale_cfg=TENSORWISE,
        offs=_offs([2, 2]),
        ragged_dim=-2,
    ),
]


@pytest.mark.parametrize("case", QUANTIZE_ERROR_CASES)
def test_quantize_operand_raise_error(case):
    with pytest.raises(case.exception, match=case.match):
        quantize_operand(**case.kwargs)


@pytest.mark.parametrize("input_case", INPUT_CASES)
@pytest.mark.parametrize("geometry", GEOMETRY_CASES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("scale_cfg", ALL_SCALES)
@pytest.mark.parametrize("fmt", ALL_QUANT_FORMATS)
def test_quantize_operand_precision(fmt, scale_cfg, contract_dim, geometry, input_case):
    """Check codes, scales, and layout against an independent block-loop oracle."""
    skip_unsupported_fmt_scale(fmt, scale_cfg)
    shape, ragged_dim = geometry
    dtype, init_method = input_case

    x = _make(init_method, shape, dtype)
    offs = None if ragged_dim is None else _ragged_offs(shape[ragged_dim])
    div = _ref_divisor(x, contract_dim, fmt, scale_cfg, offs, ragged_dim)
    expected = list(shape)
    expected[contract_dim] = _n_blocks(
        shape[contract_dim], scale_cfg, ragged_dim == contract_dim
    )

    rng = torch.random.get_rng_state()
    codes, scale, _ = quantize_operand(
        x, contract_dim, fmt, scale_cfg, offs=offs, ragged_dim=ragged_dim
    )

    assert codes.shape == x.shape and codes.dtype == str_to_dtype(fmt)
    if init_method == "transposed" and scale_cfg["granularity"] != "blockwise":
        assert codes.stride() == x.stride()
    assert codes.float().abs().amax() <= str_to_qmax(fmt)
    assert tuple(scale.shape) == tuple(expected)
    assert scale.dtype is scale_cfg["scale_dtype"]
    if ragged_dim is None and scale_cfg["granularity"] == "tensorwise":
        assert 0 in scale.stride()
    # E8M0 carries no comparison or log2 kernel, so check the decoded exponents.
    decoded = scale.float()
    assert torch.isfinite(decoded).all() and (decoded > 0).all()
    if scale_cfg["scale_dtype"] is torch.float8_e8m0fnu:
        log2_scale = torch.log2(decoded)
        assert torch.equal(log2_scale, log2_scale.round())

    assert torch.equal(_bits(codes), _bits(_ref_codes(x, div, fmt)))

    assert torch.equal(torch.random.get_rng_state(), rng)
    again, _, _ = quantize_operand(
        x, contract_dim, fmt, scale_cfg, offs=offs, ragged_dim=ragged_dim
    )
    assert torch.equal(_bits(again), _bits(codes))

    if len(shape) > 2:
        per_expert = [
            quantize_operand(expert, contract_dim, fmt, scale_cfg) for expert in x
        ]
        assert torch.equal(
            _bits(codes), _bits(torch.stack([q for q, _, _ in per_expert]))
        )
        assert torch.equal(scale, torch.stack([s for _, s, _ in per_expert]))

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
            unfloored = div > str_to_qmin("fp8_e4m3")
            if unfloored.any():
                assert (x.abs().float() / div)[unfloored].amax().item() > qmax / 2
        else:
            assert ratio > qmax / 2  # E8M0 rounding leaves under one binade unused.


@pytest.mark.parametrize(
    "shape", [shape for shape, ragged_dim in GEOMETRY_CASES if ragged_dim is None]
)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("fmt", ALL_QUANT_FORMATS)
def test_quantize_operand_granularity_sqnr(fmt, contract_dim, shape):
    """Check that finer scales improve SQNR on scattered outliers.

    Integer formats must be monotonic; fp8 needs only an end-to-end gain.
    """
    x = _make("outlier", shape)
    sqnrs = [
        _sqnr(x, roundtrip(x, contract_dim, fmt, cfg)) for cfg in SCALES_COARSE_TO_FINE
    ]

    if is_int8s(fmt):
        assert sqnrs == sorted(sqnrs)
    # 1.1 dB leaves margin below the measured 3.33 dB minimum.
    assert sqnrs[-1] - sqnrs[1] > 1.1


# An operand magnitude far under e4m3's min scale, and one far over its max.
E4M3_SCALE_RANGE_MAGNITUDES = [1e-6, 1e9]


@pytest.mark.parametrize("magnitude", E4M3_SCALE_RANGE_MAGNITUDES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("fmt", ALL_QUANT_FORMATS)
def test_quantize_operand_e4m3_scale_clamps_to_representable_window(
    fmt, contract_dim, magnitude
):
    """An e4m3 scale outside its window clamps to the window's edge, on any device.

    Neither edge may lean on the cast. Below the min subnormal it would round to
    zero and divide the operand by zero; above the max it is device-dependent --
    a bare CPU cast of 1e10 yields NaN (e4m3's top encoding) while CUDA saturates
    to 448 -- so an unclamped overflow would poison every scale on CPU only.
    """
    scale_cfg = scale_of("blockwise", (1, 16), torch.float8_e4m3fn)
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32) * magnitude
    codes, scale, _ = quantize_operand(x, contract_dim, fmt, scale_cfg)
    assert scale.dtype is torch.float8_e4m3fn
    scale_values = scale.float()
    assert torch.isfinite(scale_values).all()
    assert (scale_values >= str_to_qmin("fp8_e4m3")).all()
    assert (scale_values <= str_to_qmax("fp8_e4m3")).all()
    assert torch.isfinite(codes.float()).all()
    assert torch.isfinite(
        dequantize_operand(codes, scale, contract_dim, scale_cfg)
    ).all()


# The scale dtypes whose narrow range forces a clamp. An fp32 scale is deliberately
# absent: it holds an infinite amax as inf, which is pre-existing behaviour and only
# reachable once the operand itself has diverged.
CLAMPED_SCALE_DTYPES = [torch.float8_e8m0fnu, torch.float8_e4m3fn]
# Block maxima on both sides of every narrow scale dtype's representable window.
OUT_OF_WINDOW_AMAX = [0.0, 1e-30, 1e30, float("inf")]


@pytest.mark.parametrize("amax_value", OUT_OF_WINDOW_AMAX)
@pytest.mark.parametrize("scale_dtype", CLAMPED_SCALE_DTYPES)
def test_compute_scale_clamps_to_representable_window(scale_dtype, amax_value):
    """A narrow scale clamps into its own range, identically on CPU and CUDA.

    Neither edge may lean on the cast. Below e4m3's min subnormal a scale rounds to
    zero and divides its operand by zero. Above its max the cast is device-dependent:
    a bare CPU cast of 1e10 or inf yields NaN, e4m3's top encoding, while CUDA
    saturates to 448. Asserting across devices is what gives the ceiling teeth --
    a CUDA-only assertion passes whether or not the clamp is there.
    """
    scales = [
        _compute_scale(
            torch.tensor([amax_value], device=device), "fp8_e4m3", scale_dtype
        ).float()
        for device in ("cpu", "cuda")
    ]
    for scale in scales:
        assert torch.isfinite(scale).all()
        assert (scale > 0).all()
    assert torch.equal(scales[0], scales[1].cpu())


# Eight decades of operand magnitude. Without a global scale an e4m3 block scale is
# only usable near 1e0; the whole point of the scale is that these agree.
GLOBAL_SCALE_MAGNITUDES = [1e-4, 1e-2, 1.0, 1e2, 1e4]
# Worst round-trip relMSE over the nvfp4 scale x geometry x contract_dim x input grid,
# times 4.3. The worst cell is a 2D block over a logspace-spread operand, where int4's
# own resolution dominates; this bound guards the (G,) plumbing, not the magnitude
# property, which the per-format bounds below carry.
GLOBAL_SCALE_ROUNDTRIP_BOUND = 0.17
# Worst relMSE over magnitude x contract_dim with the nvfp4 recipe, times 4.3. The
# bounds are per format because a single loose one would let int8 regress 300x and
# still pass. Unscaled, the far magnitudes sit near 1.0, so every bound has teeth.
GLOBAL_SCALE_REL_MSE_BOUND = {
    "fp8_e4m3": 0.0028,  # measured 6.429e-04
    "fp8_e5m2": 0.011,  # measured 2.647e-03
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
    base = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    x = base * magnitude
    deq = roundtrip(x, contract_dim, fmt, NVFP4_16)
    assert _rel_mse(x, deq) < GLOBAL_SCALE_REL_MSE_BOUND[fmt]


# The nvfp4 cells paired with the identical config minus the global scale.
GLOBAL_SCALE_OFF_PAIRS = [
    (NVFP4_16, BLOCKWISE1D_16_E4M3),
    (NVFP4_2D_16, BLOCKWISE2D_16_E4M3),
]


@pytest.mark.parametrize(("on_cfg", "off_cfg"), GLOBAL_SCALE_OFF_PAIRS)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("fmt", ALL_QUANT_FORMATS)
def test_quantize_operand_global_scale_disabled_changes_nothing(
    fmt, contract_dim, on_cfg, off_cfg
):
    """With the flag off, the nvfp4 config is bit-identical to the plain e4m3 one.

    Pins that the feature is inert when disabled, so an existing run's numerics
    cannot shift underneath it.
    """
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32) * 3
    off = dict(on_cfg, global_scale=False)
    codes_off, scale_off, g_off = quantize_operand(x, contract_dim, fmt, off)
    codes_plain, scale_plain, g_plain = quantize_operand(x, contract_dim, fmt, off_cfg)
    assert g_off is None and g_plain is None
    assert torch.equal(_bits(codes_off), _bits(codes_plain))
    assert torch.equal(scale_off.float(), scale_plain.float())

    # And that it is not inert when on, or the test above would prove nothing.
    codes_on, _, g_on = quantize_operand(x, contract_dim, fmt, on_cfg)
    assert g_on is not None and g_on.dtype is torch.float32


GLOBAL_SCALE_SHAPE_CASES = [
    ((64, 128), None, None, 1),  # dense 2D is one group
    ((3, 64, 128), None, None, 3),  # stacked 3D is one group per expert
    ((70, 130), -2, RAGGED_GROUPS, RAGGED_GROUPS),
    ((70, 130), -1, RAGGED_GROUPS, RAGGED_GROUPS),
]


@pytest.mark.parametrize(
    ("shape", "ragged_dim", "groups", "expected"), GLOBAL_SCALE_SHAPE_CASES
)
def test_quantize_operand_global_scale_shape(shape, ragged_dim, groups, expected):
    """One fp32 global scale per group, whatever supplies the groups."""
    x = torch.randn(*shape, device="cuda", dtype=torch.float32)
    offs = None if groups is None else _ragged_offs(shape[ragged_dim])
    _, _, g = quantize_operand(
        x, -1, "int4", NVFP4_16, offs=offs, ragged_dim=ragged_dim
    )
    assert g.shape == (expected,)
    assert g.dtype is torch.float32
    assert (g > 0).all()


@pytest.mark.parametrize("input_case", INPUT_CASES)
@pytest.mark.parametrize(("shape", "ragged_dim"), GEOMETRY_CASES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("scale_cfg", NVFP4_SCALES)
def test_quantize_operand_global_scale_roundtrip(
    scale_cfg, contract_dim, shape, ragged_dim, input_case
):
    """A global scale round-trips over every geometry that supplies its groups."""
    dtype, init_method = input_case
    x = _make(init_method, shape, dtype).cuda()
    offs = None if ragged_dim is None else _ragged_offs(shape[ragged_dim])
    codes, scale, g = quantize_operand(
        x, contract_dim, "int4", scale_cfg, offs=offs, ragged_dim=ragged_dim
    )
    expected_groups = 1 if offs is None else offs.numel()
    assert g.shape == (len(x) if x.ndim == 3 else expected_groups,)
    deq = dequantize_operand(
        codes,
        scale,
        contract_dim,
        scale_cfg,
        offs=offs,
        ragged_dim=ragged_dim,
        global_scale=g,
    )
    assert torch.isfinite(deq).all()
    source = x.float()
    if init_method in ("zeros", "tiny"):
        # An operand under EPS floors the scale and flushes to zero. The plain fp32
        # scale path does exactly the same, so this is not the global scale's doing;
        # only e8m0, whose floor is 2**-127 rather than 1e-30, reaches low enough to
        # carry 2**-136 through.
        assert torch.equal(deq, torch.zeros_like(deq))
        return
    assert _rel_mse(source, deq) < GLOBAL_SCALE_ROUNDTRIP_BOUND


def test_quantize_operand_global_scale_is_per_group():
    """A loud group must not degrade a quiet one -- the reason it is per group.

    Contrasted against one scale over the whole tensor, where the loud group sets g
    and drives every quiet block's scale under e4m3's floor.
    """
    torch.manual_seed(0)
    quiet = torch.randn(32, 128, device="cuda", dtype=torch.float32)
    loud = torch.randn(32, 128, device="cuda", dtype=torch.float32) * 1e6
    x = torch.cat([quiet, loud], dim=0)
    offs = _offs([32, 32])

    def quiet_error(**ragged):
        codes, scale, g = quantize_operand(x, -1, "int4", NVFP4_16, **ragged)
        deq = dequantize_operand(codes, scale, -1, NVFP4_16, global_scale=g, **ragged)
        per_row = (x - deq).square().sum(-1) / x.square().sum(-1)
        return per_row[:32].mean().item(), g

    grouped, g_grouped = quiet_error(offs=offs, ragged_dim=-2)
    pooled, g_pooled = quiet_error()
    assert g_grouped.shape == (2,) and g_pooled.shape == (1,)
    # Measured 8.3e-3 per group against 5.0e-1 pooled: a 60x gap, the quiet group
    # wiped out because the loud group's amax sets g and floors every quiet block.
    assert grouped < 1e-2
    assert pooled > 0.3


@pytest.mark.parametrize("enable_sr", [False, True])
@pytest.mark.parametrize("fmt", ALL_QUANT_FORMATS)
def test_quantize_operand_stochastic_rounding(fmt, enable_sr):
    """Check adjacent codes, deterministic RNE, and unbiased stochastic rounding."""
    torch.manual_seed(0)
    grid = _code_grid(fmt)
    probe = torch.cat([grid, (grid[:-1] + grid[1:]) / 2]).reshape(1, -1)
    codes, scale, _ = quantize_operand(
        probe, -1, fmt, TENSORWISE, stochastic_rounding=enable_sr
    )
    assert scale.item() == 1.0  # The row peak is qmax.
    below = torch.searchsorted(grid, probe.flatten().contiguous())
    lower = grid[(below - 1).clamp(min=0)]
    upper = grid[below.clamp(max=grid.numel() - 1)]
    assert bool(
        ((codes.float().flatten() == lower) | (codes.float().flatten() == upper)).all()
    )

    qmax = str_to_qmax(fmt)
    ends, _, _ = quantize_operand(
        torch.tensor([[1.0, 0.0, -0.0, -1.0]]),
        -1,
        fmt,
        TENSORWISE,
        stochastic_rounding=enable_sr,
    )
    assert torch.equal(ends.float(), torch.tensor([[qmax, 0.0, -0.0, -qmax]]))
    if not is_int8s(fmt):
        assert torch.signbit(ends.float()[0, 2])  # Preserve negative zero.

    # The leading 1 fixes the scale; 0.3 lies between codes in every format.
    x = torch.cat([torch.ones(1), torch.full((9999,), 0.3)]).reshape(1, -1)
    draws, scale, _ = quantize_operand(
        x, -1, fmt, TENSORWISE, stochastic_rounding=enable_sr
    )
    again, _, _ = quantize_operand(
        x, -1, fmt, TENSORWISE, stochastic_rounding=enable_sr
    )
    reached = draws.float()[0, 1:].unique().numel()

    if not enable_sr:
        assert reached == 1  # RNE always picks the nearer code.
        assert torch.equal(_bits(again), _bits(draws))
        return

    assert reached == 2  # SR draws both adjacent codes.
    assert (draws.float() * scale)[0, 1:].mean().item() == pytest.approx(
        0.3, abs=0.0021
    )
    # Repeating all 9,999 stochastic draws is negligibly likely.
    assert not torch.equal(_bits(again), _bits(draws))


# --- dequantize_operand ---

DEQUANTIZE_ERROR_CASES = [
    ErrorCase(
        # Only the final two dimensions can be tiled.
        "contract_dim_out_of_range",
        "contract_dim must be -2 or -1",
        xq=torch.ones(3, 3, 4, dtype=torch.int8),
        scale=torch.arange(1.0, 13.0).reshape(1, 3, 4),
        contract_dim=0,
        scale_cfg=TENSORWISE,
    ),
    ErrorCase(
        # Both APIs share dimension validation, including ragged-axis errors.
        "offs_without_ragged_dim",
        "together",
        xq=torch.ones(4, 4, dtype=torch.int8),
        scale=torch.ones(4, 1),
        contract_dim=-1,
        scale_cfg=TENSORWISE,
        offs=_offs([2, 2]),
    ),
    ErrorCase(
        "ragged_dim_without_offs",
        "together",
        xq=torch.ones(4, 4, dtype=torch.int8),
        scale=torch.ones(4, 1),
        contract_dim=-1,
        scale_cfg=TENSORWISE,
        ragged_dim=-2,
    ),
    ErrorCase(
        "ragged_dim_out_of_range",
        "ragged_dim must be -2 or -1",
        xq=torch.ones(4, 4, dtype=torch.int8),
        scale=torch.ones(4, 1),
        contract_dim=-1,
        scale_cfg=TENSORWISE,
        offs=_offs([2, 2]),
        ragged_dim=0,
    ),
    ErrorCase(
        "ragged_3d_operand",
        "2D operand",
        xq=torch.ones(2, 4, 4, dtype=torch.int8),
        scale=torch.ones(2, 4, 1),
        contract_dim=-1,
        scale_cfg=TENSORWISE,
        offs=_offs([2, 2]),
        ragged_dim=-2,
    ),
    ErrorCase(
        # An incomplete config must not infer a block width.
        "scale_cfg_without_block_shape",
        "block_shape",
        exception=KeyError,
        xq=torch.ones(4, 4, dtype=torch.int8),
        scale=torch.ones(4, 1),
        contract_dim=-1,
        scale_cfg={"granularity": "rowwise", "global_scale": False},
    ),
    ErrorCase(
        "rotation_with_unaligned_ragged_contraction",
        "align",
        exception=AssertionError,
        xq=torch.ones(4, 8, dtype=torch.int8),
        scale=torch.ones(4, 2),
        contract_dim=-1,
        scale_cfg=TENSORWISE,
        offs=_offs([2, 6]),
        ragged_dim=-1,
        rotation=build_rotation(ROTATION_BLOCK4_CFG),
    ),
]


@pytest.mark.parametrize("case", DEQUANTIZE_ERROR_CASES)
def test_dequantize_operand_raise_error(case):
    with pytest.raises(case.exception, match=case.match):
        dequantize_operand(**case.kwargs)


@pytest.mark.parametrize("input_case", INPUT_CASES)
@pytest.mark.parametrize("geometry", GEOMETRY_CASES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("scale_cfg", ALL_SCALES)
@pytest.mark.parametrize("fmt", ALL_QUANT_FORMATS)
def test_dequantize_operand_precision(
    fmt, scale_cfg, contract_dim, geometry, input_case
):
    """Check fp32 reconstruction against the independent block-loop oracle."""
    skip_unsupported_fmt_scale(fmt, scale_cfg)
    shape, ragged_dim = geometry
    dtype, init_method = input_case

    x = _make(init_method, shape, dtype)
    offs = None if ragged_dim is None else _ragged_offs(shape[ragged_dim])
    div = _ref_divisor(x, contract_dim, fmt, scale_cfg, offs, ragged_dim)
    codes, scale, _ = quantize_operand(
        x, contract_dim, fmt, scale_cfg, offs=offs, ragged_dim=ragged_dim
    )
    deq = dequantize_operand(
        codes, scale, contract_dim, scale_cfg, offs=offs, ragged_dim=ragged_dim
    )

    assert deq.shape == x.shape
    assert deq.dtype == torch.float32 and deq.is_contiguous()
    assert torch.equal(deq, codes.float() * div)


# --- quantize/dequantize with rotation ---


@pytest.mark.parametrize("rotation_seed", ROTATION_SEEDS)
@pytest.mark.parametrize("rotation_random_sign", ROTATION_RANDOM_SIGNS)
@pytest.mark.parametrize("rotation_block_size", ROTATION_BLOCK_SIZES)
def test_quantize_operand_rotation_improves_sqnr(
    rotation_block_size, rotation_random_sign, rotation_seed
):
    """On outlier-corrupted int8, a block-32 rotation lifts SQNR by >3 dB."""

    torch.manual_seed(0)
    x = _make("outlier", (256, 256))
    baseline = _sqnr(x, roundtrip(x, -1, "int8", TENSORWISE))
    rotated = _sqnr(
        x,
        roundtrip(
            x,
            -1,
            "int8",
            TENSORWISE,
            rotation=build_rotation(
                _rotation_cfg(
                    rotation_block_size,
                    rotation_random_sign,
                    rotation_seed,
                )
            ),
        ),
    )
    # Measured gain is ~8 dB over 30 seeds; 3 dB leaves an order of magnitude.
    assert rotated > baseline + 3.0


@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
def test_quantize_operand_rotation_block1_matches_baseline(contract_dim):
    """block=1 rotation is the identity: bit-identical codes and scales.

    `contract_dim` can be -2 or -1; block=1 must never touch the fast path.
    """
    torch.manual_seed(0)
    x = _make("outlier", (64, 128))
    baseline_codes, baseline_scale, _ = quantize_operand(
        x, contract_dim, "int8", TENSORWISE
    )
    rotation = build_rotation(ROTATION_BLOCK1_CFG)
    codes, scale, _ = quantize_operand(
        x, contract_dim, "int8", TENSORWISE, rotation=rotation
    )
    assert torch.equal(codes, baseline_codes)
    assert torch.equal(scale, baseline_scale)


# (contract_dim, ragged_dim) pairs whose contraction axis stays dense.
RAGGED_DENSE_CONTRACTIONS = [(-1, -2), (-2, -1)]


@pytest.mark.parametrize("rotation_seed", ROTATION_SEEDS)
@pytest.mark.parametrize("rotation_random_sign", ROTATION_RANDOM_SIGNS)
@pytest.mark.parametrize("rotation_block_size", ROTATION_BLOCK_SIZES)
@pytest.mark.parametrize("contract_dim, ragged_dim", RAGGED_DENSE_CONTRACTIONS)
def test_quantize_operand_rotation_ragged_outer_axis(
    contract_dim,
    ragged_dim,
    rotation_block_size,
    rotation_random_sign,
    rotation_seed,
):
    """A ragged outer axis leaves the rotated contraction dense, so groups never mix."""
    torch.manual_seed(0)
    x = _make("normal", (64, 64))
    offs = _offs([21, 0, 43])
    rotation = build_rotation(
        _rotation_cfg(rotation_block_size, rotation_random_sign, rotation_seed)
    )

    codes, scale, _ = quantize_operand(
        x,
        contract_dim,
        "int8",
        TENSORWISE,
        offs=offs,
        ragged_dim=ragged_dim,
        rotation=rotation,
    )
    deq = dequantize_operand(
        codes,
        scale,
        contract_dim,
        TENSORWISE,
        offs=offs,
        ragged_dim=ragged_dim,
        rotation=rotation,
    )

    assert _sqnr(x, deq) > 10.0


@pytest.mark.parametrize("rotation_seed", ROTATION_SEEDS)
@pytest.mark.parametrize("rotation_random_sign", ROTATION_RANDOM_SIGNS)
@pytest.mark.parametrize("rotation_block_size", ROTATION_BLOCK_SIZES)
def test_quantize_operand_rotation_ragged_contraction_roundtrip(
    rotation_block_size, rotation_random_sign, rotation_seed
):
    """Aligned groups support a rotated ragged-contraction roundtrip."""
    x = torch.randn(64, 64)
    rotation = build_rotation(
        _rotation_cfg(rotation_block_size, rotation_random_sign, rotation_seed)
    )
    offs = _offs([32, 32])
    codes, scale, _ = quantize_operand(
        x,
        -2,
        "int8",
        TENSORWISE,
        offs=offs,
        ragged_dim=-2,
        rotation=rotation,
    )
    dequantized = dequantize_operand(
        codes,
        scale,
        -2,
        TENSORWISE,
        offs=offs,
        ragged_dim=-2,
        rotation=rotation,
    )
    assert _sqnr(x, dequantized) > 10.0


def test_quantize_operand_rotation_rejects_indivisible_block():
    """A block that does not divide the rotated extent is rejected."""
    x = torch.randn(8, 100)
    rotation = build_rotation(ROTATION_BLOCK32_CFG)
    with pytest.raises(ValueError, match="block 32 must divide"):
        quantize_operand(x, -1, "int8", TENSORWISE, rotation=rotation)


@cuda_sm89_or_newer
@pytest.mark.parametrize("scale_cfg", [ROWWISE, NVFP4_16])
def test_quantize_operand_compiles_fullgraph(scale_cfg):
    """Rotated FP8 quantization must have identical eager and compiled bits.

    The nvfp4 case also pins that a global scale stays a tensor: reading it back
    as a Python float would sync the device and break fullgraph here.
    """
    torch.manual_seed(0)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    rotation = build_rotation(ROTATION_BLOCK32_CFG).cuda()

    def quantize(x):
        return quantize_operand(x, -1, E4M3, scale_cfg, rotation=rotation)

    eager_codes, eager_scale, eager_global = quantize(x)
    compiled_codes, compiled_scale, compiled_global = torch.compile(
        quantize, fullgraph=True
    )(x)

    assert torch.equal(compiled_codes, eager_codes)
    assert torch.equal(compiled_scale, eager_scale)
    assert (eager_global is None) == (compiled_global is None)
    if eager_global is not None:
        assert torch.equal(compiled_global, eager_global)


def test_quantize_operand_compiled_raise_error():
    prev = torch.get_default_device()
    torch.set_default_device("cpu")
    try:
        x = torch.ones(4, 8)
        offs = _offs([2, 6])
        rotation = build_rotation(ROTATION_BLOCK4_CFG)

        def call() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
            return quantize_operand(
                x,
                -1,
                "int8",
                TENSORWISE,
                offs=offs,
                ragged_dim=-1,
                rotation=rotation,
            )

        compiled = torch.compile(call, backend="eager", fullgraph=True)
        for fn in [call, compiled]:
            with pytest.raises(
                (AssertionError, RuntimeError),
                match="ragged contraction boundaries must align with rotation blocks",
            ):
                fn()
    finally:
        torch.set_default_device(prev)


def test_dequantize_operand_compiled_raise_error():
    prev = torch.get_default_device()
    torch.set_default_device("cpu")
    try:
        xq = torch.ones(4, 8, dtype=torch.int8)
        scale = torch.ones(4, 2)
        offs = _offs([2, 6])
        rotation = build_rotation(ROTATION_BLOCK4_CFG)

        def call() -> torch.Tensor:
            return dequantize_operand(
                xq,
                scale,
                -1,
                TENSORWISE,
                offs=offs,
                ragged_dim=-1,
                rotation=rotation,
            )

        compiled = torch.compile(call, backend="eager", fullgraph=True)
        for fn in [call, compiled]:
            with pytest.raises(
                (AssertionError, RuntimeError),
                match="ragged contraction boundaries must align with rotation blocks",
            ):
                fn()
    finally:
        torch.set_default_device(prev)


@pytest.mark.parametrize("rotation_seed", ROTATION_SEEDS)
@pytest.mark.parametrize("rotation_random_sign", ROTATION_RANDOM_SIGNS)
@pytest.mark.parametrize("rotation_block_size", ROTATION_BLOCK_SIZES)
def test_quantize_operand_rotation_roundtrip_approximates_baseline(
    rotation_block_size, rotation_random_sign, rotation_seed
):
    """Rotated roundtrip error stays within fp32 reconstruction error.

    The rotation is an isometry, so the quantize->dequantize error should be
    comparable to the unrotated path on a well-scaled operand (no outliers).
    """
    torch.manual_seed(0)
    x = _make("normal", (64, 128))
    baseline_deq = roundtrip(x, -1, "int8", TENSORWISE)
    rotation = build_rotation(
        _rotation_cfg(rotation_block_size, rotation_random_sign, rotation_seed)
    )
    rotated_deq = roundtrip(x, -1, "int8", TENSORWISE, rotation=rotation)
    # Both must reconstruct x reasonably; neither is allowed to blow up.
    assert _sqnr(x, baseline_deq) > 10.0
    assert _sqnr(x, rotated_deq) > 10.0
