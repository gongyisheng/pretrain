# todo: test refactory
import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.kernel.ops.gemm import grouped_mm
from src.layers.mlp import SparseMoEBlock
from src.metrics.functional import compute_quantization_metrics
from src.metrics.quant import QuantizationStats, set_quantization_monitoring_status
from src.model import build_model
from src.quant.convert import apply_quantization, enable_quantization
from src.quant.moe import (
    QuantizedSparseMoEBlock,
    ScaledGroupedGemmFn,
    quantized_grouped_mm,
)
from src.quant.quantize import dequantize_operand, quantize_operand
from src.quant.utils import is_fp4, is_quantized, resolve_scale
from src.quant.rotation import apply_rotation_on_axes, build_rotation
from src.utils.config import (
    ModelConfig,
    QuantizationConfig,
    TrainConfig,
    TrainingConfig,
)
from tests.fast.helper import cuda_capability_at_least, cuda_sm89_or_newer
from tests.fast.quant.helper import (
    ALL_FORMATS,
    FORWARD_DTYPES,
    BACKWARD_DTYPES,
    INT4_W8A16_DTYPES,
    FP8_E4M3_W8A8_E5M2_G8_DTYPES,
    FP4_E2M1_W4A4G4_DTYPES,
    FP4_E2M1_4OVER6_W4A4G4_DTYPES,
    BLOCKWISE1D_16,
    BLOCKWISE1D_32_E8M0,
    BLOCKWISE1D_16_E2M1,
    BLOCKWISE2D_16,
    BLOCKWISE2D_32_E8M0,
    BLOCKWISE2D_16_E2M1,
    ROWWISE,
    TENSORWISE,
    mm_ref,
    operand_fmt,
    rel,
    rule,
    SCALE_PAIRS,
    SCALE_TRIPLES,
    scale_combinations,
    skip_unsupported_dtype_scale,
    skip_unsupported_ragged_k_scale,
    skip_unsupported_fmt_scale,
    uses_fp4_gemm,
)


# Worst relative error across format, scale, and layout grids: 2.93e-4 (3.41x).
PRECISION_BOUND = 1e-3

# 299 rows across four experts; expert 1 is empty to test expert boundaries.
COUNTS = [128, 0, 130, 41]
FP4_COUNTS = [128, 0, 128, 48]
GROUPED_COUNTS = [COUNTS, FP4_COUNTS]


def _cfg(scale_cfg=ROWWISE, dtype=None):
    return rule(dtype or FP8_E4M3_W8A8_E5M2_G8_DTYPES, scale_cfg)


def _per_tensor_scale(act_scale, weight_scale, grad_out_scale):
    return {
        "weight": weight_scale,
        "act": act_scale,
        "grad_out": grad_out_scale,
    }


def _expert_mm(cfg, a, b, offs, bias=None, rotation=None):
    """Run the quantized expert GEMM without block metadata or statistics."""
    return ScaledGroupedGemmFn.apply(a, b, bias, offs, cfg, {}, rotation)


def _make(counts, K, N, seed=0):
    torch.manual_seed(seed)
    n_experts, n_rows = len(counts), sum(counts)
    a = torch.randn(n_rows, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n_experts, K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    return a, b, offs


# --- quantized_grouped_mm ---


# Exercise ragged-M, ragged-K, and ragged-N layouts.
GROUPED_LAYOUTS = ["ragged_m", "ragged_k", "ragged_n"]
# MXFP8 ragged-K needs 32-wide contraction blocks.
MXFP8_RAGGED_K_SCALES = [BLOCKWISE1D_32_E8M0, BLOCKWISE2D_32_E8M0]
GROUPED_SCALE_PAIRS = SCALE_PAIRS + scale_combinations(MXFP8_RAGGED_K_SCALES, 2)
GROUPED_SCALE_TRIPLES = SCALE_TRIPLES + scale_combinations(MXFP8_RAGGED_K_SCALES, 3)


def test_quantized_grouped_mm_raise_error():
    a = torch.ones(2, 4, device="cpu", dtype=torch.bfloat16)
    b = torch.ones(1, 4, 3, device="cpu", dtype=torch.float16)
    offs = torch.tensor([2], device="cpu", dtype=torch.int32)

    with pytest.raises(ValueError):
        quantized_grouped_mm(a, b, offs, "bf16", "fp16", a.dtype, ROWWISE, ROWWISE)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_quantized_grouped_mm_rotated_ragged_axis_raise_error(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    rotation = build_rotation(
        {"rotation_cls": "hadamard", "rotation_kwargs": {"block_size": 4}}
    )
    a = torch.ones(3, 4, device=device)
    b = torch.ones(1, 4, 2, device=device)
    offs = torch.tensor([3], device=device, dtype=torch.int32)
    a_rotation_axes = (-2,)

    with pytest.raises(AssertionError):
        quantized_grouped_mm(
            a,
            b,
            offs,
            "bf16",
            "bf16",
            a.dtype,
            TENSORWISE,
            TENSORWISE,
            rotation=rotation,
            a_rotation_axes=a_rotation_axes,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_quantized_grouped_mm_rotation_ragged_k_bias(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    rotation = build_rotation(
        {"rotation_cls": "hadamard", "rotation_kwargs": {"block_size": 4}}
    )
    torch.manual_seed(0)
    a = torch.randn(4, 4, device=device)
    b = torch.randn(4, 4, device=device)
    bias = torch.randn(1, 4, device=device)
    offs = torch.tensor([4], device=device, dtype=torch.int32)
    a_rotation_axes = (-2,)
    b_rotation_axes = (-1,)

    out = quantized_grouped_mm(
        apply_rotation_on_axes(a, rotation, a_rotation_axes),
        apply_rotation_on_axes(b, rotation, b_rotation_axes),
        offs,
        "bf16",
        "bf16",
        a.dtype,
        TENSORWISE,
        TENSORWISE,
        bias=bias,
        rotation=rotation,
        a_rotation_axes=a_rotation_axes,
        b_rotation_axes=b_rotation_axes,
    )
    torch.testing.assert_close(out, (a @ b + bias).unsqueeze(0), atol=1e-5, rtol=0)


@pytest.mark.parametrize("a_fmt", ALL_FORMATS)
@pytest.mark.parametrize("b_fmt", ALL_FORMATS)
@pytest.mark.parametrize("a_scale,b_scale", GROUPED_SCALE_PAIRS)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("counts", GROUPED_COUNTS)
def test_quantized_grouped_mm_precision(
    a_fmt,
    b_fmt,
    a_scale,
    b_scale,
    layout,
    bias,
    counts,
):
    """Compare each layout with per-expert dequantized GEMM, including optional bias."""
    if not cuda_capability_at_least((8, 9)):
        pytest.skip("CUDA SM89 or newer required")
    skip_unsupported_fmt_scale(a_fmt, a_scale)
    skip_unsupported_fmt_scale(b_fmt, b_scale)
    has_fp4 = is_fp4(a_fmt) or is_fp4(b_fmt)
    if uses_fp4_gemm(a_fmt, b_fmt, a_scale) and not cuda_capability_at_least((10, 0)):
        pytest.skip("fused FP4 requires CUDA SM100 or newer")
    if layout == "ragged_k":
        skip_unsupported_ragged_k_scale(a_scale)
        skip_unsupported_ragged_k_scale(b_scale)
    if bias and layout != "ragged_m":
        pytest.skip(
            "bias is defined per expert over the row axis, which only ragged-M has"
        )
    a, b, offs = _make(counts, K=64, N=48)

    if layout == "ragged_m":
        # Ragged-M: (R,K) x (E,K,N) -> (R,N)
        bias0 = (
            torch.randn(offs.shape[0], 48, device="cuda", dtype=torch.bfloat16) * 0.1
            if bias
            else None
        )
        out = quantized_grouped_mm(
            a, b, offs, a_fmt, b_fmt, a.dtype, a_scale, b_scale, bias=bias0
        )
        ref = torch.empty_like(out)
        lo = 0
        for group, hi in enumerate(offs.tolist()):
            if hi > lo:
                fwd = mm_ref(
                    a[lo:hi],
                    b[group],
                    a_fmt,
                    b_fmt,
                    a_scale,
                    b_scale,
                )
                if bias:
                    fwd = fwd + bias0[group].to(fwd.dtype)
                ref[lo:hi] = fwd.to(out.dtype)
            lo = hi
    elif layout == "ragged_n":
        # Ragged-N: (E,M,K) x (K,N) -> (M,N)
        slabs = torch.randn(offs.shape[0], 32, 64, device="cuda", dtype=torch.bfloat16)
        cols = torch.randn(64, a.shape[0], device="cuda", dtype=torch.bfloat16) * 0.1
        out = quantized_grouped_mm(
            slabs, cols, offs, a_fmt, b_fmt, slabs.dtype, a_scale, b_scale
        )
        ref = torch.empty_like(out)
        lo = 0
        for group, hi in enumerate(offs.tolist()):
            if hi > lo:
                ref[:, lo:hi] = mm_ref(
                    slabs[group],
                    cols[:, lo:hi],
                    a_fmt,
                    b_fmt,
                    a_scale,
                    b_scale,
                ).to(out.dtype)
            lo = hi
    else:
        # Ragged-K: (K,R) x (R,N) -> (E,K,N)
        gy = torch.randn(a.shape[0], 48, device="cuda", dtype=torch.bfloat16)
        out = quantized_grouped_mm(
            a.mT, gy, offs, a_fmt, b_fmt, a.dtype, a_scale, b_scale
        )
        ref = torch.zeros_like(out)
        lo = 0
        for group, hi in enumerate(offs.tolist()):
            if hi > lo:
                padding = (-(hi - lo)) % 16 if has_fp4 else 0
                ref[group] = mm_ref(
                    F.pad(a[lo:hi].t(), (0, padding)),
                    F.pad(gy[lo:hi], (0, 0, 0, padding)),
                    a_fmt,
                    b_fmt,
                    a_scale,
                    b_scale,
                ).to(out.dtype)
            lo = hi

    assert rel(out, ref) < PRECISION_BOUND, rel(out, ref)


# Format pairs bind the expected statistics.
# Keep each case on one line for readability.
# fmt: off
GROUPED_STATS_CONFIGS = [
    ("int8", "int8", False, False, False),
    ("int8", "bf16", True, True, False),
    ("int8", "int8", True, True, True),
    ("fp4_e2m1", "fp4_e2m1", True, True, True),
    ("fp4_e2m1_4over6", "fp4_e2m1_4over6", True, True, True),
]
# fmt: on
GROUPED_STATS_LAYOUTS = ["ragged_m", "ragged_k", "ragged_n"]
GROUPED_STATS_DEVICES = ["cpu", "cuda"]
GROUPED_STATS_WITH_PADDING = [False, True]
PADDING_STATS_CONFIG = ("fp4_e2m1", "fp4_e2m1", True, True, True)
PADDING_STATS_SCALES = (TENSORWISE, TENSORWISE)


@pytest.mark.parametrize("device", GROUPED_STATS_DEVICES)
@pytest.mark.parametrize("layout", GROUPED_STATS_LAYOUTS)
@pytest.mark.parametrize("config", GROUPED_STATS_CONFIGS)
@pytest.mark.parametrize("a_scale,b_scale", SCALE_PAIRS)
@pytest.mark.parametrize("with_padding", GROUPED_STATS_WITH_PADDING)
def test_quantized_grouped_mm_records_stats(
    device, layout, config, a_scale, b_scale, with_padding
):
    a_fmt, b_fmt, with_stats, a_folded, b_folded = config
    if with_padding and (
        device != "cpu"
        or config != PADDING_STATS_CONFIG
        or (a_scale, b_scale) != PADDING_STATS_SCALES
    ):
        pytest.skip("padding statistics cover CPU FP4 tensorwise operands")
    skip_unsupported_fmt_scale(a_fmt, a_scale)
    skip_unsupported_fmt_scale(b_fmt, b_scale)
    nvfp4 = is_fp4(a_fmt)
    if nvfp4 and layout != "ragged_m" and not with_padding:
        pytest.skip("NVFP4 statistics coverage uses ragged-M")
    if not nvfp4 and device == "cpu":
        pytest.skip("CPU statistics coverage is limited to NVFP4")
    capability = (10, 0) if nvfp4 else (8, 9)
    if device == "cuda" and not cuda_capability_at_least(capability):
        pytest.skip(f"CUDA SM{capability[0]}{capability[1]} or newer required")
    if is_fp4(a_fmt):
        torch.manual_seed(0)
        counts = [16, 0, 32] if with_padding else [16, 16]
        offs = torch.tensor(counts, device=device, dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        a = torch.randn(sum(counts), 32, device=device)
        if with_padding:
            a[0].zero_()
        b = torch.randn(len(counts), 32, 8 if with_padding else 16, device=device)
    else:
        a, b, offs = _make(COUNTS, K=64, N=48)
    if layout == "ragged_m":
        src_a, src_b = a, b
    elif layout == "ragged_k":
        src_a = a.mT
        src_b = torch.randn(a.shape[0], b.shape[-1], device=a.device, dtype=a.dtype)
        if with_padding:
            src_b[0].zero_()
    else:
        src_a, src_b = b.mT, a.mT
    a_stats = QuantizationStats("act/x", a.device) if with_stats else None
    b_stats = QuantizationStats("weight/x", b.device) if with_stats else None
    unmonitored = quantized_grouped_mm(
        src_a,
        src_b,
        offs,
        a_fmt,
        b_fmt,
        torch.float32 if is_fp4(a_fmt) else a.dtype,
        a_scale,
        b_scale,
    )
    set_quantization_monitoring_status(True)
    try:
        out = quantized_grouped_mm(
            src_a,
            src_b,
            offs,
            a_fmt,
            b_fmt,
            torch.float32 if is_fp4(a_fmt) else a.dtype,
            a_scale,
            b_scale,
            a_stats=a_stats,
            b_stats=b_stats,
        )
    finally:
        set_quantization_monitoring_status(False)
    torch.testing.assert_close(out, unmonitored, rtol=0, atol=0)
    if is_fp4(a_fmt) and not with_padding:
        expected = []
        start = 0
        for group, stop in enumerate(offs.tolist()):
            aq, sa, gsa, _ = quantize_operand(a[start:stop], -1, a_fmt, a_scale)
            bq, sb, gsb, _ = quantize_operand(b[group], -2, b_fmt, b_scale)
            expected.append(
                dequantize_operand(aq, sa, -1, a_scale, global_scale=gsa)
                @ dequantize_operand(bq, sb, -2, b_scale, global_scale=gsb)
            )
            start = stop
        torch.testing.assert_close(out, torch.cat(expected), rtol=0, atol=1e-5)
    else:
        assert torch.isfinite(out).all()
    if with_stats:
        assert a_stats.numel.shape == (1,)
        assert b_stats.numel.shape == (1,)
        # FP4 uses 16-row alignment: 48 input rows plus three 16-row slots.
        expected_padded_rows = 96
        expected_a_numel = (
            expected_padded_rows * src_a.shape[-2]
            if with_padding and layout == "ragged_k"
            else src_a.numel()
        )
        expected_b_numel = (
            expected_padded_rows * src_b.shape[-1]
            if with_padding and layout == "ragged_k"
            else src_b.numel()
        )
        assert a_stats.numel.item() == (expected_a_numel if a_folded else 0)
        assert b_stats.numel.item() == (expected_b_numel if b_folded else 0)
        for source, stats, folded in (
            (src_a, a_stats, a_folded),
            (src_b, b_stats, b_folded),
        ):
            if folded and not with_padding:
                expected = source.float().square().sum().reshape(1)
                assert torch.equal(stats.src_sq, expected)
        folded_stats = [
            stats
            for stats, folded in ((a_stats, a_folded), (b_stats, b_folded))
            if folded
        ]
        metrics = compute_quantization_metrics(torch.nn.ModuleList(folded_stats))
        assert set(metrics) == {
            f"{metric}/{stats.key}"
            for metric in ("sqnr", "underflow_rate")
            for stats in folded_stats
        }


# 168 rows over four experts: one empty, one shorter than a rotation block, and one
# that is not a block multiple -- the cases segment padding has to align.
ROTATION_COUNTS = [128, 0, 7, 33]
ROTATION_CONFIGS = [
    {"rotation_cls": "hadamard", "rotation_kwargs": {"block_size": 16, "seed": 0}},
    {"rotation_cls": "hadamard", "rotation_kwargs": {"block_size": 32, "seed": 0}},
]
ROTATION_FORMATS = ["fp8_e4m3", "int8"]
ROTATION_SCALES = [
    ROWWISE,
    BLOCKWISE1D_16,
    BLOCKWISE2D_16,
    BLOCKWISE1D_32_E8M0,
    BLOCKWISE2D_32_E8M0,
]
ROTATION_SCALE_PAIRS = scale_combinations(ROTATION_SCALES, 2)
# Worst over the valid layout grid is 0.0544553, giving a 3.3x margin. An
# expert-boundary cancellation failure remains O(1).
ROTATION_PRECISION_BOUND = 0.18


@cuda_sm89_or_newer
@pytest.mark.parametrize("fmt", ROTATION_FORMATS)
@pytest.mark.parametrize("a_scale,b_scale", ROTATION_SCALE_PAIRS)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("rotation_cfg", ROTATION_CONFIGS)
def test_quantized_grouped_mm_rotation_precision(
    fmt, a_scale, b_scale, layout, rotation_cfg
):
    """Rotated and unrotated quantized GEMMs must agree for every ragged layout."""
    skip_unsupported_fmt_scale(fmt, a_scale)
    skip_unsupported_fmt_scale(fmt, b_scale)
    if layout == "ragged_k":
        skip_unsupported_ragged_k_scale(a_scale)
        skip_unsupported_ragged_k_scale(b_scale)
    a, b, offs = _make(ROTATION_COUNTS, K=64, N=48)
    rotation = build_rotation(rotation_cfg)
    a_rotation_axes = (-1,)
    b_rotation_axes = (-2,)

    if layout == "ragged_m":
        # Ragged-M: (R,K) x (E,K,N) -> (R,N)
        out = quantized_grouped_mm(
            apply_rotation_on_axes(a, rotation, a_rotation_axes, out_dtype=a.dtype),
            apply_rotation_on_axes(b, rotation, b_rotation_axes, out_dtype=b.dtype),
            offs,
            fmt,
            fmt,
            a.dtype,
            a_scale,
            b_scale,
            rotation=rotation,
            a_rotation_axes=a_rotation_axes,
            b_rotation_axes=b_rotation_axes,
        )
        ref = quantized_grouped_mm(a, b, offs, fmt, fmt, a.dtype, a_scale, b_scale)
    elif layout == "ragged_n":
        # Ragged-N: (E,M,K) x (K,R) -> (M,R)
        slabs = torch.randn(offs.shape[0], 32, 64, device="cuda", dtype=torch.bfloat16)
        cols = torch.randn(64, a.shape[0], device="cuda", dtype=torch.bfloat16) * 0.1
        out = quantized_grouped_mm(
            apply_rotation_on_axes(
                slabs, rotation, a_rotation_axes, out_dtype=slabs.dtype
            ),
            apply_rotation_on_axes(
                cols, rotation, b_rotation_axes, out_dtype=cols.dtype
            ),
            offs,
            fmt,
            fmt,
            slabs.dtype,
            a_scale,
            b_scale,
            rotation=rotation,
            a_rotation_axes=a_rotation_axes,
            b_rotation_axes=b_rotation_axes,
        )
        ref = quantized_grouped_mm(
            slabs,
            cols,
            offs,
            fmt,
            fmt,
            slabs.dtype,
            a_scale,
            b_scale,
        )
    else:
        # Ragged-K: (K,R) x (R,N) -> (E,K,N)
        gy = torch.randn(a.shape[0], 48, device="cuda", dtype=torch.bfloat16)
        padded_a, padded_gy, padded_counts = [], [], []
        start = 0
        for stop in offs.tolist():
            padding = (-(stop - start)) % rotation.alignment
            padded_a.append(F.pad(a[start:stop], (0, 0, 0, padding)))
            padded_gy.append(F.pad(gy[start:stop], (0, 0, 0, padding)))
            padded_counts.append(stop - start + padding)
            start = stop
        padded_a = torch.cat(padded_a)
        padded_gy = torch.cat(padded_gy)
        padded_offs = torch.tensor(
            padded_counts, device=offs.device, dtype=offs.dtype
        ).cumsum(0, dtype=offs.dtype)
        out = quantized_grouped_mm(
            apply_rotation_on_axes(
                padded_a.mT, rotation, a_rotation_axes, out_dtype=a.dtype
            ),
            apply_rotation_on_axes(
                padded_gy, rotation, b_rotation_axes, out_dtype=a.dtype
            ),
            padded_offs,
            fmt,
            fmt,
            a.dtype,
            a_scale,
            b_scale,
            rotation=rotation,
            a_rotation_axes=a_rotation_axes,
            b_rotation_axes=b_rotation_axes,
        )
        ref = quantized_grouped_mm(
            padded_a.mT,
            padded_gy,
            padded_offs,
            fmt,
            fmt,
            a.dtype,
            a_scale,
            b_scale,
        )

    assert rel(out, ref) < ROTATION_PRECISION_BOUND, rel(out, ref)


# --- ScaledGroupedGemmFn ---

COMPILE_DTYPES = [
    FP8_E4M3_W8A8_E5M2_G8_DTYPES,
    FP4_E2M1_W4A4G4_DTYPES,
    FP4_E2M1_4OVER6_W4A4G4_DTYPES,
]
COMPILE_SCALES = [ROWWISE, BLOCKWISE1D_16_E2M1, BLOCKWISE2D_16_E2M1]
COMPILE_SCALE_TRIPLES = scale_combinations(COMPILE_SCALES, 3)
COMPILE_COUNTS = [COUNTS, [0, 7, 9, 0]]
COMPILE_ROTATIONS = [None, True]
# Compiler arithmetic can cross quantization thresholds; worst relative norm
# across this grid is 0.01550, with a 3.55x margin.
COMPILE_REL_BOUND = 0.055

INPUT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
FORWARD_PRECISION_ROTATIONS = [
    None,
    {
        "rotation_cls": "hadamard",
        "rotation_kwargs": {"block_size": 16},
        "rotation_axes": {"act": {"fwd": [-1]}},
    },
    {
        "rotation_cls": "hadamard",
        "rotation_kwargs": {"block_size": 16},
        "rotation_axes": {"weight": {"fwd": [-1]}},
    },
]
BACKWARD_PRECISION_ROTATIONS = [
    None,
    {
        "rotation_cls": "hadamard",
        "rotation_kwargs": {"block_size": 16},
        "rotation_axes": {"grad_out": {"dgrad": [-1]}},
    },
    {
        "rotation_cls": "hadamard",
        "rotation_kwargs": {"block_size": 16},
        "rotation_axes": {"weight": {"dgrad": [-2]}},
    },
]


@cuda_sm89_or_newer
@pytest.mark.parametrize("dtype", COMPILE_DTYPES)
@pytest.mark.parametrize("act_scale,weight_scale,grad_out_scale", COMPILE_SCALE_TRIPLES)
@pytest.mark.parametrize("counts", COMPILE_COUNTS)
@pytest.mark.parametrize("with_rotation", COMPILE_ROTATIONS)
def test_scaled_grouped_gemm_fn_compiles_fullgraph(
    with_rotation,
    counts,
    dtype,
    act_scale,
    weight_scale,
    grad_out_scale,
):
    """Check fullgraph forward/backward and zero gradients for empty experts."""
    if uses_fp4_gemm(
        dtype["act"], dtype["weight"], act_scale
    ) and not cuda_capability_at_least((10, 0)):
        pytest.skip("fused FP4 requires CUDA SM100 or newer")
    a, b, offs = _make(counts, K=64, N=48)
    weight = b.mT
    cfg = _cfg(_per_tensor_scale(act_scale, weight_scale, grad_out_scale), dtype)
    rotation = None
    if with_rotation:
        rotation_cfg = {
            "rotation_cls": "hadamard",
            "rotation_kwargs": {"block_size": 16, "seed": 0},
            "rotation_axes": {
                "weight": {"fwd": [-1], "dgrad": [-1]},
                "act": {"fwd": [-1], "wgrad": []},
                "grad_out": {"dgrad": [], "wgrad": []},
            },
        }
        cfg.rotation = rotation_cfg
        rotation = build_rotation(rotation_cfg).cuda()

    def fwd(a, weight):
        return _expert_mm(cfg, a, weight, offs, rotation=rotation)

    def run(fn):
        a_ = a.clone().requires_grad_(True)
        weight_ = weight.clone().requires_grad_(True)
        y = fn(a_, weight_)
        y.sum().backward()
        return y, a_.grad, weight_.grad

    eager = run(fwd)
    torch.compiler.reset()
    try:
        compiled = run(torch.compile(fwd, fullgraph=True))
    finally:
        torch.compiler.reset()
    for got, ref in zip(compiled, eager):
        assert got.shape == ref.shape and got.dtype == ref.dtype
        if dtype == FP8_E4M3_W8A8_E5M2_G8_DTYPES and act_scale == ROWWISE:
            torch.testing.assert_close(got, ref, atol=0, rtol=0)
        else:
            assert rel(got, ref) < COMPILE_REL_BOUND
    for group, count in enumerate(counts):
        if count == 0:
            assert torch.count_nonzero(compiled[2][group]) == 0


@cuda_sm89_or_newer
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("act_scale,weight_scale", SCALE_PAIRS)
@pytest.mark.parametrize("dtype", FORWARD_DTYPES)
@pytest.mark.parametrize("counts", GROUPED_COUNTS)
@pytest.mark.parametrize("rotation_config", FORWARD_PRECISION_ROTATIONS)
@pytest.mark.parametrize("input_dtype", INPUT_DTYPES)
def test_scaled_grouped_gemm_fn_forward_precision(
    dtype, act_scale, weight_scale, bias, counts, rotation_config, input_dtype
):
    """Check forward precision across formats, scales, bias, and empty experts."""
    if rotation_config is None and input_dtype is not torch.bfloat16:
        pytest.skip("unrotated precision already uses bfloat16 inputs")
    skip_unsupported_dtype_scale(dtype, act_scale)
    skip_unsupported_dtype_scale(dtype, weight_scale)
    if uses_fp4_gemm(
        operand_fmt(dtype, "act", "fwd"), operand_fmt(dtype, "weight", "fwd"), act_scale
    ) and not cuda_capability_at_least((10, 0)):
        pytest.skip("fused FP4 requires CUDA SM100 or newer")
    a, b, offs = _make(counts, K=64, N=48)
    a, b = a.to(input_dtype), b.to(input_dtype)
    weight = b.mT
    bias0 = torch.randn(offs.shape[0], 48, device="cuda", dtype=input_dtype) * 0.1
    cfg = rule(
        dtype,
        _per_tensor_scale(act_scale, weight_scale, act_scale),
        rotation=rotation_config,
    )
    rotation = build_rotation(cfg.rotation)
    y = _expert_mm(
        cfg,
        a,
        weight,
        offs,
        bias=bias0.clone() if bias else None,
        rotation=rotation,
    )

    if rotation is not None:
        act_axes = cfg.rotation["rotation_axes"]["act"]["fwd"]
        weight_axes = cfg.rotation["rotation_axes"]["weight"]["fwd"]
        rotated_a = apply_rotation_on_axes(a, rotation, act_axes)
        rotated_weight = apply_rotation_on_axes(weight, rotation, weight_axes)
        fwd_a = _oracle_qdq(
            rotated_a,
            -1,
            operand_fmt(dtype, "act", "fwd"),
            act_scale,
            offs,
            -2,
        )
        fwd_weight = _oracle_qdq(
            rotated_weight.mT,
            -2,
            operand_fmt(dtype, "weight", "fwd"),
            weight_scale,
        )
        fwd_a = _undo_oracle_rotations(fwd_a, rotation, act_axes)
        fwd_weight = _undo_oracle_rotations(
            fwd_weight, rotation, tuple(-3 - axis for axis in weight_axes)
        )
        y_ref = grouped_mm(fwd_a, fwd_weight, offs, bias=bias0 if bias else None)
        torch.testing.assert_close(y, y_ref, atol=0, rtol=0)
        return

    y_ref = torch.empty_like(y)
    lo = 0
    for group, hi in enumerate(offs.tolist()):
        if hi > lo:
            fwd = mm_ref(
                a[lo:hi],
                b[group],
                operand_fmt(dtype, "act", "fwd"),
                operand_fmt(dtype, "weight", "fwd"),
                act_scale,
                weight_scale,
            )
            if bias:
                fwd = fwd + bias0[group].to(fwd.dtype)
            y_ref[lo:hi] = fwd.to(y.dtype)
        lo = hi
    assert rel(y, y_ref) < PRECISION_BOUND


@cuda_sm89_or_newer
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("act_scale,weight_scale,grad_out_scale", GROUPED_SCALE_TRIPLES)
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
@pytest.mark.parametrize("counts", GROUPED_COUNTS)
@pytest.mark.parametrize("rotation_config", BACKWARD_PRECISION_ROTATIONS)
def test_scaled_grouped_gemm_fn_backward_precision(
    dtype,
    act_scale,
    weight_scale,
    grad_out_scale,
    bias,
    counts,
    rotation_config,
):
    """Check backward precision and fp32 bias-gradient accumulation."""
    skip_unsupported_dtype_scale(dtype, act_scale)
    skip_unsupported_dtype_scale(dtype, weight_scale)
    skip_unsupported_dtype_scale(dtype, grad_out_scale)
    fwd_uses_fused_fp4 = uses_fp4_gemm(
        operand_fmt(dtype, "act", "fwd"), operand_fmt(dtype, "weight", "fwd"), act_scale
    )
    dgrad_uses_fused_fp4 = uses_fp4_gemm(
        operand_fmt(dtype, "grad_out", "dgrad"),
        operand_fmt(dtype, "weight", "dgrad"),
        grad_out_scale,
    )
    wgrad_uses_fused_fp4 = uses_fp4_gemm(
        operand_fmt(dtype, "act", "wgrad"),
        operand_fmt(dtype, "grad_out", "wgrad"),
        act_scale,
    )
    if (
        fwd_uses_fused_fp4 or dgrad_uses_fused_fp4 or wgrad_uses_fused_fp4
    ) and not cuda_capability_at_least((10, 0)):
        pytest.skip("fused FP4 requires CUDA SM100 or newer")
    wgrad_uses_fp4 = is_fp4(operand_fmt(dtype, "act", "wgrad")) or is_fp4(
        operand_fmt(dtype, "grad_out", "wgrad")
    )
    # The wgrad of a grouped GEMM contracts over the ragged axis.
    skip_unsupported_ragged_k_scale(act_scale)
    skip_unsupported_ragged_k_scale(grad_out_scale)
    a, b, offs = _make(counts, K=64, N=48)
    weight = b.mT
    bias0 = torch.randn(offs.shape[0], 48, device="cuda", dtype=torch.bfloat16) * 0.1
    a_q = a.clone().requires_grad_(True)
    weight_q = weight.clone().requires_grad_(True)
    bias_q = bias0.clone().requires_grad_(True) if bias else None
    cfg = rule(
        dtype,
        _per_tensor_scale(act_scale, weight_scale, grad_out_scale),
        rotation=rotation_config,
    )
    rotation = build_rotation(cfg.rotation)
    y = _expert_mm(
        cfg,
        a_q,
        weight_q,
        offs,
        bias=bias_q,
        rotation=rotation,
    )
    gy = torch.randn_like(y)
    y.backward(gy)

    ga_ref, gb_ref = torch.empty_like(a), torch.zeros_like(b)
    if rotation is not None:
        grad_out_axes = cfg.rotation["rotation_axes"]["grad_out"]["dgrad"]
        weight_axes = cfg.rotation["rotation_axes"]["weight"]["dgrad"]
        dgrad_y = _oracle_qdq(
            apply_rotation_on_axes(gy, rotation, grad_out_axes),
            -1,
            operand_fmt(dtype, "grad_out", "dgrad"),
            grad_out_scale,
            offs,
            -2,
        )
        dgrad_weight = _oracle_qdq(
            apply_rotation_on_axes(weight, rotation, weight_axes),
            -2,
            operand_fmt(dtype, "weight", "dgrad"),
            weight_scale,
        )
        ga_ref = grouped_mm(
            _undo_oracle_rotations(dgrad_y, rotation, grad_out_axes),
            _undo_oracle_rotations(dgrad_weight, rotation, weight_axes),
            offs,
        )
    lo = 0
    for group, hi in enumerate(offs.tolist()):
        if hi > lo:
            if rotation is None:
                ga_ref[lo:hi] = mm_ref(
                    gy[lo:hi],
                    b[group].t().contiguous(),
                    operand_fmt(dtype, "grad_out", "dgrad"),
                    operand_fmt(dtype, "weight", "dgrad"),
                    grad_out_scale,
                    weight_scale,
                ).to(y.dtype)
            padding = (-(hi - lo)) % 16 if wgrad_uses_fp4 else 0
            gb_ref[group] = mm_ref(
                F.pad(a[lo:hi].t(), (0, padding)),
                F.pad(gy[lo:hi], (0, 0, 0, padding)),
                operand_fmt(dtype, "act", "wgrad"),
                operand_fmt(dtype, "grad_out", "wgrad"),
                act_scale,
                grad_out_scale,
            ).to(y.dtype)
        lo = hi

    if rotation is not None:
        torch.testing.assert_close(a_q.grad, ga_ref, atol=0, rtol=0)
    expected = [("grad_weight", weight_q.grad, gb_ref.mT)]
    if rotation is None:
        expected.append(("grad_a", a_q.grad, ga_ref))
    if bias:
        rows = torch.arange(y.shape[0], device=offs.device)
        acc = torch.zeros_like(bias0, dtype=torch.float32)
        acc.index_add_(0, torch.searchsorted(offs, rows, right=True), gy.float())
        expected.append(("grad_bias", bias_q.grad, acc.to(bias0.dtype)))
    for name, got, ref in expected:
        assert rel(got, ref) < PRECISION_BOUND, (name, rel(got, ref))


ORACLE_COUNTS = [8, 0, 8, 0]
UNALIGNED_ROW_ROTATION_AXES = [
    {"act": {"fwd": [-2]}},
    {"act": {"wgrad": [-2]}},
    {"grad_out": {"dgrad": [-2]}},
    {"grad_out": {"wgrad": [-2]}},
]
UNALIGNED_ROW_CASES = [([15, 15], ValueError), ([15, 17], AssertionError)]
# Worst relative errors are 1.36e-7, 6.40e-4, and 5.77e-3 respectively.
ORACLE_BOUNDS = {
    torch.float32: 5.7e-7,
    torch.float16: 0.0043,
    torch.bfloat16: 0.031,
}


@cuda_sm89_or_newer
@pytest.mark.parametrize("rotation_axes", UNALIGNED_ROW_ROTATION_AXES)
@pytest.mark.parametrize("counts,error", UNALIGNED_ROW_CASES)
def test_scaled_grouped_gemm_fn_rotation_axes_raise_error(rotation_axes, counts, error):
    rotation_config = {
        "rotation_cls": "hadamard",
        "rotation_kwargs": {"block_size": 16},
        "rotation_axes": rotation_axes,
    }
    cfg = rule(INT4_W8A16_DTYPES, rotation=rotation_config)
    rotation = build_rotation(rotation_config)
    a, b, offs = _make(counts, K=32, N=32)
    a.requires_grad_()
    weight = b.mT.detach().clone().requires_grad_()

    with pytest.raises(error):
        _expert_mm(cfg, a, weight, offs, rotation=rotation).sum().backward()


def _apply_oracle_rotations(tensor, rotation, axes):
    for axis in axes:
        tensor = rotation(tensor, axis, tensor.dtype)
    return tensor


def _undo_oracle_rotations(tensor, rotation, axes):
    for axis in reversed(axes):
        tensor = rotation.inverse(tensor, axis, tensor.dtype)
    return tensor


def _oracle_qdq(tensor, axis, fmt, scale, offs=None, ragged_dim=None):
    if not is_quantized(fmt):
        return tensor
    quantized, scales, global_scale, _ = quantize_operand(
        tensor,
        axis,
        fmt,
        scale,
        offs=offs,
        ragged_dim=ragged_dim,
    )
    return dequantize_operand(
        quantized,
        scales,
        axis,
        scale,
        offs=offs,
        ragged_dim=ragged_dim,
        global_scale=global_scale,
    ).to(tensor.dtype)


def _oracle_grouped_mm(a, b, offs, bias=None):
    output = a.new_zeros(a.shape[0], b.shape[-1])
    start = 0
    for group, stop in enumerate(offs.tolist()):
        output[start:stop] = a[start:stop] @ b[group]
        if bias is not None:
            output[start:stop] += bias[group]
        start = stop
    return output


@cuda_sm89_or_newer
@pytest.mark.parametrize("input_dtype", INPUT_DTYPES)
def test_scaled_grouped_gemm_fn_rotation_axes_precision(input_dtype):
    """Compare ragged tensor rotations with an independent QDQ-restoration oracle."""
    rotation_cfg = {
        "rotation_cls": "hadamard",
        "rotation_kwargs": {"block_size": 4, "random_sign": True, "seed": 17},
        "rotation_axes": {
            "weight": {"fwd": [-2, -1], "dgrad": [-2, -1]},
            "act": {"fwd": [-2, -1], "wgrad": [-2, -1]},
            "grad_out": {"dgrad": [-2, -1], "wgrad": [-2, -1]},
        },
    }
    cfg = _cfg(TENSORWISE, FP8_E4M3_W8A8_E5M2_G8_DTYPES)
    cfg.rotation = rotation_cfg
    a, b, offs = _make(ORACLE_COUNTS, K=8, N=8, seed=17)
    a = a.to(input_dtype).requires_grad_()
    weight = b.mT.to(input_dtype).detach().clone().requires_grad_()
    bias = torch.randn(4, 8, device="cuda", dtype=input_dtype).requires_grad_()
    rotation = build_rotation(rotation_cfg).cuda()
    out = ScaledGroupedGemmFn.apply(a, weight, bias, offs, cfg, {}, rotation)
    grad_y = torch.randn_like(out)
    out.backward(grad_y)

    act_fwd_axes = [-2, -1]
    weight_fwd_axes = [-2, -1]
    grad_out_dgrad_axes = [-2, -1]
    weight_dgrad_axes = [-2, -1]
    act_wgrad_axes = [-2, -1]
    grad_out_wgrad_axes = [-2, -1]
    act_scale = resolve_scale(cfg.scale, "act")
    weight_scale = resolve_scale(cfg.scale, "weight")
    grad_out_scale = resolve_scale(cfg.scale, "grad_out")

    rotated_a = _apply_oracle_rotations(a.detach(), rotation, act_fwd_axes)
    rotated_weight = _apply_oracle_rotations(weight.detach(), rotation, weight_fwd_axes)
    fwd_a = _oracle_qdq(
        rotated_a,
        -1,
        cfg.dtype["act"]["fwd"],
        act_scale,
        offs,
        -2,
    )
    fwd_weight = _oracle_qdq(
        rotated_weight.mT,
        -2,
        cfg.dtype["weight"]["fwd"],
        weight_scale,
    ).mT
    out_ref = _oracle_grouped_mm(
        _undo_oracle_rotations(fwd_a, rotation, act_fwd_axes),
        _undo_oracle_rotations(fwd_weight, rotation, weight_fwd_axes).mT,
        offs,
        bias.detach(),
    )

    rotated_grad_y = _apply_oracle_rotations(grad_y, rotation, grad_out_dgrad_axes)
    rotated_weight_dgrad = _apply_oracle_rotations(
        weight.detach(), rotation, weight_dgrad_axes
    )
    dgrad_y = _oracle_qdq(
        rotated_grad_y,
        -1,
        cfg.dtype["grad_out"]["dgrad"],
        grad_out_scale,
        offs,
        -2,
    )
    dgrad_weight = _oracle_qdq(
        rotated_weight_dgrad,
        -2,
        cfg.dtype["weight"]["dgrad"],
        weight_scale,
    )
    grad_a_ref = _oracle_grouped_mm(
        _undo_oracle_rotations(dgrad_y, rotation, grad_out_dgrad_axes),
        _undo_oracle_rotations(dgrad_weight, rotation, weight_dgrad_axes),
        offs,
    )

    wgrad_a = _oracle_qdq(
        _apply_oracle_rotations(a.detach(), rotation, act_wgrad_axes).mT,
        -1,
        cfg.dtype["act"]["wgrad"],
        act_scale,
        offs,
        -1,
    ).mT
    rotated_grad_y_wgrad = _apply_oracle_rotations(
        grad_y, rotation, grad_out_wgrad_axes
    )
    wgrad_y = _oracle_qdq(
        rotated_grad_y_wgrad,
        -2,
        cfg.dtype["grad_out"]["wgrad"],
        grad_out_scale,
        offs,
        -2,
    )
    restored_a = _undo_oracle_rotations(wgrad_a, rotation, act_wgrad_axes)
    restored_grad_y = _undo_oracle_rotations(wgrad_y, rotation, grad_out_wgrad_axes)
    grad_b_ref = torch.stack(
        [
            restored_a[start:stop].mT @ restored_grad_y[start:stop]
            for start, stop in zip([0, *offs[:-1].tolist()], offs.tolist())
        ]
    )
    grad_bias_ref = torch.zeros_like(bias, dtype=torch.float32)
    grad_bias_ref.index_add_(
        0,
        torch.searchsorted(
            offs, torch.arange(grad_y.shape[0], device="cuda"), right=True
        ),
        grad_y.float(),
    )

    for name, actual, reference in (
        ("output", out, out_ref),
        ("grad_a", a.grad, grad_a_ref),
        ("grad_weight", weight.grad, grad_b_ref.mT),
        ("grad_bias", bias.grad, grad_bias_ref.to(input_dtype)),
    ):
        error = rel(actual, reference)
        assert error < ORACLE_BOUNDS[input_dtype], (name, error)


@cuda_sm89_or_newer
def test_scaled_grouped_gemm_fn_bias_grad_precision():
    rows = 257
    a = torch.ones(rows, 1, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(1, 1, 1, device="cuda", dtype=torch.bfloat16).requires_grad_()
    bias = torch.zeros(1, 1, device="cuda", requires_grad=True)
    offs = torch.tensor([rows], device="cuda", dtype=torch.int32)
    dtype = {"weight": "bf16", "act": "bf16", "grad_out": "bf16"}

    out = _expert_mm(_cfg(dtype=dtype), a, weight, offs, bias=bias)
    out.backward(torch.ones_like(out))

    assert bias.grad.dtype is torch.float32
    torch.testing.assert_close(bias.grad, torch.full_like(bias, rows), atol=0, rtol=0)


AUTOCAST_DTYPES = [None, torch.float16, torch.bfloat16]
AUTOCAST_INPUT_DTYPES = [torch.float32, torch.bfloat16]
AUTOCAST_QUANTIZATION_DTYPES = [
    FP8_E4M3_W8A8_E5M2_G8_DTYPES,
    INT4_W8A16_DTYPES,
]
AUTOCAST_ROTATIONS = [
    None,
    {
        "rotation_cls": "hadamard",
        "rotation_kwargs": {"block_size": 16, "seed": 0},
        "rotation_axes": {
            "weight": {"fwd": [-1], "dgrad": [-1]},
            "act": {"fwd": [-1], "wgrad": []},
            "grad_out": {"dgrad": [], "wgrad": []},
        },
    },
]


@cuda_sm89_or_newer
@pytest.mark.parametrize("autocast_dtype", AUTOCAST_DTYPES)
@pytest.mark.parametrize("input_dtype", AUTOCAST_INPUT_DTYPES)
@pytest.mark.parametrize("quantization_dtype", AUTOCAST_QUANTIZATION_DTYPES)
@pytest.mark.parametrize("rotation_config", AUTOCAST_ROTATIONS)
def test_scaled_grouped_gemm_fn_autocast_precision(
    autocast_dtype, input_dtype, quantization_dtype, rotation_config
):
    torch.manual_seed(0)
    a, b, offs = _make([8, 0, 12], K=32, N=16)
    x = a.to(input_dtype).detach().clone().requires_grad_()
    weight = b.mT.float().detach().clone().requires_grad_()
    bias = torch.randn(3, 16, device="cuda", requires_grad=True)
    x_before = x.detach().clone()
    weight_before = weight.detach().clone()
    bias_before = bias.detach().clone()
    cfg = rule(quantization_dtype, ROWWISE, rotation=rotation_config)
    rotation = build_rotation(cfg.rotation)
    compute_dtype = input_dtype if autocast_dtype is None else autocast_dtype

    with torch.amp.autocast(
        "cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None
    ):
        output = ScaledGroupedGemmFn.apply(x, weight, bias, offs, cfg, {}, rotation)
    grad_output = torch.randint(-4, 5, output.shape, device=output.device).to(
        compute_dtype
    )
    output.backward(grad_output)

    ref_x = x.detach().to(compute_dtype).requires_grad_()
    ref_weight = weight.detach().to(compute_dtype).requires_grad_()
    ref_bias = bias.detach().clone().requires_grad_()
    reference = ScaledGroupedGemmFn.apply(
        ref_x, ref_weight, ref_bias, offs, cfg, {}, rotation
    )
    reference.backward(grad_output)

    assert output.dtype is compute_dtype
    torch.testing.assert_close(output, reference, atol=0, rtol=0)
    for actual, expected, master in (
        (x.grad, ref_x.grad, x),
        (weight.grad, ref_weight.grad, weight),
        (bias.grad, ref_bias.grad, bias),
    ):
        assert actual.dtype is master.dtype
        torch.testing.assert_close(actual, expected.to(actual.dtype), atol=0, rtol=0)
        assert torch.isfinite(actual).all()
    assert torch.equal(x, x_before)
    assert torch.equal(weight, weight_before)
    assert torch.equal(bias, bias_before)


# --- QuantizedSparseMoEBlock ---


ENABLED_AFTER_STEPS = [0, 2]
QUANTIZATION_ENABLED = [False, True]


@pytest.mark.parametrize("enabled_after_steps", ENABLED_AFTER_STEPS)
@pytest.mark.parametrize("enabled", QUANTIZATION_ENABLED)
def test_quantized_sparse_moe_block_only_quantizes_during_training(
    enabled_after_steps, enabled
):
    """Check quantized training and exact unquantized evaluation."""
    torch.manual_seed(0)
    source = SparseMoEBlock(
        d_model=4,
        intermediate_size=8,
        n_routed_experts=2,
        n_routed_experts_per_token=1,
        aux_loss=False,
    )
    cfg = TrainConfig(
        training=TrainingConfig(mixed_precision="no"),
        quantization=QuantizationConfig(
            **{
                "enabled": True,
                "enabled_after_steps": enabled_after_steps,
                "dtype": INT4_W8A16_DTYPES,
            }
        ),
    ).quantization
    block = QuantizedSparseMoEBlock.from_module(source, cfg)
    args = (torch.randn(4, 4), torch.randn(2, 8, 4), torch.tensor([2, 4]))
    plain = SparseMoEBlock.expert_mm(block, *args, projection="gate")

    block.train()
    if enabled:
        enable_quantization(block)
    assert block.quantization_enabled is enabled
    out = block.expert_mm(*args, projection="gate")
    if enabled:
        assert not torch.equal(out, plain)
    else:
        assert torch.equal(out, plain)
    block.eval()
    assert torch.equal(block.expert_mm(*args, projection="gate"), plain)


def test_quantized_sparse_moe_block_rotation_axes():
    source = SparseMoEBlock(
        d_model=4,
        intermediate_size=8,
        n_routed_experts=2,
        n_routed_experts_per_token=1,
        aux_loss=False,
    )
    cfg = QuantizationConfig(
        enabled=True,
        dtype=INT4_W8A16_DTYPES,
        rotation={
            "rotation_cls": "hadamard",
            "rotation_kwargs": {"block_size": 4},
            "rotation_axes": {"act": {"fwd": [-1]}},
        },
    )
    rotation = build_rotation(cfg.rotation)
    block = QuantizedSparseMoEBlock.from_module(source, cfg, rotation)

    assert (
        block.quantization_config.rotation["rotation_axes"]
        == cfg.rotation["rotation_axes"]
    )
    assert block.rotation is rotation


@cuda_sm89_or_newer
def test_quantized_sparse_moe_block_autocast():
    """Check autocast GEMMs, caller-dtype outputs, and fp32-master gradients."""
    torch.manual_seed(0)
    source = SparseMoEBlock(
        d_model=32,
        intermediate_size=48,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
        aux_loss=True,
        aux_loss_coef=1e-3,
    ).cuda()  # Keep the master weights in fp32.
    with torch.no_grad():  # Initialize the expert weights for a nonzero signal.
        nn.init.normal_(source.expert_gate, std=0.02)
        nn.init.normal_(source.expert_up, std=0.02)
        nn.init.normal_(source.expert_down, std=0.02)
    block = QuantizedSparseMoEBlock.from_module(
        source, rule(FP8_E4M3_W8A8_E5M2_G8_DTYPES, ROWWISE)
    )
    enable_quantization(block)
    x = torch.randn(2, 8, 32, device="cuda", dtype=torch.float32, requires_grad=True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out, _ = block(x)
    assert out.dtype == torch.float32  # The block preserves the caller's dtype.
    out.square().mean().backward()
    assert block.expert_gate.grad.dtype == torch.float32  # Matches the fp32 master.
    assert torch.isfinite(block.expert_gate.grad).all()
    assert torch.isfinite(block.expert_up.grad).all()
    assert torch.isfinite(block.expert_down.grad).all()
    assert torch.isfinite(x.grad).all()


# Feature rotations support arbitrary per-expert token counts.
E2E_QUANTIZATION = [
    {
        "dtype": {
            "weight": "fp8_e4m3",
            "act": "fp8_e4m3",
            "grad_out": "fp8_e5m2",
        },
        "scale": {
            "weight": {"granularity": "rowwise", "block_shape": (1, 0)},
            "act": {"granularity": "rowwise", "block_shape": (1, 0)},
            "grad_out": {"granularity": "rowwise", "block_shape": (1, 0)},
            "scale_dtype": "fp32",
            "enable_global_scale": False,
        },
    },
    {
        "dtype": {
            "weight": "fp4_e2m1",
            "act": "fp4_e2m1",
            "grad_out": "fp4_e2m1",
        },
        "scale": {
            "weight": {"granularity": "blockwise", "block_shape": [16, 16]},
            "act": {"granularity": "blockwise", "block_shape": [1, 16]},
            "grad_out": {"granularity": "blockwise", "block_shape": [1, 16]},
            "scale_dtype": "fp8_e4m3",
            "enable_global_scale": True,
        },
    },
]
E2E_ROTATIONS = [
    {"rotation_cls": None},
    {
        "rotation_cls": "hadamard",
        "rotation_kwargs": {"block_size": 16},
        "rotation_axes": {
            "act": {"wgrad": [-1]},
            "grad_out": {"wgrad": [-1]},
        },
    },
    {
        "rotation_cls": "hadamard",
        "rotation_kwargs": {"block_size": 16},
        "rotation_axes": {
            "weight": {"fwd": [-1], "dgrad": [-1]},
            "act": {"fwd": [-1], "wgrad": [-1]},
            "grad_out": {"wgrad": [-1]},
        },
    },
]


@cuda_sm89_or_newer
@pytest.mark.parametrize("quantization", E2E_QUANTIZATION)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("rotation", E2E_ROTATIONS)
def test_quantized_sparse_moe_block_trains_a_full_model(bias, rotation, quantization):
    """Check one converted MoE step and fused per-expert bias gradients."""
    if is_fp4(quantization["dtype"]["weight"]) and not cuda_capability_at_least(
        (10, 0)
    ):
        pytest.skip("fused FP4 requires CUDA SM100 or newer")
    config = TrainConfig(
        max_seq_len=64,
        model=ModelConfig(
            d_model=64,
            n_layers=2,
            vocab_size=128,
            attn=[{"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4, "n_kv_heads": 2}}],
            mlp=[
                {
                    "mlp_cls": "moe",
                    "mlp_kwargs": {
                        "intermediate_size": 48,
                        "n_routed_experts": 4,
                        "n_routed_experts_per_token": 2,
                        "bias": bias,
                        "aux_loss": True,
                        "aux_loss_coef": 1e-3,
                        "activation_cls": "swiglu",
                    },
                }
            ],
            norm_cls="rmsnorm",
            pos_emb_cls="rope",
        ),
        training=TrainingConfig(mixed_precision="bf16"),
        quantization=QuantizationConfig(
            **{
                "enabled": True,
                **copy.deepcopy(quantization),
                "rotation": rotation,
            }
        ),
    )
    model = build_model(config)
    apply_quantization(model, config)
    enable_quantization(model)
    model.cuda().to(torch.bfloat16)
    blocks = [m for m in model.modules() if isinstance(m, QuantizedSparseMoEBlock)]
    assert blocks  # Conversion installed the quantized seam.
    # Pin the rotation to the block: a config that silently resolves to None would
    # leave every assertion below passing without a rotation ever running.
    for block in blocks:
        assert (block.rotation is not None) == (rotation["rotation_cls"] is not None)

    ids = torch.randint(0, 128, (2, 64), device="cuda")
    position_ids = torch.arange(64, device="cuda").unsqueeze(0).expand(2, 64)
    attn_masks = [None] * config.model.n_layers
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(ids, position_ids, attn_masks)
        loss = logits.float().log_softmax(-1).mean().neg()
    loss.backward()

    assert torch.isfinite(loss).item()
    for block in blocks:
        for name, p in block.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert p.grad.abs().sum() > 0, name
