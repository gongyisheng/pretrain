import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.mlp import SparseMoEBlock
from src.metrics.quant import QuantizationStats, set_quantization_monitoring_status
from src.model import build_model
from src.quant.convert import apply_quantization
from src.quant.moe import (
    QuantizedSparseMoEBlock,
    ScaledGroupedGemmFn,
    quantized_grouped_mm,
)
from src.quant.quantize import dequantize_operand, quantize_operand
from src.quant.utils import is_fp4
from src.quant.constants import GEMM_OPS
from src.quant.rotation import build_rotation
from src.utils.config import (
    ModelConfig,
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
        **act_scale,
        "block_shape": {
            "act": act_scale["block_shape"],
            "weight": weight_scale["block_shape"],
            "grad_out": grad_out_scale["block_shape"],
        },
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
GROUPED_STATS_LAYOUTS = ["ragged_m", "ragged_k"]
GROUPED_STATS_DEVICES = ["cpu", "cuda"]


@pytest.mark.parametrize("device", GROUPED_STATS_DEVICES)
@pytest.mark.parametrize("layout", GROUPED_STATS_LAYOUTS)
@pytest.mark.parametrize("config", GROUPED_STATS_CONFIGS)
@pytest.mark.parametrize("a_scale,b_scale", SCALE_PAIRS)
def test_quantized_grouped_mm_records_stats(device, layout, config, a_scale, b_scale):
    a_fmt, b_fmt, with_stats, a_folded, b_folded = config
    skip_unsupported_fmt_scale(a_fmt, a_scale)
    skip_unsupported_fmt_scale(b_fmt, b_scale)
    nvfp4 = is_fp4(a_fmt)
    if nvfp4 and layout != "ragged_m":
        pytest.skip("NVFP4 statistics coverage uses ragged-M")
    if not nvfp4 and device == "cpu":
        pytest.skip("CPU statistics coverage is limited to NVFP4")
    capability = (10, 0) if nvfp4 else (8, 9)
    if device == "cuda" and not cuda_capability_at_least(capability):
        pytest.skip(f"CUDA SM{capability[0]}{capability[1]} or newer required")
    if is_fp4(a_fmt):
        torch.manual_seed(0)
        counts = [16, 16]
        offs = torch.tensor(counts, device=device, dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        a = torch.randn(sum(counts), 32, device=device)
        b = torch.randn(len(counts), 32, 16, device=device)
    else:
        a, b, offs = _make(COUNTS, K=64, N=48)
    if layout == "ragged_m":
        src_a, src_b = a, b
    else:
        src_a = a.mT
        src_b = torch.randn(a.shape[0], b.shape[-1], device=a.device, dtype=a.dtype)
    # Allocate one stats slot per expert to catch cold experts.
    experts = len(offs)
    a_stats = QuantizationStats("act/x", experts, a.device) if with_stats else None
    b_stats = QuantizationStats("weight/x", experts, b.device) if with_stats else None
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
    if is_fp4(a_fmt):
        expected = []
        start = 0
        for group, stop in enumerate(offs.tolist()):
            aq, sa, gsa = quantize_operand(a[start:stop], -1, a_fmt, a_scale)
            bq, sb, gsb = quantize_operand(b[group], -2, b_fmt, b_scale)
            expected.append(
                dequantize_operand(aq, sa, -1, a_scale, global_scale=gsa)
                @ dequantize_operand(bq, sb, -2, b_scale, global_scale=gsb)
            )
            start = stop
        torch.testing.assert_close(out, torch.cat(expected), rtol=0, atol=1e-5)
        assert a_stats.numel.tolist() == [16 * 32, 16 * 32]
        assert b_stats.numel.tolist() == [32 * 16, 32 * 16]
    else:
        assert torch.isfinite(out).all()
    if with_stats and not is_fp4(a_fmt):
        assert a_stats.numel.shape == (experts,)
        assert a_stats.numel.sum().item() == (src_a.numel() if a_folded else 0)
        assert b_stats.numel.sum().item() == (src_b.numel() if b_folded else 0)


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

    if layout == "ragged_m":
        # Ragged-M: (R,K) x (E,K,N) -> (R,N)
        out = quantized_grouped_mm(
            a, b, offs, fmt, fmt, a.dtype, a_scale, b_scale, rotation=rotation
        )
        ref = quantized_grouped_mm(
            a, b, offs, fmt, fmt, a.dtype, a_scale, b_scale, rotation=None
        )
    elif layout == "ragged_n":
        # Ragged-N: (E,M,K) x (K,R) -> (M,R)
        slabs = torch.randn(offs.shape[0], 32, 64, device="cuda", dtype=torch.bfloat16)
        cols = torch.randn(64, a.shape[0], device="cuda", dtype=torch.bfloat16) * 0.1
        out = quantized_grouped_mm(
            slabs,
            cols,
            offs,
            fmt,
            fmt,
            slabs.dtype,
            a_scale,
            b_scale,
            rotation=rotation,
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
            rotation=None,
        )
    else:
        # Ragged-K: (K,R) x (R,N) -> (E,K,N)
        gy = torch.randn(a.shape[0], 48, device="cuda", dtype=torch.bfloat16)
        out = quantized_grouped_mm(
            a.mT, gy, offs, fmt, fmt, a.dtype, a_scale, b_scale, rotation=rotation
        )
        ref = quantized_grouped_mm(
            a.mT, gy, offs, fmt, fmt, a.dtype, a_scale, b_scale, rotation=None
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
COMPILE_ROTATION_GEMMS = [None, list(GEMM_OPS)]
# Compiler arithmetic can cross quantization thresholds; worst relative norm
# across this grid is 0.01550, with a 3.55x margin.
COMPILE_REL_BOUND = 0.055


@cuda_sm89_or_newer
@pytest.mark.parametrize("dtype", COMPILE_DTYPES)
@pytest.mark.parametrize("act_scale,weight_scale,grad_out_scale", COMPILE_SCALE_TRIPLES)
@pytest.mark.parametrize("counts", COMPILE_COUNTS)
@pytest.mark.parametrize("rotation_gemms", COMPILE_ROTATION_GEMMS)
def test_scaled_grouped_gemm_fn_compiles_fullgraph(
    rotation_gemms,
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
    cfg = _cfg(_per_tensor_scale(act_scale, weight_scale, grad_out_scale), dtype)
    rotation = None
    if rotation_gemms is not None:
        rotation_cfg = {
            "rotation_cls": "hadamard",
            "rotation_kwargs": {"block_size": 16, "seed": 0},
            "gemms": rotation_gemms,
        }
        cfg.rotation = rotation_cfg
        rotation = build_rotation(rotation_cfg).cuda()

    def fwd(a, b):
        return _expert_mm(cfg, a, b, offs, rotation=rotation)

    def run(fn):
        a_, b_ = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
        y = fn(a_, b_)
        y.sum().backward()
        return y, a_.grad, b_.grad

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
def test_scaled_grouped_gemm_fn_forward_precision(
    dtype, act_scale, weight_scale, bias, counts
):
    """Check forward precision across formats, scales, bias, and empty experts."""
    skip_unsupported_dtype_scale(dtype, act_scale)
    skip_unsupported_dtype_scale(dtype, weight_scale)
    if uses_fp4_gemm(
        operand_fmt(dtype, "act", "fwd"), operand_fmt(dtype, "weight", "fwd"), act_scale
    ) and not cuda_capability_at_least((10, 0)):
        pytest.skip("fused FP4 requires CUDA SM100 or newer")
    a, b, offs = _make(counts, K=64, N=48)
    bias0 = torch.randn(offs.shape[0], 48, device="cuda", dtype=torch.bfloat16) * 0.1
    y = _expert_mm(
        _cfg(_per_tensor_scale(act_scale, weight_scale, act_scale), dtype),
        a,
        b,
        offs,
        bias=bias0.clone() if bias else None,
    )

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
def test_scaled_grouped_gemm_fn_backward_precision(
    dtype,
    act_scale,
    weight_scale,
    grad_out_scale,
    bias,
    counts,
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
    bias0 = torch.randn(offs.shape[0], 48, device="cuda", dtype=torch.bfloat16) * 0.1
    a_q, b_q = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
    bias_q = bias0.clone().requires_grad_(True) if bias else None
    y = _expert_mm(
        _cfg(_per_tensor_scale(act_scale, weight_scale, grad_out_scale), dtype),
        a_q,
        b_q,
        offs,
        bias=bias_q,
    )
    gy = torch.randn_like(y)
    y.backward(gy)

    ga_ref, gb_ref = torch.empty_like(a), torch.zeros_like(b)
    lo = 0
    for group, hi in enumerate(offs.tolist()):
        if hi > lo:
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

    expected = [("grad_a", a_q.grad, ga_ref), ("grad_b", b_q.grad, gb_ref)]
    if bias:
        rows = torch.arange(y.shape[0], device=offs.device)
        acc = torch.zeros_like(bias0, dtype=torch.float32)
        acc.index_add_(0, torch.searchsorted(offs, rows, right=True), gy.float())
        expected.append(("grad_bias", bias_q.grad, acc.to(bias0.dtype)))
    for name, got, ref in expected:
        assert rel(got, ref) < PRECISION_BOUND, (name, rel(got, ref))


@cuda_sm89_or_newer
def test_scaled_grouped_gemm_fn_bias_grad_precision():
    rows = 257
    a = torch.ones(rows, 1, device="cuda", dtype=torch.bfloat16)
    b = torch.ones(1, 1, 1, device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros(1, 1, device="cuda", requires_grad=True)
    offs = torch.tensor([rows], device="cuda", dtype=torch.int32)
    dtype = {"weight": "bf16", "act": "bf16", "grad_out": "bf16"}

    out = _expert_mm(_cfg(dtype=dtype), a, b, offs, bias=bias)
    out.backward(torch.ones_like(out))

    assert bias.grad.dtype is torch.float32
    torch.testing.assert_close(bias.grad, torch.full_like(bias, rows), atol=0, rtol=0)


# --- QuantizedSparseMoEBlock ---


def test_quantized_sparse_moe_block_only_quantizes_during_training():
    """Check quantized training and exact unquantized evaluation."""
    torch.manual_seed(0)
    source = SparseMoEBlock(
        d_model=4,
        intermediate_size=8,
        n_routed_experts=2,
        n_routed_experts_per_token=1,
        aux_loss=False,
    )
    block = QuantizedSparseMoEBlock.from_module(source, rule(INT4_W8A16_DTYPES))
    args = (torch.randn(4, 4), torch.randn(2, 4, 8), torch.tensor([2, 4]))
    plain = SparseMoEBlock.expert_mm(block, *args, projection="gate")

    block.train()
    assert not torch.equal(block.expert_mm(*args, projection="gate"), plain)
    block.eval()
    assert torch.equal(block.expert_mm(*args, projection="gate"), plain)


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


# Block 16 divides both expert contractions below (64 and 48); wgrad reaches the
# ragged contraction, so it is the case segment padding has to carry.
E2E_RECIPES = ["fp8", "nvfp4"]
E2E_ROTATIONS = [
    None,
    {
        "rotation_cls": "hadamard",
        "rotation_kwargs": {"block_size": 16},
        "gemms": ["wgrad"],
    },
    {"rotation_cls": "hadamard", "rotation_kwargs": {"block_size": 16}},
]


@cuda_sm89_or_newer
@pytest.mark.parametrize("recipe", E2E_RECIPES)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("rotation", E2E_ROTATIONS)
def test_quantized_sparse_moe_block_trains_a_full_model(bias, rotation, recipe):
    """Check one converted MoE step and fused per-expert bias gradients."""
    if recipe == "nvfp4" and not cuda_capability_at_least((10, 0)):
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
        training=TrainingConfig(
            mixed_precision="bf16",
            quantization={
                "enabled": True,
                "dtype": {"recipe": recipe},
                "scale": {"recipe": "rowwise"} if recipe == "fp8" else {},
                "rotation": rotation,
            },
        ),
    )
    model = build_model(config)
    apply_quantization(model, config)
    model.cuda().to(torch.bfloat16)
    blocks = [m for m in model.modules() if isinstance(m, QuantizedSparseMoEBlock)]
    assert blocks  # Conversion installed the quantized seam.
    # Pin the rotation to the block: a config that silently resolves to None would
    # leave every assertion below passing without a rotation ever running.
    for block in blocks:
        assert (block.rotation is not None) == (rotation is not None)

    ids = torch.randint(0, 128, (2, 64), device="cuda")
    position_ids = torch.arange(64, device="cuda").unsqueeze(0).expand(2, 64)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(ids, position_ids)
        loss = logits.float().log_softmax(-1).mean().neg()
    loss.backward()

    assert torch.isfinite(loss).item()
    for block in blocks:
        for name, p in block.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert p.grad.abs().sum() > 0, name
