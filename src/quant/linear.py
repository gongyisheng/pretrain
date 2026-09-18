import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.kernel.ops.gemm import SCALED_MM_OPS, _is_kernel_available
from src.metrics.quant import QuantizationStats, quantize_and_record
from src.quant.quantize import dequantize_operand
from src.quant.rotation import Rotation
from src.quant.utils import is_fp4, is_quantized, resolve_scale, scaled_mm_op
from src.utils.config import QuantizationConfig


def quantized_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_fmt: str,
    b_fmt: str,
    out_dtype: torch.dtype,
    a_scale: dict,
    b_scale: dict,
    bias: torch.Tensor | None = None,
    a_stochastic_rounding: bool = False,
    b_stochastic_rounding: bool = False,
    a_stats: QuantizationStats | None = None,
    b_stats: QuantizationStats | None = None,
    rotation: Rotation | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    """Quantized 2D GEMM with optional per-operand quantization statistics.

    Bias is added before casting to `out_dtype`. `rotation` preconditions both
    operands with the same block-Hadamard; scaled kernels consume those codes
    directly, while emulation inverts each dequantized operand.
    """
    if a.dtype != b.dtype:
        raise ValueError(
            f"a and b must have the same dtype, got {a.dtype} and {b.dtype}"
        )
    op = scaled_mm_op(
        a_fmt,
        b_fmt,
        a_scale["scale_dtype"],
        a_scale["block_shape"],
    )
    scale_layout = "row_major"
    if (
        backend == "cuda"
        and a.ndim == b.ndim == 2
        and a.shape[1] == b.shape[0]
        and a.shape[1] % 16 == 0
        and b.shape[1] % 16 == 0
        and (bias is None or (bias.shape == (b.shape[1],) and bias.is_contiguous()))
        and rotation is None
        and (
            (
                op == "gemm.mxfp8_scaled_mm"
                and a_fmt == b_fmt == "fp8_e4m3"
                and a_scale["scale_dtype"] is torch.float8_e8m0fnu
                and b_scale["scale_dtype"] is torch.float8_e8m0fnu
                and a_scale["block_shape"][1] == b_scale["block_shape"][1] == 32
                and out_dtype is torch.bfloat16
            )
            or (
                op == "gemm.nvfp4_scaled_mm"
                and is_fp4(a_fmt)
                and is_fp4(b_fmt)
                and a_scale["scale_dtype"] is torch.float8_e4m3fn
                and b_scale["scale_dtype"] is torch.float8_e4m3fn
                and a_scale["block_shape"][1] == b_scale["block_shape"][1] == 16
                and a.shape[1] % 32 == 0
                and out_dtype in (torch.bfloat16, torch.float16, torch.float32)
                and (bias is None or out_dtype is not torch.float32)
            )
        )
        and _is_kernel_available(op, "cuda", a.device)
    ):
        scale_layout = "swizzled_32_4_4"
    aq = sa = gsa = bq = sb = gsb = None
    if is_quantized(a_fmt):
        aq, sa, gsa = quantize_and_record(
            a_stats,
            a,
            -1,
            a_fmt,
            a_scale,
            stochastic_rounding=a_stochastic_rounding,
            rotation=rotation,
            output_layout="row_major",
            backend=backend,
            scale_layout=scale_layout,
        )
    if is_quantized(b_fmt):
        bq, sb, gsb = quantize_and_record(
            b_stats,
            b,
            -2,
            b_fmt,
            b_scale,
            stochastic_rounding=b_stochastic_rounding,
            rotation=rotation,
            output_layout="column_major" if op is not None else "row_major",
            backend=backend,
            scale_layout=scale_layout,
        )

    if scale_layout == "swizzled_32_4_4":
        return SCALED_MM_OPS[op](
            aq,
            bq,
            sa,
            sb,
            out_dtype,
            a_scale["block_shape"][1],
            bias=None if bias is None else bias.to(out_dtype),
            gsa=gsa,
            gsb=gsb,
            scale_layout=scale_layout,
        )
    if op is not None:
        return SCALED_MM_OPS[op](
            aq,
            bq,
            sa,
            sb,
            out_dtype,
            a_scale["block_shape"][1],
            bias=None if bias is None else bias.to(out_dtype),
            gsa=gsa,
            gsb=gsb,
        )

    if aq is not None:
        a = dequantize_operand(
            aq, sa, -1, a_scale, rotation=rotation, global_scale=gsa
        ).to(a.dtype)
    if bq is not None:
        b = dequantize_operand(
            bq, sb, -2, b_scale, rotation=rotation, global_scale=gsb
        ).to(b.dtype)
    y = a @ b if bias is None else torch.addmm(bias.to(a.dtype), a, b)
    return y.to(out_dtype)


class QuantizedLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, cfg: QuantizationConfig, stats, rotation):
        # Compute in the active autocast dtype if any, else x's dtype.
        device_type = x.device.type
        if torch.is_autocast_enabled(device_type):
            compute_dtype = torch.get_autocast_dtype(device_type)
        else:
            compute_dtype = x.dtype

        x2d = x.reshape(-1, x.shape[-1]).to(compute_dtype)
        w = weight.to(compute_dtype)
        y = quantized_mm(
            x2d,
            w.t(),
            cfg.dtype["act"]["fwd"],
            cfg.dtype["weight"]["fwd"],
            compute_dtype,
            resolve_scale(cfg.scale, "act"),
            resolve_scale(cfg.scale, "weight"),
            bias=bias,
            a_stochastic_rounding=cfg.rounding["act"] == "SR",
            b_stochastic_rounding=cfg.rounding["weight"] == "SR",
            a_stats=stats.get("act"),
            b_stats=stats.get("weight"),
            rotation=(
                rotation
                if cfg.rotation is not None and "fwd" in cfg.rotation["gemms"]
                else None
            ),
            backend=cfg.backend,
        )

        ctx.save_for_backward(x2d, w)
        ctx.cfg = cfg
        # Stats are metadata, not saved tensors.
        ctx.stats = stats
        ctx.rotation = rotation
        ctx.has_bias = bias is not None
        ctx.x_shape = x.shape
        ctx.x_dtype = x.dtype
        ctx.w_dtype = weight.dtype
        ctx.bias_dtype = None if bias is None else bias.dtype
        return y.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        x2d, w = ctx.saved_tensors
        cfg = ctx.cfg
        stats = ctx.stats
        compute_dtype = x2d.dtype
        g = grad_out.reshape(-1, grad_out.shape[-1]).to(compute_dtype)  # (M, N)
        # dX = g @ W, (M,N)@(N,K) -> (M,K)
        dx = quantized_mm(
            g,
            w,
            cfg.dtype["grad_out"]["dgrad"],
            cfg.dtype["weight"]["dgrad"],
            compute_dtype,
            resolve_scale(cfg.scale, "grad_out"),
            resolve_scale(cfg.scale, "weight"),
            a_stochastic_rounding=cfg.rounding["grad_out"] == "SR",
            b_stochastic_rounding=cfg.rounding["weight"] == "SR",
            a_stats=stats.get("grad_out"),
            b_stats=stats.get("weight"),
            rotation=(
                ctx.rotation
                if cfg.rotation is not None and "dgrad" in cfg.rotation["gemms"]
                else None
            ),
            backend=cfg.backend,
        )
        # dW = gᵀ @ X, (N,M)@(M,K) -> (N,K)
        dw = quantized_mm(
            g.t(),
            x2d,
            cfg.dtype["grad_out"]["wgrad"],
            cfg.dtype["act"]["wgrad"],
            compute_dtype,
            resolve_scale(cfg.scale, "grad_out"),
            resolve_scale(cfg.scale, "act"),
            a_stochastic_rounding=cfg.rounding["grad_out"] == "SR",
            b_stochastic_rounding=cfg.rounding["act"] == "SR",
            a_stats=stats.get("grad_out"),
            b_stats=stats.get("act"),
            rotation=(
                ctx.rotation
                if cfg.rotation is not None and "wgrad" in cfg.rotation["gemms"]
                else None
            ),
            backend=cfg.backend,
        )
        db = g.sum(dim=0, dtype=torch.float32) if ctx.has_bias else None

        dx = dx.reshape(*ctx.x_shape).to(ctx.x_dtype)
        dw = dw.to(ctx.w_dtype)
        db = db.to(ctx.bias_dtype) if db is not None else None
        return dx, dw, db, None, None, None


class QuantizedLinear(nn.Linear):
    @classmethod
    def from_module(
        cls,
        module: nn.Linear,
        quantization_config: QuantizationConfig,
        rotation: Rotation | None = None,
    ) -> "QuantizedLinear":
        q = cls.__new__(cls)
        q.__dict__ = copy.deepcopy(module).__dict__
        q.quantization_config = quantization_config
        object.__setattr__(q, "rotation", rotation)
        # Monitoring attaches stats later; an empty dict disables recording.
        q.quant_stats = {}
        return q

    def forward(self, x):
        if not self.training:
            return F.linear(x, self.weight, self.bias)
        return QuantizedLinearFn.apply(
            x,
            self.weight,
            self.bias,
            self.quantization_config,
            self.quant_stats,
            self.rotation,
        )
