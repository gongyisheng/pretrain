import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.kernel.ops.gemm import SCALED_MM_OPS
from src.metrics.quant import (
    QuantizationStats,
    get_quantization_monitoring_status,
    record_operand,
)
from src.quant.quantize import dequantize_operand, quantize_operand
from src.quant.rotation import (
    Rotation,
    apply_rotation_on_axes,
    transpose_rotation_axes,
)
from src.quant.utils import is_quantized, resolve_scale, scaled_mm_op
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
    a_rotation_axes: tuple[int, ...] | list[int] = (),
    b_rotation_axes: tuple[int, ...] | list[int] = (),
) -> torch.Tensor:
    """Quantize already rotated operands and restore the result's coordinates.

    Bias is added before casting to `out_dtype`.
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
    if a_scale["block_shape"][1] != b_scale["block_shape"][1]:
        op = None
    if rotation is None:
        a_rotation_axes = b_rotation_axes = ()
    a_inner = -1 in a_rotation_axes
    b_inner = -2 in b_rotation_axes
    a_outer = -2 in a_rotation_axes
    b_outer = -1 in b_rotation_axes
    if a_inner != b_inner:
        op = None
    use_fused_gemm_bias = bias is not None and not a_outer and not b_outer
    aq = sa = gsa = bq = sb = gsb = None
    if is_quantized(a_fmt):
        collect_stats = a_stats is not None and get_quantization_monitoring_status()
        aq, sa, gsa, a_quantization_stats = quantize_operand(
            a,
            -1,
            a_fmt,
            a_scale,
            stochastic_rounding=a_stochastic_rounding,
            return_quantization_stats=collect_stats,
        )
        if a_quantization_stats is not None:
            record_operand(a_stats, a_quantization_stats)

    if is_quantized(b_fmt):
        collect_stats = b_stats is not None and get_quantization_monitoring_status()
        bq, sb, gsb, b_quantization_stats = quantize_operand(
            b,
            -2,
            b_fmt,
            b_scale,
            stochastic_rounding=b_stochastic_rounding,
            return_quantization_stats=collect_stats,
        )
        if b_quantization_stats is not None:
            record_operand(b_stats, b_quantization_stats)

    if op is not None:
        y = SCALED_MM_OPS[op](
            aq,
            bq,
            sa,
            sb,
            out_dtype,
            a_scale["block_shape"][1],
            bias=bias.to(out_dtype) if use_fused_gemm_bias else None,
            gsa=gsa,
            gsb=gsb,
        )
    else:
        if aq is not None:
            a = dequantize_operand(aq, sa, -1, a_scale, global_scale=gsa).to(a.dtype)
        if bq is not None:
            b = dequantize_operand(bq, sb, -2, b_scale, global_scale=gsb).to(b.dtype)
        if a_inner and not b_inner:
            a = rotation.inverse(a, -1)
        elif b_inner and not a_inner:
            b = rotation.inverse(b, -2)
        y = torch.addmm(bias.to(a.dtype), a, b) if use_fused_gemm_bias else a @ b
    if a_outer:
        y = rotation.inverse(y, -2)
    if b_outer:
        y = rotation.inverse(y, -1)
    if bias is not None and not use_fused_gemm_bias:
        y = y.float() + bias.float()
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
        weight = weight.to(compute_dtype)
        rotation_axes = cfg.rotation["rotation_axes"]
        act_rotation_axes = rotation_axes["act"]["fwd"]
        weight_rotation_axes = rotation_axes["weight"]["fwd"]
        fwd_x = apply_rotation_on_axes(x2d, rotation, act_rotation_axes)
        fwd_weight = apply_rotation_on_axes(weight, rotation, weight_rotation_axes)
        y = quantized_mm(
            fwd_x,
            fwd_weight.t(),
            cfg.dtype["act"]["fwd"],
            cfg.dtype["weight"]["fwd"],
            compute_dtype,
            resolve_scale(cfg.scale, "act"),
            resolve_scale(cfg.scale, "weight"),
            bias=bias,
            a_stochastic_rounding=cfg.rounding["act"] == "SR",
            b_stochastic_rounding=cfg.rounding["weight"] == "SR",
            a_stats=stats.get("fwd/act"),
            b_stats=stats.get("fwd/weight"),
            rotation=rotation,
            a_rotation_axes=act_rotation_axes,
            b_rotation_axes=transpose_rotation_axes(weight_rotation_axes),
        )
        y = y.reshape(*x.shape[:-1], weight.shape[0])

        ctx.save_for_backward(x2d.reshape(x.shape), weight)
        ctx.cfg = cfg
        ctx.stats = stats
        ctx.rotation = rotation
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        x2d = x.reshape(-1, x.shape[-1])
        cfg = ctx.cfg
        stats = ctx.stats
        compute_dtype = x2d.dtype
        grad_out = grad_out.reshape(-1, grad_out.shape[-1]).to(compute_dtype)  # (M, N)
        rotation = ctx.rotation
        rotation_axes = cfg.rotation["rotation_axes"]
        act_rotation_axes = rotation_axes["act"]["wgrad"]
        weight_rotation_axes = rotation_axes["weight"]["dgrad"]
        grad_out_rotation_axes = rotation_axes["grad_out"]["dgrad"]
        bwd_grad_out = apply_rotation_on_axes(
            grad_out, rotation, grad_out_rotation_axes
        )
        bwd_weight = apply_rotation_on_axes(weight, rotation, weight_rotation_axes)
        # dX = g @ W, (M,N)@(N,K) -> (M,K)
        dx = quantized_mm(
            bwd_grad_out,
            bwd_weight,
            cfg.dtype["grad_out"]["dgrad"],
            cfg.dtype["weight"]["dgrad"],
            compute_dtype,
            resolve_scale(cfg.scale, "grad_out"),
            resolve_scale(cfg.scale, "weight"),
            a_stochastic_rounding=cfg.rounding["grad_out"] == "SR",
            b_stochastic_rounding=cfg.rounding["weight"] == "SR",
            a_stats=stats.get("dgrad/grad_out"),
            b_stats=stats.get("dgrad/weight"),
            rotation=rotation,
            a_rotation_axes=grad_out_rotation_axes,
            b_rotation_axes=weight_rotation_axes,
        )
        dx = dx.reshape(x.shape)
        # dW = gᵀ @ X, (N,M)@(M,K) -> (N,K)
        grad_out_rotation_axes = rotation_axes["grad_out"]["wgrad"]
        bwd_grad_out = apply_rotation_on_axes(
            grad_out, rotation, grad_out_rotation_axes
        )
        bwd_act = apply_rotation_on_axes(x2d, rotation, act_rotation_axes)
        dw = quantized_mm(
            bwd_grad_out.t(),
            bwd_act,
            cfg.dtype["grad_out"]["wgrad"],
            cfg.dtype["act"]["wgrad"],
            compute_dtype,
            resolve_scale(cfg.scale, "grad_out"),
            resolve_scale(cfg.scale, "act"),
            a_stochastic_rounding=cfg.rounding["grad_out"] == "SR",
            b_stochastic_rounding=cfg.rounding["act"] == "SR",
            a_stats=stats.get("wgrad/grad_out"),
            b_stats=stats.get("wgrad/act"),
            rotation=rotation,
            a_rotation_axes=transpose_rotation_axes(grad_out_rotation_axes),
            b_rotation_axes=act_rotation_axes,
        )
        db = None
        if ctx.needs_input_grad[2]:
            db = grad_out.sum(dim=0, dtype=torch.float32)
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
        q.quantization_enabled = False
        object.__setattr__(q, "rotation", rotation)
        # Monitoring attaches stats later; an empty dict disables recording.
        q.quant_stats = {}
        return q

    def forward(self, x):
        if not self.training or not self.quantization_enabled:
            return F.linear(x, self.weight, self.bias)
        return QuantizedLinearFn.apply(
            x,
            self.weight,
            self.bias,
            self.quantization_config,
            self.quant_stats,
            self.rotation,
        )
