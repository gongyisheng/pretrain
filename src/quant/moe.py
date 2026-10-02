import copy

import torch

from src.kernel.ops.gemm import SCALED_MM_OPS, grouped_mm
from src.layers.mlp import SparseMoEBlock
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
from src.quant.utils import is_fp4, is_quantized, resolve_scale, scaled_grouped_mm_op
from src.utils.config import QuantizationConfig


def quantized_grouped_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    offs: torch.Tensor,
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
    """Quantized grouped GEMM on already rotated operands.

    Operand ranks select the layout. `bias` has shape (E, N), broadcasts over output
    rows, and is not quantized. Rotated ragged axes must align per expert.
    """
    if a.dtype != b.dtype:
        raise ValueError(
            f"a and b must have the same dtype, got {a.dtype} and {b.dtype}"
        )
    op = scaled_grouped_mm_op(
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
    ragged_k = a.ndim == 2 and b.ndim == 2
    ragged_alignment = (
        rotation.alignment
        if rotation is not None
        and (
            (a.ndim == 2 and (a_inner if ragged_k else a_outer))
            or (b.ndim == 2 and (b_inner if ragged_k else b_outer))
        )
        else 1
    )
    if ragged_alignment > 1:
        valid = torch.all(offs.remainder(ragged_alignment) == 0)
        message = (
            "each expert's rotated extent must be divisible by the rotation alignment"
        )
        if torch.compiler.is_compiling():
            torch._assert_async(valid, message)
        else:
            torch._assert(valid, message)

    # Map the operand containing the ragged axis.
    if a.ndim == 3:
        # Ragged N: only B is mapped on its outer axis.
        contract_a, a_offs, a_ragged_dim = -1, None, None
        b_offs, b_ragged_dim = offs, -1
    elif ragged_k:
        # Ragged K: both operands share their contraction axis.
        contract_a, a_ragged_dim = -1, -1
        b_ragged_dim = -2
        has_fp4 = is_fp4(a_fmt) or is_fp4(b_fmt)
        alignment = 16 if has_fp4 else 1
        if alignment > 1:
            n_rows, n_groups = a.shape[-1], offs.shape[0]
            starts = torch.cat([offs.new_zeros(1), offs[:-1]])
            counts = offs - starts
            ceil_inputs = counts + alignment - 1
            padded_blocks = torch.div(ceil_inputs, alignment, rounding_mode="floor")
            padded_counts = padded_blocks * alignment
            padded_offs = padded_counts.cumsum(0).to(offs.dtype)
            padded_starts = torch.cat([padded_offs.new_zeros(1), padded_offs[:-1]])
            rows = torch.arange(n_rows, device=offs.device, dtype=offs.dtype)
            group = torch.searchsorted(offs, rows, right=True)
            group.clamp_(max=n_groups - 1)
            index = (padded_starts[group] + rows - starts[group]).long()
            max_padded_rows = n_rows + n_groups * alignment
            n_padded = -(-max_padded_rows // alignment) * alignment
            a_shape = (a.shape[-2], n_padded)
            padded_a = a.new_zeros(a_shape, dtype=torch.float32)
            padded_a.index_copy_(1, index, a.float())
            a = padded_a
            b_shape = (n_padded, b.shape[-1])
            padded_b = b.new_zeros(b_shape, dtype=torch.float32)
            padded_b.index_copy_(0, index, b.float())
            b = padded_b
            if has_fp4:
                # Keep allocation static and include the zero tail in the last group.
                padded_offs[-1] = n_padded
            offs = padded_offs
        a_offs, b_offs = offs, offs
    else:
        # Ragged M: only A is mapped.
        contract_a, a_offs, a_ragged_dim = -1, offs, -2
        b_offs, b_ragged_dim = None, None

    aq = sa = gsa = bq = sb = gsb = None
    if a_inner != b_inner and is_quantized(a_fmt) and is_quantized(b_fmt):
        op = None

    if is_quantized(a_fmt):
        collect_stats = a_stats is not None and get_quantization_monitoring_status()
        aq, sa, gsa, a_quantization_stats = quantize_operand(
            a,
            contract_a,
            a_fmt,
            a_scale,
            offs=a_offs,
            ragged_dim=a_ragged_dim,
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
            offs=b_offs,
            ragged_dim=b_ragged_dim,
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
            offs,
            out_dtype,
            a_scale["block_shape"][1],
            bias=bias.to(out_dtype) if use_fused_gemm_bias else None,
            gsa=gsa,
            gsb=gsb,
        )
    else:
        if aq is not None:
            a = dequantize_operand(
                aq,
                sa,
                contract_a,
                a_scale,
                offs=a_offs,
                ragged_dim=a_ragged_dim,
                global_scale=gsa,
            ).to(a.dtype)
        if bq is not None:
            b = dequantize_operand(
                bq,
                sb,
                -2,
                b_scale,
                offs=b_offs,
                ragged_dim=b_ragged_dim,
                global_scale=gsb,
            ).to(b.dtype)
        if a_inner and not b_inner:
            a = rotation.inverse(a, -1, out_dtype)
        elif b_inner and not a_inner:
            b = rotation.inverse(b, -2, out_dtype)
        y = grouped_mm(
            a.to(out_dtype),
            b.to(out_dtype),
            offs,
            bias=bias.to(out_dtype) if use_fused_gemm_bias else None,
        )
    if a_outer:
        y = rotation.inverse(y, -2, out_dtype)
    if b_outer:
        y = rotation.inverse(y, -1, out_dtype)
    if bias is not None and not use_fused_gemm_bias:
        if ragged_k:
            y = y + bias[:, None, :].to(out_dtype)
        elif a.ndim == 2 and b.ndim == 3:
            rows = torch.arange(y.shape[0], device=offs.device)
            groups = torch.searchsorted(offs, rows, right=True)
            y = y + bias.index_select(0, groups).to(out_dtype)
        else:
            raise NotImplementedError("bias is not supported for ragged-N grouped GEMM")
    return y.to(out_dtype)


class ScaledGroupedGemmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, offs, cfg: QuantizationConfig, stats, rotation):
        device_type = x.device.type
        if torch.is_autocast_enabled(device_type):
            compute_dtype = torch.get_autocast_dtype(device_type)
        else:
            compute_dtype = x.dtype

        x = x.to(compute_dtype)
        weight = weight.to(compute_dtype)
        rotation_axes = cfg.rotation["rotation_axes"]
        act_rotation_axes = rotation_axes["act"]["fwd"]
        weight_rotation_axes = rotation_axes["weight"]["fwd"]
        fwd_x = apply_rotation_on_axes(x, rotation, act_rotation_axes)
        fwd_weight = apply_rotation_on_axes(weight, rotation, weight_rotation_axes)
        y = quantized_grouped_mm(
            fwd_x,
            fwd_weight.mT,
            offs,
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
        ctx.save_for_backward(x, weight, offs)
        ctx.cfg = cfg
        ctx.stats = stats
        ctx.rotation = rotation
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, offs = ctx.saved_tensors
        cfg = ctx.cfg
        stats = ctx.stats
        compute_dtype = x.dtype
        grad_out = grad_out.to(compute_dtype)
        rotation = ctx.rotation
        rotation_axes = cfg.rotation["rotation_axes"]
        act_rotation_axes = rotation_axes["act"]["wgrad"]
        weight_rotation_axes = rotation_axes["weight"]["dgrad"]
        grad_out_rotation_axes = rotation_axes["grad_out"]["dgrad"]
        bwd_grad_out = apply_rotation_on_axes(
            grad_out, rotation, grad_out_rotation_axes
        )
        bwd_weight = apply_rotation_on_axes(weight, rotation, weight_rotation_axes)
        # dgrad: dx = grad_out @ weight.
        dx = quantized_grouped_mm(
            bwd_grad_out,
            bwd_weight,
            offs,
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
        # Wgrad uses the ragged-K layout.
        grad_out_rotation_axes = rotation_axes["grad_out"]["wgrad"]
        bwd_grad_out = apply_rotation_on_axes(
            grad_out, rotation, grad_out_rotation_axes
        )
        bwd_act = apply_rotation_on_axes(x, rotation, act_rotation_axes)
        dw = quantized_grouped_mm(
            bwd_act.mT,
            bwd_grad_out,
            offs,
            cfg.dtype["act"]["wgrad"],
            cfg.dtype["grad_out"]["wgrad"],
            compute_dtype,
            resolve_scale(cfg.scale, "act"),
            resolve_scale(cfg.scale, "grad_out"),
            a_stochastic_rounding=cfg.rounding["act"] == "SR",
            b_stochastic_rounding=cfg.rounding["grad_out"] == "SR",
            a_stats=stats.get("wgrad/act"),
            b_stats=stats.get("wgrad/grad_out"),
            rotation=rotation,
            a_rotation_axes=transpose_rotation_axes(act_rotation_axes),
            b_rotation_axes=grad_out_rotation_axes,
        )
        db = None
        if ctx.needs_input_grad[2]:
            # Sum each expert's rows in fp32 to avoid bf16 reduction error.
            rows = torch.arange(grad_out.shape[0], device=offs.device)
            group_of_row = torch.searchsorted(offs, rows, right=True)
            db = grad_out.new_zeros(
                offs.shape[0], grad_out.shape[1], dtype=torch.float32
            )
            db.index_add_(0, group_of_row, grad_out.float())
        return dx, dw.mT, db, None, None, None, None


class QuantizedSparseMoEBlock(SparseMoEBlock):
    @classmethod
    def from_module(
        cls,
        module: SparseMoEBlock,
        quantization_config: QuantizationConfig,
        rotation: Rotation | None = None,
    ) -> "QuantizedSparseMoEBlock":
        q = cls.__new__(cls)
        q.__dict__ = copy.deepcopy(module).__dict__
        q.quantization_config = quantization_config
        q.quantization_enabled = False
        object.__setattr__(q, "rotation", rotation)
        # Monitoring populates this; empty disables statistics.
        q.quant_stats = {}
        return q

    def expert_mm(self, a, b, offs, bias=None, projection=None):
        """Quantize training expert GEMMs; use the base block in eval."""
        if not self.training or not self.quantization_enabled:
            return super().expert_mm(a, b, offs, bias=bias, projection=projection)
        return ScaledGroupedGemmFn.apply(
            a,
            b,
            bias,
            offs,
            self.quantization_config,
            self.quant_stats.get(projection, {}),
            self.rotation,
        )
