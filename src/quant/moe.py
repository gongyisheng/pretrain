from __future__ import annotations

import copy

import torch

from src.kernel.gemm import grouped_gemm, scaled_grouped_gemm
from src.layers.mlp import SparseMoEBlock
from src.metrics.quant import record_operand
from src.quant.quantize import dequantize_operand, quantize_operand
from src.quant.utils import is_fp8, is_int8s, is_quantized
from src.utils.config import QuantizationConfig


def quantized_grouped_gemm(
    a, b, offs, a_fmt, b_fmt, out_dtype, scaling, bias=None, a_stats=None, b_stats=None
):
    """Quantized ragged grouped GEMM, layout picked from the operand ranks.

        (M,K) x (E,K,N) -> (M,N)    ragged M: A's row axis, not the contraction
        (M,K) x (K,N)   -> (E,M,N)  ragged K: the contraction itself (the wgrad)

    The ragged-N layout (3D x 2D) is unimplemented: a per-group scale on B's ragged
    column axis would need a column-wise ragged mapping, and nothing asks for it.

    `bias` is an optional (E,N) tensor broadcast over the output's row dim. It is
    never quantized -- it rides the epilogue, past the scales.
    """
    block_size = scaling["block_shape"][1]
    same_family = (is_fp8(a_fmt) and is_fp8(b_fmt)) or (
        is_int8s(a_fmt) and is_int8s(b_fmt)
    )
    if a.ndim != 2:
        raise NotImplementedError("the ragged-N layout (3D x 2D) is not supported")
    ragged_k = b.ndim == 2

    # ragged K: the contraction is ragged, so both operands carry the mapping, and A is
    # quantized in the (ragged,K) orientation -- only there is the reduction ragged --
    # then transposed back below with its scale. ragged M: the contraction is dense, so
    # the mapping reaches A alone, where only tensorwise has a tile wide enough to span
    # groups.
    src_a = a.mT if ragged_k else a
    contract_a = -2 if ragged_k else -1
    a_ragged_dim = -2
    # B carries the mapping only when the contraction is the ragged axis
    b_ragged_dim, b_offs = (-2, offs) if ragged_k else (None, None)

    aq = sa = bq = sb = None
    if is_quantized(a_fmt):
        aq, sa = quantize_operand(
            src_a, contract_a, a_fmt, scaling, offs=offs, ragged_dim=a_ragged_dim
        )
        record_operand(
            a_stats,
            src_a,
            aq,
            sa,
            contract_a,
            scaling,
            offs=offs,
            ragged_dim=a_ragged_dim,
        )
    if is_quantized(b_fmt):
        bq, sb = quantize_operand(
            b, -2, b_fmt, scaling, offs=b_offs, ragged_dim=b_ragged_dim
        )
        # `offs` here is the real offs, not `b_offs` -- deliberately, even though
        # b_ragged_dim is None in the forward/dgrad case (B stacked (E,K,N)), which
        # looks like it violates the given-together rule _check_dims enforces. It
        # doesn't: record_operand narrows offs to None for its own dequantize when
        # ragged_dim is None, while the accumulated sums branch on whether offs is
        # None to decide global vs. per-expert reduction. Swapping this to b_offs "to
        # match" would silently drop the MoE weight metrics to a global reduction
        # whenever b_ragged_dim is None.
        record_operand(
            b_stats,
            b,
            bq,
            sb,
            -2,
            scaling,
            offs=offs,
            ragged_dim=b_ragged_dim,
        )

    if same_family and a.is_cuda:
        y = scaled_grouped_gemm(
            aq.mT if ragged_k else aq,
            bq,
            sa.mT if ragged_k else sa,
            sb,
            offs,
            out_dtype,
            block_size,
            bias=bias,
            scale_dtype=scaling.get("scale_dtype"),
        )
        return y

    if aq is not None:
        src_a = dequantize_operand(
            aq, sa, contract_a, scaling, offs=offs, ragged_dim=a_ragged_dim
        ).to(a.dtype)
    if bq is not None:
        b = dequantize_operand(
            bq, sb, -2, scaling, offs=b_offs, ragged_dim=b_ragged_dim
        ).to(b.dtype)
    y = grouped_gemm(
        (src_a.mT if ragged_k else src_a).to(out_dtype),
        b.to(out_dtype),
        offs,
        bias=None if bias is None else bias.to(out_dtype),
    )
    return y.to(out_dtype)


class ScaledGroupedGemmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, bias, offs, cfg: QuantizationConfig, stats):
        out_dtype = a.dtype
        y = quantized_grouped_gemm(
            a,
            b,
            offs,
            cfg.dtype["act"],
            cfg.dtype["weight"],
            out_dtype,
            cfg.scaling,
            bias=bias,
            a_stats=stats.get("fwd_act"),
            b_stats=stats.get("fwd_weight"),
        )
        ctx.save_for_backward(a, b, offs)
        ctx.cfg = cfg
        ctx.stats = stats
        ctx.bias_needs_grad = ctx.needs_input_grad[2]
        return y

    @staticmethod
    def backward(ctx, grad_y):
        a, b, offs = ctx.saved_tensors
        cfg = ctx.cfg
        stats = ctx.stats
        out_dtype = a.dtype
        # dgrad: grad_a = grad_y @ b^T
        grad_a = quantized_grouped_gemm(
            grad_y,
            b.transpose(-2, -1).contiguous(),
            offs,
            cfg.dtype["grad_input"],
            cfg.dtype["weight"],
            out_dtype,
            cfg.scaling,
            a_stats=stats.get("dgrad_grad"),
            b_stats=stats.get("dgrad_weight"),
        )
        # wgrad: grad_b[g] = a[g]^T @ grad_y[g] — the ragged token axis is the
        # contraction, i.e. the ragged-K layout
        grad_b = quantized_grouped_gemm(
            a.mT,
            grad_y,
            offs,
            cfg.dtype["act"],
            cfg.dtype["grad_weight"],
            out_dtype,
            cfg.scaling,
            a_stats=stats.get("wgrad_act"),
            b_stats=stats.get("wgrad_grad"),
        )
        grad_bias = None
        if ctx.bias_needs_grad:
            # bias is (E,N) broadcast over the ragged row axis, so its grad sums that
            # axis. The fp32 accumulator is the point: autograd's own index backward
            # sums each expert's hundreds of rows in grad_y's dtype, and bf16 drifts
            # percent-level over that many terms. Unquantized on purpose -- a plain
            # reduction of the incoming grad has no operand worth quantizing.
            rows = torch.arange(grad_y.shape[0], device=offs.device)
            group_of_row = torch.searchsorted(offs, rows, right=True)
            acc = grad_y.new_zeros(offs.shape[0], grad_y.shape[1], dtype=torch.float32)
            acc.index_add_(0, group_of_row, grad_y.float())
            grad_bias = acc.to(grad_y.dtype)
        return grad_a, grad_b, grad_bias, None, None, None


def scaled_grouped_gemm_fn(cfg: QuantizationConfig, stats=None):
    """Build the block's expert GEMM seam.

    `stats` maps `<projection>.<gemm>.<operand>` to an accumulator. Rebinding this
    closure with a populated map is how metric collection is installed -- routing
    happens inside the block's forward, so no module hook can reach these operands.
    """
    stats = stats or {}

    def expert_mm(a, b, offs, bias=None, projection=None):
        return ScaledGroupedGemmFn.apply(
            a, b, bias, offs, cfg, stats.get(projection, {})
        )

    return expert_mm


class QuantizedSparseMoEBlock(SparseMoEBlock):
    @classmethod
    def from_module(
        cls, module: SparseMoEBlock, quantization_config: QuantizationConfig
    ) -> "QuantizedSparseMoEBlock":
        q = cls.__new__(cls)
        q.__dict__ = copy.deepcopy(module).__dict__
        q.quantization_config = quantization_config
        q.expert_mm = scaled_grouped_gemm_fn(quantization_config)
        return q
