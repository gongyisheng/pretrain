from __future__ import annotations

import copy

import torch

from src.kernel.gemm import grouped_gemm, scaled_grouped_gemm
from src.layers.mlp import SparseMoEBlock
from src.quant.quantize import (
    QuantizationSnapshot,
    dequantize_operand,
    quantize_operand,
    ragged_scale_blocks,
)
from src.quant.utils import is_fp8, is_int8s, is_quantized
from src.utils.config import QuantizationConfig


def _quantize_b(b, granularity, block_size, fmt, scale_dtype, ragged):
    """Quantize the right operand along its contraction axis (dim -2).

    A 3D b is quantized per expert, so no scale spans two of them; a 2D b shares the
    ragged mapping with the left operand.
    """
    if b.ndim == 2:
        return quantize_operand(
            b, 0, granularity, block_size, fmt, scale_dtype=scale_dtype, ragged=ragged
        )
    per_expert = [
        quantize_operand(b[g], 0, granularity, block_size, fmt, scale_dtype=scale_dtype)
        for g in range(b.shape[0])
    ]
    return (
        torch.stack([q for q, _ in per_expert]),
        torch.stack([s for _, s in per_expert]),
    )


def quantized_grouped_gemm(
    a, b, offs, a_fmt, b_fmt, out_dtype, scaling, bias=None, enable_snapshot=False
):
    """Quantized ragged grouped GEMM, layout picked from the operand ranks.

        (M,K) x (E,K,N) -> (M,N)    ragged M: A's row axis, not the contraction
        (M,K) x (K,N)   -> (E,M,N)  ragged K: the contraction itself (the wgrad)

    The ragged-N layout (3D x 2D) is unimplemented: a per-group scale on B's ragged
    column axis would need a column-wise ragged mapping, and nothing asks for it.

    `bias` is an optional (E,N) tensor broadcast over the output's row dim. It is
    never quantized -- it rides the epilogue, past the scales.
    """
    granularity = scaling["granularity"]
    block_size = scaling.get("block_size", 0)
    scale_dtype = scaling.get("scale_dtype")
    same_family = (is_fp8(a_fmt) and is_fp8(b_fmt)) or (
        is_int8s(a_fmt) and is_int8s(b_fmt)
    )
    if a.ndim != 2:
        raise NotImplementedError("the ragged-N layout (3D x 2D) is not supported")
    ragged_k = b.ndim == 2

    # ragged K: the contraction is ragged, so both operands need the mapping, and A
    # is quantized in the (ragged,K) orientation — only there is the reduction ragged
    # — then transposed back below with its scale. ragged M: the contraction is dense,
    # so only tensorwise has an amax wide enough to span groups (rowwise/blockwise are
    # already per row).
    src_a = a.mT if ragged_k else a
    contract_a = 0 if ragged_k else -1
    contract_b = 0 if ragged_k else 1
    ragged = (
        ragged_scale_blocks(offs, src_a.shape[0], block_size)
        if ragged_k or granularity == "tensorwise"
        else None
    )
    # the mapping only reaches the contraction axis — and so B, and the dequantized
    # fallback — when the contraction is the ragged one
    contract_ragged = ragged if ragged_k else None

    a_snap = b_snap = None
    aq = sa = bq = sb = None
    if is_quantized(a_fmt):
        aq, sa = quantize_operand(
            src_a,
            contract_a,
            granularity,
            block_size,
            a_fmt,
            scale_dtype=scale_dtype,
            ragged=ragged,
        )
        if enable_snapshot:
            a_snap = QuantizationSnapshot(
                src_a, aq, sa, contract_a, block_size, offs, a_fmt
            )
    if is_quantized(b_fmt):
        bq, sb = _quantize_b(
            b, granularity, block_size, b_fmt, scale_dtype, contract_ragged
        )
        if enable_snapshot:
            b_snap = QuantizationSnapshot(
                b, bq, sb, contract_b, block_size, offs, b_fmt
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
        )
        return y, a_snap, b_snap

    if aq is not None:
        src_a = dequantize_operand(
            aq, sa, contract_a, block_size, ragged=contract_ragged
        ).to(a.dtype)
    if bq is not None:
        b = dequantize_operand(
            bq, sb, contract_b, block_size, ragged=contract_ragged
        ).to(b.dtype)
    y = grouped_gemm(
        (src_a.mT if ragged_k else src_a).to(out_dtype),
        b.to(out_dtype),
        offs,
        bias=None if bias is None else bias.to(out_dtype),
    )
    return y.to(out_dtype), a_snap, b_snap


class ScaledGroupedGemmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, bias, offs, cfg: QuantizationConfig):
        out_dtype = a.dtype
        y, _, _ = quantized_grouped_gemm(
            a,
            b,
            offs,
            cfg.dtype["act"],
            cfg.dtype["weight"],
            out_dtype,
            cfg.scaling,
            bias=bias,
        )
        ctx.save_for_backward(a, b, offs)
        ctx.cfg = cfg
        ctx.bias_needs_grad = ctx.needs_input_grad[2]
        return y

    @staticmethod
    def backward(ctx, grad_y):
        a, b, offs = ctx.saved_tensors
        cfg = ctx.cfg
        out_dtype = a.dtype
        # dgrad: grad_a = grad_y @ b^T
        grad_a, _, _ = quantized_grouped_gemm(
            grad_y,
            b.transpose(-2, -1).contiguous(),
            offs,
            cfg.dtype["grad_input"],
            cfg.dtype["weight"],
            out_dtype,
            cfg.scaling,
        )
        # wgrad: grad_b[g] = a[g]^T @ grad_y[g] — the ragged token axis is the
        # contraction, i.e. the ragged-K layout
        grad_b, _, _ = quantized_grouped_gemm(
            a.mT,
            grad_y,
            offs,
            cfg.dtype["act"],
            cfg.dtype["grad_weight"],
            out_dtype,
            cfg.scaling,
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
        return grad_a, grad_b, grad_bias, None, None


def quantized_expert_mm(cfg: QuantizationConfig):

    def expert_mm(a, b, offs, bias=None, projection=None):
        return ScaledGroupedGemmFn.apply(a, b, bias, offs, cfg)

    return expert_mm


class QuantizedSparseMoEBlock(SparseMoEBlock):
    @classmethod
    def from_module(
        cls, module: SparseMoEBlock, quantization_config: QuantizationConfig
    ) -> "QuantizedSparseMoEBlock":
        q = cls.__new__(cls)
        q.__dict__ = copy.deepcopy(module).__dict__
        q.quantization_config = quantization_config
        q.expert_mm = quantized_expert_mm(quantization_config)
        return q
