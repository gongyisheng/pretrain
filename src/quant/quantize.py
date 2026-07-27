from __future__ import annotations

import torch
import torch.nn.functional as F

from src.quant.utils import fmt_emax, fmt_qmax, fmt_store_dtype, is_int8s

_EPS = 1e-30
# E8M0 (bias 127) exponent range for the power-of-two (MX-style) scale.
_E8M0_EXP_MIN, _E8M0_EXP_MAX = -127, 127


def effective_block_size(granularity: str, block_size: int, K: int) -> int:
    """Elements per scale block along the contraction axis (K for row/tensorwise)."""
    return block_size if granularity == "blockwise" else K


def _amax_to_scale(
    amax: torch.Tensor, fmt: str, scale_dtype: str | None
) -> torch.Tensor:
    """fp32 dequant scale from per-block amax, dispatched by scale_dtype.

    The element format `fmt` supplies the alignment targets from the max map in
    utils: `emax` (top octave) for the power-of-two scale, `qmax` otherwise.

    "fp8_e8m0" -> power-of-two (MX-style) scale: align the block's amax to the
    element's top octave `emax`, snapped to an integer exponent E8M0 can store.
    Any other scale_dtype (fp32 / None) -> arbitrary-real amax/qmax scale.
    """
    if scale_dtype == "fp8_e8m0":
        exp = (torch.floor(torch.log2(amax.clamp_min(_EPS))) - fmt_emax(fmt)).clamp(
            _E8M0_EXP_MIN, _E8M0_EXP_MAX
        )
        return torch.exp2(exp)
    return (amax / fmt_qmax(fmt)).clamp_min(_EPS)


def _quantize_last(xf, K, bs, *, fmt, scale_dtype):
    """Blockwise quantize a (..., K) fp32 tensor along the last axis.

    Returns (xq (..., K) store_dtype, scale (..., nkb) fp32).
    """
    nkb = (K + bs - 1) // bs
    pad = nkb * bs - K
    xp = F.pad(xf, (0, pad)) if pad else xf
    xb = xp.reshape(*xp.shape[:-1], nkb, bs)  # (..., nkb, bs)
    amax = xb.abs().amax(dim=-1)  # (..., nkb)
    scale = _amax_to_scale(amax, fmt, scale_dtype)
    q = xb / scale.unsqueeze(-1)
    if is_int8s(fmt):
        q = torch.round(q)
    qmax = fmt_qmax(fmt)
    q = q.clamp(-qmax, qmax).to(fmt_store_dtype(fmt))
    xq = q.flatten(-2)[..., :K].contiguous()
    return xq, scale


def quantize_operand(
    x,
    contract_dim,
    granularity,
    block_size,
    fmt,
    *,
    scale_dtype=None,
):
    """Quantize 2D operand x for scaled_gemm along contract_dim (-1: A, 0: B).

    `fmt` is the quantized element format (fp8_e4m3 / int7 / ...); its qmax,
    emax, and storage dtype come from the max map in utils.

    Returns (xq store_dtype, scale fp32) canonical: A -> (M, nkb), B -> (nkb, N).
    """
    xf = x.movedim(contract_dim, -1).float().contiguous()  # (..., K)
    K = xf.shape[-1]
    if granularity == "tensorwise":
        amax = xf.abs().amax()
        scale = _amax_to_scale(amax, fmt, scale_dtype)  # 0-dim
        q = xf / scale
        if is_int8s(fmt):
            q = torch.round(q)
        qmax = fmt_qmax(fmt)
        q = q.clamp(-qmax, qmax).to(fmt_store_dtype(fmt))
        xq = q.movedim(-1, contract_dim).contiguous()
        # canonical broadcast: A -> (M,1), B -> (1,N)
        rows = xq.shape[0]
        cols = xq.shape[1]
        s = scale.reshape(1, 1)
        scale2d = s.expand(rows, 1) if contract_dim == -1 else s.expand(1, cols)
        return xq, scale2d.contiguous()

    bs = effective_block_size(granularity, block_size, K)
    xq_last, scale = _quantize_last(xf, K, bs, fmt=fmt, scale_dtype=scale_dtype)
    xq = xq_last.movedim(-1, contract_dim).contiguous()
    scale = scale.movedim(-1, contract_dim).contiguous()  # A:(M,nkb) B:(nkb,N)
    return xq, scale


def fake_quantize_operand(
    x,
    contract_dim,
    granularity,
    block_size,
    fmt,
    *,
    scale_dtype=None,
):
    """dequant(quant(x)) in x.dtype — CPU/mixed fallback and numeric oracle."""
    xq, scale = quantize_operand(
        x, contract_dim, granularity, block_size, fmt, scale_dtype=scale_dtype
    )
    qf = xq.movedim(contract_dim, -1).float()  # (..., K)
    K = qf.shape[-1]
    sf = scale.movedim(contract_dim, -1).float()  # (..., nkb)
    nkb = sf.shape[-1]
    # Recompute the exact block size quantize_operand used (not reverse-
    # engineered from nkb/K, which is ambiguous whenever the last block is
    # partial, e.g. K=100, block_size=32 -> nkb=4 but ceil(100/4)=25 != 32).
    bs = effective_block_size(granularity, block_size, K)
    pad = nkb * bs - K
    qp = F.pad(qf, (0, pad)) if pad else qf
    deq = (qp.reshape(*qp.shape[:-1], nkb, bs) * sf.unsqueeze(-1)).flatten(-2)[..., :K]
    return deq.movedim(-1, contract_dim).to(x.dtype).contiguous()
