from __future__ import annotations

import torch

_INT8_MAX = 127
_EPS = 1e-12
_warned_shape_fallback = False


def quantize_int8(
    x: torch.Tensor, dim: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if dim is None:
        abs_max = x.detach().abs().max().float()
    else:
        abs_max = x.detach().abs().amax(dim=dim, keepdim=True).float()
    scale = (abs_max / _INT8_MAX).clamp_min(_EPS)
    x_i8 = torch.round(x.float() / scale).clamp(-_INT8_MAX, _INT8_MAX).to(torch.int8)
    return x_i8, scale


def fake_quantize_int8(x: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    x_i8, scale = quantize_int8(x, dim=dim)
    return (x_i8.float() * scale).to(x.dtype)


def int8_gemm(a, b, out_dtype, rowwise=False):
    a_dim = -1 if rowwise else None
    b_dim = 0 if rowwise else None
    M, K = a.shape
    N = b.shape[1]
    # torch._int_mm needs CUDA and M > 16, K/N multiples of 8. For anything it
    # rejects, fall back to the fake-quant path (same numerics, no kernel).
    if not (a.is_cuda and M > 16 and K % 8 == 0 and N % 8 == 0):
        global _warned_shape_fallback
        if not _warned_shape_fallback:
            print(
                f"int8_gemm: (device={a.device}, M={M}, K={K}, N={N}) unsupported "
                "by torch._int_mm (needs CUDA, M>16, K/N multiples of 8); using "
                "the fake-quant fallback (no int8 kernel speedup). Warned once."
            )
            _warned_shape_fallback = True
        return (fake_quantize_int8(a, dim=a_dim) @ fake_quantize_int8(b, dim=b_dim)).to(
            out_dtype
        )

    a_i8, a_scale = quantize_int8(a, dim=a_dim)  # rowwise: (M,1); else scalar
    b_i8, b_scale = quantize_int8(b, dim=b_dim)  # rowwise: (1,N); else scalar
    b_i8_col_major = b_i8.t().contiguous().t()
    acc = torch._int_mm(a_i8.contiguous(), b_i8_col_major)
    return (acc.float() * a_scale * b_scale).to(out_dtype)
