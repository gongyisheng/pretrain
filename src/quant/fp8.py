from __future__ import annotations

import torch

_FP8_MAX = {dt: torch.finfo(dt).max for dt in (torch.float8_e4m3fn, torch.float8_e5m2)}
_EPS = 1e-12
HARDWARE_REQUIREMENT = (
    "a CUDA GPU with compute capability >= 8.9 (Ada/Hopper/Blackwell)"
)


def is_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


def quantize_fp8(
    x: torch.Tensor, fp8_dtype: torch.dtype, dim: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic fp8 quantization.

    dim=None -> per-tensor (0-dim scale). dim=i -> per-slice scale reduced over
    axis i (keepdim), broadcastable back over x. Returns (x_fp8, scale) with
    x ~= x_fp8.float() * scale.
    """
    fp8_max = _FP8_MAX[fp8_dtype]
    if dim is None:
        abs_max = x.detach().abs().max().float()
    else:
        abs_max = x.detach().abs().amax(dim=dim, keepdim=True).float()
    scale = (abs_max / fp8_max).clamp_min(_EPS)  # dequant scale
    x_fp8 = (x.float() / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return x_fp8, scale


def fp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    rowwise: bool = False,
) -> torch.Tensor:
    """Compute `a @ b` via torch._scaled_mm with fp8 scaling.

    a: (M, K) high precision, b: (K, N) high precision. Returns (M, N) in
    out_dtype. `a_dtype`/`b_dtype` pick the fp8 dtype each operand casts to; at
    least one must be e4m3 (cuBLAS rejects e5m2 x e5m2). `rowwise` scales a per
    row (M,1) and b per column (1,N); else a single per-tensor scale each.
    _scaled_mm requires the second operand column-major, so b is passed as a
    transposed contiguous view.
    """
    if rowwise:
        a_fp8, a_scale = quantize_fp8(a, a_dtype, dim=-1)  # (M, 1)
        b_fp8, b_scale = quantize_fp8(b, b_dtype, dim=0)  # (1, N)
    else:
        a_fp8, a_scale = quantize_fp8(a, a_dtype)
        b_fp8, b_scale = quantize_fp8(b, b_dtype)
    b_fp8_col_major = b_fp8.t().contiguous().t()  # (K, N) stored column-major
    return torch._scaled_mm(
        a_fp8.contiguous(),
        b_fp8_col_major,
        scale_a=a_scale,
        scale_b=b_scale,
        out_dtype=out_dtype,
    )


def fake_quantize_fp8(
    x: torch.Tensor, fp8_dtype: torch.dtype, dim: int | None = None
) -> torch.Tensor:
    """dequant(quant(x)) in x.dtype — numerical oracle for the real GEMM path.

    The dequant multiply runs in fp32 (accumulation dtype) to mirror
    torch._scaled_mm, which applies the fp32 scale to an fp32 accumulator;
    only the final value is cast back to x.dtype.
    """
    x_fp8, scale = quantize_fp8(x, fp8_dtype, dim=dim)
    return (x_fp8.float() * scale).to(x.dtype)
