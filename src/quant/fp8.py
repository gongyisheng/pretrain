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
