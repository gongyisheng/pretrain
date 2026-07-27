from __future__ import annotations

import torch

_EPS = 1e-12


def quantize_int8(
    x: torch.Tensor, qmax: int, dim: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if dim is None:
        abs_max = x.detach().abs().max().float()
    else:
        abs_max = x.detach().abs().amax(dim=dim, keepdim=True).float()
    scale = (abs_max / qmax).clamp_min(_EPS)
    x_i8 = torch.round(x.float() / scale).clamp(-qmax, qmax).to(torch.int8)
    return x_i8, scale


def fake_quantize_int8(
    x: torch.Tensor, qmax: int, dim: int | None = None
) -> torch.Tensor:
    x_i8, scale = quantize_int8(x, qmax, dim=dim)
    return (x_i8.float() * scale).to(x.dtype)
