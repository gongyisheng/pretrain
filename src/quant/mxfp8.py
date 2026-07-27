from __future__ import annotations

import torch
import torch.nn.functional as F

MXFP8_BLOCK_SIZE = 32
# Largest normal exponent per fp8 element format; the shared E8M0 scale aligns a
# block's amax to this so the block's max value lands near the format's top.
_ELEM_EMAX = {torch.float8_e4m3fn: 8, torch.float8_e5m2: 15}
_FP8_MAX = {dt: torch.finfo(dt).max for dt in (torch.float8_e4m3fn, torch.float8_e5m2)}
# E8M0 (bias 127) represents 2^e for e in [-127, 127]; 255 is NaN.
_E8M0_EXP_MIN, _E8M0_EXP_MAX = -127, 127
_EPS = 1e-30

HARDWARE_REQUIREMENT = "a CUDA GPU with compute capability >= 10.0 (Blackwell)"


def is_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


def quantize_mxfp8(
    x: torch.Tensor, fp8_dtype: torch.dtype, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blockwise mxfp8 quantization along `dim`; returns (x_fp8, E8M0 scale)."""
    emax = _ELEM_EMAX[fp8_dtype]
    fp8_max = _FP8_MAX[fp8_dtype]
    xf = x.movedim(dim, -1).contiguous()
    k = xf.shape[-1]
    assert k % MXFP8_BLOCK_SIZE == 0, (
        f"mxfp8 block dim {k} is not a multiple of {MXFP8_BLOCK_SIZE}"
    )
    xb = xf.unflatten(-1, (k // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)).float()
    amax = xb.abs().amax(dim=-1)  # (..., n_blocks)
    exp = (torch.floor(torch.log2(amax.clamp_min(_EPS))) - emax).clamp(
        _E8M0_EXP_MIN, _E8M0_EXP_MAX
    )
    scale = torch.exp2(exp)  # dequant multiplier (exact power of two)
    xq = (xb / scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    xq = xq.flatten(-2).movedim(-1, dim).contiguous()
    scale_e8 = scale.movedim(-1, dim).to(torch.float8_e8m0fnu)
    return xq, scale_e8


def fake_quantize_mxfp8(
    x: torch.Tensor, fp8_dtype: torch.dtype, dim: int
) -> torch.Tensor:
    """dequant(quant(x)) in x.dtype — numerical oracle / CPU fallback for mxfp8_gemm."""
    xq, scale = quantize_mxfp8(x, fp8_dtype, dim)
    xf = xq.movedim(dim, -1).float()
    k = xf.shape[-1]
    sf = scale.movedim(dim, -1).float()
    xr = xf.unflatten(-1, (k // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)) * sf.unsqueeze(-1)
    return xr.flatten(-2).movedim(-1, dim).to(x.dtype)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def to_blocked(scale_2d: torch.Tensor) -> torch.Tensor:
    """Rearrange a (rows, n_blocks) E8M0 scale into the SWIZZLE_32_4_4 layout the MMA expects."""
    # From torch.testing._internal.common_quantized.to_blocked (no public helper).
    rows, cols = scale_2d.shape
    n_row_tiles, n_col_tiles = _ceil_div(rows, 128), _ceil_div(cols, 4)
    padded_rows, padded_cols = n_row_tiles * 128, n_col_tiles * 4
    padded = scale_2d
    if (rows, cols) != (padded_rows, padded_cols):  # F.pad lacks e8m0 support
        padded = torch.zeros(
            (padded_rows, padded_cols), device=scale_2d.device, dtype=scale_2d.dtype
        )
        padded[:rows, :cols] = scale_2d
    blocks = padded.view(n_row_tiles, 128, n_col_tiles, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def mxfp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
) -> torch.Tensor:
    """`a @ b` (M,K)@(K,N) via `F.scaled_mm` mxfp8 (BlockWise1x32, E8M0); e4m3 only, K auto-padded to a multiple of 32."""
    pad = (-a.shape[-1]) % MXFP8_BLOCK_SIZE
    if pad:
        a = F.pad(a, (0, pad))  # pad a's last dim (contraction)
        b = F.pad(b, (0, 0, 0, pad))  # pad b's first dim (contraction)
    a_fp8, a_scale = quantize_mxfp8(a, a_dtype, dim=-1)  # scale (M, K//32)
    b_fp8, b_scale = quantize_mxfp8(b, b_dtype, dim=0)  # scale (K//32, N)
    st = F.ScalingType.BlockWise1x32
    sw = F.SwizzleType.SWIZZLE_32_4_4
    return F.scaled_mm(
        a_fp8.contiguous(),
        b_fp8.t().contiguous().t(),  # (K, N) stored column-major
        to_blocked(a_scale),
        st,
        to_blocked(b_scale.t().contiguous()),  # (N, K//32) row-major before swizzle
        st,
        swizzle_a=sw,
        swizzle_b=sw,
        output_dtype=out_dtype,
    )
