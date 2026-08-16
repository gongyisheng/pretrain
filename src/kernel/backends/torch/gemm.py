import torch
import torch.nn.functional as F

from src.kernel.ops.gemm import (
    can_implement_grouped_gemm_torch,
    can_implement_scaled_gemm_torch,
    can_implement_scaled_grouped_gemm_torch,
)
from src.kernel.registry import register_kernel
from src.kernel.utils import to_column_major


@register_kernel(
    op="gemm.grouped",
    backend="torch",
    can_implement=can_implement_grouped_gemm_torch,
    build="eager",
    autograd=False,
)
def grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    offs: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    out = F.grouped_mm(a, b, offs=offs, bias=None)
    if bias is None:
        return out
    if out.ndim == 3:
        return out + bias[:, None, :]
    rows = torch.arange(out.shape[0], device=offs.device)
    groups = torch.searchsorted(offs, rows, right=True)
    return out + bias[groups]


def _scaled_gemm_int8(aq, bq, sa, sb, out_dtype, block_size, bias):
    width = block_size or aq.shape[1]
    out = torch.zeros((aq.shape[0], bq.shape[1]), device=aq.device, dtype=torch.float32)
    for index, start in enumerate(range(0, aq.shape[1], width)):
        stop = min(start + width, aq.shape[1])
        partial = torch._int_mm(
            aq[:, start:stop].contiguous(), bq[start:stop].contiguous()
        )
        out += partial.float() * sa[:, index : index + 1] * sb[index : index + 1]
    if bias is not None:
        out += bias.float()
    return out.to(out_dtype)


def _scaled_gemm_fp8(aq, bq, sa, sb, out_dtype, block_size, bias):
    width = block_size or aq.shape[1]
    out = torch.zeros((aq.shape[0], bq.shape[1]), device=aq.device, dtype=torch.float32)
    for index, start in enumerate(range(0, aq.shape[1], width)):
        stop = min(start + width, aq.shape[1])
        partial = F.scaled_mm(
            aq[:, start:stop].contiguous(),
            to_column_major(bq[start:stop]),
            sa[:, index : index + 1].contiguous(),
            F.ScalingType.RowWise,
            sb[index : index + 1],
            F.ScalingType.RowWise,
            bias=None,
            output_dtype=torch.bfloat16,
        )
        out += partial.float()
    if bias is not None:
        out += bias.float()
    return out.to(out_dtype)


def _to_blocked_scale(scale):
    rows, columns = scale.shape
    row_tiles = (rows + 127) // 128
    column_tiles = (columns + 3) // 4
    padded = torch.zeros(
        (row_tiles * 128, column_tiles * 4),
        device=scale.device,
        dtype=scale.dtype,
    )
    padded[:rows, :columns] = scale
    blocked = padded.view(row_tiles, 128, column_tiles, 4)
    blocked = blocked.permute(0, 2, 1, 3)
    return blocked.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _scaled_gemm_mxfp8(aq, bq, sa, sb, out_dtype, block_size, bias):
    repeat = block_size // 32
    groups = (aq.shape[1] + 31) // 32
    scale_a = sa.repeat_interleave(repeat, 1)[:, :groups]
    scale_b = sb.repeat_interleave(repeat, 0)[:groups]
    scale_a = _to_blocked_scale(scale_a.to(torch.float8_e8m0fnu))
    scale_b = _to_blocked_scale(scale_b.t().contiguous().to(torch.float8_e8m0fnu))
    recipe = F.ScalingType.BlockWise1x32
    swizzle = F.SwizzleType.SWIZZLE_32_4_4
    return F.scaled_mm(
        aq,
        to_column_major(bq),
        scale_a,
        recipe,
        scale_b,
        recipe,
        swizzle_a=swizzle,
        swizzle_b=swizzle,
        bias=bias,
        output_dtype=out_dtype,
    )


def _can_use_native_mxfp8(aq, bq, block_size, scale_dtype):
    return (
        aq.dtype == torch.float8_e4m3fn
        and bq.dtype == torch.float8_e4m3fn
        and scale_dtype == torch.float8_e8m0fnu
        and block_size in (32, 64, 128)
        and torch.cuda.get_device_capability(aq.device)[0] in (10, 12)
        and getattr(torch, "float8_e8m0fnu", None) is not None
        and getattr(getattr(F, "ScalingType", None), "BlockWise1x32", None) is not None
        and getattr(getattr(F, "SwizzleType", None), "SWIZZLE_32_4_4", None) is not None
    )


@register_kernel(
    op="gemm.scaled",
    backend="torch",
    can_implement=can_implement_scaled_gemm_torch,
    build="eager",
    autograd=False,
)
def scaled_gemm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    scale_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if aq.dtype == torch.int8 and bq.dtype == torch.int8:
        return _scaled_gemm_int8(aq, bq, sa, sb, out_dtype, block_size, bias)
    if _can_use_native_mxfp8(aq, bq, block_size, scale_dtype):
        return _scaled_gemm_mxfp8(aq, bq, sa, sb, out_dtype, block_size, bias)
    return _scaled_gemm_fp8(aq, bq, sa, sb, out_dtype, block_size, bias)


@register_kernel(
    op="gemm.scaled_grouped",
    backend="torch",
    can_implement=can_implement_scaled_grouped_gemm_torch,
    build="eager",
    autograd=False,
)
def scaled_grouped_gemm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    scale_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    del block_size, scale_dtype
    return F.scaled_grouped_mm(
        aq,
        to_column_major(bq),
        sa.squeeze(1),
        F.ScalingType.RowWise,
        sb.squeeze(1),
        F.ScalingType.RowWise,
        bias=bias,
        offs=offs,
        output_dtype=out_dtype,
    )
