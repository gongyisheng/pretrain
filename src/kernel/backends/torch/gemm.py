import torch
import torch.nn.functional as F

from src.kernel.registry import register_kernel
from src.kernel.spec import SupportResult


def _functional_available(name) -> SupportResult:
    if hasattr(F, name):
        return SupportResult(True)
    return SupportResult(False, f"torch.nn.functional.{name} is unavailable")


def _grouped_available() -> SupportResult:
    return _functional_available("grouped_mm")


def _scaled_available() -> SupportResult:
    return _functional_available("scaled_mm")


def _scaled_grouped_available() -> SupportResult:
    return _functional_available("scaled_grouped_mm")


def _is_aligned(x) -> bool:
    alignment = 16 // x.dtype.itemsize
    return all(stride == 1 or stride % alignment == 0 for stride in x.stride()[-2:])


def _is_row_major(x) -> bool:
    return x.is_contiguous() and x.stride(-1) == 1


def grouped_gemm_eligibility(args, kwargs) -> SupportResult:
    del kwargs
    a, b, offs, bias = args
    if bias is not None and bias.dtype != a.dtype:
        return SupportResult(False, "bias dtype must match operand dtype")
    if bias is not None:
        return SupportResult(False, "torch grouped GEMM does not support bias")
    if a.device.type != "cuda" or b.device.type != "cuda":
        return SupportResult(False, "torch grouped GEMM requires CUDA operands")
    if a.device != b.device or offs.device != a.device:
        return SupportResult(False, "operands and offsets must share one device")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        return SupportResult(False, "torch grouped GEMM requires bf16 operands")
    if offs.dtype != torch.int32:
        return SupportResult(False, "torch grouped GEMM requires int32 offsets")
    if a.ndim not in (2, 3) or b.ndim not in (2, 3):
        return SupportResult(False, "grouped GEMM operands must be 2D or 3D")
    if a.ndim == 3 and b.ndim == 3:
        return SupportResult(False, "3D x 3D has no ragged dimension")
    if not _is_aligned(a) or not _is_aligned(b):
        return SupportResult(False, "torch grouped GEMM requires 16-byte strides")
    return SupportResult(True)


@register_kernel(
    op="gemm.grouped",
    backend="torch",
    priority=50,
    availability=_grouped_available,
    eligibility=grouped_gemm_eligibility,
    build="eager",
    autograd=False,
)
def grouped_gemm(a, b, offs, bias=None):
    return F.grouped_mm(a, b, offs=offs, bias=bias)


def _scaled_common_eligibility(
    aq,
    bq,
    sa,
    sb,
    out_dtype,
    block_size,
    bias,
    scale_dtype,
) -> SupportResult:
    if aq.device.type != "cuda" or bq.device.type != "cuda":
        return SupportResult(False, "torch scaled GEMM requires CUDA operands")
    if torch.cuda.get_device_capability(aq.device) < (8, 9):
        return SupportResult(False, "torch scaled GEMM requires SM89 or newer")
    if aq.device != bq.device or sa.device != aq.device or sb.device != aq.device:
        return SupportResult(False, "operands and scales must share one device")
    if aq.dtype != torch.float8_e4m3fn or bq.dtype != torch.float8_e4m3fn:
        return SupportResult(False, "torch scaled GEMM requires fp8 e4m3 operands")
    if aq.ndim != 2 or bq.ndim != 2:
        return SupportResult(False, "torch scaled GEMM requires 2D operands")
    if not _is_row_major(aq):
        return SupportResult(False, "torch scaled GEMM requires row-major A")
    if aq.shape[1] != bq.shape[0]:
        return SupportResult(False, "operand contraction dimensions must match")
    if any(dim % 16 != 0 for dim in (aq.shape[0], aq.shape[1], bq.shape[1])):
        return SupportResult(
            False, "torch scaled GEMM dimensions must be multiples of 16"
        )
    if block_size != 0:
        return SupportResult(False, "torch scaled GEMM supports rowwise scales only")
    if scale_dtype not in (None, "fp32"):
        return SupportResult(False, "torch scaled GEMM requires fp32 scales")
    if sa.dtype != torch.float32 or sb.dtype != torch.float32:
        return SupportResult(False, "torch scaled GEMM requires float32 scale tensors")
    if sa.shape != (aq.shape[0], 1) or sb.shape != (1, bq.shape[1]):
        return SupportResult(False, "torch scaled GEMM requires rowwise scale shapes")
    if not sa.is_contiguous() or not sb.is_contiguous():
        return SupportResult(False, "torch scaled GEMM requires contiguous scales")
    if out_dtype not in (torch.bfloat16, torch.float16):
        return SupportResult(False, "torch scaled GEMM output must be bf16 or fp16")
    if bias is not None and bias.dtype != out_dtype:
        return SupportResult(False, "bias dtype must match output dtype")
    if bias is not None and (bias.shape != (bq.shape[1],) or not bias.is_contiguous()):
        return SupportResult(False, "bias must be a contiguous output-width vector")
    return SupportResult(True)


def scaled_gemm_eligibility(args, kwargs) -> SupportResult:
    del kwargs
    return _scaled_common_eligibility(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]
    )


def _column_major(x):
    if x.stride(0) == 1:
        return x
    return x.t().contiguous().t()


@register_kernel(
    op="gemm.scaled",
    backend="torch",
    priority=50,
    availability=_scaled_available,
    eligibility=scaled_gemm_eligibility,
    build="eager",
    autograd=False,
)
def scaled_gemm(
    aq,
    bq,
    sa,
    sb,
    out_dtype,
    block_size,
    bias=None,
    scale_dtype=None,
):
    del block_size, scale_dtype
    return F.scaled_mm(
        aq,
        _column_major(bq),
        sa,
        F.ScalingType.RowWise,
        sb,
        F.ScalingType.RowWise,
        bias=bias,
        output_dtype=out_dtype,
    )


def scaled_grouped_gemm_eligibility(args, kwargs) -> SupportResult:
    del kwargs
    aq, bq, sa, sb, offs, out_dtype, block_size, bias, scale_dtype = args
    if aq.device.type != "cuda" or bq.device.type != "cuda":
        return SupportResult(False, "torch scaled grouped GEMM requires CUDA operands")
    if aq.device != bq.device or sa.device != aq.device or sb.device != aq.device:
        return SupportResult(False, "operands and scales must share one device")
    if (
        offs.device != aq.device
        or offs.dtype != torch.int32
        or offs.ndim != 1
        or not offs.is_contiguous()
    ):
        return SupportResult(False, "offsets must be int32 on the operand device")
    if aq.dtype != torch.float8_e4m3fn or bq.dtype != torch.float8_e4m3fn:
        return SupportResult(
            False, "torch scaled grouped GEMM requires fp8 e4m3 operands"
        )
    if aq.ndim != 2 or bq.ndim != 3:
        return SupportResult(False, "torch scaled grouped GEMM supports ragged-M only")
    if not _is_row_major(aq):
        return SupportResult(False, "torch scaled grouped GEMM requires row-major A")
    if aq.shape[1] != bq.shape[1] or offs.numel() != bq.shape[0]:
        return SupportResult(False, "grouped operand dimensions must match offsets")
    if any(dim % 16 != 0 for dim in (aq.shape[1], bq.shape[2])):
        return SupportResult(
            False, "torch scaled grouped GEMM dimensions must be multiples of 16"
        )
    if block_size != 0:
        return SupportResult(
            False, "torch scaled grouped GEMM supports rowwise scales only"
        )
    if scale_dtype not in (None, "fp32"):
        return SupportResult(False, "torch scaled grouped GEMM requires fp32 scales")
    if sa.dtype != torch.float32 or sb.dtype != torch.float32:
        return SupportResult(False, "torch scaled grouped GEMM requires float32 scales")
    if sa.shape != (aq.shape[0], 1) or sb.shape != (
        bq.shape[0],
        1,
        bq.shape[2],
    ):
        return SupportResult(False, "torch scaled grouped GEMM requires rowwise scales")
    if not sa.is_contiguous() or not sb.is_contiguous():
        return SupportResult(
            False, "torch scaled grouped GEMM requires contiguous scales"
        )
    if out_dtype not in (torch.bfloat16, torch.float16):
        return SupportResult(False, "torch scaled grouped output must be bf16 or fp16")
    if bias is not None:
        return SupportResult(False, "torch scaled grouped GEMM does not support bias")
    capability = torch.cuda.get_device_capability(aq.device)
    if capability[0] not in (9, 10):
        return SupportResult(False, "torch scaled grouped GEMM requires SM90 or SM100")
    return SupportResult(True)


def _group_column_major(x):
    if x.stride(-2) == 1:
        return x
    return x.transpose(-1, -2).contiguous().transpose(-1, -2)


@register_kernel(
    op="gemm.scaled_grouped",
    backend="torch",
    priority=50,
    availability=_scaled_grouped_available,
    eligibility=scaled_grouped_gemm_eligibility,
    build="eager",
    autograd=False,
)
def scaled_grouped_gemm(
    aq,
    bq,
    sa,
    sb,
    offs,
    out_dtype,
    block_size,
    bias=None,
    scale_dtype=None,
):
    del block_size, scale_dtype
    return F.scaled_grouped_mm(
        aq,
        _group_column_major(bq),
        sa,
        F.ScalingType.RowWise,
        sb,
        F.ScalingType.RowWise,
        bias=bias,
        offs=offs,
        output_dtype=out_dtype,
    )
