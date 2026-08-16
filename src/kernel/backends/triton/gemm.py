import functools
from collections.abc import Mapping
from typing import Any

import torch
import triton
import triton.language as tl
from torch._library.triton import triton_op, wrap_triton

from src.kernel.registry import register_kernel
from src.kernel.spec import CheckResult
from src.kernel.utils import (
    check_compute_capability_at_least,
    check_contiguous,
    check_cuda_tensors,
    check_dimension_sizes_match,
    check_dtypes,
    check_ndim,
    check_nonempty,
    check_same_device,
    check_shape,
    check_values,
    first_failure,
)

# ---------------------------------------------------------------------------
# Config + helpers
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _num_sms(device: torch.device | None = None) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


# ---------------------------------------------------------------------------
# Grouped GEMM (bf16 ragged MoE): ragged over M, N, or K
# ---------------------------------------------------------------------------


_GROUPED_CFG = [
    (64, 32, 32, 4, 3),
    (64, 64, 32, 4, 3),
    (64, 128, 32, 4, 3),
    (64, 256, 32, 8, 3),
    (128, 64, 32, 4, 4),
    (128, 128, 32, 8, 4),
    (32, 64, 64, 4, 3),
    (32, 128, 64, 4, 3),
    (64, 64, 64, 4, 4),
    (64, 128, 64, 8, 4),
    (64, 256, 64, 8, 4),
    (128, 64, 64, 8, 4),
    (128, 128, 64, 8, 3),
    (128, 256, 64, 8, 3),
    (256, 128, 64, 8, 3),
]

_GROUPED_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
        num_warps=w,
        num_stages=s,
    )
    for (bm, bn, bk, w, s) in _GROUPED_CFG
]


@triton.autotune(configs=_GROUPED_CONFIGS, key=["M", "N", "K", "A_IS_2D", "B_IS_2D"])
@triton.jit
def _grouped_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    offs_ptr,
    G,
    M,
    N,
    K,
    stride_ag,
    stride_am,
    stride_ak,
    stride_bg,
    stride_bk,
    stride_bn,
    stride_cg,
    stride_cm,
    stride_cn,
    stride_biasg,
    stride_biasn,
    NUM_SMS: tl.constexpr,
    A_IS_2D: tl.constexpr,
    B_IS_2D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Ragged grouped GEMM over any one of M, N, K, with an optional (G,N) bias.

    Which dim `offs` partitions follows from the operand ranks, as in
    torch._grouped_mm. The tile grid is always (M, N) and the reduction always over
    K, so one tile body serves all three layouts. The ragged dim's extent is passed
    as 0 (its bounds come from `offs`) to keep the autotune key stable.

    The bias is per group, broadcast over the output's row dim -- a superset of
    torch._grouped_mm, which rejects bias.
    """
    M_VARY: tl.constexpr = A_IS_2D and not B_IS_2D
    N_VARY: tl.constexpr = not A_IS_2D and B_IS_2D
    K_VARY: tl.constexpr = A_IS_2D and B_IS_2D

    # this persistent program owns pid, pid+NUM_SMS, ...
    global_tile_idx = tl.program_id(0)
    group_tile_start = 0  # flat index of the current group's first tile
    m_end = 0  # offs are END-offsets, so the previous end is the next start
    n_end = 0
    k_end = 0
    for g in range(G):
        if M_VARY:
            m_start = m_end
            m_end = tl.load(offs_ptr + g)
            m_size = m_end - m_start
        else:
            m_start = 0
            m_size = M
        if N_VARY:
            n_start = n_end
            n_end = tl.load(offs_ptr + g)
            n_size = n_end - n_start
        else:
            n_start = 0
            n_size = N
        if K_VARY:
            k_start = k_end
            k_end = tl.load(offs_ptr + g)
            k_size = k_end - k_start
        else:
            k_start = 0
            k_size = K

        num_n_tiles = tl.cdiv(n_size, BLOCK_N)
        num_m_tiles = tl.cdiv(m_size, BLOCK_M)
        group_tile_count = num_m_tiles * num_n_tiles
        # an empty ragged M/N group has no tiles; an empty ragged K group still
        # stores its zero slice
        while global_tile_idx < group_tile_start + group_tile_count:
            group_tile_idx = global_tile_idx - group_tile_start
            tile_m = group_tile_idx // num_n_tiles
            tile_n = group_tile_idx % num_n_tiles
            offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
            m_mask = offs_m < m_size
            n_mask = offs_n < n_size
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, k_size, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                k_mask = offs_k < k_size
                a_ptrs = (
                    a_ptr
                    + (0 if A_IS_2D else g * stride_ag)
                    + (m_start + offs_m)[:, None] * stride_am
                    + (k_start + offs_k)[None, :] * stride_ak
                )
                b_ptrs = (
                    b_ptr
                    + (0 if B_IS_2D else g * stride_bg)
                    + (k_start + offs_k)[:, None] * stride_bk
                    + (n_start + offs_n)[None, :] * stride_bn
                )
                a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
                acc += tl.dot(a, b)
            if HAS_BIAS:
                acc += tl.load(
                    bias_ptr + g * stride_biasg + (n_start + offs_n) * stride_biasn,
                    mask=n_mask,
                    other=0.0,
                )[None, :]
            c_ptrs = (
                c_ptr
                + (g * stride_cg if K_VARY else 0)
                + (m_start + offs_m)[:, None] * stride_cm
                + (n_start + offs_n)[None, :] * stride_cn
            )
            tl.store(
                c_ptrs,
                acc.to(c_ptr.dtype.element_ty),
                mask=m_mask[:, None] & n_mask[None, :],
            )
            global_tile_idx += NUM_SMS
        group_tile_start += group_tile_count


@triton_op("jit_kernel::grouped_gemm", mutates_args={})
def _grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    offs: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    a_is_2d, b_is_2d = a.ndim == 2, b.ndim == 2
    G = offs.shape[0]
    # the ragged dim's extent is passed as 0: unused, and keeps the autotune key stable
    if a_is_2d and not b_is_2d:  # (M,K) x (G,K,N) -> (M,N), ragged M
        M, N, K = 0, b.shape[2], a.shape[1]
        output_size = (a.shape[0], N)
    elif a_is_2d and b_is_2d:  # (M,K) x (K,N) -> (G,M,N), ragged K
        M, N, K = a.shape[0], b.shape[1], 0
        output_size = (G, M, N)
    else:  # (G,M,K) x (K,N) -> (M,N), ragged N
        if bias is not None:
            raise NotImplementedError("bias is not supported for the ragged-N layout")
        M, N, K = a.shape[1], 0, a.shape[2]
        output_size = (M, b.shape[1])

    # pad the last dim to 16 bytes like torch._grouped_mm, so the output is itself a
    # legal torch._grouped_mm operand when backward feeds it back
    align = 16 // a.dtype.itemsize
    padded = (output_size[-1] + align - 1) // align * align
    stride = (M * padded, padded, 1) if len(output_size) == 3 else (padded, 1)
    c = torch.empty_strided(output_size, stride, device=a.device, dtype=a.dtype)
    num_sms = _num_sms(a.device)
    wrap_triton(_grouped_gemm_kernel)[(num_sms,)](
        a,
        b,
        c,
        bias,
        offs,
        G,
        M,
        N,
        K,
        0 if a_is_2d else a.stride(0),
        a.stride(-2),
        a.stride(-1),
        0 if b_is_2d else b.stride(0),
        b.stride(-2),
        b.stride(-1),
        c.stride(0) if c.ndim == 3 else 0,
        c.stride(-2),
        c.stride(-1),
        0 if bias is None else bias.stride(0),
        0 if bias is None else bias.stride(1),
        NUM_SMS=num_sms,
        A_IS_2D=a_is_2d,
        B_IS_2D=b_is_2d,
        HAS_BIAS=bias is not None,
    )
    return c


def can_implement_grouped_gemm(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    a, b, offs = args[:3]
    bias = args[3] if len(args) > 3 else kwargs.get("bias")
    if bias is not None:
        result = check_dtypes((bias,), frozenset({a.dtype}), "Triton grouped GEMM bias")
        if not result.ok:
            return result
    tensors = (a, b, offs) if bias is None else (a, b, offs, bias)
    result = first_failure(
        (
            check_same_device(tensors, "Triton grouped GEMM"),
            check_cuda_tensors(tensors, "Triton grouped GEMM"),
            check_dtypes(
                (a, b), frozenset({torch.bfloat16}), "Triton grouped GEMM operands"
            ),
            check_ndim(a, frozenset({2, 3}), "Triton grouped GEMM A"),
            check_ndim(b, frozenset({2, 3}), "Triton grouped GEMM B"),
            check_ndim(offs, frozenset({1}), "Triton grouped GEMM offsets"),
            check_contiguous(offs, "Triton grouped GEMM offsets"),
            check_dtypes(
                (offs,), frozenset({torch.int32}), "Triton grouped GEMM offsets"
            ),
            check_nonempty(offs, "Triton grouped GEMM offsets"),
        )
    )
    if not result.ok:
        return result
    result = check_compute_capability_at_least(a.device, (8, 0), "Triton grouped GEMM")
    if not result.ok:
        return result
    if a.ndim == 3 and b.ndim == 3:
        return CheckResult(False, "3D x 3D has no ragged dimension")
    result = check_dimension_sizes_match(
        ((a, -1), (b, -2)), "Triton grouped GEMM contraction dimensions"
    )
    if not result.ok:
        return result
    if a.ndim == 3:
        result = check_dimension_sizes_match(
            ((a, 0), (offs, 0)), "Triton grouped GEMM A group count"
        )
        if not result.ok:
            return result
    if b.ndim == 3:
        result = check_dimension_sizes_match(
            ((b, 0), (offs, 0)), "Triton grouped GEMM B group count"
        )
        if not result.ok:
            return result
    if bias is not None and a.ndim == 3:
        return CheckResult(False, "bias is not supported for the ragged-N layout")
    if bias is not None:
        result = check_shape(
            bias, (offs.numel(), b.shape[-1]), "Triton grouped GEMM bias"
        )
        if not result.ok:
            return result
    return CheckResult(True)


@register_kernel(
    op="gemm.grouped",
    backend="triton",
    priority=100,
    can_implement=can_implement_grouped_gemm,
    build="jit",
    autograd=False,
)
def grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    offs: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ragged grouped GEMM matching torch._grouped_mm's layout convention.

    Layouts, by which dim `offs` (1-D int32 cumulative end-offsets) partitions:
        (M,K) x (G,K,N) -> (M,N)    ragged M
        (M,K) x (K,N)   -> (G,M,N)  ragged K
        (G,M,K) x (K,N) -> (M,N)    ragged N

    `bias` is an optional (G,N) tensor broadcast over the output's row dim, fused
    into the Triton epilogue.
    """
    if bias is not None and a.ndim == 3:
        raise NotImplementedError("bias is not supported for the ragged-N layout")

    assert a.ndim in (2, 3) and b.ndim in (2, 3) and offs.ndim == 1
    assert a.ndim == 2 or b.ndim == 2, "3D x 3D has no ragged dim; use torch.bmm"
    assert a.shape[-1] == b.shape[-2], (
        f"contraction mismatch: a {a.shape[-1]}, b {b.shape[-2]}"
    )
    if a.ndim == 3:
        assert offs.shape[0] == a.shape[0], (
            f"offs len {offs.shape[0]} != G {a.shape[0]}"
        )
    if b.ndim == 3:
        assert offs.shape[0] == b.shape[0], (
            f"offs len {offs.shape[0]} != G {b.shape[0]}"
        )
    assert offs.dtype == torch.int32
    assert a.dtype == b.dtype, f"dtype mismatch: a {a.dtype}, b {b.dtype}"
    assert a.device == b.device == offs.device, "a, b, offs must share same device"

    if bias is not None:
        assert bias.shape == (offs.shape[0], b.shape[-1]), (
            f"bias must be (G,N)=({offs.shape[0]},{b.shape[-1]}), got {tuple(bias.shape)}"
        )
        assert bias.dtype == a.dtype, f"bias dtype {bias.dtype} != a {a.dtype}"

    return _grouped_gemm(a, b, offs, bias)


# ---------------------------------------------------------------------------
# Scaled GEMM (quantized): fp8 / int8, tensorwise|rowwise|blockwise scaling
# ---------------------------------------------------------------------------

_SCALED_CFG = [
    (32, 128, 32, 4, 4),
    (64, 128, 32, 4, 3),
    (128, 128, 32, 4, 3),
    (128, 128, 32, 8, 4),
    (64, 64, 64, 4, 3),
    (128, 128, 64, 4, 3),
    (128, 128, 64, 4, 4),
    (128, 256, 64, 8, 4),
    (256, 128, 64, 8, 3),
    (64, 128, 128, 4, 3),
    (128, 128, 128, 8, 3),
]
_SCALED_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
        num_warps=w,
        num_stages=s,
    )
    for (bm, bn, bk, w, s) in _SCALED_CFG
]


def _early_prune_scaled_configs(
    configs, named_args, SCALE_BLOCK_SIZE, HAS_BIAS, grid=None, warmup=None
):
    del named_args, HAS_BIAS, grid, warmup
    scale_block = SCALE_BLOCK_SIZE
    if not scale_block:
        return configs
    return [c for c in configs if c.kwargs["BLOCK_K"] <= max(32, scale_block)]


@triton.autotune(
    configs=_SCALED_CONFIGS,
    key=["M", "N", "K", "SCALE_BLOCK_SIZE"],
    prune_configs_by={"early_config_prune": _early_prune_scaled_configs},
)
@triton.jit
def _scaled_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sa_ptr,
    sb_ptr,
    bias_ptr,
    M,
    N,
    K,
    n_scale_blocks,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_sam,
    stride_sak,
    stride_sbk,
    stride_sbn,
    stride_biasn,
    SCALE_BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """C[m,n] = Σ_b sa[m,b] · sb[b,n] · (Σ_{k∈b} aq[m,k] · bq[k,n]) + bias[n]."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for scale_block_idx in range(n_scale_blocks):
        # BLOCK_K floors at 32 (fp8/int8 tl.dot rejects K < 32), so a 16-wide scale
        # block runs one 32-wide tile; block_end, not the trip count, keeps the extra
        # lanes out of this block's scale.
        block_end = tl.minimum(K, scale_block_idx * SCALE_BLOCK_SIZE + SCALE_BLOCK_SIZE)
        block_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_in_block in range(0, SCALE_BLOCK_SIZE, BLOCK_K):
            offs_k = (
                scale_block_idx * SCALE_BLOCK_SIZE + k_in_block + tl.arange(0, BLOCK_K)
            )
            k_mask = offs_k < block_end
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            block_acc += tl.dot(a, b).to(tl.float32)
        sa = tl.load(
            sa_ptr + offs_m * stride_sam + scale_block_idx * stride_sak,
            mask=m_mask,
            other=0.0,
        )
        sb = tl.load(
            sb_ptr + scale_block_idx * stride_sbk + offs_n * stride_sbn,
            mask=n_mask,
            other=0.0,
        )
        acc += sa[:, None] * block_acc * sb[None, :]
    # on acc, not block_acc: the bias is unscaled, so folding it into a per-block
    # partial would scale it and add it once per block
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n * stride_biasn, mask=n_mask, other=0.0)
        acc += bias[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :]
    )


_SCALED_MXFP8_CFG = [
    (64, 64, 64, 4, 3),
    (64, 128, 64, 4, 3),
    (64, 128, 128, 4, 3),
    (64, 256, 64, 4, 3),
    (128, 128, 64, 4, 3),
    (128, 128, 64, 8, 4),
    (128, 256, 64, 8, 4),
    (256, 128, 64, 8, 3),
]
_SCALED_MXFP8_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
        num_warps=w,
        num_stages=s,
    )
    for (bm, bn, bk, w, s) in _SCALED_MXFP8_CFG
]

# tl.dot_scaled names its operand format; the e8m0 scale width is fixed at 32
_MXFP8_FORMAT = {torch.float8_e4m3fn: "e4m3", torch.float8_e5m2: "e5m2"}
_MXFP8_BLOCK_SIZE = 32


@triton.autotune(configs=_SCALED_MXFP8_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _scaled_gemm_mxfp8_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sa_ptr,
    sb_ptr,
    bias_ptr,
    M,
    N,
    K,
    n_scale_blocks,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_sam,
    stride_sak,
    stride_sbk,
    stride_sbn,
    stride_biasn,
    A_FORMAT: tl.constexpr,
    B_FORMAT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C[m,n] = Σ_k sa[m,k/32] · sb[k/32,n] · aq[m,k] · bq[k,n] + bias[n], mxfp8 only.

    The scales never reach the accumulator here: `tl.dot_scaled` lowers to
    `mma.sync...kind::mxf8f6f4.block_scale` (SASS `QMMA.SF`), which applies the e8m0
    factors per 32-element group inside the MMA. That is why there is no scale-block
    loop -- and why this runs on the faster of the two fp8 tensor-core paths, ~2.4x
    the epilogue-scaling kernel next door on the same shapes.

    `sa`/`sb` are the e8m0 exponent bytes viewed as uint8. `sb` is read transposed
    through its strides -- the instruction wants B's scales indexed (N, K/32) while
    `quantize_operand` produces (K/32, N) -- which costs nothing on a tensor this small.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_sk = tl.arange(0, BLOCK_K // 32)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        # a masked-off scale rides along with a zeroed operand, so its value is moot
        sk = k0 // 32 + offs_sk
        sk_mask = sk < n_scale_blocks
        sa = tl.load(
            sa_ptr + offs_m[:, None] * stride_sam + sk[None, :] * stride_sak,
            mask=m_mask[:, None] & sk_mask[None, :],
            other=0,
        )
        sb = tl.load(
            sb_ptr + offs_n[:, None] * stride_sbn + sk[None, :] * stride_sbk,
            mask=n_mask[:, None] & sk_mask[None, :],
            other=0,
        )
        acc = tl.dot_scaled(a, sa, A_FORMAT, b, sb, B_FORMAT, acc=acc)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n * stride_biasn, mask=n_mask, other=0.0)[
            None, :
        ]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :]
    )


def _check_triton_quantized_operands(
    aq: torch.Tensor, bq: torch.Tensor, feature: str
) -> CheckResult:
    quantized_dtypes = frozenset({torch.int8, torch.float8_e4m3fn, torch.float8_e5m2})
    result = check_dtypes((aq, bq), quantized_dtypes, feature)
    if not result.ok:
        return result
    if aq.dtype.is_floating_point != bq.dtype.is_floating_point:
        return CheckResult(
            False, f"{feature} requires operands from the same dtype family"
        )
    minimum = (8, 9) if aq.dtype.is_floating_point else (8, 0)
    return check_compute_capability_at_least(aq.device, minimum, feature)


def can_implement_scaled_gemm(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    aq, bq, sa, sb, out_dtype, block_size = args[:6]
    bias = args[6] if len(args) > 6 else kwargs.get("bias")
    scale_dtype = args[7] if len(args) > 7 else kwargs.get("scale_dtype")
    result = check_values(
        (scale_dtype,),
        frozenset({None, "fp32", "fp8_e8m0"}),
        "Triton scaled GEMM scale_dtype",
    )
    if not result.ok:
        return result
    tensors = (aq, bq, sa, sb) + (() if bias is None else (bias,))
    result = first_failure(
        (
            check_same_device(tensors, "Triton scaled GEMM"),
            check_cuda_tensors(tensors, "Triton scaled GEMM"),
            check_ndim(aq, frozenset({2}), "Triton scaled GEMM A"),
            check_ndim(bq, frozenset({2}), "Triton scaled GEMM B"),
            check_ndim(sa, frozenset({2}), "Triton scaled GEMM A scale"),
            check_ndim(sb, frozenset({2}), "Triton scaled GEMM B scale"),
            check_dtypes(
                (sa, sb), frozenset({torch.float32}), "Triton scaled GEMM scales"
            ),
        )
    )
    if not result.ok:
        return result
    result = _check_triton_quantized_operands(aq, bq, "Triton scaled GEMM")
    if not result.ok:
        return result
    result = check_dimension_sizes_match(
        ((aq, -1), (bq, -2)), "Triton scaled GEMM contraction dimensions"
    )
    if not result.ok:
        return result
    if block_size < 0:
        return CheckResult(False, "block_size must be non-negative")
    nblocks = (aq.shape[1] + block_size - 1) // block_size if block_size else 1
    result = first_failure(
        (
            check_shape(sa, (aq.shape[0], nblocks), "Triton scaled GEMM A scale"),
            check_shape(sb, (nblocks, bq.shape[1]), "Triton scaled GEMM B scale"),
        )
    )
    if not result.ok:
        return result
    if nblocks > 1 and (block_size < 16 or block_size & (block_size - 1)):
        return CheckResult(False, "block_size must be a power of two >= 16")
    result = check_dtypes(
        (out_dtype,),
        frozenset({torch.float32, torch.float16, torch.bfloat16}),
        "Triton scaled GEMM output dtype",
    )
    if not result.ok:
        return result
    if bias is not None:
        result = first_failure(
            (
                check_shape(bias, (bq.shape[1],), "Triton scaled GEMM bias"),
                check_dtypes(
                    (bias,),
                    frozenset({torch.float32, torch.float16, torch.bfloat16}),
                    "Triton scaled GEMM bias",
                ),
            )
        )
        if not result.ok:
            return result
    return CheckResult(True)


@register_kernel(
    op="gemm.scaled",
    backend="triton",
    priority=100,
    can_implement=can_implement_scaled_gemm,
    build="jit",
    autograd=False,
)
@triton_op("jit_kernel::scaled_gemm", mutates_args={})
def scaled_gemm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None = None,
    scale_dtype: str | None = None,
) -> torch.Tensor:
    """Scaled GEMM, (M,K) x (K,N) -> (M,N), with per-scale-block accumulation.

    `scale_dtype` "fp8_e8m0" with fp8 operands and `block_size` any multiple of 32 is
    mxfp8: the hardware's block-scaled MMA reads a 32-wide scale vector natively, and a
    wider block is the same scale replicated across consecutive 32-wide groups, so it
    routes to `_scaled_gemm_mxfp8_kernel` too. Everything else -- a non-multiple-of-32
    width, fp32 scales, int8 -- keeps the epilogue-scaling kernel, which is where those
    recipes are fastest anyway.

    `block_size` 0 means one scale block spanning the whole contraction.
    `SCALE_BLOCK_SIZE` below is derived from `n_scale_blocks` rather than from that
    sentinel because a blockwise width >= K also collapses to one block, and would
    otherwise leave the kernel restarting its accumulator on a boundary K never
    reaches.

    `bias` is an optional (N,) tensor broadcast over the rows, added to the fp32
    accumulator in the epilogue so it never passes through the scales.

    Shapes come from `aq`, `bq` and `sa` alone; every other operand is read through
    those, so the asserts below are what stands between a mismatch and an
    out-of-bounds load. Device is not among them -- Triton's pointer check already
    rejects an operand on another device.
    """
    M, K = aq.shape
    N = bq.shape[1]
    assert sa.ndim == 2 and sb.ndim == 2, (
        f"sa/sb must be 2-D, got {sa.ndim}-D and {sb.ndim}-D"
    )
    n_scale_blocks = sa.shape[1]
    assert bq.shape[0] == K, f"contraction mismatch: aq {K}, bq {bq.shape[0]}"
    # e4m3 x e5m2 is deliberately allowed, fp8 x int8 is not: tl.dot needs one family
    assert aq.dtype.is_floating_point == bq.dtype.is_floating_point, (
        f"aq {aq.dtype} and bq {bq.dtype} are not the same family"
    )
    # a short sa would silently leave the tail of K out of the block loop
    expected_blocks = triton.cdiv(K, block_size) if block_size else 1
    assert n_scale_blocks == expected_blocks, (
        f"sa has {n_scale_blocks} scale blocks, block_size {block_size} over K={K} "
        f"needs {expected_blocks}"
    )
    assert sa.shape[0] == M and sb.shape[0] == n_scale_blocks and sb.shape[1] == N, (
        f"scales must be sa (M,B)=({M},{n_scale_blocks}) and "
        f"sb (B,N)=({n_scale_blocks},{N}), got {tuple(sa.shape)} and {tuple(sb.shape)}"
    )
    if bias is not None:
        assert bias.ndim == 1 and bias.shape[0] == N, (
            f"bias must be (N,)=({N},), got {tuple(bias.shape)}"
        )
    if n_scale_blocks > 1 and (block_size < 16 or block_size & (block_size - 1)):
        # block_size is the scale-block width, and BLOCK_K -- always a power of two --
        # tiles it. A width that is not one leaves every scale block ending on a
        # masked partial tile, and a width under 16 cannot fill even the narrowest
        # tile fp8/int8 tl.dot accepts.
        raise ValueError(
            f"block_size must be a power of two >= 16 when it tiles K, got {block_size}"
        )
    SCALE_BLOCK_SIZE = block_size if n_scale_blocks > 1 else K
    c = torch.empty((M, N), device=aq.device, dtype=out_dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    if (
        scale_dtype == "fp8_e8m0"
        # block_size 0 is the "one block spans all of K" sentinel, not a real width --
        # 0 % 32 == 0 in Python, so it must be excluded explicitly or it would collapse
        # rep_k to 0 below and hand the kernel a zero-width scale tensor.
        and block_size != 0
        and block_size % _MXFP8_BLOCK_SIZE == 0
        and aq.dtype in _MXFP8_FORMAT
        and bq.dtype in _MXFP8_FORMAT
    ):
        # The MMA's scale vector is fixed at 32. A wider scale block is the same scale
        # repeated across consecutive 32-wide groups, so replicating here reaches the
        # block-scaled MMA rather than falling back to epilogue scaling. block_size 32
        # leaves both repeats as no-ops.
        rep_k = block_size // _MXFP8_BLOCK_SIZE
        # ceil, not floor: the quantizer pads a trailing partial block whenever
        # block_size does not divide K, and sa/sb legitimately carry that column.
        n_mx = triton.cdiv(K, _MXFP8_BLOCK_SIZE)
        sa_mx = sa.repeat_interleave(rep_k, 1)[:, :n_mx]
        sb_mx = sb.repeat_interleave(rep_k, 0)[:n_mx]
        # e8m0 holds a bare exponent, so `_compute_scale` already produced exact powers
        # of two and this cast is lossless; the kernel wants those bytes, not floats.
        sa8 = sa_mx.contiguous().to(torch.float8_e8m0fnu).view(torch.uint8)
        sb8 = sb_mx.contiguous().to(torch.float8_e8m0fnu).view(torch.uint8)
        wrap_triton(_scaled_gemm_mxfp8_kernel)[grid](
            aq,
            bq,
            c,
            sa8,
            sb8,
            bias,
            M,
            N,
            K,
            n_mx,
            aq.stride(0),
            aq.stride(1),
            bq.stride(0),
            bq.stride(1),
            c.stride(0),
            c.stride(1),
            sa8.stride(0),
            sa8.stride(1),
            sb8.stride(0),
            sb8.stride(1),
            0 if bias is None else bias.stride(0),
            A_FORMAT=_MXFP8_FORMAT[aq.dtype],
            B_FORMAT=_MXFP8_FORMAT[bq.dtype],
            HAS_BIAS=bias is not None,
        )
        return c

    wrap_triton(_scaled_gemm_kernel)[grid](
        aq,
        bq,
        c,
        sa,
        sb,
        bias,
        M,
        N,
        K,
        n_scale_blocks,
        aq.stride(0),
        aq.stride(1),
        bq.stride(0),
        bq.stride(1),
        c.stride(0),
        c.stride(1),
        sa.stride(0),
        sa.stride(1),
        sb.stride(0),
        sb.stride(1),
        0 if bias is None else bias.stride(0),
        SCALE_BLOCK_SIZE=SCALE_BLOCK_SIZE,
        HAS_BIAS=bias is not None,
    )
    return c


# ---------------------------------------------------------------------------
# Scaled grouped GEMM (quantized ragged MoE): fp8 / int8 with block scaling
# ---------------------------------------------------------------------------


_SCALED_GROUPED_CFG = [
    (64, 32, 32, 4, 3),
    (64, 64, 32, 4, 3),
    (64, 128, 32, 4, 3),
    (64, 256, 32, 8, 3),
    (128, 64, 32, 4, 4),
    (128, 128, 32, 8, 2),
    (128, 128, 32, 8, 3),
    (128, 128, 32, 8, 4),
    (32, 64, 64, 4, 3),
    (32, 128, 64, 4, 3),
    (64, 64, 64, 4, 4),
    (64, 128, 64, 8, 4),
    (64, 256, 64, 8, 4),
    (128, 64, 64, 8, 4),
    (128, 128, 64, 8, 3),
    (128, 128, 64, 8, 4),
    (128, 256, 64, 8, 3),
    (256, 128, 64, 8, 3),
    (64, 64, 128, 4, 3),
    (64, 128, 128, 8, 4),
    (128, 64, 128, 8, 4),
    (128, 128, 128, 8, 3),
]

_SCALED_GROUPED_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s
    )
    for (bm, bn, bk, w, s) in _SCALED_GROUPED_CFG
]


def _early_prune_scaled_grouped_configs(
    configs,
    named_args,
    NUM_SMS,
    A_IS_2D,
    B_IS_2D,
    SCALE_BLOCK_SIZE,
    HAS_BIAS,
    grid=None,
    warmup=None,
):
    del named_args, NUM_SMS, A_IS_2D, B_IS_2D, HAS_BIAS, grid, warmup
    scale_block = SCALE_BLOCK_SIZE
    if not scale_block:
        return configs
    return [c for c in configs if c.kwargs["BLOCK_K"] <= max(32, scale_block)]


@triton.autotune(
    configs=_SCALED_GROUPED_CONFIGS,
    key=["M", "N", "K", "A_IS_2D", "B_IS_2D", "SCALE_BLOCK_SIZE"],
    prune_configs_by={"early_config_prune": _early_prune_scaled_grouped_configs},
)
@triton.jit
def _scaled_grouped_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sa_ptr,
    sb_ptr,
    bias_ptr,
    offs_ptr,
    G,
    M,
    N,
    K,
    stride_ag,
    stride_am,
    stride_ak,
    stride_bg,
    stride_bk,
    stride_bn,
    stride_cg,
    stride_cm,
    stride_cn,
    stride_sag,
    stride_sam,
    stride_sak,
    stride_sbg,
    stride_sbk,
    stride_sbn,
    stride_biasg,
    stride_biasn,
    NUM_SMS: tl.constexpr,
    A_IS_2D: tl.constexpr,
    B_IS_2D: tl.constexpr,
    SCALE_BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Ragged scaled grouped GEMM over any one of M, N, K, with block scaling.

    Same layout model as _grouped_gemm_kernel. The quantized addition is
    `scale_block_idx`, the index along the contraction axis -- global when the
    contraction is dense, per-group re-tiled and carried across the group loop when
    the contraction is itself ragged.

    SCALE_BLOCK_SIZE is the scale block's width along the contraction axis; 0 means
    one block spanning the whole segment. row/tensorwise are that case and cannot
    state a width instead: a ragged contraction has a per-group segment length known
    only at runtime, while SCALE_BLOCK_SIZE is a constexpr. An empty group also owns
    one scale row, which cdiv(0, width) would skip, leaving the cursor a row behind
    for every group after it.

    The optional bias is per group, broadcast over the output's row dim, and added
    to the fp32 accumulator after the scales so it never passes through them.
    """
    M_VARY: tl.constexpr = A_IS_2D and not B_IS_2D
    N_VARY: tl.constexpr = not A_IS_2D and B_IS_2D
    K_VARY: tl.constexpr = A_IS_2D and B_IS_2D

    # this persistent program owns pid, pid+NUM_SMS, ...
    global_tile_idx = tl.program_id(0)
    group_tile_start = 0  # flat index of the current group's first tile
    m_end = 0  # offs are END-offsets, so the previous end is the next start
    n_end = 0
    k_end = 0
    scale_block_end = 0  # cursor, only advanced when the contraction is ragged
    for g in range(G):
        if M_VARY:
            m_start = m_end
            m_end = tl.load(offs_ptr + g)
            m_size = m_end - m_start
        else:
            m_start = 0
            m_size = M
        if N_VARY:
            n_start = n_end
            n_end = tl.load(offs_ptr + g)
            n_size = n_end - n_start
        else:
            n_start = 0
            n_size = N
        if K_VARY:
            k_start = k_end
            k_end = tl.load(offs_ptr + g)
            k_size = k_end - k_start
            # blocks re-tile inside each group, so the cursor carries across g; the
            # group loop visits g in order, so no prefix scan is needed
            scale_block_start = scale_block_end
            scale_block_end = scale_block_start + (
                1 if SCALE_BLOCK_SIZE == 0 else tl.cdiv(k_size, SCALE_BLOCK_SIZE)
            )
        else:
            k_start = 0
            k_size = K
            scale_block_start = 0
            scale_block_end = (
                1 if SCALE_BLOCK_SIZE == 0 else tl.cdiv(K, SCALE_BLOCK_SIZE)
            )

        num_n_tiles = tl.cdiv(n_size, BLOCK_N)
        num_m_tiles = tl.cdiv(m_size, BLOCK_M)
        group_tile_count = num_m_tiles * num_n_tiles
        while global_tile_idx < group_tile_start + group_tile_count:
            group_tile_idx = global_tile_idx - group_tile_start
            tile_m = group_tile_idx // num_n_tiles
            tile_n = group_tile_idx % num_n_tiles
            offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
            m_mask = offs_m < m_size
            n_mask = offs_n < n_size
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for scale_block_idx in range(scale_block_start, scale_block_end):
                if SCALE_BLOCK_SIZE == 0:
                    r0 = k_start
                    r1 = k_start + k_size
                else:
                    r0 = (
                        k_start
                        + (scale_block_idx - scale_block_start) * SCALE_BLOCK_SIZE
                    )
                    r1 = tl.minimum(r0 + SCALE_BLOCK_SIZE, k_start + k_size)
                # one scale block: accumulate in fp32, then apply both scales once
                block_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                # a computed trip count rather than `while k < r1`: Triton's software
                # pipeliner runs on scf.for, so a while loop would serialize the
                # global loads instead of overlapping them with the dots
                for step in range(tl.cdiv(r1 - r0, BLOCK_K)):
                    offs_k = r0 + step * BLOCK_K + tl.arange(0, BLOCK_K)  # absolute
                    k_mask = offs_k < r1
                    a_ptrs = (
                        a_ptr
                        + (0 if A_IS_2D else g * stride_ag)
                        + (m_start + offs_m)[:, None] * stride_am
                        + offs_k[None, :] * stride_ak
                    )
                    b_ptrs = (
                        b_ptr
                        + (0 if B_IS_2D else g * stride_bg)
                        + offs_k[:, None] * stride_bk
                        + (n_start + offs_n)[None, :] * stride_bn
                    )
                    a = tl.load(
                        a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0
                    )
                    b = tl.load(
                        b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0
                    )
                    block_acc += tl.dot(a, b).to(tl.float32)
                sa = tl.load(
                    sa_ptr
                    + (0 if A_IS_2D else g * stride_sag)
                    + (m_start + offs_m) * stride_sam
                    + scale_block_idx * stride_sak,
                    mask=m_mask,
                    other=0.0,
                )
                sb = tl.load(
                    sb_ptr
                    + (0 if B_IS_2D else g * stride_sbg)
                    + scale_block_idx * stride_sbk
                    + (n_start + offs_n) * stride_sbn,
                    mask=n_mask,
                    other=0.0,
                )
                acc += sa[:, None] * block_acc * sb[None, :]
            # on acc, not block_acc: the bias is unscaled, so folding it into a
            # per-block partial would scale it and add it once per block
            if HAS_BIAS:
                bias_ptrs = (
                    bias_ptr + g * stride_biasg + (n_start + offs_n) * stride_biasn
                )
                acc += tl.load(bias_ptrs, mask=n_mask, other=0.0)[None, :]
            c_ptrs = (
                c_ptr
                + (g * stride_cg if K_VARY else 0)
                + (m_start + offs_m)[:, None] * stride_cm
                + (n_start + offs_n)[None, :] * stride_cn
            )
            tl.store(
                c_ptrs,
                acc.to(c_ptr.dtype.element_ty),
                mask=m_mask[:, None] & n_mask[None, :],
            )
            global_tile_idx += NUM_SMS
        group_tile_start += group_tile_count


_SCALED_GROUPED_MXFP8_CFG = [
    (bm, bn, bk, w, s) for (bm, bn, bk, w, s) in _SCALED_GROUPED_CFG if bk >= 64
]
_SCALED_GROUPED_MXFP8_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s
    )
    for (bm, bn, bk, w, s) in _SCALED_GROUPED_MXFP8_CFG
]


@triton.autotune(
    configs=_SCALED_GROUPED_MXFP8_CONFIGS,
    key=["M", "N", "K", "A_IS_2D", "B_IS_2D"],
)
@triton.jit
def _scaled_grouped_gemm_mxfp8_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sa_ptr,
    sb_ptr,
    bias_ptr,
    offs_ptr,
    G,
    M,
    N,
    K,
    stride_ag,
    stride_am,
    stride_ak,
    stride_bg,
    stride_bk,
    stride_bn,
    stride_cg,
    stride_cm,
    stride_cn,
    stride_sag,
    stride_sam,
    stride_sak,
    stride_sbg,
    stride_sbk,
    stride_sbn,
    stride_biasg,
    stride_biasn,
    NUM_SMS: tl.constexpr,
    A_IS_2D: tl.constexpr,
    B_IS_2D: tl.constexpr,
    A_FORMAT: tl.constexpr,
    B_FORMAT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Ragged scaled grouped GEMM, mxfp8 only: same layout model as the kernel below,
    but the e8m0 scales go into the MMA (`QMMA.SF`) instead of the accumulator.

    The scale-block index is `scale_block_start + k0 // 32 + i` for the i-th 32-wide
    group of the tile. `scale_block_start` is 0 whenever the contraction is dense, and
    otherwise carries across the group loop exactly as it does next door -- a ragged
    contraction re-tiles its scale blocks inside each group, so group g's blocks begin
    where g-1's ended. BLOCK_K is a multiple of 32, so a tile's 32-wide groups never
    straddle a scale block.
    """
    M_VARY: tl.constexpr = A_IS_2D and not B_IS_2D
    N_VARY: tl.constexpr = not A_IS_2D and B_IS_2D
    K_VARY: tl.constexpr = A_IS_2D and B_IS_2D

    global_tile_idx = tl.program_id(0)
    group_tile_start = 0
    m_end = 0
    n_end = 0
    k_end = 0
    scale_block_end = 0
    offs_sk = tl.arange(0, BLOCK_K // 32)
    for g in range(G):
        if M_VARY:
            m_start = m_end
            m_end = tl.load(offs_ptr + g)
            m_size = m_end - m_start
        else:
            m_start = 0
            m_size = M
        if N_VARY:
            n_start = n_end
            n_end = tl.load(offs_ptr + g)
            n_size = n_end - n_start
        else:
            n_start = 0
            n_size = N
        if K_VARY:
            k_start = k_end
            k_end = tl.load(offs_ptr + g)
            k_size = k_end - k_start
            scale_block_start = scale_block_end
            scale_block_end = scale_block_start + tl.cdiv(k_size, 32)
        else:
            k_start = 0
            k_size = K
            scale_block_start = 0
            scale_block_end = tl.cdiv(K, 32)

        num_n_tiles = tl.cdiv(n_size, BLOCK_N)
        num_m_tiles = tl.cdiv(m_size, BLOCK_M)
        group_tile_count = num_m_tiles * num_n_tiles
        while global_tile_idx < group_tile_start + group_tile_count:
            group_tile_idx = global_tile_idx - group_tile_start
            tile_m = group_tile_idx // num_n_tiles
            tile_n = group_tile_idx % num_n_tiles
            offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
            m_mask = offs_m < m_size
            n_mask = offs_n < n_size
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, k_size, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                k_mask = offs_k < k_size
                a = tl.load(
                    a_ptr
                    + (0 if A_IS_2D else g * stride_ag)
                    + (m_start + offs_m)[:, None] * stride_am
                    + (k_start + offs_k)[None, :] * stride_ak,
                    mask=m_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                b = tl.load(
                    b_ptr
                    + (0 if B_IS_2D else g * stride_bg)
                    + (k_start + offs_k)[:, None] * stride_bk
                    + (n_start + offs_n)[None, :] * stride_bn,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )
                sbi = scale_block_start + k0 // 32 + offs_sk
                sk_mask = sbi < scale_block_end
                sa = tl.load(
                    sa_ptr
                    + (0 if A_IS_2D else g * stride_sag)
                    + (m_start + offs_m)[:, None] * stride_sam
                    + sbi[None, :] * stride_sak,
                    mask=m_mask[:, None] & sk_mask[None, :],
                    other=0,
                )
                sb = tl.load(
                    sb_ptr
                    + (0 if B_IS_2D else g * stride_sbg)
                    + (n_start + offs_n)[:, None] * stride_sbn
                    + sbi[None, :] * stride_sbk,
                    mask=n_mask[:, None] & sk_mask[None, :],
                    other=0,
                )
                acc = tl.dot_scaled(a, sa, A_FORMAT, b, sb, B_FORMAT, acc=acc)
            if HAS_BIAS:
                acc += tl.load(
                    bias_ptr + g * stride_biasg + (n_start + offs_n) * stride_biasn,
                    mask=n_mask,
                    other=0.0,
                )[None, :]
            c_ptrs = (
                c_ptr
                + (g * stride_cg if K_VARY else 0)
                + (m_start + offs_m)[:, None] * stride_cm
                + (n_start + offs_n)[None, :] * stride_cn
            )
            tl.store(
                c_ptrs,
                acc.to(c_ptr.dtype.element_ty),
                mask=m_mask[:, None] & n_mask[None, :],
            )
            global_tile_idx += NUM_SMS
        group_tile_start += group_tile_count


def can_implement_scaled_grouped_gemm(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    aq, bq, sa, sb, offs, out_dtype, block_size = args[:7]
    bias = args[7] if len(args) > 7 else kwargs.get("bias")
    scale_dtype = args[8] if len(args) > 8 else kwargs.get("scale_dtype")
    result = check_values(
        (scale_dtype,),
        frozenset({None, "fp32", "fp8_e8m0"}),
        "Triton scaled grouped GEMM scale_dtype",
    )
    if not result.ok:
        return result
    tensors = (aq, bq, sa, sb, offs) + (() if bias is None else (bias,))
    result = first_failure(
        (
            check_same_device(tensors, "Triton scaled grouped GEMM"),
            check_cuda_tensors(tensors, "Triton scaled grouped GEMM"),
            check_ndim(aq, frozenset({2, 3}), "Triton scaled grouped GEMM A"),
            check_ndim(bq, frozenset({2, 3}), "Triton scaled grouped GEMM B"),
            check_ndim(offs, frozenset({1}), "Triton scaled grouped GEMM offsets"),
            check_contiguous(offs, "Triton scaled grouped GEMM offsets"),
            check_dtypes(
                (sa, sb),
                frozenset({torch.float32}),
                "Triton scaled grouped GEMM scales",
            ),
        )
    )
    if not result.ok:
        return result
    result = _check_triton_quantized_operands(aq, bq, "Triton scaled grouped GEMM")
    if not result.ok:
        return result
    if aq.ndim == 3 and bq.ndim == 3:
        return CheckResult(False, "3D x 3D has no ragged dimension")
    result = first_failure(
        (
            check_dtypes(
                (offs,),
                frozenset({torch.int32}),
                "Triton scaled grouped GEMM offsets",
            ),
            check_nonempty(offs, "Triton scaled grouped GEMM offsets"),
        )
    )
    if not result.ok:
        return result
    result = check_dimension_sizes_match(
        ((aq, -1), (bq, -2)),
        "Triton scaled grouped GEMM contraction dimensions",
    )
    if not result.ok:
        return result
    if block_size < 0:
        return CheckResult(False, "block_size must be non-negative")
    if aq.ndim == 3:
        result = check_dimension_sizes_match(
            ((aq, 0), (offs, 0)), "Triton scaled grouped GEMM A group count"
        )
        if not result.ok:
            return result
    if bq.ndim == 3:
        result = check_dimension_sizes_match(
            ((bq, 0), (offs, 0)), "Triton scaled grouped GEMM B group count"
        )
        if not result.ok:
            return result

    if aq.ndim == 2 and bq.ndim == 3:
        nblocks = (aq.shape[1] + block_size - 1) // block_size if block_size else 1
        valid_scales = sa.shape == (aq.shape[0], nblocks) and sb.shape == (
            bq.shape[0],
            nblocks,
            bq.shape[2],
        )
    elif aq.ndim == 2 and bq.ndim == 2:
        valid_scales = (
            sa.ndim == 2
            and sb.ndim == 2
            and sa.shape[0] == aq.shape[0]
            and sa.shape[1] == sb.shape[0]
            and sb.shape[1] == bq.shape[1]
        )
    else:
        nblocks = (aq.shape[2] + block_size - 1) // block_size if block_size else 1
        valid_scales = sa.shape == (aq.shape[0], aq.shape[1], nblocks) and sb.shape == (
            nblocks,
            bq.shape[1],
        )
    if not valid_scales:
        return CheckResult(False, "invalid scale layout")
    if bias is not None and aq.ndim == 3:
        return CheckResult(False, "bias is not supported for the ragged-N layout")
    if bias is not None:
        result = first_failure(
            (
                check_shape(
                    bias,
                    (offs.shape[0], bq.shape[-1]),
                    "Triton scaled grouped GEMM bias",
                ),
                check_dtypes(
                    (bias,),
                    frozenset({torch.float32, torch.float16, torch.bfloat16}),
                    "Triton scaled grouped GEMM bias",
                ),
            )
        )
        if not result.ok:
            return result
    result = check_dtypes(
        (out_dtype,),
        frozenset({torch.float32, torch.float16, torch.bfloat16}),
        "Triton scaled grouped GEMM output dtype",
    )
    if not result.ok:
        return result
    return CheckResult(True)


@register_kernel(
    op="gemm.scaled_grouped",
    backend="triton",
    priority=100,
    can_implement=can_implement_scaled_grouped_gemm,
    build="jit",
    autograd=False,
)
@triton_op("jit_kernel::scaled_grouped_gemm", mutates_args={})
def scaled_grouped_gemm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None = None,
    scale_dtype: str | None = None,
) -> torch.Tensor:
    """Ragged scaled grouped GEMM, layout picked from the operand ranks.

    `scale_dtype` "fp8_e8m0" with fp8 operands routes to `_scaled_grouped_gemm_mxfp8_kernel`,
    which lets the MMA apply the scales, at `block_size` 32 always and any multiple of 32
    for the ragged-M/ragged-N layouts (see the ragged-K note below `is_ragged_k` in the
    body). Every other combination keeps the epilogue-scaling kernel -- see `scaled_gemm`.

    Layouts, by which dim `offs` (1-D int32 cumulative end-offsets) partitions:
        (M,K) x (E,K,N) -> (M,N)    ragged M
        (M,K) x (K,N)   -> (E,M,N)  ragged K
        (E,M,K) x (K,N) -> (M,N)    ragged N

    `sa`/`sb` mirror their operand's axis order with the contraction axis replaced by
    the scale-block axis, so transposing an operand transposes its scale too.
    `block_size` 0 means one scale block spanning the whole contraction segment.

    `bias` is an optional (E,N) tensor broadcast over the output's row dim, added to
    the fp32 accumulator in the epilogue so it never passes through the scales.
    """
    a_is_2d, b_is_2d = aq.ndim == 2, bq.ndim == 2
    E = offs.shape[0]
    if not a_is_2d and not b_is_2d:
        raise NotImplementedError("3D x 3D has no ragged dim; use torch.bmm")
    if bias is not None and not a_is_2d:
        raise NotImplementedError("bias is not supported for the ragged-N layout")
    # the ragged dim's extent is passed as 0: unused, and keeps the autotune key stable
    if a_is_2d and not b_is_2d:  # (M,K) x (E,K,N) -> (M,N), ragged M
        M, N, K = 0, bq.shape[2], aq.shape[1]
        output_size = (aq.shape[0], N)
    elif a_is_2d and b_is_2d:  # (M,K) x (K,N) -> (E,M,N), ragged K
        M, N, K = aq.shape[0], bq.shape[1], 0
        output_size = (E, M, N)
    else:  # (E,M,K) x (K,N) -> (M,N), ragged N
        M, N, K = aq.shape[1], 0, aq.shape[2]
        output_size = (M, bq.shape[1])

    c = torch.empty(output_size, device=aq.device, dtype=out_dtype)
    num_sms = _num_sms(aq.device)

    # The ragged-K layout (a_is_2d and b_is_2d) re-tiles its scale blocks *inside* each
    # group at runtime -- group g's blocks start where group g-1's ended, sized off
    # `offs` via cdiv(k_size_g, 32) -- so a flat repeat_interleave over the whole
    # scale-block axis would smear a replicated scale across a group boundary. Keep
    # that layout locked to the kernel's native 32-wide block; the other two layouts
    # (ragged M, ragged N) have a single non-ragged K shared by every group, where one
    # repeat_interleave over the whole axis is exact, so they get the generalized
    # multiple-of-32 condition. Do not simplify this to one shared condition.
    is_ragged_k = a_is_2d and b_is_2d
    mxfp8_ok = (
        scale_dtype == "fp8_e8m0"
        and aq.dtype in _MXFP8_FORMAT
        and bq.dtype in _MXFP8_FORMAT
        and (
            block_size == _MXFP8_BLOCK_SIZE
            if is_ragged_k
            # block_size 0 is the "one block spans the whole segment" sentinel, not a
            # real width -- 0 % 32 == 0 in Python, so it must be excluded explicitly or
            # it would collapse rep_k to 0 below and hand the kernel a zero-width
            # scale tensor.
            else block_size != 0 and block_size % _MXFP8_BLOCK_SIZE == 0
        )
    )
    if mxfp8_ok:
        if is_ragged_k:
            sa8 = sa.to(torch.float8_e8m0fnu).view(torch.uint8)
            sb8 = sb.to(torch.float8_e8m0fnu).view(torch.uint8)
        else:
            # Same replication trick as scaled_gemm: K is shared by every group here,
            # so one repeat_interleave over the whole scale-block axis is exact. The
            # axis is always last for sa and second-to-last for sb, in both the 2-D
            # and 3-D (E, ...) shapes, so this covers ragged M and ragged N alike.
            rep_k = block_size // _MXFP8_BLOCK_SIZE
            # ceil, not floor: the quantizer pads a trailing partial block whenever
            # block_size does not divide K, and sa/sb legitimately carry that column.
            n_mx = triton.cdiv(K, _MXFP8_BLOCK_SIZE)
            sa_mx = sa.repeat_interleave(rep_k, dim=-1).narrow(-1, 0, n_mx)
            sb_mx = sb.repeat_interleave(rep_k, dim=-2).narrow(-2, 0, n_mx)
            sa8 = sa_mx.contiguous().to(torch.float8_e8m0fnu).view(torch.uint8)
            sb8 = sb_mx.contiguous().to(torch.float8_e8m0fnu).view(torch.uint8)
        wrap_triton(_scaled_grouped_gemm_mxfp8_kernel)[(num_sms,)](
            aq,
            bq,
            c,
            sa8,
            sb8,
            bias,
            offs,
            E,
            M,
            N,
            K,
            0 if a_is_2d else aq.stride(0),
            aq.stride(-2),
            aq.stride(-1),
            0 if b_is_2d else bq.stride(0),
            bq.stride(-2),
            bq.stride(-1),
            c.stride(0) if c.ndim == 3 else 0,
            c.stride(-2),
            c.stride(-1),
            0 if a_is_2d else sa8.stride(0),
            sa8.stride(-2),
            sa8.stride(-1),
            0 if b_is_2d else sb8.stride(0),
            sb8.stride(-2),
            sb8.stride(-1),
            0 if bias is None else bias.stride(0),
            0 if bias is None else bias.stride(1),
            NUM_SMS=num_sms,
            A_IS_2D=a_is_2d,
            B_IS_2D=b_is_2d,
            A_FORMAT=_MXFP8_FORMAT[aq.dtype],
            B_FORMAT=_MXFP8_FORMAT[bq.dtype],
            HAS_BIAS=bias is not None,
        )
        return c

    wrap_triton(_scaled_grouped_gemm_kernel)[(num_sms,)](
        aq,
        bq,
        c,
        sa,
        sb,
        bias,
        offs,
        E,
        M,
        N,
        K,
        0 if a_is_2d else aq.stride(0),
        aq.stride(-2),
        aq.stride(-1),
        0 if b_is_2d else bq.stride(0),
        bq.stride(-2),
        bq.stride(-1),
        c.stride(0) if c.ndim == 3 else 0,
        c.stride(-2),
        c.stride(-1),
        0 if a_is_2d else sa.stride(0),
        sa.stride(-2),
        sa.stride(-1),
        0 if b_is_2d else sb.stride(0),
        sb.stride(-2),
        sb.stride(-1),
        0 if bias is None else bias.stride(0),
        0 if bias is None else bias.stride(1),
        NUM_SMS=num_sms,
        A_IS_2D=a_is_2d,
        B_IS_2D=b_is_2d,
        SCALE_BLOCK_SIZE=block_size,
        HAS_BIAS=bias is not None,
    )
    return c
