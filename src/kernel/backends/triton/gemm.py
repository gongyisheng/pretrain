import functools
import torch
import triton
import triton.language as tl
from torch._library.triton import triton_op, wrap_triton

from src.kernel.registry import register_kernel
from src.kernel.spec import cuda


@functools.lru_cache(maxsize=None)
def _num_sms(device: torch.device | None = None) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


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
def _grouped_mm_kernel(
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
    """Ragged grouped GEMM over one of M, N, or K, with optional (G,N) bias.

    Operand ranks select the partitioned dimension as in torch._grouped_mm. The tile
    grid is (M,N), reduction is K, and the ragged extent is 0 to stabilize autotuning.

    Bias is per group and broadcasts over output rows.
    """
    M_VARY: tl.constexpr = A_IS_2D and not B_IS_2D
    N_VARY: tl.constexpr = not A_IS_2D and B_IS_2D
    K_VARY: tl.constexpr = A_IS_2D and B_IS_2D

    # Persistent program: pid, pid + NUM_SMS, ...
    global_tile_idx = tl.program_id(0)
    group_tile_start = tl.zeros((), dtype=tl.int64)
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
        # Empty M/N groups have no tiles; empty K groups still store zeros.
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


@register_kernel(
    op="gemm.grouped_mm",
    backend="triton",
    build="jit",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(8, 0))}),
)
@triton_op("jit_kernel::grouped_mm", mutates_args={})
def grouped_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    offs: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ragged grouped GEMM using torch._grouped_mm's layout convention.

    Layouts, by which dim `offs` (1-D int32 cumulative end-offsets) partitions:
        (M,K) x (G,K,N) -> (M,N)    ragged M
        (M,K) x (K,N)   -> (G,M,N)  ragged K
        (G,M,K) x (K,N) -> (M,N)    ragged N

    `bias` is an optional (G,N) tensor broadcast over the output's row dim, fused
    into the Triton epilogue.

    Callers must supply bf16 or fp16 CUDA `a`/`b` on one device, a contiguous int32
    `offs` of length G, and no bias for the ragged-N layout.
    """
    a_is_2d, b_is_2d = a.ndim == 2, b.ndim == 2
    if not a_is_2d and not b_is_2d:
        raise NotImplementedError("3D x 3D not supported")
    G = offs.shape[0]
    # Pass the ragged extent as 0; it is unused and keeps the autotune key stable.
    if a_is_2d and not b_is_2d:  # (M,K) x (G,K,N) -> (M,N), ragged M
        M, N, K = 0, b.shape[2], a.shape[1]
        output_size = (a.shape[0], N)
    elif a_is_2d and b_is_2d:  # (M,K) x (K,N) -> (G,M,N), ragged K
        M, N, K = a.shape[0], b.shape[1], 0
        output_size = (G, M, N)
    else:  # (G,M,K) x (K,N) -> (M,N), ragged N
        if bias is not None:
            raise NotImplementedError("bias not supported for ragged-N")
        M, N, K = a.shape[1], 0, a.shape[2]
        output_size = (M, b.shape[1])

    # Match torch._grouped_mm's 16-byte trailing alignment for backward operands.
    align = 16 // a.dtype.itemsize
    padded = (output_size[-1] + align - 1) // align * align
    stride = (M * padded, padded, 1) if len(output_size) == 3 else (padded, 1)
    c = torch.empty_strided(output_size, stride, device=a.device, dtype=a.dtype)
    num_sms = _num_sms(a.device)
    wrap_triton(_grouped_mm_kernel)[(num_sms,)](
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
def _scaled_mm_kernel(
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
    """Scaled GEMM with per-block accumulation and optional bias."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for scale_block_idx in range(n_scale_blocks):
        # tl.dot requires K >= 32. A 16-wide scale block therefore uses one 32-wide
        # tile; block_end keeps its extra lanes out of this block's scale.
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
    # Bias is unscaled: apply it to acc, not each scaled block partial.
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
def _mxfp8_scaled_mm_kernel(
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
    """MXFP8 scaled GEMM; scales apply per 32-element K group, then bias is added.

    `tl.dot_scaled` applies e8m0 factors per 32-element group inside the MMA, so
    scales do not enter the accumulator or require a scale-block loop.

    `sa`/`sb` are e8m0 exponent bytes viewed as uint8. `sb` is read through transposed
    strides because the instruction indexes B scales as (N, K/32).
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
        # A masked scale multiplies a zeroed operand, so its value is irrelevant.
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


@register_kernel(
    op="gemm.int8_scaled_mm",
    backend="triton",
    build="jit",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(8, 0))}),
)
@register_kernel(
    op="gemm.fp8_scaled_mm",
    backend="triton",
    build="jit",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(8, 9))}),
)
@triton_op("jit_kernel::scaled_mm", mutates_args={})
def scaled_mm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled GEMM, (M,K) x (K,N) -> (M,N).

    `block_size` 0 means one scale block over K; nonzero multi-block widths must be
    powers of two at least 16. Optional `(N,)` bias is broadcast over rows and added
    after scaling. Inputs must be compatible CUDA tensors.
    """
    M, K = aq.shape
    N = bq.shape[1]
    n_scale_blocks = sa.shape[1]
    # This width tiles K below; invalid widths are not rejected by Triton but miscompute.
    if n_scale_blocks > 1 and (block_size < 16 or block_size & (block_size - 1) != 0):
        raise ValueError(
            f"block_size must be a power of two >= 16 when it tiles K, got {block_size}"
        )
    SCALE_BLOCK_SIZE = block_size if n_scale_blocks > 1 else K
    c = torch.empty((M, N), device=aq.device, dtype=out_dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    wrap_triton(_scaled_mm_kernel)[grid](
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


@register_kernel(
    op="gemm.mxfp8_scaled_mm",
    backend="triton",
    build="jit",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(10, 0))}),
)
@triton_op("jit_kernel::mxfp8_scaled_mm", mutates_args={})
def mxfp8_scaled_mm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """MXFP8 scaled GEMM, (M,K) x (K,N) -> (M,N).

    Inputs must be compatible CUDA tensors; `block_size` must be a nonzero multiple
    of 32.
    """
    # Zero means one block over K, not a width; reject it before modulo/replication.
    if block_size == 0 or block_size % _MXFP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"mxfp8 block_size must be a nonzero multiple of {_MXFP8_BLOCK_SIZE}, got {block_size}"
        )
    M, K = aq.shape
    N = bq.shape[1]
    c = torch.empty((M, N), device=aq.device, dtype=out_dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    # The MMA scale vector is 32 wide; replicate wider host blocks across 32-wide groups.
    rep_k = block_size // _MXFP8_BLOCK_SIZE
    # Use ceil: quantization pads and retains a trailing partial scale block.
    n_mx = triton.cdiv(K, _MXFP8_BLOCK_SIZE)
    sa_mx = sa.repeat_interleave(rep_k, 1)[:, :n_mx]
    sb_mx = sb.repeat_interleave(rep_k, 0)[:n_mx]
    # e8m0 stores a bare exponent; the kernel consumes its bytes, not float scales.
    sa8 = sa_mx.contiguous().to(torch.float8_e8m0fnu).view(torch.uint8)
    sb8 = sb_mx.contiguous().to(torch.float8_e8m0fnu).view(torch.uint8)
    wrap_triton(_mxfp8_scaled_mm_kernel)[grid](
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
def _scaled_grouped_mm_kernel(
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
    """Ragged grouped GEMM over one of M, N, or K, with block scaling.

    Layouts match `_grouped_mm_kernel`. `scale_block_idx` is global for dense K and
    re-tiled per group for ragged K, with a cursor carried across groups.

    `SCALE_BLOCK_SIZE` is the contraction-axis width; 0 means one block per segment.
    Ragged K cannot use that sentinel because segment lengths are runtime values. Empty
    groups still own one scale row; otherwise `cdiv(0, width)` leaves the cursor behind.

    Bias is per group, broadcasts over output rows, and is added after scaling.
    """
    M_VARY: tl.constexpr = A_IS_2D and not B_IS_2D
    N_VARY: tl.constexpr = not A_IS_2D and B_IS_2D
    K_VARY: tl.constexpr = A_IS_2D and B_IS_2D

    # Persistent program: pid, pid + NUM_SMS, ...
    global_tile_idx = tl.program_id(0)
    group_tile_start = tl.zeros((), dtype=tl.int64)
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
            # Blocks re-tile inside each group; the ordered cursor needs no prefix scan.
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
                # Accumulate one block in fp32, then apply both scales once.
                block_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                # Use a computed range: Triton's pipeliner overlaps loads with dots,
                # whereas a while loop would serialize them.
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
            # Bias is unscaled: apply it to acc, not each scaled block partial.
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
def _mxfp8_scaled_grouped_mm_kernel(
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
    SCALE_BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Ragged MXFP8 grouped GEMM; e8m0 scales enter the MMA (`QMMA.SF`).

    Each 32-wide MMA group uses `scale_block_start + (k0 + 32 * i) // SCALE_BLOCK_SIZE`.
    Ragged-K uses native blocks and carries the cursor across groups; dense-K uses 32,
    after host-side scale replication. Both widths are multiples of 32, so groups do
    not straddle scale blocks.
    """
    M_VARY: tl.constexpr = A_IS_2D and not B_IS_2D
    N_VARY: tl.constexpr = not A_IS_2D and B_IS_2D
    K_VARY: tl.constexpr = A_IS_2D and B_IS_2D

    global_tile_idx = tl.program_id(0)
    group_tile_start = tl.zeros((), dtype=tl.int64)
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
            scale_block_end = scale_block_start + tl.cdiv(k_size, SCALE_BLOCK_SIZE)
        else:
            k_start = 0
            k_size = K
            scale_block_start = 0
            scale_block_end = tl.cdiv(K, SCALE_BLOCK_SIZE)

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
                sbi = scale_block_start + (k0 + offs_sk * 32) // SCALE_BLOCK_SIZE
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


@register_kernel(
    op="gemm.mxfp8_scaled_grouped_mm",
    backend="triton",
    build="jit",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(10, 0))}),
)
@triton_op("jit_kernel::mxfp8_scaled_grouped_mm", mutates_args={})
def mxfp8_scaled_grouped_mm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ragged MXFP8 grouped GEMM; layout follows operand ranks.

    Layouts, by which dim `offs` (1-D int32 cumulative end-offsets) partitions:
        (M,K) x (E,K,N) -> (M,N)    ragged M
        (M,K) x (K,N)   -> (E,M,N)  ragged K
        (E,M,K) x (K,N) -> (M,N)    ragged N

    `sa`/`sb` mirror their operand axes with K replaced by the scale-block axis; a
    transposed operand must have its scale tensor transposed accordingly.

    `bias` is optional (E,N), broadcasts over output rows, and is added after scaling.

    Callers must supply compatible CUDA tensors. `block_size` may be any nonzero
    multiple of 32.
    """
    is_ragged_k = aq.ndim == 2 and bq.ndim == 2
    if block_size == 0 or block_size % _MXFP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"mxfp8 grouped GEMM requires block_size a nonzero multiple of {_MXFP8_BLOCK_SIZE}, got {block_size}"
        )
    a_is_2d, b_is_2d = aq.ndim == 2, bq.ndim == 2
    E = offs.shape[0]
    # Pass the ragged extent as 0; it is unused and keeps the autotune key stable.
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

    if is_ragged_k:
        # Ragged-K re-tiles blocks per group; global replication would misplace later
        # groups, so the kernel indexes native-width blocks directly.
        sa8 = sa.to(torch.float8_e8m0fnu).view(torch.uint8)
        sb8 = sb.to(torch.float8_e8m0fnu).view(torch.uint8)
    else:
        # K is shared by every group, so replication over each scale axis is exact for
        # ragged M and N.
        rep_k = block_size // _MXFP8_BLOCK_SIZE
        # Use ceil: quantization pads and retains a trailing partial scale block.
        n_mx = triton.cdiv(K, _MXFP8_BLOCK_SIZE)
        sa_mx = sa.repeat_interleave(rep_k, dim=-1).narrow(-1, 0, n_mx)
        sb_mx = sb.repeat_interleave(rep_k, dim=-2).narrow(-2, 0, n_mx)
        sa8 = sa_mx.contiguous().to(torch.float8_e8m0fnu).view(torch.uint8)
        sb8 = sb_mx.contiguous().to(torch.float8_e8m0fnu).view(torch.uint8)
    wrap_triton(_mxfp8_scaled_grouped_mm_kernel)[(num_sms,)](
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
        SCALE_BLOCK_SIZE=block_size if is_ragged_k else _MXFP8_BLOCK_SIZE,
    )
    return c


@register_kernel(
    op="gemm.int8_scaled_grouped_mm",
    backend="triton",
    build="jit",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(8, 0))}),
)
@register_kernel(
    op="gemm.fp8_scaled_grouped_mm",
    backend="triton",
    build="jit",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(8, 9))}),
)
@triton_op("jit_kernel::scaled_grouped_mm", mutates_args={})
def scaled_grouped_mm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ragged scaled grouped GEMM; layout follows operand ranks.

    Layouts, by which dim `offs` (1-D int32 cumulative end-offsets) partitions:
        (M,K) x (E,K,N) -> (M,N)    ragged M
        (M,K) x (K,N)   -> (E,M,N)  ragged K
        (E,M,K) x (K,N) -> (M,N)    ragged N

    `sa`/`sb` mirror their operand axes with K replaced by the scale-block axis; a
    transposed operand must have its scale tensor transposed accordingly.
    `block_size` 0 means one scale block spanning the whole contraction segment.

    `bias` is optional (E,N), broadcasts over output rows, and is added after scaling.

    Callers must supply compatible CUDA tensors. `block_size` may be zero for one
    scale block; nonzero widths must be powers of two of at least 16, matching dense
    `scaled_mm`.
    """
    if block_size != 0 and (block_size < 16 or block_size & (block_size - 1) != 0):
        raise ValueError(
            f"block_size must be 0 or a power of two >= 16, got {block_size}"
        )
    a_is_2d, b_is_2d = aq.ndim == 2, bq.ndim == 2
    if not a_is_2d and not b_is_2d:
        raise NotImplementedError("3D x 3D not supported")
    E = offs.shape[0]
    # Pass the ragged extent as 0; it is unused and keeps the autotune key stable.
    if a_is_2d and not b_is_2d:  # (M,K) x (E,K,N) -> (M,N), ragged M
        M, N, K = 0, bq.shape[2], aq.shape[1]
        output_size = (aq.shape[0], N)
    elif a_is_2d and b_is_2d:  # (M,K) x (K,N) -> (E,M,N), ragged K
        M, N, K = aq.shape[0], bq.shape[1], 0
        output_size = (E, M, N)
    else:  # (E,M,K) x (K,N) -> (M,N), ragged N
        if bias is not None:
            raise NotImplementedError("bias not supported for ragged-N")
        M, N, K = aq.shape[1], 0, aq.shape[2]
        output_size = (M, bq.shape[1])

    c = torch.empty(output_size, device=aq.device, dtype=out_dtype)
    num_sms = _num_sms(aq.device)

    wrap_triton(_scaled_grouped_mm_kernel)[(num_sms,)](
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
