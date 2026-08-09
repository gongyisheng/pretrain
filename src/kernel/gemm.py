import functools

import torch
import triton
import triton.language as tl
from torch._library.triton import triton_op, wrap_triton

# ---------------------------------------------------------------------------
# Config + helpers
# ---------------------------------------------------------------------------

_CFG = [
    (32, 64, 64, 4, 3),
    (64, 32, 32, 4, 3),
    (64, 64, 32, 4, 3),
    (64, 64, 64, 4, 4),
    (128, 64, 32, 4, 4),
    (128, 64, 64, 8, 4),
    (32, 128, 64, 4, 3),
    (64, 128, 32, 4, 3),
    (64, 128, 64, 8, 4),
    (128, 128, 32, 8, 4),
    (128, 128, 64, 8, 3),
    (64, 256, 32, 8, 3),
    (64, 256, 64, 8, 4),
    (128, 256, 64, 8, 3),
]

_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
        num_warps=w,
        num_stages=s,
    )
    for (bm, bn, bk, w, s) in _CFG
]


@functools.lru_cache(maxsize=None)
def _num_sms(device: torch.device | None = None) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K", "A_IS_2D", "B_IS_2D"])
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

    Which dim `offs` partitions follows from the operand ranks, exactly as in
    torch._grouped_mm. The tile grid is always (M, N) and the reduction always
    over K, so one tile body serves all three layouts; the extent of the ragged
    dim is passed as 0 (unused -- its bounds come from `offs`) to keep the
    autotune key stable as that dim changes size step to step.

    The bias is per group, broadcast over the output's row dim; torch._grouped_mm
    rejects bias entirely, so this is a deliberate superset of its behaviour.
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
        # process every tile of group g that this program owns. An empty ragged M or
        # N group has no tiles; an empty ragged K group still stores its zero slice.
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


# ---------------------------------------------------------------------------
# Torch wrappers / ops
# ---------------------------------------------------------------------------


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

    # allocate like torch._grouped_mm: last dim padded to a 16-byte boundary, so the
    # output is itself a legal torch._grouped_mm operand when backward feeds it back
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


def _grouped_gemm_setup_context(ctx, inputs, output):
    a, b, offs, _bias = inputs
    ctx.save_for_backward(a, b, offs)
    ctx.bias_needs_grad = ctx.needs_input_grad[3]


def _grouped_gemm_backward(ctx, grad_c):
    # each layout's dgrad/wgrad lands in another layout of the same set, so one
    # formula serves all three
    a, b, offs = ctx.saved_tensors
    grad_bias = None
    if ctx.bias_needs_grad:
        # bias is (G,N) broadcast over the output's row dim, so its grad sums that
        # dim. Both branches accumulate in fp32: over thousands of rows a bf16
        # accumulator drifts percent-level.
        if grad_c.ndim == 3:
            grad_bias = grad_c.sum(1, dtype=torch.float32).to(grad_c.dtype)
        else:
            # a segmented column-sum IS the ragged-K layout, with a row of ones as
            # the left operand -- reuses the kernel's fp32 accumulator and reads
            # grad_c once, where index_add_ would need an fp32 copy of it
            ones = grad_c.new_ones(1, grad_c.shape[0])
            grad_bias = _grouped_gemm(ones, grad_c, offs).squeeze(1)
    return (
        _grouped_gemm(grad_c, b.mT, offs),
        _grouped_gemm(a.mT, grad_c, offs),
        None,
        grad_bias,
    )


_grouped_gemm.register_autograd(
    _grouped_gemm_backward, setup_context=_grouped_gemm_setup_context
)


def grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    offs: torch.Tensor,
    bias: torch.Tensor | None = None,
    impl: str = "auto",
    **_,
) -> torch.Tensor:
    """Ragged grouped GEMM matching torch._grouped_mm's layout convention.

    Layouts, by which dim `offs` (1-D int32 cumulative end-offsets) partitions:
        (M,K) x (G,K,N) -> (M,N)    ragged M
        (M,K) x (K,N)   -> (G,M,N)  ragged K
        (G,M,K) x (K,N) -> (M,N)    ragged N

    `bias` is an optional (G,N) tensor broadcast over the output's row dim, fused
    into the Triton epilogue. It goes beyond torch._grouped_mm, which rejects bias,
    so `impl="torch"` adds it separately and unfused.

    Note `impl="torch"` additionally requires the non-unit stride of the last two
    dims to be a multiple of 16 bytes, which the Triton path does not — for the
    ragged-K layout with a contiguous source that means K % 8 == 0 in bf16.
    """
    if impl not in ("auto", "triton", "torch"):
        raise ValueError(f"impl must be auto|triton|torch, got {impl!r}")
    if bias is not None and a.ndim == 3:
        raise NotImplementedError("bias is not supported for the ragged-N layout")

    if impl == "auto":
        impl = (
            "triton"
            if (
                a.is_cuda
                and a.dtype == torch.bfloat16
                and torch.cuda.get_device_capability(a.device)[0] not in (9, 10)
            )
            else "torch"
        )

    if impl == "torch":
        out = torch._grouped_mm(a, b, offs=offs)
        if bias is not None:
            rows = torch.arange(out.shape[0], device=offs.device)
            out = out + (
                bias[:, None, :]
                if out.ndim == 3
                else bias[torch.searchsorted(offs, rows, right=True)]
            )
        return out

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

    if not (a.is_cuda and a.dtype == torch.bfloat16):
        raise ValueError("impl='triton' requires bf16 CUDA tensors")
    return _grouped_gemm(a, b, offs, bias)


# ---------------------------------------------------------------------------
# Scaled GEMM (quantized): fp8 / int8, tensorwise|rowwise|blockwise scaling
# ---------------------------------------------------------------------------

_SCALED_CFG = [
    (64, 64, 4, 3),
    (128, 64, 4, 4),
    (64, 128, 4, 3),
    (128, 128, 8, 4),
    (64, 256, 8, 3),
    (256, 64, 8, 3),
    (128, 256, 8, 4),
]
_SCALED_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
    for (bm, bn, w, s) in _SCALED_CFG
]

# _CONFIGS plus BLOCK_K=128 entries. The pre-merge forward derived
# BLOCK_K = min(128, next_pow2(K)) for tensorwise/rowwise, and _CONFIGS tops out at
# 64 -- without these the forward would lose its 128-wide reduction on the K=512-1408
# MoE shapes. Kept separate from _CONFIGS so the bf16 grouped GEMM's tuning space
# does not change.
_SCALED_GROUPED_CONFIGS = _CONFIGS + [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": 128}, num_warps=w, num_stages=s
    )
    for (bm, bn, w, s) in [
        (64, 64, 4, 3),
        (128, 64, 8, 4),
        (64, 128, 8, 4),
        (128, 128, 8, 3),
    ]
]


@triton.autotune(configs=_SCALED_CONFIGS, key=["N", "K", "SCALE_BLOCK_SIZE"])
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
        block_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_in_block in range(0, SCALE_BLOCK_SIZE, BLOCK_K):
            offs_k = (
                scale_block_idx * SCALE_BLOCK_SIZE + k_in_block + tl.arange(0, BLOCK_K)
            )
            k_mask = offs_k < K
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
    # on acc, not block_acc: the bias is unscaled and belongs to the whole row, so
    # folding it into a per-scale-block partial would scale it and add it once per block
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n * stride_biasn, mask=n_mask, other=0.0)
        acc += bias[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :]
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
) -> torch.Tensor:
    """Scaled GEMM, (M,K) x (K,N) -> (M,N), with per-scale-block accumulation.

    `block_size` follows the same convention as scaled_grouped_gemm: 0 means one
    scale block spanning the whole contraction. The branch tests `n_scale_blocks`
    rather than that sentinel because a blockwise width >= K also collapses to one
    block, and would otherwise set a BLOCK_K that tl.arange cannot take.

    `bias` is an optional (N,) tensor broadcast over the rows, added to the fp32
    accumulator in the epilogue so it never passes through the scales.
    """
    M, K = aq.shape
    N = bq.shape[1]
    n_scale_blocks = sa.shape[1]
    if n_scale_blocks > 1:
        SCALE_BLOCK_SIZE, BLOCK_K = block_size, block_size
    else:
        SCALE_BLOCK_SIZE, BLOCK_K = K, min(128, triton.next_power_of_2(K))
    c = torch.empty((M, N), device=aq.device, dtype=out_dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

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
        BLOCK_K=BLOCK_K,
    )
    return c


# ---------------------------------------------------------------------------
# Scaled grouped GEMM (quantized ragged MoE): fp8 / int8 with block scaling
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_SCALED_GROUPED_CONFIGS,
    key=["M", "N", "K", "A_IS_2D", "B_IS_2D", "SCALE_BLOCK_SIZE"],
)
@triton.jit
def _scaled_grouped_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sa_ptr,
    sb_ptr,
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
    NUM_SMS: tl.constexpr,
    A_IS_2D: tl.constexpr,
    B_IS_2D: tl.constexpr,
    SCALE_BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Ragged scaled grouped GEMM over any one of M, N, K, with block scaling.

    Same layout model as _grouped_gemm_kernel: the tile grid is (M, N) and the
    reduction is over K for every layout, so one tile body serves all three. The
    quantized addition is `scale_block_idx`, the index along the contraction axis --
    a global index when the contraction is dense, and a per-group re-tiled index
    carried across the group loop when the contraction is itself ragged.

    SCALE_BLOCK_SIZE is the scale block's width along the contraction axis; 0 means one
    block spanning the whole segment. row/tensorwise are that case, and cannot state
    the width numerically instead: when the contraction is the ragged axis the segment
    length is per group and only known at runtime, while SCALE_BLOCK_SIZE is a constexpr.
    An empty group also owns one scale row, which cdiv(0, width) would skip, leaving
    the cursor a row behind the scale buffer for every group after it.
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
            # blocks re-tile inside each group, so the cursor carries across g -- the
            # group loop already visits g in order, so no prefix scan is needed
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
                # pipeliner runs on scf.for, so a while loop here serializes the global
                # loads instead of overlapping them with the dots
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


@triton_op("jit_kernel::scaled_grouped_gemm", mutates_args={})
def scaled_grouped_gemm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
) -> torch.Tensor:
    """Ragged scaled grouped GEMM, layout picked from the operand ranks.

    Layouts, by which dim `offs` (1-D int32 cumulative end-offsets) partitions:
        (M,K) x (E,K,N) -> (M,N)    ragged M
        (M,K) x (K,N)   -> (E,M,N)  ragged K
        (E,M,K) x (K,N) -> (M,N)    ragged N

    `sa`/`sb` mirror their operand's axis order with the contraction axis replaced by
    the scale-block axis, so transposing an operand transposes its scale with it.
    `block_size` 0 means one scale block spanning the whole contraction segment.
    """
    a_is_2d, b_is_2d = aq.ndim == 2, bq.ndim == 2
    E = offs.shape[0]
    if not a_is_2d and not b_is_2d:
        raise NotImplementedError("3D x 3D has no ragged dim; use torch.bmm")
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
    wrap_triton(_scaled_grouped_gemm_kernel)[(num_sms,)](
        aq,
        bq,
        c,
        sa,
        sb,
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
        NUM_SMS=num_sms,
        A_IS_2D=a_is_2d,
        B_IS_2D=b_is_2d,
        SCALE_BLOCK_SIZE=block_size,
    )
    return c
