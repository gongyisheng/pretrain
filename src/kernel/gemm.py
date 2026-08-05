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
    NUM_SMS: tl.constexpr,
    A_IS_2D: tl.constexpr,
    B_IS_2D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Ragged grouped GEMM over any one of M, N, K.

    Which dim `offs` partitions follows from the operand ranks, exactly as in
    torch._grouped_mm. The tile grid is always (M, N) and the reduction always
    over K, so one tile body serves all three layouts; the extent of the ragged
    dim is passed as 0 (unused -- its bounds come from `offs`) to keep the
    autotune key stable as that dim changes size step to step.
    """
    M_VARY: tl.constexpr = A_IS_2D and not B_IS_2D
    N_VARY: tl.constexpr = not A_IS_2D and B_IS_2D
    K_VARY: tl.constexpr = A_IS_2D and B_IS_2D

    tile_idx = tl.program_id(0)  # this persistent program owns pid, pid+NUM_SMS, ...
    last_end = 0  # running count of tiles before the current group
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
        num_tiles_g = tl.cdiv(m_size, BLOCK_M) * num_n_tiles
        # process every tile of group g that this program owns. An empty ragged M or
        # N group has no tiles; an empty ragged K group still stores its zero slice.
        while (tile_idx >= last_end) and (tile_idx < last_end + num_tiles_g):
            local = tile_idx - last_end
            tile_m = local // num_n_tiles
            tile_n = local % num_n_tiles
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
            tile_idx += NUM_SMS
        last_end += num_tiles_g


# ---------------------------------------------------------------------------
# Torch wrappers / ops
# ---------------------------------------------------------------------------


def _empty_like_grouped_mm(size, device, dtype):
    """Allocate like torch._grouped_mm: last dim padded to a 16-byte boundary."""
    align = 16 // dtype.itemsize
    padded = (size[-1] + align - 1) // align * align
    stride = (size[1] * padded, padded, 1) if len(size) == 3 else (padded, 1)
    return torch.empty_strided(size, stride, device=device, dtype=dtype)


@triton_op("jit_kernel::grouped_gemm", mutates_args={})
def _grouped_gemm(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    a_is_2d, b_is_2d = a.ndim == 2, b.ndim == 2
    G = offs.shape[0]
    # the ragged dim's extent is passed as 0: unused, and keeps the autotune key stable
    if a_is_2d and not b_is_2d:  # (M,K) x (G,K,N) -> (M,N), ragged M
        size, M, N, K = (a.shape[0], b.shape[2]), 0, b.shape[2], a.shape[1]
    elif a_is_2d and b_is_2d:  # (M,K) x (K,N) -> (G,M,N), ragged K
        size, M, N, K = (G, a.shape[0], b.shape[1]), a.shape[0], b.shape[1], 0
    else:  # (G,M,K) x (K,N) -> (M,N), ragged N
        size, M, N, K = (a.shape[1], b.shape[1]), a.shape[1], 0, a.shape[2]

    c = _empty_like_grouped_mm(size, a.device, a.dtype)
    num_sms = _num_sms(a.device)
    wrap_triton(_grouped_gemm_kernel)[(num_sms,)](
        a,
        b,
        c,
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
        NUM_SMS=num_sms,
        A_IS_2D=a_is_2d,
        B_IS_2D=b_is_2d,
    )
    return c


def _grouped_gemm_setup_context(ctx, inputs, output):
    a, b, offs = inputs
    ctx.save_for_backward(a, b, offs)


def _grouped_gemm_backward(ctx, grad_c):
    # each layout's grads land in another layout of the same set, so one formula
    # serves all three
    a, b, offs = ctx.saved_tensors
    return (
        _grouped_gemm(grad_c, b.mT, offs),
        _grouped_gemm(a.mT, grad_c, offs),
        None,
    )


_grouped_gemm.register_autograd(
    _grouped_gemm_backward, setup_context=_grouped_gemm_setup_context
)


def grouped_gemm(
    a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor, impl: str = "auto", **_
) -> torch.Tensor:
    """Ragged grouped GEMM matching torch._grouped_mm's layout convention.

    Layouts, by which dim `offs` (1-D int32 cumulative end-offsets) partitions:
        (M,K) x (G,K,N) -> (M,N)    ragged M
        (M,K) x (K,N)   -> (G,M,N)  ragged K
        (G,M,K) x (K,N) -> (M,N)    ragged N

    Note `impl="torch"` additionally requires the non-unit stride of the last two
    dims to be a multiple of 16 bytes, which the Triton path does not — for the
    ragged-K layout with a contiguous source that means K % 8 == 0 in bf16.
    """
    if impl not in ("auto", "triton", "torch"):
        raise ValueError(f"impl must be auto|triton|torch, got {impl!r}")

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
        return torch._grouped_mm(a, b, offs=offs)

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

    if not (a.is_cuda and a.dtype == torch.bfloat16):
        raise ValueError("impl='triton' requires bf16 CUDA tensors")
    return _grouped_gemm(a, b, offs)


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


@triton.autotune(configs=_SCALED_CONFIGS, key=["N", "K", "BLOCK_SIZE"])
@triton.jit
def _scaled_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sa_ptr,
    sb_ptr,
    M,
    N,
    K,
    nkb,
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
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(nkb):
        blk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kk in range(0, BLOCK_SIZE, BLOCK_K):
            offs_k = kb * BLOCK_SIZE + kk + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            blk += tl.dot(a, b).to(tl.float32)
        sa = tl.load(
            sa_ptr + offs_m * stride_sam + kb * stride_sak, mask=m_mask, other=0.0
        )
        sb = tl.load(
            sb_ptr + kb * stride_sbk + offs_n * stride_sbn, mask=n_mask, other=0.0
        )
        acc += sa[:, None] * blk * sb[None, :]
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
) -> torch.Tensor:
    M, K = aq.shape
    N = bq.shape[1]
    nkb = sa.shape[1]
    if nkb > 1:
        BLOCK_SIZE, BLOCK_K = block_size, block_size
    else:
        BLOCK_SIZE, BLOCK_K = K, min(128, triton.next_power_of_2(K))
    c = torch.empty((M, N), device=aq.device, dtype=out_dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    wrap_triton(_scaled_gemm_kernel)[grid](
        aq,
        bq,
        c,
        sa,
        sb,
        M,
        N,
        K,
        nkb,
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
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_K=BLOCK_K,
    )
    return c


# ---------------------------------------------------------------------------
# Scaled grouped GEMM (quantized ragged MoE): fp8 / int8 with block scaling
# ---------------------------------------------------------------------------


@triton.autotune(configs=_SCALED_CONFIGS, key=["N", "K", "BLOCK_SIZE"])
@triton.jit
def _scaled_grouped_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sa_ptr,
    sb_ptr,
    offs_ptr,
    E,
    N,
    K,
    nkb,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_sam,
    stride_sak,
    stride_sbe,
    stride_sbk,
    stride_sbn,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    tile_idx = pid  # this persistent program handles tiles pid, pid+NUM_SMS, ...
    last_end = 0  # running count of tiles before the current group
    m_start = 0  # first row of the current group (prev group's end-offset)
    for g in range(E):
        m_end = tl.load(offs_ptr + g)
        m_g = m_end - m_start
        num_m_tiles = tl.cdiv(m_g, BLOCK_M)
        num_tiles_g = num_m_tiles * num_n_tiles
        while (tile_idx >= last_end) and (tile_idx < last_end + num_tiles_g):
            local = tile_idx - last_end
            tile_m = local // num_n_tiles
            tile_n = local % num_n_tiles
            offs_m = m_start + tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
            m_mask = offs_m < m_end
            n_mask = offs_n < N
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kb in range(nkb):
                blk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for kk in range(0, BLOCK_SIZE, BLOCK_K):
                    offs_k = kb * BLOCK_SIZE + kk + tl.arange(0, BLOCK_K)
                    k_mask = offs_k < K
                    a_ptrs = (
                        a_ptr
                        + offs_m[:, None] * stride_am
                        + offs_k[None, :] * stride_ak
                    )
                    b_ptrs = (
                        b_ptr
                        + g * stride_be
                        + offs_k[:, None] * stride_bk
                        + offs_n[None, :] * stride_bn
                    )
                    a = tl.load(
                        a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0
                    )
                    b = tl.load(
                        b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0
                    )
                    blk += tl.dot(a, b).to(tl.float32)
                sa = tl.load(
                    sa_ptr + offs_m * stride_sam + kb * stride_sak,
                    mask=m_mask,
                    other=0.0,
                )
                sb = tl.load(
                    sb_ptr + g * stride_sbe + kb * stride_sbk + offs_n * stride_sbn,
                    mask=n_mask,
                    other=0.0,
                )
                acc += sa[:, None] * blk * sb[None, :]
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(
                c_ptrs,
                acc.to(c_ptr.dtype.element_ty),
                mask=m_mask[:, None] & n_mask[None, :],
            )
            tile_idx += NUM_SMS
        last_end += num_tiles_g
        m_start = m_end


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
    R, K = aq.shape
    E, _, N = bq.shape
    nkb = sa.shape[1]
    if nkb > 1:
        BLOCK_SIZE, BLOCK_K = block_size, block_size
    else:
        BLOCK_SIZE, BLOCK_K = K, min(128, triton.next_power_of_2(K))
    c = torch.empty((R, N), device=aq.device, dtype=out_dtype)
    num_sms = _num_sms(aq.device)
    wrap_triton(_scaled_grouped_gemm_kernel)[(num_sms,)](
        aq,
        bq,
        c,
        sa,
        sb,
        offs,
        E,
        N,
        K,
        nkb,
        aq.stride(0),
        aq.stride(1),
        bq.stride(0),
        bq.stride(1),
        bq.stride(2),
        c.stride(0),
        c.stride(1),
        sa.stride(0),
        sa.stride(1),
        sb.stride(0),
        sb.stride(1),
        sb.stride(2),
        NUM_SMS=num_sms,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_K=BLOCK_K,
    )
    return c


@triton.autotune(configs=_CONFIGS, key=["N", "K", "BLOCK_SIZE"])
@triton.jit
def _scaled_grouped_gemm_wgrad_kernel(
    a_ptr,
    g_ptr,
    gb_ptr,
    sa_ptr,
    sg_ptr,
    offs_ptr,
    E,
    N,
    K,
    stride_am,
    stride_ak,
    stride_gm,
    stride_gn,
    stride_ge,
    stride_gk,
    stride_gn2,
    stride_sam,
    stride_sak,
    stride_sgm,
    stride_sgn,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    tiles_per_group = num_k_tiles * num_n_tiles
    total_tiles = E * tiles_per_group
    # first_block[g] on the fly, mirroring ragged_scale_blocks: a group owns
    # ceil(rows / BLOCK_SIZE) scale rows, or exactly one when BLOCK_SIZE == 0 (even if
    # empty), so a block never straddles a group boundary. tile_id ascends, hence g
    # never decreases and the running total only advances as groups are reached --
    # O(E) scalar loads per program in total, no per-tile prefix sum.
    seen_groups = 0
    jb_start = 0
    for tile_id in range(pid, total_tiles, NUM_SMS):
        g = tile_id // tiles_per_group
        while seen_groups < g:
            done_end = tl.load(offs_ptr + seen_groups)
            done_start = tl.load(
                offs_ptr + seen_groups - 1, mask=seen_groups > 0, other=0
            )
            if BLOCK_SIZE == 0:
                jb_start += 1
            else:
                jb_start += tl.cdiv(done_end - done_start, BLOCK_SIZE)
            seen_groups += 1
        local = tile_id % tiles_per_group
        tile_k = local // num_n_tiles
        tile_n = local % num_n_tiles
        m_start = tl.load(offs_ptr + g - 1, mask=g > 0, other=0)
        m_end = tl.load(offs_ptr + g)
        offs_k = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_mask = offs_k < K
        n_mask = offs_n < N
        # jb_start now counts every earlier group's rows; group g's own count comes from
        # the row bounds loaded above. An empty blockwise group owns none, so the jb
        # loop is skipped and the tile falls through to the zero store.
        if BLOCK_SIZE == 0:
            jb_end = jb_start + 1
        else:
            jb_end = jb_start + tl.cdiv(m_end - m_start, BLOCK_SIZE)
        acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
        for jb in range(jb_start, jb_end):
            if BLOCK_SIZE == 0:
                m = m_start
                m_stop = m_end
            else:
                m = m_start + (jb - jb_start) * BLOCK_SIZE
                m_stop = tl.minimum(m + BLOCK_SIZE, m_end)
            # clamping to m_stop keeps an M tile inside one scale block and one expert
            blk = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
            while m < m_stop:
                offs_m = m + tl.arange(0, BLOCK_M)
                m_mask = offs_m < m_stop
                a_ptrs = (
                    a_ptr + offs_k[:, None] * stride_ak + offs_m[None, :] * stride_am
                )
                g_ptrs = (
                    g_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
                )
                a = tl.load(a_ptrs, mask=k_mask[:, None] & m_mask[None, :], other=0.0)
                gc = tl.load(g_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
                blk += tl.dot(a, gc).to(tl.float32)
                m += BLOCK_M
            sa = tl.load(
                sa_ptr + jb * stride_sam + offs_k * stride_sak, mask=k_mask, other=0.0
            )
            sg = tl.load(
                sg_ptr + jb * stride_sgm + offs_n * stride_sgn, mask=n_mask, other=0.0
            )
            acc += sa[:, None] * blk * sg[None, :]
        gb_ptrs = (
            gb_ptr
            + g * stride_ge
            + offs_k[:, None] * stride_gk
            + offs_n[None, :] * stride_gn2
        )
        tl.store(
            gb_ptrs,
            acc.to(gb_ptr.dtype.element_ty),
            mask=k_mask[:, None] & n_mask[None, :],
        )


@triton_op("jit_kernel::scaled_grouped_gemm_wgrad", mutates_args={})
def scaled_grouped_gemm_wgrad(
    aq: torch.Tensor,
    gq: torch.Tensor,
    sa: torch.Tensor,
    sg: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
) -> torch.Tensor:
    _, K = aq.shape
    N = gq.shape[1]
    E = offs.shape[0]
    grad_b = torch.empty((E, K, N), device=aq.device, dtype=out_dtype)
    num_sms = _num_sms(aq.device)
    assert sa.shape[0] == sg.shape[0], (
        f"scale block count mismatch: sa {sa.shape[0]}, sg {sg.shape[0]}"
    )
    wrap_triton(_scaled_grouped_gemm_wgrad_kernel)[(num_sms,)](
        aq,
        gq,
        grad_b,
        sa,
        sg,
        offs,
        E,
        N,
        K,
        aq.stride(0),
        aq.stride(1),
        gq.stride(0),
        gq.stride(1),
        grad_b.stride(0),
        grad_b.stride(1),
        grad_b.stride(2),
        sa.stride(0),
        sa.stride(1),
        sg.stride(0),
        sg.stride(1),
        NUM_SMS=num_sms,
        BLOCK_SIZE=block_size,
    )
    return grad_b
