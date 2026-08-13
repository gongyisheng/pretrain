"""CuTe DSL mxfp8 dense GEMM for sm120.

One CTA computes a 128x128 output tile by walking K in 128-wide steps: stage the
A/B tile and its scale factors in shared memory, pull them into registers, and run
the warp-level `m16n8k32` block-scaled MMA (`kind::mxf8`), which applies the e8m0
scales inside the tensor core. Accumulation is fp32; `out_dtype` is only reached in
the epilogue.

There is deliberately no pipelining here: the shared-memory stage is single-buffered
and fed by ordinary loads, not `cp.async`. That makes this slower than the Triton
mxfp8 kernel, which is the expected intermediate state -- this module exists to
establish a correct CuTe mainloop for the feed path to be built on.

Scale factors are held in shared memory in the cuBLASLt-compatible 32x4x4 swizzle
(`sm120_make_smem_layout_sfa`/`sfb`), the layout the MMA's scale-factor fragment
partitioning reads back.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as blackwell_helpers
import cutlass.utils.blockscaled_layout as blockscaled_layout
import torch
from cutlass.cute.nvgpu import warp as cute_warp
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import Float8E4M3FN, Float8E8M0FNU, Float32

TILE_M = 128
TILE_N = 128
TILE_K = 128
SCALE_VEC = 32
# scale factors per row (column) per K tile
TILE_SF = TILE_K // SCALE_VEC
THREADS = 256

# Compiled host functions, keyed on everything baked into the artifact: K, N and the
# operand/output dtypes. M is dynamic, so a training loop that changes batch size
# reuses the same compile rather than paying JIT again.
_COMPILED: dict[tuple, object] = {}


def _make_mxfp8_tiled_mma() -> cute.TiledMma:
    """The sm120 block-scaled tiled MMA: 8 warps over a (128, 32, 32) tile.

    The atom layout and permutation mirror the C++ collective builder
    (`sm120_blockscaled_mma_builder.inl`): the N permutation reorders columns within
    each 32-wide group, which is what makes the scale-factor fragment land where the
    instruction expects it.
    """
    op = cute_warp.MmaMXF8Op(Float8E4M3FN, Float32, Float8E8M0FNU)
    permutation = blackwell_helpers.get_permutation_mnk(
        (TILE_M, TILE_N, TILE_K), SCALE_VEC, True
    )
    return cute.make_tiled_mma(op, (4, 2, 1), permutation_mnk=permutation)


def _partition_mxfp8_sf(sf: cute.Tensor, tiled_mma: cute.TiledMma, tidx, is_a: bool):
    """This thread's slice of a scale-factor tile, as the MMA wants to read it.

    `partition_fragment_SFA`/`SFB` build the register fragment but drop the source
    view, and the mainloop needs both, so the partitioning is restated here.
    """
    thrfrg = blackwell_helpers.thrfrg_SFA if is_a else blackwell_helpers.thrfrg_SFB
    # `thrfrg_*` wants a domain in A/B *element* coordinates, not scale-block ones --
    # which is why a 128x4 tile of scales is legal here: the smem layout spans all 128
    # k-elements per scale with a stride-0 mode, so 32 k's read back the same byte.
    thr = cute.make_tensor(sf.iterator, thrfrg(sf.layout, tiled_mma))
    vmnk = tiled_mma.thr_layout_vmnk.get_flat_coord(tidx)
    # this thread's (V, M, K) coordinate for A, (V, N, K) for B -- vmnk is (V, M, N, K)
    coord = (vmnk[0], (vmnk[1], vmnk[3])) if is_a else (vmnk[0], (vmnk[2], vmnk[3]))
    part = cute.group_modes(cute.flatten(thr[coord, (None, None)]), 0, 2)
    # B alone needs its N modes regrouped: the tiled MMA's tile is (128, 32, 32), so the
    # CTA's 128-wide N spans 4 of them where its M spans exactly 1, and B's flattened
    # fragment carries one mode more than A's -- ((32,1),(2,4),4) against ((32,1),2,4).
    return part if is_a else cute.group_modes(part, 1, 3)


def _tiled_copy(dtype, thr_shape, thr_stride, val_shape) -> cute.TiledCopy:
    """A plain global->shared tiled copy; `thr_*` maps a tile coord to its thread."""
    return cute.make_tiled_copy_tv(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype),
        cute.make_layout(thr_shape, stride=thr_stride),
        cute.make_layout(val_shape),
    )


@cute.kernel
def _scaled_gemm_mxfp8_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mSFA: cute.Tensor,
    mSFB: cute.Tensor,
    mD: cute.Tensor,
    tiled_mma: cute.TiledMma,
    sfa_smem_layout: cute.Layout,
    sfb_smem_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    tile_m, tile_n, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(
        Float8E4M3FN, cute.make_ordered_layout((TILE_M, TILE_K), order=(1, 0)), 16
    )
    sB = smem.allocate_tensor(
        Float8E4M3FN, cute.make_ordered_layout((TILE_N, TILE_K), order=(0, 1)), 16
    )
    sSFA = smem.allocate_tensor(Float8E8M0FNU, sfa_smem_layout, 16)
    sSFB = smem.allocate_tensor(Float8E8M0FNU, sfb_smem_layout, 16)
    # Write view of the same two buffers, indexed as (row, scale) instead of the
    # (row, k-element) the MMA reads them by. It restates the 32x4x4 swizzle above:
    # offset = 16 * (row % 32) + 4 * (row // 32) + scale.
    swizzle = cute.make_layout(((32, 4), TILE_SF), stride=((16, 4), 1))
    sSFA_w = cute.make_tensor(sSFA.iterator, swizzle)
    sSFB_w = cute.make_tensor(sSFB.iterator, swizzle)

    # this CTA's column of A / row of B tiles, and its one output tile
    gA = cute.zipped_divide(mA, (TILE_M, TILE_K))[(None, None), (tile_m, None)]
    gB = cute.zipped_divide(mB, (TILE_N, TILE_K))[(None, None), (tile_n, None)]
    gSFA = cute.zipped_divide(mSFA, (TILE_M, TILE_SF))[(None, None), (tile_m, None)]
    gSFB = cute.zipped_divide(mSFB, (TILE_N, TILE_SF))[(None, None), (tile_n, None)]
    gD = cute.zipped_divide(mD, (TILE_M, TILE_N))[(None, None), (tile_m, tile_n)]

    # 16 bytes per thread along whichever axis is contiguous in global memory: K for
    # A, N for B, and 2 scale bytes per thread for the (128, 4) scale tiles.
    copy_a = _tiled_copy(Float8E4M3FN, (32, 8), (8, 1), (1, 16)).get_slice(tidx)
    copy_b = _tiled_copy(Float8E4M3FN, (8, 32), (1, 8), (16, 1)).get_slice(tidx)
    copy_sf = _tiled_copy(Float8E8M0FNU, (128, 2), (1, 128), (1, 2)).get_slice(tidx)

    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCgD = thr_mma.partition_C(gD)
    tCrA = tiled_mma.make_fragment_A(tCsA)
    tCrB = tiled_mma.make_fragment_B(tCsB)
    tCsSFA = _partition_mxfp8_sf(sSFA, tiled_mma, tidx, True)
    tCsSFB = _partition_mxfp8_sf(sSFB, tiled_mma, tidx, False)
    tCrSFA = cute.make_fragment_like(tCsSFA)
    tCrSFB = cute.make_fragment_like(tCsSFB)

    acc = cute.make_rmem_tensor(tCgD.shape, Float32)
    acc.fill(0.0)

    for k_tile in cutlass.range(cute.size(gA, mode=[2]), unroll=1):
        cute.autovec_copy(
            copy_a.partition_S(gA[None, None, k_tile]), copy_a.partition_D(sA)
        )
        cute.autovec_copy(
            copy_b.partition_S(gB[None, None, k_tile]), copy_b.partition_D(sB)
        )
        cute.autovec_copy(
            copy_sf.partition_S(gSFA[None, None, k_tile]), copy_sf.partition_D(sSFA_w)
        )
        cute.autovec_copy(
            copy_sf.partition_S(gSFB[None, None, k_tile]), copy_sf.partition_D(sSFB_w)
        )
        cute.arch.barrier()

        cute.autovec_copy(tCsA, tCrA)
        cute.autovec_copy(tCsB, tCrB)
        cute.autovec_copy(tCsSFA, tCrSFA)
        cute.autovec_copy(tCsSFB, tCrSFB)
        cute.gemm(tiled_mma, acc, [tCrA, tCrSFA], [tCrB, tCrSFB], acc)
        # the next tile overwrites the stage, so no thread may run ahead of the MMA
        cute.arch.barrier()

    tCgD.store(acc.load().to(mD.element_type))


@cute.jit
def _scaled_gemm_mxfp8_host(
    mA: cute.Tensor,
    mB: cute.Tensor,
    sfa: cute.Tensor,
    sfb: cute.Tensor,
    mD: cute.Tensor,
):
    tiled_mma = _make_mxfp8_tiled_mma()
    M, K = mA.shape
    N = mB.shape[0]
    n_scale_blocks = K // SCALE_VEC
    # Both scale tensors are read as (MN, scale-block); sb arrives transposed, as
    # (K // 32, N), so its view just swaps the strides rather than the data.
    mSFA = cute.make_tensor(
        sfa.iterator, cute.make_layout((M, n_scale_blocks), stride=(n_scale_blocks, 1))
    )
    mSFB = cute.make_tensor(
        sfb.iterator, cute.make_layout((N, n_scale_blocks), stride=(1, N))
    )
    tile = (TILE_M, TILE_N, TILE_K)
    sfa_smem_layout = cute.select(
        blockscaled_layout.sm120_make_smem_layout_sfa(tiled_mma, tile, SCALE_VEC, 1),
        mode=[0, 1],
    )
    sfb_smem_layout = cute.select(
        blockscaled_layout.sm120_make_smem_layout_sfb(tiled_mma, tile, SCALE_VEC, 1),
        mode=[0, 1],
    )
    _scaled_gemm_mxfp8_kernel(
        mA, mB, mSFA, mSFB, mD, tiled_mma, sfa_smem_layout, sfb_smem_layout
    ).launch(
        grid=[cute.ceil_div(M, TILE_M), cute.ceil_div(N, TILE_N), 1],
        block=[THREADS, 1, 1],
    )


def scaled_gemm_mxfp8(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """mxfp8 GEMM, (M,K) x (K,N) -> (M,N), scales applied inside the MMA.

    `sa` is (M, K//32) and `sb` is (K//32, N), both fp32 holding exact powers of two
    -- what `quantize_operand` emits for `scale_dtype="fp8_e8m0"` at block size 32 --
    so the cast to e8m0 below is lossless.

    This first version takes e4m3 operands and M, K, N that are multiples of 128.
    """
    M, K = aq.shape
    N = bq.shape[1]
    e4m3 = torch.float8_e4m3fn
    assert aq.dtype == e4m3 and bq.dtype == e4m3, (
        f"the CuTe mxfp8 GEMM is e4m3 x e4m3 only, got {aq.dtype} x {bq.dtype}"
    )
    assert bq.shape[0] == K, f"contraction mismatch: aq {K}, bq {bq.shape[0]}"
    # the operand layouts are baked into the compile, so a non-contiguous operand
    # would be read through the wrong strides rather than rejected
    assert aq.is_contiguous() and bq.is_contiguous(), "aq and bq must be contiguous"
    assert M % TILE_M == 0 and K % TILE_K == 0 and N % TILE_N == 0, (
        f"M, K, N must be multiples of {TILE_M}, got ({M}, {K}, {N})"
    )
    assert sa.shape == (M, K // SCALE_VEC) and sb.shape == (K // SCALE_VEC, N), (
        f"scale shapes {tuple(sa.shape)} / {tuple(sb.shape)} do not match "
        f"({M}, {K // SCALE_VEC}) / ({K // SCALE_VEC}, {N})"
    )

    out = torch.empty(M, N, device=aq.device, dtype=out_dtype)
    sa8 = sa.to(torch.float8_e8m0fnu).contiguous()
    sb8 = sb.to(torch.float8_e8m0fnu).contiguous()
    # bq is (K, N) row-major; the MMA reads B as (N, K), which is the same bytes
    args = (
        from_dlpack(aq, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=aq.dim_order(), divisibility=TILE_M
        ),
        from_dlpack(bq.t(), assumed_align=16),
        from_dlpack(sa8, assumed_align=4).mark_compact_shape_dynamic(
            mode=0, stride_order=sa8.dim_order(), divisibility=TILE_M
        ),
        from_dlpack(sb8, assumed_align=4),
        from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=out.dim_order(), divisibility=TILE_M
        ),
    )
    key = (K, N, aq.dtype, bq.dtype, out_dtype)
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(_scaled_gemm_mxfp8_host, *args)
        _COMPILED[key] = compiled
    compiled(*args)
    return out
