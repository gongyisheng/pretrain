"""CuTe DSL mxfp8 dense GEMM for sm120.

One CTA computes a 128x128 output tile by walking K in 128-wide steps: stage the
A/B tile and its scale factors in shared memory, pull them into registers, and run
the warp-level `m16n8k32` block-scaled MMA (`kind::mxf8`), which applies the e8m0
scales inside the tensor core. Either operand may be e4m3 or e5m2. Accumulation is
fp32; `out_dtype` is only reached in the epilogue.

M and N are free of the tile: a trailing tile runs off the end of the tensor, and
both the loads and the store are predicated against the real extent. K only has to
be a multiple of the 32-element scale block.

There is deliberately no pipelining here: the shared-memory stage is single-buffered
and fed by ordinary loads, not `cp.async`. That makes this slower than the Triton
mxfp8 kernel, which is the expected intermediate state -- this module exists to
establish a correct CuTe mainloop for the feed path to be built on.

Scale factors are held in shared memory in the cuBLASLt-compatible 32x4x4 swizzle
(`sm120_make_smem_layout_sfa`/`sfb`), the layout the MMA's scale-factor fragment
partitioning reads back. That compatibility is about the *smem* layout only: the
kernel consumes scales in `quantize_operand`'s native `(M, K//32)` layout straight
from the host and performs the 32x4x4 swizzle itself while staging into shared
memory, so no pre-swizzled global input is required.
"""

from __future__ import annotations

import cutlass
import cutlass._mlir.dialects.cute_nvgpu as cute_nvgpu_ir
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as blackwell_helpers
import cutlass.utils.blockscaled_layout as blockscaled_layout
import torch
from cutlass.cute.atom import make_atom
from cutlass.cute.core import _pack_shape
from cutlass.cute.nvgpu import warp as cute_warp
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import Float8E8M0FNU, Float32

TILE_M = 128
TILE_N = 128
TILE_K = 128
assert TILE_M == TILE_N == 128 and TILE_K == 128, (
    "the SF write view, the SFB mode regroup and copy_a's layout are derived "
    "for a 128^3 tile"
)
SCALE_VEC = 32
# scale factors per row (column) per K tile
TILE_SF = TILE_K // SCALE_VEC
THREADS = 256

# Compiled host functions, keyed on everything baked into the artifact: K, N and the
# operand/output dtypes. M is dynamic, so a training loop that changes batch size
# reuses the same compile rather than paying JIT again.
_COMPILED: dict[tuple, object] = {}


# the two fp8 formats the block-scaled MMA takes; either may sit in either operand
FP8_FORMATS = frozenset({torch.float8_e4m3fn, torch.float8_e5m2})

# dtypes the epilogue's store can write; anything else reaches `cute.compile` unchecked
OUT_DTYPES = frozenset({torch.float32, torch.float16, torch.bfloat16})


class _MmaMXF8Op(cute_warp.MmaMXF8Op):
    """`MmaMXF8Op` with the A and B formats chosen independently.

    The stock op emits one dtype for both operands, and the DSL's mixed-precision
    sibling `MmaMXF8F6F4Op` rejects two 8-bit formats outright. The atom type
    underneath takes the two separately, though, and e4m3 and e5m2 differ only in the
    instruction's operand qualifiers -- same width, same fragment layouts -- so this
    passes them through and leaves everything else on the stock op.

    `b_dtype` is not a field of the base dataclass, so two of these compare equal when
    only it differs. Nothing keys off op identity today -- `_COMPILED` keys on both
    torch dtypes -- but a third format would want to revisit that.
    """

    def __init__(self, a_dtype, b_dtype):
        super().__init__(a_dtype, Float32, Float8E8M0FNU)
        object.__setattr__(self, "b_dtype", b_dtype)  # the base is a frozen dataclass

    def _make_trait(self, *, loc=None, ip=None, **kwargs) -> cute_warp.MmaMXF8Trait:
        ty = cute_nvgpu_ir.MmaAtomSM120BlockScaledType.get(
            _pack_shape(self.shape_mnk, loc=loc, ip=ip).type.attribute,
            self.sf_vec_size,
            self.use_sf_layout_TV,
            self.ab_dtype.mlir_type,
            self.b_dtype.mlir_type,
            self.acc_dtype.mlir_type,
            self.sf_type.mlir_type,
        )
        return cute_warp.MmaMXF8Trait(make_atom(ty, loc=loc, ip=ip))


def _make_mxfp8_tiled_mma(a_dtype, b_dtype) -> cute.TiledMma:
    """The sm120 block-scaled tiled MMA: 8 warps over a (128, 32, 32) tile.

    The atom layout and permutation mirror the C++ collective builder
    (`sm120_blockscaled_mma_builder.inl`): the N permutation reorders columns within
    each 32-wide group, which is what makes the scale-factor fragment land where the
    instruction expects it.
    """
    op = _MmaMXF8Op(a_dtype, b_dtype)
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
    """A plain global->shared tiled copy; `thr_*` maps a tile coord to its thread.

    The atom is sized to the whole per-thread vector so that predication is decided
    once per vector rather than per element.
    """
    return cute.make_tiled_copy_tv(
        cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dtype,
            num_bits_per_copy=dtype.width * val_shape[0] * val_shape[1],
        ),
        cute.make_layout(thr_shape, stride=thr_stride),
        cute.make_layout(val_shape),
    )


def _group_rest(t: cute.Tensor) -> cute.Tensor:
    """Collapse a partitioned tensor's trailing modes, leaving (V, rest)."""
    return cute.group_modes(t, 1, cute.rank(t))


def _tile_coords(shape, tiler, tile):
    """The global coordinates of one tile, shaped exactly like the tile itself."""
    return cute.zipped_divide(cute.make_identity_tensor(shape), tiler)[
        (None, None), tile
    ]


def _copy_tile(tiled_copy, tidx, gS, cS, sD, bound, zero_first: bool):
    """Stage one tile in shared memory, skipping whatever falls outside `bound`.

    `cS` is the tile's global coordinates, so a copy vector is dropped exactly when
    its source lies past the end of the tensor. One comparison per vector stands for
    the whole vector, which holds because no vector straddles an edge: A and the
    scales are cut along K, a multiple of 32, and B's N vector width divides N.

    `zero_first` blanks the stage before the copy, and is set only when K leaves a
    short last tile. That tail multiplies into every output element, so stale bytes
    there would corrupt the result -- an uninitialised e8m0 byte can even be the 0xFF
    NaN encoding. The M and N edges need no such care: garbage rows and columns reach
    only their own accumulator rows and columns, which the epilogue never stores.
    """
    copy = tiled_copy.get_slice(tidx)
    src = _group_rest(copy.partition_S(gS))
    crd = _group_rest(copy.partition_S(cS))
    dst = _group_rest(copy.partition_D(sD))
    if zero_first:
        zeros = cute.make_fragment_like(dst)
        zeros.fill(0)
        cute.autovec_copy(zeros, dst)
    n_vec = cute.size(src, mode=[1])
    pred = cute.make_rmem_tensor((1, n_vec), cutlass.Boolean)
    for i in range(n_vec):
        pred[0, i] = cute.elem_less(crd[0, i], bound)
    cute.copy(tiled_copy, src, dst, pred=pred)


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
        mA.element_type, cute.make_ordered_layout((TILE_M, TILE_K), order=(1, 0)), 16
    )
    sB = smem.allocate_tensor(
        mB.element_type, cute.make_ordered_layout((TILE_N, TILE_K), order=(0, 1)), 16
    )
    sSFA = smem.allocate_tensor(Float8E8M0FNU, sfa_smem_layout, 16)
    sSFB = smem.allocate_tensor(Float8E8M0FNU, sfb_smem_layout, 16)
    # Write view of the same two buffers, indexed as (row, scale) instead of the
    # (row, k-element) the MMA reads them by. It restates the 32x4x4 swizzle above:
    # offset = 16 * (row % 32) + 4 * (row // 32) + scale.
    swizzle = cute.make_layout(((32, 4), TILE_SF), stride=((16, 4), 1))
    sSFA_w = cute.make_tensor(sSFA.iterator, swizzle)
    sSFB_w = cute.make_tensor(sSFB.iterator, swizzle)

    # this CTA's column of A / row of B tiles, and its one output tile. M and N need
    # not divide the tile: the trailing tiles then run off the end of the tensor, and
    # the matching coordinate tiles below say by how much.
    gA = cute.zipped_divide(mA, (TILE_M, TILE_K))[(None, None), (tile_m, None)]
    gB = cute.zipped_divide(mB, (TILE_N, TILE_K))[(None, None), (tile_n, None)]
    gSFA = cute.zipped_divide(mSFA, (TILE_M, TILE_SF))[(None, None), (tile_m, None)]
    gSFB = cute.zipped_divide(mSFB, (TILE_N, TILE_SF))[(None, None), (tile_n, None)]
    gD = cute.zipped_divide(mD, (TILE_M, TILE_N))[(None, None), (tile_m, tile_n)]
    cA = _tile_coords(mA.shape, (TILE_M, TILE_K), (tile_m, None))
    cB = _tile_coords(mB.shape, (TILE_N, TILE_K), (tile_n, None))
    cSFA = _tile_coords(mSFA.shape, (TILE_M, TILE_SF), (tile_m, None))
    cSFB = _tile_coords(mSFB.shape, (TILE_N, TILE_SF), (tile_n, None))
    cD = _tile_coords(mD.shape, (TILE_M, TILE_N), (tile_m, tile_n))

    # 16 bytes per thread along whichever axis is contiguous in global memory: K for
    # A, N for B. B's width drops to the largest power of two dividing N, so that a
    # vector never straddles the N edge nor loads across a row start it is not aligned
    # to. The scale tiles are copied a byte at a time, which sidesteps straddling and
    # alignment for them entirely and costs little: 512 bytes against 32 KiB of operand.
    b_vec = max(v for v in (16, 8, 4, 2, 1) if mB.shape[0] % v == 0)
    b_thr = TILE_N // b_vec
    copy_a = _tiled_copy(mA.element_type, (32, 8), (8, 1), (1, 16))
    copy_b = _tiled_copy(
        mB.element_type, (b_thr, THREADS // b_thr), (1, b_thr), (b_vec, 1)
    )
    copy_sf = _tiled_copy(Float8E8M0FNU, (128, 2), (1, 128), (1, 1))

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

    # K's last tile is short exactly when the tile does not divide it, and K is static
    k_tail = mA.shape[1] % TILE_K != 0
    for k_tile in cutlass.range(cute.size(gA, mode=[2]), unroll=1):
        k = (None, None, k_tile)
        _copy_tile(copy_a, tidx, gA[k], cA[k], sA, mA.shape, k_tail)
        _copy_tile(copy_b, tidx, gB[k], cB[k], sB, mB.shape, k_tail)
        _copy_tile(copy_sf, tidx, gSFA[k], cSFA[k], sSFA_w, mSFA.shape, k_tail)
        _copy_tile(copy_sf, tidx, gSFB[k], cSFB[k], sSFB_w, mSFB.shape, k_tail)
        cute.arch.barrier()

        cute.autovec_copy(tCsA, tCrA)
        cute.autovec_copy(tCsB, tCrB)
        cute.autovec_copy(tCsSFA, tCrSFA)
        cute.autovec_copy(tCsSFB, tCrSFB)
        cute.gemm(tiled_mma, acc, [tCrA, tCrSFA], [tCrB, tCrSFB], acc)
        # the next tile overwrites the stage, so no thread may run ahead of the MMA
        cute.arch.barrier()

    # fp32 all the way through the mainloop; the caller's dtype is reached here, and
    # only the elements inside (M, N) are written back
    tCrD = cute.make_fragment_like(tCgD)
    tCrD.store(acc.load().to(mD.element_type))
    tCcD = thr_mma.partition_C(cD)
    for i in cutlass.range(cute.size(tCgD), unroll_full=True):
        if cute.elem_less(tCcD[i], mD.shape):
            tCgD[i] = tCrD[i]


@cute.jit
def _scaled_gemm_mxfp8_host(
    mA: cute.Tensor,
    mB: cute.Tensor,
    sfa: cute.Tensor,
    sfb: cute.Tensor,
    mD: cute.Tensor,
):
    tiled_mma = _make_mxfp8_tiled_mma(mA.element_type, mB.element_type)
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

    Either operand may be e4m3 or e5m2. M and N are arbitrary -- partial tiles are
    predicated -- but K must be a multiple of the 32-element scale block, which is
    what mxfp8 means rather than a limitation of this kernel.
    """
    M, K = aq.shape
    N = bq.shape[1]
    assert aq.dtype in FP8_FORMATS and bq.dtype in FP8_FORMATS, (
        f"the CuTe mxfp8 GEMM takes e4m3 or e5m2 operands, got {aq.dtype} x {bq.dtype}"
    )
    assert bq.shape[0] == K, f"contraction mismatch: aq {K}, bq {bq.shape[0]}"
    # the operand layouts are baked into the compile, so a non-contiguous operand
    # would be read through the wrong strides rather than rejected
    assert aq.is_contiguous() and bq.is_contiguous(), "aq and bq must be contiguous"
    # M and N are predicated, but a scale block is indivisible, so K is not
    assert K % SCALE_VEC == 0, f"K must be a multiple of {SCALE_VEC}, got {K}"
    assert sa.shape == (M, K // SCALE_VEC) and sb.shape == (K // SCALE_VEC, N), (
        f"scale shapes {tuple(sa.shape)} / {tuple(sb.shape)} do not match "
        f"({M}, {K // SCALE_VEC}) / ({K // SCALE_VEC}, {N})"
    )
    assert out_dtype in OUT_DTYPES, f"unsupported out_dtype {out_dtype}"

    out = torch.empty(M, N, device=aq.device, dtype=out_dtype)
    sa8 = sa.to(torch.float8_e8m0fnu).contiguous()
    sb8 = sb.to(torch.float8_e8m0fnu).contiguous()
    # bq is (K, N) row-major; the MMA reads B as (N, K), which is the same bytes
    args = (
        from_dlpack(aq, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=aq.dim_order(), divisibility=1
        ),
        from_dlpack(bq.t(), assumed_align=16),
        from_dlpack(sa8, assumed_align=4).mark_compact_shape_dynamic(
            mode=0, stride_order=sa8.dim_order(), divisibility=1
        ),
        from_dlpack(sb8, assumed_align=4),
        from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=out.dim_order(), divisibility=1
        ),
    )
    key = (K, N, aq.dtype, bq.dtype, out_dtype)
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(_scaled_gemm_mxfp8_host, *args)
        _COMPILED[key] = compiled
    compiled(*args)
    return out
