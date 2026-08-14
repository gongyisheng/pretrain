"""CuTe DSL mxfp8 dense GEMM for sm120.

One CTA computes a 128x128 output tile by walking K in 128-wide steps: stage the
A/B tile and its scale factors in shared memory, pull them into registers, and run
the warp-level `m16n8k32` block-scaled MMA (`kind::mxf8`), which applies the e8m0
scales inside the tensor core. Either operand may be e4m3 or e5m2. Accumulation is
fp32; `out_dtype` is only reached in the epilogue.

M and N are free of the tile: a trailing tile runs off the end of the tensor, and
both the loads and the store are predicated against the real extent. K only has to
be a multiple of the 32-element scale block.

Both operands are taken K-contiguous -- A as `(M, K)`, B as `(N, K)` -- which is
the layout the MMA's fragments want (four consecutive k per load) and, for B, the
layout an `nn.Linear` weight is already stored in. Both tiles are held in shared
memory K-major and XOR-swizzled, the swizzle spreading the rows a warp reads
together across all 32 banks. Nothing is relayouted on the host.

The mainloop is a `STAGES`-deep software pipeline. Shared memory holds `STAGES`
copies of each of the four staged tensors, every global->shared transfer is a
`cp.async` group, and iteration `i` waits only for the group it is about to feed the
MMA -- the copies for tiles `i+1 .. i+STAGES-1` stay in flight across the MMA.

Scale factors for both operands are held in shared memory in the cuBLASLt-compatible
32x4x4 swizzle (`sm120_make_smem_layout_sfa`/`_sfb`), whose scale axis is contiguous
-- as is the scale axis of both global scale tensors -- so a 4-byte `cp.async` moves
a whole k-tile's scales per thread. The kernel consumes scales in
`quantize_operand`'s native `(M, K//32)` / `(N, K//32)` layout straight from the host
and lays them out itself while staging, so no pre-swizzled global input is required.
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
# bytes one k-tile of scale factors occupies in shared memory, for either operand
SF_TILE_BYTES = TILE_M * TILE_SF
# XOR swizzle for the operand tiles. Both are K-major with TILE_K-byte rows, read by
# the MMA a 4-byte k-group at a time from 8 different rows at once, so unswizzled
# every lane of that group lands in the same bank. `Swizzle<3, 4, 3>` permutes the
# eight 16-byte granules of a row by the low 3 bits of the row index, which spreads
# those 8 rows across all 32 banks. It is a bijection on the same allocation, so it
# adds no bytes -- SMEM_BYTES below stays exact. The 16-byte granule is also what
# keeps the staging copy legal: a `cp.async` vector is exactly one granule, so the
# swizzle moves it whole rather than splitting it.
#
# `Swizzle<b, m, s>` XORs the b bits above bit m into the b bits above bit (m + s), so
# all three numbers are pinned, not just two: the granules it permutes have to be
# exactly one row (BASE + BITS == log2 TILE_K) *and* the bits it XORs in have to be
# the row index (BASE + SHIFT == log2 TILE_K, i.e. SHIFT == BITS). SHIFT is the one a
# sweep would vary on its own, and it fails in both directions -- measured here at
# TILE_K = 128, everything else fixed:
#
#   Swizzle<3,4,0/1/2>   26/26 tests pass, 45.5 TF/s at 4096^3
#   Swizzle<3,4,3>       26/26 tests pass, 73.6 TF/s      <- shipped
#   Swizzle<3,4,4>       25/26 tests FAIL
#
# Too small and the XOR source overlaps the granule index instead of being the row
# index, so rows stop spreading over the banks: still correct, but the swizzle's
# entire 62% is silently gone and only a benchmark would notice. Too large and it
# reaches past the row index and the numbers go wrong. Neither crashes, so the assert
# is the only thing standing between a sweep and a bad result.
SWIZZLE_BITS, SWIZZLE_BASE, SWIZZLE_SHIFT = 3, 4, 3
assert (
    TILE_K == (1 << SWIZZLE_BASE) << SWIZZLE_BITS and SWIZZLE_SHIFT == SWIZZLE_BITS
), (
    f"Swizzle<{SWIZZLE_BITS},{SWIZZLE_BASE},{SWIZZLE_SHIFT}> is not a permutation of "
    f"one {TILE_K}-byte K-major row: need BASE + BITS == BASE + SHIFT == "
    f"log2(TILE_K) == {TILE_K.bit_length() - 1}, so that the swizzle XORs the row "
    f"index into the granule index and touches nothing else"
)
# Pipeline depth. Shared memory per CTA is STAGES * (32 KiB of operand + 1 KiB of
# scales); 3 is the deepest that fits sm120's 99 KiB shared-memory budget at a 128^3
# tile. STAGES-1 tiles are in flight while the MMA consumes the oldest.
STAGES = 3
assert STAGES >= 2, "the mainloop needs at least one tile in flight behind the MMA"
# The shipped configuration sits exactly on the limit -- 3 * (16384 + 16384 + 512 +
# 512) = 101376 = 99 KiB, with every allocation already a multiple of 16 so alignment
# adds no slack. There is no headroom at all, so anything that grows a staged buffer
# (a wider tile, a padded or swizzled layout that rounds up) has to buy the room back
# by dropping a stage or shrinking TILE_K. Without this the overflow surfaces only as
# an opaque launch failure that names nothing.
SMEM_BYTES = STAGES * (TILE_M * TILE_K + TILE_N * TILE_K + 2 * SF_TILE_BYTES)
assert SMEM_BYTES <= 99 * 1024, (
    f"staged shared memory is {SMEM_BYTES} bytes, over sm120's 99 KiB limit -- "
    f"lower STAGES or shrink the tile (TILE_K halves it fastest)"
)

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


def _operand_smem_layout(tile_mn: int) -> cute.ComposedLayout:
    """A staged operand tile: `(MN, K, STAGES)`, K-major, XOR-swizzled.

    Both operands are K-major so that the MMA's fragment -- which wants four
    consecutive k per load -- reads whole 4-byte words instead of single bytes, and
    both are swizzled so that the eight rows a warp reads at once fall in different
    banks. The stage sits above the swizzle's bits (a stage is `tile_mn * TILE_K`
    bytes, far past bit 9), so staging and swizzling do not interact.
    """
    return cute.make_composed_layout(
        cute.make_swizzle(SWIZZLE_BITS, SWIZZLE_BASE, SWIZZLE_SHIFT),
        0,
        cute.make_ordered_layout((tile_mn, TILE_K, STAGES), order=(1, 0, 2)),
    )


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


def _tiled_copy(dtype, thr_shape, thr_stride, val_shape, is_async) -> cute.TiledCopy:
    """A global->shared tiled copy; `thr_*` maps a tile coord to its thread.

    The atom is sized to the whole per-thread vector so that predication is decided
    once per vector rather than per element.

    `is_async` picks `cp.async` over an ordinary load/store pair. `cp.async` moves
    4, 8 or 16 bytes and needs the vector contiguous and aligned at both ends, which
    every copy here satisfies for any shape: the operand vector is 16 bytes along K,
    contiguous in global memory and one swizzle granule in shared memory, and each
    scale vector is the k-tile's 4 scales, contiguous in global memory because the
    host pads that axis to a multiple of `TILE_SF` and contiguous in shared memory in
    both scale layouts.
    """
    op = cute.nvgpu.cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    return cute.make_tiled_copy_tv(
        cute.make_copy_atom(
            op,
            dtype,
            num_bits_per_copy=dtype.width * val_shape[0] * val_shape[1],
        ),
        cute.make_layout(thr_shape, stride=thr_stride),
        cute.make_layout(val_shape),
    )


def _sf_stage(buf: cute.Tensor, stage, layout: cute.Layout) -> cute.Tensor:
    """One stage's view of a flat scale buffer, under `layout`.

    The offset is a whole multiple of `SF_TILE_BYTES` off a 16-byte-aligned
    allocation, but pointer arithmetic drops the alignment down to the 1-byte
    element, and `cp.async` will not accept a destination it believes is unaligned.
    `align` restates what the allocation already guarantees.
    """
    return cute.make_tensor((buf.iterator + stage * SF_TILE_BYTES).align(16), layout)


def _group_rest(t: cute.Tensor) -> cute.Tensor:
    """Collapse a partitioned tensor's trailing modes, leaving (V, rest)."""
    return cute.group_modes(t, 1, cute.rank(t))


def _tile_coords(shape, tiler, tile):
    """The global coordinates of one tile, shaped exactly like the tile itself."""
    return cute.zipped_divide(cute.make_identity_tensor(shape), tiler)[
        (None, None), tile
    ]


def _copy_tile(tiled_copy, tidx, gS, cS, sD, bound, zero_k_tail: bool):
    """Stage one tile in shared memory, skipping whatever falls outside `bound`.

    `cS` is the tile's global coordinates, so a copy vector is dropped exactly when
    its source lies past the end of the tensor. One comparison per vector stands for
    the whole vector, which holds because no vector straddles an edge: every copy
    here is cut along K -- a multiple of 32 for the operands, and padded to a
    multiple of `TILE_SF` for either operand's scales.

    **Drop-never-substitute.** A dropped vector leaves whatever the stage buffer
    already held in *that* slot -- never a neighbour's data. Under staging "already
    held" means "written by the tile that used this stage `STAGES` iterations ago",
    which is a different value than in the unpipelined kernel but sits at the same
    (row, column) of the tile, so the argument is unchanged: a stale element only
    ever reaches the accumulator row (M edge) or column (N edge) it belongs to, and
    the epilogue stores neither. The operand swizzle does not disturb this: it is a
    bijection on the stage's own bytes, so distinct (row, column) pairs still land at
    distinct addresses and a slot still belongs to exactly one of them.

    **K-tail zeroing.** A short last K tile is the one case where stale data is not
    confined: it multiplies into every output element, and an uninitialised e8m0 byte
    can be the 0xFF NaN encoding, which `NaN * 0` spreads across an accumulator row.
    `zero_k_tail` therefore stores zeros over exactly the vectors dropped for running
    past K -- into the same slots, in the same stage buffer, as part of staging that
    tile. That makes zeroing a property of the stage buffer rather than of the loop
    iteration, which is what staging requires: the tail tile is copied several
    iterations before the MMA reads it, so anything done once per iteration would
    zero the wrong buffer. The zero stores and the copy are disjoint by construction
    (complementary predicates), so they cannot race each other in the async copy's
    destination. Both go through the same `partition_D`, so the swizzle sends the
    zeros to exactly the addresses the dropped copy would have written -- the
    predicates are compared in tile coordinates and never in addresses, so the
    swizzle cannot move a zero off the slot it is meant to cover. The scales need no
    equivalent -- `scaled_gemm_mxfp8` pads their k axis with zeros so their copy has
    no dropped vectors to begin with.
    """
    copy = tiled_copy.get_slice(tidx)
    src = _group_rest(copy.partition_S(gS))
    crd = _group_rest(copy.partition_S(cS))
    dst = _group_rest(copy.partition_D(sD))
    n_vec = cute.size(src, mode=[1])
    pred = cute.make_rmem_tensor((1, n_vec), cutlass.Boolean)
    for i in range(n_vec):
        pred[0, i] = cute.elem_less(crd[0, i], bound)
    cute.copy(tiled_copy, src, dst, pred=pred)
    if zero_k_tail:
        zeros = cute.make_fragment_like(dst)
        zeros.fill(0)
        # mode 1 of the coordinate is K for both A (M, K) and B (N, K)
        zpred = cute.make_rmem_tensor((1, n_vec), cutlass.Boolean)
        for i in range(n_vec):
            zpred[0, i] = crd[0, i][1] >= bound[1]
        cute.copy(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                sD.element_type,
                num_bits_per_copy=sD.element_type.width * cute.size(dst, mode=[0]),
            ),
            zeros,
            dst,
            pred=zpred,
        )


@cute.jit
def _stage_k_tile(k_tile, stage, tidx, k_tail, a, b, sfa, sfb):
    """Issue every global->shared copy for one k-tile into one stage buffer.

    `@cute.jit` rather than bare: the scale split below branches on `tidx`, and only a
    decorated body gets its control flow staged. It is still device code -- the
    decorator marks the function for AST preprocessing, and the call is inlined into
    whichever `@cute.kernel` reaches it.

    `a` and `b` are `(tiled_copy, global tiles, coordinate tiles, smem, bound)`, the
    smem buffer carrying the stage as a trailing mode. `sfa` and `sfb` add the write
    view between smem and bound, since the scale buffers are flat and are addressed
    through a byte offset plus a `(row, scale)` layout.

    A k-tile of scales is 128 groups of 4 bytes, exactly one `cp.async` per thread
    for half a CTA, so the halves split the work: threads below the midpoint stage
    A's scales, the rest stage B's. A shape that falls back to the synchronous scale
    copy keeps the same split, just at a narrower vector.
    """
    k = (None, None, k_tile)
    _copy_tile(a[0], tidx, a[1][k], a[2][k], a[3][None, None, stage], a[4], k_tail)
    _copy_tile(b[0], tidx, b[1][k], b[2][k], b[3][None, None, stage], b[4], k_tail)
    half = THREADS // 2
    if tidx < half:
        dst = _sf_stage(sfa[3], stage, sfa[4])
        _copy_tile(sfa[0], tidx, sfa[1][k], sfa[2][k], dst, sfa[5], False)
    else:
        dst = _sf_stage(sfb[3], stage, sfb[4])
        _copy_tile(sfb[0], tidx - half, sfb[1][k], sfb[2][k], dst, sfb[5], False)


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

    # Every staged buffer carries a leading pipeline depth: STAGES k-tiles are
    # resident at once so the copies for the next STAGES-1 can be in flight.
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(mA.element_type, _operand_smem_layout(TILE_M), 16)
    sB = smem.allocate_tensor(mB.element_type, _operand_smem_layout(TILE_N), 16)
    sSFA = smem.allocate_tensor(
        Float8E8M0FNU, cute.make_layout(SF_TILE_BYTES * STAGES), 16
    )
    sSFB = smem.allocate_tensor(
        Float8E8M0FNU, cute.make_layout(SF_TILE_BYTES * STAGES), 16
    )
    # Write view of a scale buffer, indexed as (row, scale) instead of the
    # (row, k-element) the MMA reads it by. It restates the 32x4x4 swizzle both
    # `sfa_smem_layout` and `sfb_smem_layout` carry: offset = 16 * (row % 32) +
    # 4 * (row // 32) + scale. One view serves both, since a k-tile of either
    # operand's scales is the same 128x4 block in the same layout.
    sf_write = cute.make_layout(((32, 4), TILE_SF), stride=((16, 4), 1))

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

    # 16 bytes per thread along K, the contiguous axis of both operands in global
    # memory -- and, after the swizzle, exactly one granule of the destination row, so
    # the vector lands whole. K is a multiple of 32, so a 16-wide vector never
    # straddles the K edge and one predicate per vector is enough. Each scale tile
    # moves a whole k-tile's 4 bytes per thread along the scale axis, contiguous in
    # both `mSFA` and `mSFB` and in both smem layouts, so one copy serves both and
    # half the CTA stages A's scales while the other half stages B's.
    copy_a = _tiled_copy(mA.element_type, (32, 8), (8, 1), (1, 16), True)
    copy_b = _tiled_copy(mB.element_type, (32, 8), (8, 1), (1, 16), True)
    copy_sf = _tiled_copy(Float8E8M0FNU, (TILE_M, 1), (1, 1), (1, TILE_SF), True)

    thr_mma = tiled_mma.get_slice(tidx)
    tCgD = thr_mma.partition_C(gD)
    tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA[None, None, 0]))
    tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB[None, None, 0]))
    tCrSFA = cute.make_fragment_like(
        _partition_mxfp8_sf(_sf_stage(sSFA, 0, sfa_smem_layout), tiled_mma, tidx, True)
    )
    tCrSFB = cute.make_fragment_like(
        _partition_mxfp8_sf(_sf_stage(sSFB, 0, sfb_smem_layout), tiled_mma, tidx, False)
    )

    acc = cute.make_rmem_tensor(tCgD.shape, Float32)
    acc.fill(0.0)

    # K's last tile is short exactly when the tile does not divide it, and K is static
    k_tail = mA.shape[1] % TILE_K != 0
    n_k = cute.size(gA, mode=[2])

    job_a = (copy_a, gA, cA, sA, mA.shape)
    job_b = (copy_b, gB, cB, sB, mB.shape)
    job_sfa = (copy_sf, gSFA, cSFA, sSFA, sf_write, mSFA.shape)
    job_sfb = (copy_sf, gSFB, cSFB, sSFB, sf_write, mSFB.shape)

    # Prologue: put the first STAGES-1 tiles in flight. Each iteration below then
    # waits for exactly one group, so every stage must contribute a group even when K
    # is too short to fill it -- an empty `commit_group` keeps the count aligned.
    for s in range(STAGES - 1):
        if s < n_k:
            _stage_k_tile(s, s, tidx, k_tail, job_a, job_b, job_sfa, job_sfb)
        cute.arch.cp_async_commit_group()

    for k_tile in cutlass.range(n_k, unroll=1):
        stage = k_tile % STAGES
        # STAGES-2 groups may still be pending; the oldest, this tile's, must not be
        cute.arch.cp_async_wait_group(STAGES - 2)
        # `wait_group` only covers this thread's own copies, and the stage was last
        # read STAGES-1 iterations ago -- the barrier closes both gaps at once
        cute.arch.barrier()

        # The barrier just freed the stage STAGES-1 tiles behind, so refill it before
        # anything else: the transfer then runs underneath both the shared->register
        # loads and the MMA. Issuing it first also keeps its address registers out of
        # the live range of the A/B fragments, which is what stops the mainloop
        # spilling. Reading stage `stage` below is unaffected -- a different buffer.
        nxt = k_tile + STAGES - 1
        if nxt < n_k:
            _stage_k_tile(
                nxt, nxt % STAGES, tidx, k_tail, job_a, job_b, job_sfa, job_sfb
            )
        cute.arch.cp_async_commit_group()

        cute.autovec_copy(thr_mma.partition_A(sA[None, None, stage]), tCrA)
        cute.autovec_copy(thr_mma.partition_B(sB[None, None, stage]), tCrB)
        cute.autovec_copy(
            _partition_mxfp8_sf(
                _sf_stage(sSFA, stage, sfa_smem_layout), tiled_mma, tidx, True
            ),
            tCrSFA,
        )
        cute.autovec_copy(
            _partition_mxfp8_sf(
                _sf_stage(sSFB, stage, sfb_smem_layout), tiled_mma, tidx, False
            ),
            tCrSFB,
        )
        cute.gemm(tiled_mma, acc, [tCrA, tCrSFA], [tCrB, tCrSFB], acc)

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
    mSFB: cute.Tensor,
    mD: cute.Tensor,
):
    tiled_mma = _make_mxfp8_tiled_mma(mA.element_type, mB.element_type)
    M, K = mA.shape
    N = mB.shape[0]
    # Both scale tensors arrive as (MN, scale-block) row-major, so B's is already the
    # tensor the kernel reads and is taken as `mSFB` unchanged. `sfa` is restated only
    # because M is dynamic: the extent of its block axis must be the *padded* count
    # the caller allocated, which is also its row pitch. Reading the padding is
    # deliberate -- those columns hold zeros, which is what spares the mainloop from
    # having to zero a short K tile's scales itself, and it leaves the scale copies
    # with no dropped vectors at all. The padding widens each row rather than
    # appending rows, so the row pitch is the padded count for *both* operands;
    # taking `mSFB`'s own layout keeps that automatic.
    padded_blocks = sfa.shape[1]
    mSFA = cute.make_tensor(
        sfa.iterator, cute.make_layout((M, padded_blocks), stride=(padded_blocks, 1))
    )
    tile = (TILE_M, TILE_N, TILE_K)
    # Both operands' scales are staged in the cuBLASLt-compatible 32x4x4 swizzle. Its
    # scale axis is contiguous, which is what makes the 4-byte `cp.async` legal now
    # that both scale tensors are scale-contiguous in global memory too. The two
    # layouts come out identical at this tile -- (((32,4),1),((32,1),4,1)) with
    # strides (((16,4),512),((0,1),1,512)) -- but they are built from their own
    # helpers rather than shared, since only the helpers know that. Measured worth
    # 4-5%: staging B's scales row-major instead cost 91.4 vs 95.0 TF/s at 4096^3.
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
    """mxfp8 GEMM, (M,K) x (N,K)^T -> (M,N), scales applied inside the MMA.

    Both operands are taken K-contiguous: `aq` is (M, K) and `bq` is (N, K), the
    layout an `nn.Linear` weight already has and the one the MMA wants, so neither
    is relayouted anywhere. Callers holding a (K, N) B pass `bq.t()`, a free view.

    `sa` is (M, K//32) and `sb` is (N, K//32), both fp32 holding exact powers of two
    -- what `quantize_operand(x, -1, fmt, ...)` emits for `scale_dtype="fp8_e8m0"` at
    block size 32 -- so the cast to e8m0 below is lossless.

    Either operand may be e4m3 or e5m2. M and N are arbitrary -- partial tiles are
    predicated -- but K must be a multiple of the 32-element scale block, which is
    what mxfp8 means rather than a limitation of this kernel.
    """
    M, K = aq.shape
    N = bq.shape[0]
    assert aq.dtype in FP8_FORMATS and bq.dtype in FP8_FORMATS, (
        f"the CuTe mxfp8 GEMM takes e4m3 or e5m2 operands, got {aq.dtype} x {bq.dtype}"
    )
    assert bq.shape[1] == K, f"contraction mismatch: aq {K}, bq {bq.shape[1]}"
    # the operand layouts are baked into the compile, so a non-contiguous operand
    # would be read through the wrong strides rather than rejected
    assert aq.is_contiguous() and bq.is_contiguous(), "aq and bq must be contiguous"
    # M and N are predicated, but a scale block is indivisible, so K is not
    assert K % SCALE_VEC == 0, f"K must be a multiple of {SCALE_VEC}, got {K}"
    assert sa.shape == (M, K // SCALE_VEC) and sb.shape == (N, K // SCALE_VEC), (
        f"scale shapes {tuple(sa.shape)} / {tuple(sb.shape)} do not match "
        f"({M}, {K // SCALE_VEC}) / ({N}, {K // SCALE_VEC})"
    )
    assert out_dtype in OUT_DTYPES, f"unsupported out_dtype {out_dtype}"

    out = torch.empty(M, N, device=aq.device, dtype=out_dtype)
    sa8 = sa.to(torch.float8_e8m0fnu).contiguous()
    sb8 = sb.to(torch.float8_e8m0fnu).contiguous()
    # cp.async moves a whole TILE_SF-scale group (4 bytes, one per k-tile) per copy.
    # When K % TILE_K != 0 the last k-tile's group only has a real scale in its first
    # block(s); the rest of that 4-byte group would read past the end of sa8/sb8.
    # Pad the scale-block axis up to a whole number of groups so the read is always
    # in-bounds. Zero, not empty or 0xFF (the e8m0 NaN encoding): the operand elements
    # those padding blocks would scale are zeroed in shared memory by `_copy_tile`'s
    # K-tail store, and 0 * anything = 0, but 0 * NaN = NaN would poison a whole
    # accumulator row. Do not remove: the mainloop's 4-byte scale `cp.async` reads
    # these columns on every short K tile, both to stay in bounds and to get the
    # zeros -- it does not predicate the scale copy along k at all.
    #
    # The block axis is the contiguous axis of both scale tensors, so padding widens
    # each row rather than appending rows: the row pitch becomes `padded_blocks` for
    # both, which is what `_scaled_gemm_mxfp8_host` reads back as the extent.
    n_scale_blocks = K // SCALE_VEC
    padded_blocks = -(-n_scale_blocks // TILE_SF) * TILE_SF
    if padded_blocks != n_scale_blocks:
        sa_padded = torch.zeros(M, padded_blocks, device=sa8.device, dtype=sa8.dtype)
        sa_padded[:, :n_scale_blocks] = sa8
        sa8 = sa_padded
        sb_padded = torch.zeros(N, padded_blocks, device=sb8.device, dtype=sb8.dtype)
        sb_padded[:, :n_scale_blocks] = sb8
        sb8 = sb_padded
    args = (
        from_dlpack(aq, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=aq.dim_order(), divisibility=1
        ),
        from_dlpack(bq, assumed_align=16),
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
