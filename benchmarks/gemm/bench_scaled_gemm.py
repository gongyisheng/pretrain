"""Scaled GEMM latency: Triton `scaled_gemm` vs the native torch ops it replaces.

Sweeps fp8 tensorwise, fp8 rowwise, int8 rowwise, and mxfp8 (block-32) scaling
across three linear shapes from a 51M Qwen3-style config (d_model=512,
intermediate=1536) at two token counts M in {4096, 16384}. Native calls
(`torch._scaled_mm`, `torch._int_mm`, an inlined mxfp8 GEMM recovered from the
deleted `src/quant/mxfp8.py`) are each wrapped in try/except: combos this
GPU/cuBLAS rejects (e.g. e5m2 x e5m2, non-Blackwell MX) print `n/a` instead of
crashing the sweep. Latency is measured via `_time`; accuracy is relative
error against an fp32 reference `a.float() @ b.float()`. Prints a table and
(unless --no-plot) writes a matplotlib grouped-bar chart.

    uv run python benchmarks/gemm/bench_scaled_gemm.py
    uv run python benchmarks/gemm/bench_scaled_gemm.py --no-plot
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from src.kernel.gemm import scaled_gemm
from src.quant.quantize import quantize_operand

E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2
FP8_MAX = {E4M3: torch.finfo(E4M3).max, E5M2: torch.finfo(E5M2).max}

MXFP8_BLOCK = 32

# Shapes from a real 51M Qwen3-style config.
D_MODEL = 512
INTERMEDIATE = 1536
M_LIST = [4096, 16384]

# (name, K, N) — three linear shapes: attn proj (d->d), MLP up (d->4d-ish
# gate/up proj), MLP down (inter->d).
SHAPES = [
    ("attn_proj", D_MODEL, D_MODEL),
    ("mlp_up", D_MODEL, INTERMEDIATE),
    ("mlp_down", INTERMEDIATE, D_MODEL),
]

SCHEMES = ["fp8_tensorwise", "fp8_rowwise", "int8_rowwise", "mxfp8"]

DEFAULT_OUT = "benchmarks/results/scaled_gemm.png"

# reference-palette blue/orange pair (validated colorblind-safe, matches
# bench_grouped_gemm.py's _IMPL_COLOR)
_SCHEME_COLOR = {"triton": "#2a78d6", "native": "#eb6834"}

_MX_SUPPORTED = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (
    10,
    0,
)


# ---------------------------------------------------------------------------
# Inlined native mxfp8 path (recovered from the deleted src/quant/mxfp8.py via
# `git show 8e339cf:src/quant/mxfp8.py`) — self-contained, no src.quant.mxfp8
# import since that module no longer exists.
# ---------------------------------------------------------------------------

_MX_E8M0_EXP_MIN, _MX_E8M0_EXP_MAX = -127, 127
_MX_EPS = 1e-30


def _quantize_mxfp8_native(x, fp8_dtype, dim):
    """Blockwise mxfp8 quantization along `dim`; returns (x_fp8, E8M0 scale)."""
    fp8_max = FP8_MAX[fp8_dtype]
    xf = x.movedim(dim, -1).contiguous()
    k = xf.shape[-1]
    assert k % MXFP8_BLOCK == 0, (
        f"mxfp8 block dim {k} is not a multiple of {MXFP8_BLOCK}"
    )
    xb = xf.unflatten(-1, (k // MXFP8_BLOCK, MXFP8_BLOCK)).float()
    amax = xb.abs().amax(dim=-1)
    # matches src.quant.quantize._compute_scale: ceil against fp8_max, so amax/scale
    # stays at or below fp8_max and the block maximum never saturates
    exp = torch.ceil(torch.log2(amax.clamp_min(_MX_EPS) / fp8_max)).clamp(
        _MX_E8M0_EXP_MIN, _MX_E8M0_EXP_MAX
    )
    scale = torch.exp2(exp)
    xq = (xb / scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    xq = xq.flatten(-2).movedim(-1, dim).contiguous()
    scale_e8 = scale.movedim(-1, dim).to(torch.float8_e8m0fnu)
    return xq, scale_e8


def _to_blocked_native(scale_2d):
    """Rearrange a (rows, n_blocks) E8M0 scale into the SWIZZLE_32_4_4 layout."""

    def _ceil_div(a, b):
        return (a + b - 1) // b

    rows, cols = scale_2d.shape
    n_row_tiles, n_col_tiles = _ceil_div(rows, 128), _ceil_div(cols, 4)
    padded_rows, padded_cols = n_row_tiles * 128, n_col_tiles * 4
    padded = scale_2d
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols), device=scale_2d.device, dtype=scale_2d.dtype
        )
        padded[:rows, :cols] = scale_2d
    blocks = padded.view(n_row_tiles, 128, n_col_tiles, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _time(fn, iters=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def _relerr(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


def _colmajor(x):
    """Col-major-strided view of contiguous (K, N) x, for torch._scaled_mm's mat2."""
    return x.t().contiguous().t()


def _make(M, K, N):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    return a, b


_FMT = {E4M3: "fp8_e4m3", E5M2: "fp8_e5m2"}


def _scaling(gran, bs=0, scale_dtype=None):
    return {
        "granularity": gran,
        "block_shape": (1, bs) if bs else (0, 0),
        "scale_dtype": scale_dtype,
    }


# ---------------------------------------------------------------------------
# Per-scheme benchmarks: {triton_ms, native_ms, triton_relerr, native_relerr}
# native_* are None (printed as n/a) when the native call is unsupported.
# ---------------------------------------------------------------------------


def _bench_fp8_tensorwise(a, b, ref):
    scaling = _scaling("tensorwise")
    aq, sa = quantize_operand(a, -1, _FMT[E4M3], scaling)
    bq, sb = quantize_operand(b, -2, _FMT[E4M3], scaling)
    bs = 0  # one scale block per contraction segment

    def triton_fn():
        return scaled_gemm(aq, bq, sa, sb, torch.bfloat16, bs)

    triton_out = triton_fn()
    triton_ms = _time(triton_fn)
    triton_relerr = _relerr(triton_out, ref)

    native_ms = native_relerr = None
    try:
        sa_n = sa.reshape(-1)[:1].contiguous()
        sb_n = sb.reshape(-1)[:1].contiguous()
        bq_col = _colmajor(bq)

        def native_fn():
            return torch._scaled_mm(
                aq.contiguous(), bq_col, sa_n, sb_n, out_dtype=torch.bfloat16
            )

        native_out = native_fn()
        native_ms = _time(native_fn)
        native_relerr = _relerr(native_out, ref)
    except Exception:
        pass
    return dict(
        triton_ms=triton_ms,
        native_ms=native_ms,
        triton_relerr=triton_relerr,
        native_relerr=native_relerr,
    )


def _bench_fp8_rowwise(a, b, ref):
    scaling = _scaling("rowwise")
    aq, sa = quantize_operand(a, -1, _FMT[E4M3], scaling)
    bq, sb = quantize_operand(b, -2, _FMT[E4M3], scaling)
    bs = 0  # one scale block per contraction segment

    def triton_fn():
        return scaled_gemm(aq, bq, sa, sb, torch.bfloat16, bs)

    triton_out = triton_fn()
    triton_ms = _time(triton_fn)
    triton_relerr = _relerr(triton_out, ref)

    native_ms = native_relerr = None
    try:
        bq_col = _colmajor(bq)

        def native_fn():
            return torch._scaled_mm(
                aq.contiguous(),
                bq_col,
                sa.contiguous(),
                sb.contiguous(),
                out_dtype=torch.bfloat16,
            )

        native_out = native_fn()
        native_ms = _time(native_fn)
        native_relerr = _relerr(native_out, ref)
    except Exception:
        pass
    return dict(
        triton_ms=triton_ms,
        native_ms=native_ms,
        triton_relerr=triton_relerr,
        native_relerr=native_relerr,
    )


def _bench_int8_rowwise(a, b, ref, M, K, N):
    if M <= 16 or K % 8 or N % 8:
        return None  # torch._int_mm's shape constraints aren't met — skip entirely

    scaling = _scaling("rowwise")
    aq, sa = quantize_operand(a, -1, "int8", scaling)
    bq, sb = quantize_operand(b, -2, "int8", scaling)
    bs = 0  # one scale block per contraction segment

    def triton_fn():
        return scaled_gemm(aq, bq, sa, sb, torch.bfloat16, bs)

    triton_out = triton_fn()
    triton_ms = _time(triton_fn)
    triton_relerr = _relerr(triton_out, ref)

    native_ms = native_relerr = None
    try:
        aqc, bqc = aq.contiguous(), bq.contiguous()

        def native_fn():
            return (torch._int_mm(aqc, bqc).float() * sa * sb).to(torch.bfloat16)

        native_out = native_fn()
        native_ms = _time(native_fn)
        native_relerr = _relerr(native_out, ref)
    except Exception:
        pass
    return dict(
        triton_ms=triton_ms,
        native_ms=native_ms,
        triton_relerr=triton_relerr,
        native_relerr=native_relerr,
    )


def _bench_mxfp8(a, b, ref):
    scaling = _scaling("blockwise", MXFP8_BLOCK, "fp8_e8m0")
    aq, sa = quantize_operand(a, -1, _FMT[E4M3], scaling)
    bq, sb = quantize_operand(b, -2, _FMT[E4M3], scaling)
    bs = MXFP8_BLOCK

    def triton_fn():
        return scaled_gemm(aq, bq, sa, sb, torch.bfloat16, bs, scale_dtype="fp8_e8m0")

    triton_out = triton_fn()
    triton_ms = _time(triton_fn)
    triton_relerr = _relerr(triton_out, ref)

    native_ms = native_relerr = None
    if _MX_SUPPORTED:
        try:
            # hoisted out of the timed region so this measures the GEMM, not the
            # quantize+swizzle, which is what the Triton arm above also excludes
            a_native, b_native = a, b
            pad = (-a_native.shape[-1]) % MXFP8_BLOCK
            if pad:  # K auto-padded to a multiple of 32, as mxfp8 quantization requires
                a_native = F.pad(a_native, (0, pad))
                b_native = F.pad(b_native, (0, 0, 0, pad))
            a_fp8, a_scale = _quantize_mxfp8_native(a_native, E4M3, dim=-1)
            b_fp8, b_scale = _quantize_mxfp8_native(b_native, E4M3, dim=0)
            a_blk = _to_blocked_native(a_scale)
            b_blk = _to_blocked_native(b_scale.t().contiguous())
            a_arg = a_fp8.contiguous()
            b_arg = b_fp8.t().contiguous().t()  # (K, N) stored column-major
            st = F.ScalingType.BlockWise1x32
            sw = F.SwizzleType.SWIZZLE_32_4_4

            def native_fn():
                return F.scaled_mm(
                    a_arg,
                    b_arg,
                    a_blk,
                    st,
                    b_blk,
                    st,
                    swizzle_a=sw,
                    swizzle_b=sw,
                    output_dtype=torch.bfloat16,
                )

            native_out = native_fn()
            native_ms = _time(native_fn)
            native_relerr = _relerr(native_out, ref)
        except Exception:
            pass
    return dict(
        triton_ms=triton_ms,
        native_ms=native_ms,
        triton_relerr=triton_relerr,
        native_relerr=native_relerr,
    )


def _bench_blockwise_2d(a, b, ref, tile):
    """Square tile vs the 1D block of the same contract extent, fp32 scales.

    Both arms take the epilogue kernel, so the difference is purely what the outer-axis
    pooling costs -- and the tile sweep shows what the K extent costs.
    """
    scaling_2d = {
        "granularity": "blockwise",
        "block_shape": (tile, tile),
        "scale_dtype": "fp32",
    }
    scaling_1d = {
        "granularity": "blockwise",
        "block_shape": (1, tile),
        "scale_dtype": "fp32",
    }

    def arm(scaling):
        aq, sa = quantize_operand(a, -1, _FMT[E4M3], scaling)
        bq, sb = quantize_operand(b, -2, _FMT[E4M3], scaling)

        def fn():
            return scaled_gemm(aq, bq, sa, sb, torch.bfloat16, tile)

        return _time(fn), _relerr(fn(), ref)

    ms_2d, relerr_2d = arm(scaling_2d)
    ms_1d, relerr_1d = arm(scaling_1d)
    return dict(
        tile=tile,
        ms_2d=ms_2d,
        ms_1d=ms_1d,
        relerr_2d=relerr_2d,
        relerr_1d=relerr_1d,
    )


def _bench_blockwise_2d_sweep(name, M, K, N):
    """Run the 2D-vs-1D tile sweep for one shape, returning per-tile result rows."""
    a, b = _make(M, K, N)
    ref = a.float() @ b.float()
    return [
        {"shape": name, "M": M, "K": K, "N": N, **_bench_blockwise_2d(a, b, ref, tile)}
        for tile in (16, 32, 64, 128)
    ]


def _bench_shape(name, M, K, N):
    """Return a list of per-scheme result rows for one (name, M, K, N) shape."""
    a, b = _make(M, K, N)
    ref = a.float() @ b.float()
    rows = []
    for scheme, fn in [
        ("fp8_tensorwise", lambda: _bench_fp8_tensorwise(a, b, ref)),
        ("fp8_rowwise", lambda: _bench_fp8_rowwise(a, b, ref)),
        ("int8_rowwise", lambda: _bench_int8_rowwise(a, b, ref, M, K, N)),
        ("mxfp8", lambda: _bench_mxfp8(a, b, ref)),
    ]:
        res = fn()
        if res is None:
            continue  # shape skipped for this scheme (e.g. int8 constraints)
        rows.append({"shape": name, "M": M, "K": K, "N": N, "scheme": scheme, **res})
    return rows


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot(results, path, device=""):
    """Write a grouped-bar chart (rows = M, cols = shape) from `results`.

    Each result: {"shape","M","K","N","scheme","triton_ms","native_ms",...}.
    Import-safe without CUDA.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lut = {(r["shape"], r["M"], r["scheme"]): r for r in results}
    shape_names = [name for name, _, _ in SHAPES]
    x = np.arange(len(SCHEMES))
    bw = 0.4

    fig, axes = plt.subplots(
        len(M_LIST), len(shape_names), figsize=(13, 8), constrained_layout=True
    )
    for ri, M in enumerate(M_LIST):
        for ci, shape_name in enumerate(shape_names):
            ax = axes[ri][ci]
            for j, impl in enumerate(("triton", "native")):
                key = f"{impl}_ms"
                vals = []
                for scheme in SCHEMES:
                    r = lut.get((shape_name, M, scheme))
                    vals.append(r[key] if r and r[key] is not None else 0.0)
                bars = ax.bar(
                    x + (j - 0.5) * bw,
                    vals,
                    bw,
                    label=impl,
                    color=_SCHEME_COLOR[impl],
                    zorder=3,
                )
                labels = [f"{v:.3f}" if v else "n/a" for v in vals]
                ax.bar_label(bars, labels=labels, fontsize=7, padding=2)
            ax.set_title(f"{shape_name}  ·  M={M:,}", fontsize=10, fontweight="bold")
            ax.set_xticks(x, SCHEMES, fontsize=7, rotation=15)
            ax.set_ylabel("latency (ms)", fontsize=9)
            ax.margins(y=0.18)
            ax.grid(axis="y", color="#e1e0d9", zorder=0)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncol=2, fontsize=10)
    sup = "Scaled GEMM latency — Triton vs native"
    sub = "lower is faster · n/a = native call unsupported on this GPU"
    if device:
        sub += f" · {device}"
    fig.suptitle(f"{sup}\n{sub}", fontsize=12, fontweight="bold")

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _fmt(v, prec=3):
    return f"{v:.{prec}f}" if v is not None else "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", default=DEFAULT_OUT, help="path to write the latency chart"
    )
    ap.add_argument("--no-plot", action="store_true", help="skip writing the chart")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.cuda.get_device_name(0)
    print(device)
    print(
        f"d_model={D_MODEL} intermediate={INTERMEDIATE} mx_supported={_MX_SUPPORTED}\n"
    )

    hdr = (
        f"{'shape':12s} {'M':>7s} {'scheme':16s} {'triton_ms':>10s} {'native_ms':>10s} "
        f"{'triton_relerr':>14s} {'native_relerr':>14s}"
    )
    print(hdr)
    results = []
    for name, K, N in SHAPES:
        for M in M_LIST:
            rows = _bench_shape(name, M, K, N)
            for r in rows:
                print(
                    f"{r['shape']:12s} {r['M']:>7d} {r['scheme']:16s} "
                    f"{_fmt(r['triton_ms']):>10s} {_fmt(r['native_ms']):>10s} "
                    f"{_fmt(r['triton_relerr'], 4):>14s} {_fmt(r['native_relerr'], 4):>14s}"
                )
                results.append(r)

    if not args.no_plot:
        plot(results, args.out, device=device)
        print(f"\nwrote {args.out}")

    print("\n2D blockwise (square tile) vs 1D at the same contract extent, fp32 scales")
    tile_hdr = (
        f"{'shape':12s} {'M':>7s} {'tile':>6s} {'ms_2d':>10s} {'ms_1d':>10s} "
        f"{'relerr_2d':>12s} {'relerr_1d':>12s}"
    )
    print(tile_hdr)
    for name, K, N in SHAPES:
        for M in M_LIST:
            for r in _bench_blockwise_2d_sweep(name, M, K, N):
                print(
                    f"{r['shape']:12s} {r['M']:>7d} {r['tile']:>6d} "
                    f"{_fmt(r['ms_2d']):>10s} {_fmt(r['ms_1d']):>10s} "
                    f"{_fmt(r['relerr_2d'], 4):>12s} {_fmt(r['relerr_1d'], 4):>12s}"
                )


if __name__ == "__main__":
    main()
