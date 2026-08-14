"""Scaled GEMM latency: Triton `scaled_gemm` vs the native torch ops it replaces.

Sweeps FP8/INT8 tensorwise, rowwise, blockwise-1D, and blockwise-2D scaling
across three linear shapes from a 51M Qwen3-style config (d_model=512,
intermediate=1536) at two token counts M in {4096, 16384}. The mxfp8 sweep
compares CuTe, Triton, and native calls. Native calls
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
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from src.kernel.gemm import scaled_gemm
from src.kernel.cute.gemm import scaled_gemm_mxfp8
from src.quant.quantize import quantize_operand

E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2
MXFP8_BLOCK = 32

# Shapes from a real 51M Qwen3-style config.
D_MODEL = 512
INTERMEDIATE = 1536
M_LIST = [4096, 16384]

# (name, K, N) — three linear shapes: attn proj (d->d), MLP fused gate/up
# (d->6d), MLP down (inter->d).
SHAPES = [
    ("attn_proj", D_MODEL, D_MODEL),
    ("mlp_up", D_MODEL, 3072),
    ("mlp_down", INTERMEDIATE, D_MODEL),
]

BLOCK_SIZES = (16, 32, 64, 128)


def _scheme_configs():
    configs = []
    for dtype in ("fp8", "int8"):
        configs.extend(
            [
                {
                    "dtype": dtype,
                    "granularity": "tensorwise",
                    "block_size": None,
                    "label": f"{dtype}_tensorwise",
                },
                {
                    "dtype": dtype,
                    "granularity": "rowwise",
                    "block_size": None,
                    "label": f"{dtype}_rowwise",
                },
            ]
        )
        for granularity in ("blockwise1d", "blockwise2d"):
            for block_size in BLOCK_SIZES:
                label = (
                    "mxfp8"
                    if (dtype, granularity, block_size) == ("fp8", "blockwise1d", 32)
                    else f"{dtype}_{granularity}_{block_size}"
                )
                configs.append(
                    {
                        "dtype": dtype,
                        "granularity": granularity,
                        "block_size": block_size,
                        "label": label,
                    }
                )
    return configs


SCHEMES = [config["label"] for config in _scheme_configs()]

DEFAULT_OUT = "benchmarks/results/scaled_gemm.png"

# reference-palette blue/orange pair (validated colorblind-safe, matches
# bench_grouped_gemm.py's _IMPL_COLOR)
_SCHEME_COLOR = {"cute": "#3fae5c", "triton": "#2a78d6", "native": "#eb6834"}

_MX_SUPPORTED = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (
    10,
    0,
)


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


def _bench_mxfp8(a, b, ref, cute_schedule="cpasync"):
    scaling = _scaling("blockwise", MXFP8_BLOCK, "fp8_e8m0")
    aq, sa = quantize_operand(a, -1, _FMT[E4M3], scaling)
    bq, sb = quantize_operand(b, -2, _FMT[E4M3], scaling)
    bs = MXFP8_BLOCK
    sa_e8 = sa.to(torch.float8_e8m0fnu).contiguous()
    sb_e8 = sb.t().contiguous().to(torch.float8_e8m0fnu)
    bq_cute = bq.t().contiguous()

    # Native scale blocking and the column-major B view are one-time setup, not
    # part of any timed arm.
    a_blk = _to_blocked_native(sa_e8)
    b_blk = _to_blocked_native(sb_e8)
    b_arg = bq_cute.t()

    def triton_fn():
        return scaled_gemm(aq, bq, sa, sb, torch.bfloat16, bs, scale_dtype="fp8_e8m0")

    triton_out = triton_fn()
    triton_ms = _time(triton_fn)
    triton_relerr = _relerr(triton_out, ref)

    cute_ms = cute_relerr = None
    if torch.cuda.get_device_capability() >= (12, 0):

        def cute_fn():
            return scaled_gemm_mxfp8(
                aq.contiguous(),
                bq_cute,
                sa_e8,
                sb_e8,
                torch.bfloat16,
                schedule=cute_schedule,
            )

        cute_out = cute_fn()
        cute_ms = _time(cute_fn)
        cute_relerr = _relerr(cute_out, ref)

    native_ms = native_relerr = None
    if _MX_SUPPORTED:
        try:
            st = F.ScalingType.BlockWise1x32
            sw = F.SwizzleType.SWIZZLE_32_4_4

            def native_fn():
                return F.scaled_mm(
                    aq.contiguous(),
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
        cute_ms=cute_ms,
        triton_ms=triton_ms,
        native_ms=native_ms,
        cute_relerr=cute_relerr,
        triton_relerr=triton_relerr,
        native_relerr=native_relerr,
    )


def _bench_scheme(a, b, ref, config, cute_schedule):
    """Benchmark one quantization contract with preparation outside timed closures."""
    result = {
        "dtype": config["dtype"],
        "granularity": config["granularity"],
        "block_size": config["block_size"],
        "scheme": config["label"],
        "cute_ms": None,
        "triton_ms": None,
        "native_ms": None,
        "cute_relerr": None,
        "triton_relerr": None,
        "native_relerr": None,
    }
    if config["label"] == "mxfp8":
        result.update(_bench_mxfp8(a, b, ref, cute_schedule))
        if result["cute_ms"] is not None and result["native_ms"] is not None:
            result["cute_native_ratio"] = result["native_ms"] / result["cute_ms"]
        return result

    block_size = config["block_size"] or 0
    if config["granularity"] == "tensorwise":
        scaling = _scaling("tensorwise")
    elif config["granularity"] == "rowwise":
        scaling = _scaling("rowwise")
    else:
        shape = (
            (1, block_size)
            if config["granularity"] == "blockwise1d"
            else (block_size, block_size)
        )
        scaling = {
            "granularity": "blockwise",
            "block_shape": shape,
            "scale_dtype": "fp32",
        }
    fmt = _FMT[E4M3] if config["dtype"] == "fp8" else "int8"
    aq, sa = quantize_operand(a, -1, fmt, scaling)
    bq, sb = quantize_operand(b, -2, fmt, scaling)

    def triton_fn():
        return scaled_gemm(aq, bq, sa, sb, torch.bfloat16, block_size)

    triton_out = triton_fn()
    result["triton_ms"] = _time(triton_fn)
    result["triton_relerr"] = _relerr(triton_out, ref)
    try:
        if config["dtype"] == "fp8" and config["granularity"] in {
            "tensorwise",
            "rowwise",
        }:
            bq_col = _colmajor(bq)
            sa_n = (
                sa.reshape(-1)[:1].contiguous()
                if config["granularity"] == "tensorwise"
                else sa.contiguous()
            )
            sb_n = (
                sb.reshape(-1)[:1].contiguous()
                if config["granularity"] == "tensorwise"
                else sb.contiguous()
            )

            def native_fn():
                return torch._scaled_mm(
                    aq.contiguous(), bq_col, sa_n, sb_n, out_dtype=torch.bfloat16
                )

        elif config["dtype"] == "int8" and config["granularity"] in {
            "tensorwise",
            "rowwise",
        }:
            aqc, bqc = aq.contiguous(), bq.contiguous()

            def native_fn():
                return (torch._int_mm(aqc, bqc).float() * sa * sb).to(torch.bfloat16)

        else:
            return result
        native_out = native_fn()
        result["native_ms"] = _time(native_fn)
        result["native_relerr"] = _relerr(native_out, ref)
    except Exception:
        pass
    return result


def _bench_shape(name, M, K, N, cute_schedule):
    """Return all 20 matrix rows for one (name, M, K, N) shape."""
    a, b = _make(M, K, N)
    ref = a.float() @ b.float()
    return [
        {
            "shape": name,
            "M": M,
            "K": K,
            "N": N,
            **_bench_scheme(a, b, ref, config, cute_schedule),
        }
        for config in _scheme_configs()
    ]


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
    bw = 0.25

    fig, axes = plt.subplots(
        len(M_LIST), len(shape_names), figsize=(13, 8), constrained_layout=True
    )
    for ri, M in enumerate(M_LIST):
        for ci, shape_name in enumerate(shape_names):
            ax = axes[ri][ci]
            for j, impl in enumerate(("cute", "triton", "native")):
                key = f"{impl}_ms"
                vals = []
                for scheme in SCHEMES:
                    r = lut.get((shape_name, M, scheme))
                    value = r.get(key) if r else None
                    vals.append(value if value is not None else 0.0)
                bars = ax.bar(
                    x + (j - 1) * bw,
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
    fig.legend(handles, labels, loc="outside upper right", ncol=3, fontsize=10)
    sup = "Scaled GEMM latency — CuTe vs Triton vs native"
    sub = "lower is faster · n/a = call unsupported on this GPU"
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
    ap.add_argument("--json-out", help="write matrix rows as JSON")
    ap.add_argument(
        "--cute-schedule",
        default="cpasync",
        help="CuTe MXFP8 execution schedule",
    )
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.cuda.get_device_name(0)
    print(device)
    print(
        f"d_model={D_MODEL} intermediate={INTERMEDIATE} mx_supported={_MX_SUPPORTED}\n"
    )

    hdr = (
        f"{'shape':12s} {'M':>7s} {'scheme':16s} {'cute_ms':>10s} {'triton_ms':>10s} "
        f"{'native_ms':>10s} {'cute_relerr':>14s} {'triton_relerr':>14s} "
        f"{'native_relerr':>14s}"
    )
    print(hdr)
    results = []
    for name, K, N in SHAPES:
        for M in M_LIST:
            rows = _bench_shape(name, M, K, N, args.cute_schedule)
            for r in rows:
                print(
                    f"{r['shape']:12s} {r['M']:>7d} {r['scheme']:16s} "
                    f"{_fmt(r.get('cute_ms')):>10s} {_fmt(r['triton_ms']):>10s} "
                    f"{_fmt(r['native_ms']):>10s} {_fmt(r.get('cute_relerr'), 4):>14s} "
                    f"{_fmt(r['triton_relerr'], 4):>14s} {_fmt(r['native_relerr'], 4):>14s}"
                )
                results.append(r)

    if args.json_out:
        with open(args.json_out, "w") as output_file:
            json.dump({"rows": results}, output_file, indent=2)
        print(f"wrote {args.json_out}")

    if not args.no_plot:
        plot(results, args.out, device=device)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
