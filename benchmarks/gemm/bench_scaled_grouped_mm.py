"""Scaled grouped GEMM latency: Triton vs native PyTorch.

Sweeps the expert-GEMM shapes of the latent-MoE configs (gate_up over K, down
over N) at two expert counts E in {64, 256}, holding total rows M fixed, under
fp8-rowwise and int8-rowwise quantization. Accuracy is relative error against
the eager backend for identical quantized operands. Unsupported native PyTorch
contracts are reported as `n/a`.

    uv run python benchmarks/gemm/bench_scaled_grouped_mm.py
    uv run python benchmarks/gemm/bench_scaled_grouped_mm.py --no-plot
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from src.kernel.selector import dispatch
from src.kernel.utils import to_column_major
from src.quant.quantize import quantize_operand
from src.quant.utils import scaled_grouped_mm_op

# Fixed total rows M = tokens * top_k (bs 8 * seq 1024 * top-k 8); rows/group = M/E.
M_FIXED = 8 * 1024 * 8
E_LIST = [64, 256]

# (role, K, N) — expert-GEMM shapes from the latent-MoE configs.
SHAPES = [
    ("gate_up", 64, 384),
    ("gate_up", 256, 384),
    ("gate_up", 512, 384),
    ("down", 192, 64),
    ("down", 192, 256),
    ("down", 192, 512),
]

SCHEMES = ["fp8_rowwise", "int8_rowwise"]

DEFAULT_OUT = "benchmarks/results/scaled_grouped_mm.png"

# blue/orange pair (validated colorblind-safe, matches sibling benchmarks)
_IMPL_COLOR = {"pytorch": "#eb6834", "triton": "#2a78d6"}


def _fmt(scheme):
    return "fp8_e4m3" if scheme == "fp8_rowwise" else "int8"


_ROWWISE = {
    "granularity": "rowwise",
    "block_shape": (0, 0),
    "scale_dtype": torch.float32,
}


def _make(E, M, K, N, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(E, K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    counts = torch.full((E,), M // E, device="cuda", dtype=torch.int64)
    counts[-1] += M - int(counts.sum())
    offs = counts.cumsum(0).to(torch.int32)
    return a, b, offs


def _quant(a, b, scheme):
    """Quantize A (M,K) and per-expert B (E,K,N) rowwise; return aq,bq,sa,sb,bs."""
    fmt = _fmt(scheme)
    bs = 0  # rowwise: one scale block per contraction segment
    aq, sa = quantize_operand(a, -1, fmt, _ROWWISE)
    bq, sb = quantize_operand(b, -2, fmt, _ROWWISE)  # (E,K,N) in one call
    return aq, bq, sa, sb, bs


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


def _scaled_grouped_mm(fmt, aq, bq, sa, sb, offs, out_dtype, block_size, backend=None):
    op = scaled_grouped_mm_op(fmt, fmt, torch.float32, _ROWWISE["block_shape"])
    return dispatch(
        op,
        (aq, bq, sa, sb, offs, out_dtype, block_size, None),
        {},
        backend,
        device=aq.device,
    )


def _pytorch_scaled_grouped_mm(aq, bq, sa, sb, offs, out_dtype):
    return F.scaled_grouped_mm(
        aq,
        to_column_major(bq),
        sa.squeeze(1),
        F.ScalingType.RowWise,
        sb.squeeze(1),
        F.ScalingType.RowWise,
        bias=None,
        offs=offs,
        output_dtype=out_dtype,
    )


def _bench_point(E, K, N, scheme):
    """Return optimized latency and error for one forward benchmark point."""
    a, b, offs = _make(E, M_FIXED, K, N)
    aq, bq, sa, sb, bs = _quant(a, b, scheme)
    fmt = _fmt(scheme)
    ref = _scaled_grouped_mm(
        fmt, aq, bq, sa, sb, offs, torch.bfloat16, bs, backend="eager"
    )

    def triton_fn():
        return _scaled_grouped_mm(
            fmt, aq, bq, sa, sb, offs, torch.bfloat16, bs, backend="triton"
        )

    triton_out = triton_fn()
    triton_ms = _time(triton_fn)
    triton_relerr = _relerr(triton_out, ref)

    def pytorch_fn():
        return _pytorch_scaled_grouped_mm(aq, bq, sa, sb, offs, torch.bfloat16)

    try:
        pytorch_out = pytorch_fn()
    except (AttributeError, NotImplementedError, RuntimeError, ValueError):
        pytorch_ms = None
        pytorch_relerr = None
    else:
        pytorch_ms = _time(pytorch_fn)
        pytorch_relerr = _relerr(pytorch_out, ref)
    return dict(
        triton_ms=triton_ms,
        pytorch_ms=pytorch_ms,
        triton_relerr=triton_relerr,
        pytorch_relerr=pytorch_relerr,
    )


def _bench_wgrad_point(E, K, N, scheme):
    """Return optimized latency and error for one weight-gradient benchmark point.

    Both operands are indexed by the same ragged token axis, so one block mapping
    covers them -- matching what src/quant/moe.py does. The bf16 baseline is the
    ragged-K contract is intentionally unsupported by native PyTorch.
    """
    a, _, offs = _make(E, M_FIXED, K, N)
    g = torch.randn(M_FIXED, N, device="cuda", dtype=torch.bfloat16) * 0.1
    fmt = _fmt(scheme)
    aq, sa = quantize_operand(a, -2, fmt, _ROWWISE, offs=offs, ragged_dim=-2)
    gq, sg = quantize_operand(g, -2, fmt, _ROWWISE, offs=offs, ragged_dim=-2)
    ref = _scaled_grouped_mm(
        fmt, aq.mT, gq, sa.mT, sg, offs, torch.bfloat16, 0, backend="eager"
    )

    def triton_fn():
        return _scaled_grouped_mm(
            fmt, aq.mT, gq, sa.mT, sg, offs, torch.bfloat16, 0, backend="triton"
        )

    triton_out = triton_fn()
    triton_ms = _time(triton_fn)
    return dict(
        triton_ms=triton_ms,
        pytorch_ms=None,
        triton_relerr=_relerr(triton_out, ref),
        pytorch_relerr=None,
    )


def plot(results, path, device=""):
    """Write a grouped-bar chart (rows = scheme, cols = E) from `results`.

    Each result: {"role","K","N","E","scheme","triton_ms","pytorch_ms",...}.
    Import-safe without CUDA.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lut = {(r["role"], r["K"], r["N"], r["E"], r["scheme"]): r for r in results}
    ticklabels = [f"K{K}" if role == "gate_up" else f"N{N}" for role, K, N in SHAPES]
    x = np.arange(len(SHAPES))
    bw = 0.4

    fig, axes = plt.subplots(
        len(SCHEMES), len(E_LIST), figsize=(12, 7.5), constrained_layout=True
    )
    for ri, scheme in enumerate(SCHEMES):
        for ci, E in enumerate(E_LIST):
            ax = axes[ri][ci]
            for j, impl in enumerate(("pytorch", "triton")):
                key = f"{impl}_ms"
                vals = [
                    (lut.get((role, K, N, E, scheme)) or {}).get(key, 0.0)
                    for role, K, N in SHAPES
                ]
                bars = ax.bar(
                    x + (j - 0.5) * bw,
                    vals,
                    bw,
                    label=impl,
                    color=_IMPL_COLOR[impl],
                    zorder=3,
                )
                labels = [f"{value:.2f}" if value else "n/a" for value in vals]
                ax.bar_label(bars, labels=labels, fontsize=7, padding=2)
            ax.set_title(f"{scheme}  ·  E={E}", fontsize=11, fontweight="bold")
            ax.set_xticks(x, ticklabels, fontsize=8)
            ax.set_ylabel("latency (ms)", fontsize=9)
            ax.margins(y=0.16)
            ax.grid(axis="y", color="#e1e0d9", zorder=0)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.axvline(2.5, color="#c3c2b7", lw=0.8, ls=":")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncol=2, fontsize=10)
    sup = "Scaled grouped GEMM latency — native PyTorch vs Triton"
    sub = f"eager oracle · M = {M_FIXED:,} rows · lower is faster"
    if device:
        sub += f" · {device}"
    fig.suptitle(f"{sup}\n{sub}", fontsize=12, fontweight="bold")

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _display(value, precision=3):
    return f"{value:.{precision}f}" if value is not None else "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", default=DEFAULT_OUT, help="path to write the latency chart"
    )
    ap.add_argument("--no-plot", action="store_true", help="skip writing the chart")
    ap.add_argument(
        "--direction",
        choices=["fwd", "wgrad"],
        default="fwd",
        help="fwd: (M,K)x(E,K,N). wgrad: ragged contraction, X^T @ gY -> (E,K,N). "
        "wgrad is table-only (no chart).",
    )
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.cuda.get_device_name(0)
    print(device)
    print(f"fixed M = {M_FIXED:,} rows\n")
    hdr = (
        f"{'shape':22s} {'E':>4s} {'scheme':13s} {'pytorch_ms':>10s} {'triton_ms':>10s} "
        f"{'pytorch_relerr':>14s} {'triton_relerr':>14s}"
    )
    print(hdr)
    point = _bench_wgrad_point if args.direction == "wgrad" else _bench_point
    results = []
    for role, K, N in SHAPES:
        for E in E_LIST:
            for scheme in SCHEMES:
                r = point(E, K, N, scheme)
                label = f"{role} K{K} N{N}"
                print(
                    f"{label:22s} {E:>4d} {scheme:13s} "
                    f"{_display(r['pytorch_ms']):>10s} {r['triton_ms']:>10.3f} "
                    f"{_display(r['pytorch_relerr'], 4):>14s} "
                    f"{r['triton_relerr']:>14.4f}"
                )
                results.append(
                    {"role": role, "K": K, "N": N, "E": E, "scheme": scheme, **r}
                )

    if not args.no_plot and args.direction == "fwd":
        plot(results, args.out, device=device)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
