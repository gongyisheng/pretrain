"""Grouped GEMM latency: native PyTorch vs Triton, forward and backward.

Sweeps the eight expert-GEMM shapes of the latent-MoE configs (gate_up over
K in {64,128,256,512}; down over N in {64,128,256,512}) at two expert counts
E in {64, 256}, holding total rows M fixed. Triton runs through the public
operation and PyTorch calls `F.grouped_mm` directly. The eager backend is the
correctness oracle and is not timed. Prints a table and (unless --no-plot)
writes a chart.

    uv run python benchmarks/gemm/bench_grouped_mm.py
    uv run python benchmarks/gemm/bench_grouped_mm.py --out /tmp/gg.png
    uv run python benchmarks/gemm/bench_grouped_mm.py --no-plot
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from src.kernel.ops.gemm import grouped_mm
from src.layers.mlp import grouped_mm_fn

# Fixed total rows M = tokens * top_k (bs 8 * seq 1024 * top-k 8); rows/group = M/E.
M_FIXED = 8 * 1024 * 8
E_LIST = [64, 256]

# (role, K, N) — eight expert-GEMM shapes from the latent-MoE configs.
#   gate_up: N = 2*d_ff = 384, K = latent_dim (64/128/256) or d_model (512)
#   down:    K = d_ff = 192,   N = latent_dim (64/128/256) or d_model (512)
SHAPES = [
    ("gate_up", 64, 384),
    ("gate_up", 128, 384),
    ("gate_up", 256, 384),
    ("gate_up", 512, 384),
    ("down", 192, 64),
    ("down", 192, 128),
    ("down", 192, 256),
    ("down", 192, 512),
]

DEFAULT_OUT = "benchmarks/results/grouped_mm.png"

# blue/orange pair (validated colorblind-safe)
_BACKEND_COLOR = {"pytorch": "#eb6834", "triton": "#2a78d6"}


def _pytorch_grouped_mm(a, b, offs, bias=None):
    out = F.grouped_mm(a, b, offs=offs, bias=None)
    if bias is None:
        return out
    if out.ndim == 3:
        return out + bias[:, None, :]
    rows = torch.arange(out.shape[0], device=offs.device)
    return out + bias[torch.searchsorted(offs, rows, right=True)]


class _PyTorchGroupedGemmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, offs):
        ctx.save_for_backward(a, b, offs)
        return _pytorch_grouped_mm(a, b, offs)

    @staticmethod
    def backward(ctx, grad_c):
        a, b, offs = ctx.saved_tensors
        return (
            _pytorch_grouped_mm(grad_c, b.mT, offs),
            _pytorch_grouped_mm(a.mT, grad_c, offs),
            None,
        )


def _pytorch_grouped_mm_with_autograd(a, b, offs):
    return _PyTorchGroupedGemmFn.apply(a, b, offs)


def _make(E, M, K, N, requires_grad=False):
    torch.manual_seed(0)
    a = torch.randn(
        M, K, device="cuda", dtype=torch.bfloat16, requires_grad=requires_grad
    )
    b = (
        torch.randn(E, K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    ).requires_grad_(requires_grad)
    counts = torch.full((E,), M // E, device="cuda", dtype=torch.int64)
    counts[-1] += M - int(counts.sum())
    offs = counts.cumsum(0).to(torch.int32)
    return a, b, offs


def _assert_parity(got, ref, chunk=500_000):
    # row-independent output → chunked compare bounds peak fp32/isclose memory.
    # atol dominates: bf16 reduction-order noise (both impls accumulate in fp32,
    # cast out to bf16) reaches a few ULPs on the K=512 shapes, and rtol is
    # meaningless where ref ≈ 0 — so a small absolute floor is the right guard.
    for i in range(0, got.shape[0], chunk):
        torch.testing.assert_close(
            got[i : i + chunk].float(),
            ref[i : i + chunk].float(),
            rtol=2e-2,
            atol=6e-2,
        )


def _relative_error(out, eager):
    return ((out.float() - eager.float()).norm() / eager.float().norm()).item()


def _time(fn, iters=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def _bench_point(E, K, N):
    """Return {(backend, pass): ms} for one (E, shape) point: fwd + bwd, both impls."""
    out = {}
    a, b, offs = _make(E, M_FIXED, K, N)
    with torch.no_grad():
        eager = grouped_mm(a, b, offs, backend="eager")
        triton_out = grouped_mm(a, b, offs, backend="triton")
        _assert_parity(triton_out, eager)
        out[("triton", "relerr")] = _relative_error(triton_out, eager)
        out[("triton", "fwd")] = _time(lambda: grouped_mm(a, b, offs, backend="triton"))
        pytorch_out = _pytorch_grouped_mm(a, b, offs)
        _assert_parity(pytorch_out, eager)
        out[("pytorch", "relerr")] = _relative_error(pytorch_out, eager)
        out[("pytorch", "fwd")] = _time(lambda: _pytorch_grouped_mm(a, b, offs))
    for backend in ("pytorch", "triton"):
        if out[(backend, "fwd")] is None:
            out[(backend, "bwd")] = None
            continue
        ag, bg, offg = _make(E, M_FIXED, K, N, requires_grad=True)
        go = torch.randn(M_FIXED, N, device="cuda", dtype=torch.bfloat16)
        res = (
            _pytorch_grouped_mm_with_autograd(ag, bg, offg)
            if backend == "pytorch"
            else grouped_mm_fn(ag, bg, offg, backend=backend)
        )
        # backward-only: forward already ran; time repeated grad passes (retain_graph)
        out[(backend, "bwd")] = _time(lambda: res.backward(go, retain_graph=True))
    return out


def plot(results, path, device=""):
    """Write the 2x2 grouped-bar chart (rows = pass, cols = E) from `results`.

    Each result: {"role","K","N","E","backend","pass","ms"}. Import-safe without CUDA.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lut = {
        (r["role"], r["K"], r["N"], r["E"], r["backend"], r["pass"]): r["ms"]
        for r in results
    }
    # per-shape x-tick: the varying dim (K for gate_up, N for down)
    ticklabels = [f"K{K}" if role == "gate_up" else f"N{N}" for role, K, N in SHAPES]
    x = np.arange(len(SHAPES))
    bw = 0.4
    passes = [("fwd", "forward"), ("bwd", "backward")]

    fig, axes = plt.subplots(2, len(E_LIST), figsize=(12, 7.5), constrained_layout=True)
    for ri, (pass_, pass_lbl) in enumerate(passes):
        for ci, E in enumerate(E_LIST):
            ax = axes[ri][ci]
            for j, backend in enumerate(("pytorch", "triton")):
                vals = [
                    lut.get((role, K, N, E, backend, pass_), 0.0) or 0.0
                    for role, K, N in SHAPES
                ]
                bars = ax.bar(
                    x + (j - 0.5) * bw,
                    vals,
                    bw,
                    label=backend,
                    color=_BACKEND_COLOR[backend],
                    zorder=3,
                )
                ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
            ax.set_title(f"{pass_lbl}  ·  E={E}", fontsize=11, fontweight="bold")
            ax.set_xticks(x, ticklabels, fontsize=8)
            ax.set_ylabel("latency (ms)", fontsize=9)
            ax.margins(y=0.16)
            ax.grid(axis="y", color="#e1e0d9", zorder=0)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            # gate_up | down group separator + labels
            ax.axvline(3.5, color="#c3c2b7", lw=0.8, ls=":")
            ax.text(
                1.5,
                -0.14,
                "gate_up",
                ha="center",
                va="top",
                fontsize=8,
                color="#52514e",
                transform=ax.get_xaxis_transform(),
            )
            ax.text(
                5.5,
                -0.14,
                "down",
                ha="center",
                va="top",
                fontsize=8,
                color="#52514e",
                transform=ax.get_xaxis_transform(),
            )

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncol=2, fontsize=10)
    sup = "Grouped GEMM latency — PyTorch vs Triton"
    sub = f"M = {M_FIXED:,} rows · lower is faster"
    if device:
        sub += f" · {device}"
    fig.suptitle(f"{sup}\n{sub}", fontsize=12, fontweight="bold")

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _fmt(value, precision=3):
    return f"{value:.{precision}f}" if value is not None else "n/a"


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
    print(f"fixed M = {M_FIXED:,} rows\n")
    hdr = (
        f"{'shape':22s} {'E':>4s} {'rows/grp':>9s} {'pytorch_fwd':>11s} {'triton_fwd':>11s} "
        f"{'pytorch_bwd':>11s} {'triton_bwd':>11s} {'pytorch_err':>11s} {'triton_err':>11s}"
    )
    print(hdr)
    results = []
    for role, K, N in SHAPES:
        for E in E_LIST:
            t = _bench_point(E, K, N)
            label = f"{role} K{K} N{N}"
            print(
                f"{label:22s} {E:>4d} {M_FIXED // E:>9d} "
                f"{_fmt(t[('pytorch', 'fwd')]):>11s} {_fmt(t[('triton', 'fwd')]):>11s} "
                f"{_fmt(t[('pytorch', 'bwd')]):>11s} {_fmt(t[('triton', 'bwd')]):>11s} "
                f"{_fmt(t[('pytorch', 'relerr')], 4):>11s} "
                f"{_fmt(t[('triton', 'relerr')], 4):>11s}"
            )
            for (backend, pass_), ms in t.items():
                results.append(
                    {
                        "role": role,
                        "K": K,
                        "N": N,
                        "E": E,
                        "backend": backend,
                        "pass": pass_,
                        "ms": ms,
                    }
                )

    if not args.no_plot:
        plot(results, args.out, device=device)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
