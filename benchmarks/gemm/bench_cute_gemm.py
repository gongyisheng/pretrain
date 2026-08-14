"""mxfp8 scaled GEMM latency: CuTe DSL kernel vs Triton vs cuBLASLt `F.scaled_mm`.

The CuTe DSL mxfp8 GEMM (`src/kernel/cute/gemm.py`, sm120) is the kernel under
development; this benchmark is the instrument its performance work is measured
against, and cuBLASLt is both the performance target and (via
`test_scaled_gemm_mxfp8_agrees_with_scaled_mm`) the feed-path oracle it is
cross-checked against.

Quantization and scale swizzling are hoisted out of every timed closure --
each arm's timed region contains only the GEMM call. Getting this wrong once
made cuBLASLt look 5x slower than it is; see `bench_scaled_gemm.py` for the
fix this benchmark carries forward.

Sweeps five shapes: two from a 51M Qwen3-style config (M=16384 tokens, the
attn_proj-adjacent up/down projections) and three square shapes
(2048/4096/8192). Reports latency (ms) and throughput (TFLOP/s =
2*M*K*N/ms/1e9) per arm, plus relative error against an fp32 reference.

    uv run python benchmarks/gemm/bench_cute_gemm.py
    uv run python benchmarks/gemm/bench_cute_gemm.py --no-plot
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from src.kernel.cute.gemm import scaled_gemm_mxfp8
from src.kernel.triton.gemm import scaled_gemm
from src.quant.quantize import quantize_operand

MXFP8_BLOCK = 32
MX = {"granularity": "blockwise", "block_size": MXFP8_BLOCK, "scale_dtype": "fp8_e8m0"}

# (M, K, N): mlp_up and mlp_down from a 51M Qwen3-style config (d_model=512,
# intermediate=1536) at M=16384 tokens, plus three square shapes.
SHAPES = [
    (16384, 512, 1536),
    (16384, 1536, 512),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]

ARMS = ["cute", "triton", "cublaslt"]

DEFAULT_OUT = "benchmarks/results/cute_gemm.png"

# reference-palette 3-way categorical set, consistent with _SCHEME_COLOR in
# bench_scaled_gemm.py
_ARM_COLOR = {"cute": "#2a78d6", "triton": "#eb6834", "cublaslt": "#3fae5c"}


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


def _to_blocked(scale_2d):
    """Rearrange an (rows, n_blocks) e8m0 scale into cuBLASLt's SWIZZLE_32_4_4 layout."""
    rows, cols = scale_2d.shape
    n_row_tiles, n_col_tiles = (rows + 127) // 128, (cols + 3) // 4
    padded_rows, padded_cols = n_row_tiles * 128, n_col_tiles * 4
    padded = scale_2d
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols), device=scale_2d.device, dtype=scale_2d.dtype
        )
        padded[:rows, :cols] = scale_2d
    blocks = padded.view(n_row_tiles, 128, n_col_tiles, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _make(M, K, N):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    return a, b


def _tflops(flops, ms):
    return flops / ms / 1e9


def _bench_shape(M, K, N):
    """Return one result row: {'M','K','N', '<arm>_ms', '<arm>_tflops', '<arm>_relerr', ...}."""
    a, b = _make(M, K, N)
    ref = a.float() @ b.float()
    flops = 2 * M * K * N

    # Quantization is identical across all three arms and hoisted out of every
    # timed closure below -- each arm times the GEMM call alone.
    aq, sa = quantize_operand(a, -1, "fp8_e4m3", MX)
    bq, sb = quantize_operand(b, -2, "fp8_e4m3", MX)

    # The CuTe kernel takes B as (N, K) with scales (N, K//32) -- the layout the MMA
    # wants, and the one a weight is already stored in -- so its arm times the GEMM
    # alone, like every other arm here. Same quantization, same codes, just emitted
    # in B's own orientation instead of the (K, N) one the other two arms take.
    bq_cute, sb_cute = quantize_operand(b.t().contiguous(), -1, "fp8_e4m3", MX)

    def cute_fn():
        return scaled_gemm_mxfp8(aq, bq_cute, sa, sb_cute, torch.bfloat16)

    cute_out = cute_fn()
    cute_ms = _time(cute_fn)

    def triton_fn():
        return scaled_gemm(
            aq, bq, sa, sb, torch.bfloat16, MXFP8_BLOCK, scale_dtype="fp8_e8m0"
        )

    triton_out = triton_fn()
    triton_ms = _time(triton_fn)

    # cuBLASLt needs its operands/scales in its own layout -- the swizzle and the
    # column-major view of bq are one-time setup, not part of the timed call, the
    # same footing the CuTe arm above is on now that it takes B K-contiguous too.
    aq_cublaslt = aq.contiguous()
    bq_cublaslt = bq.t().contiguous().t()  # (K, N) stored column-major
    sa_blocked = _to_blocked(sa.to(torch.float8_e8m0fnu))
    sb_blocked = _to_blocked(sb.t().contiguous().to(torch.float8_e8m0fnu))
    scaling_type = F.ScalingType.BlockWise1x32
    swizzle = F.SwizzleType.SWIZZLE_32_4_4

    def cublaslt_fn():
        return F.scaled_mm(
            aq_cublaslt,
            bq_cublaslt,
            sa_blocked,
            scaling_type,
            sb_blocked,
            scaling_type,
            swizzle_a=swizzle,
            swizzle_b=swizzle,
            output_dtype=torch.bfloat16,
        )

    cublaslt_out = cublaslt_fn()
    cublaslt_ms = _time(cublaslt_fn)

    return {
        "M": M,
        "K": K,
        "N": N,
        "cute_ms": cute_ms,
        "cute_tflops": _tflops(flops, cute_ms),
        "cute_relerr": _relerr(cute_out, ref),
        "triton_ms": triton_ms,
        "triton_tflops": _tflops(flops, triton_ms),
        "triton_relerr": _relerr(triton_out, ref),
        "cublaslt_ms": cublaslt_ms,
        "cublaslt_tflops": _tflops(flops, cublaslt_ms),
        "cublaslt_relerr": _relerr(cublaslt_out, ref),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot(results, path, device=""):
    """Write a grouped-bar chart (TFLOP/s per shape, one group of 3 bars per shape).

    Each result: {"M","K","N","<arm>_ms","<arm>_tflops","<arm>_relerr"}.
    Import-safe without CUDA.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [f"{r['M']}x{r['K']}x{r['N']}" for r in results]
    x = np.arange(len(results))
    bw = 0.25

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for j, arm in enumerate(ARMS):
        vals = [r[f"{arm}_tflops"] for r in results]
        bars = ax.bar(
            x + (j - 1) * bw,
            vals,
            bw,
            label=arm,
            color=_ARM_COLOR[arm],
            zorder=3,
        )
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in vals], fontsize=7, padding=2)

    ax.set_xticks(x, labels, fontsize=8, rotation=15)
    ax.set_ylabel("TFLOP/s", fontsize=10)
    ax.margins(y=0.18)
    ax.grid(axis="y", color="#e1e0d9", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=9)

    sup = "mxfp8 scaled GEMM throughput — CuTe vs Triton vs cuBLASLt"
    sub = "higher is faster"
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
        "--out", default=DEFAULT_OUT, help="path to write the throughput chart"
    )
    ap.add_argument("--no-plot", action="store_true", help="skip writing the chart")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.cuda.get_device_name(0)
    print(device)
    print()

    hdr = (
        f"{'M':>7s} {'K':>7s} {'N':>7s} "
        f"{'cute_ms':>9s} {'cute_tf':>8s} "
        f"{'triton_ms':>10s} {'triton_tf':>10s} "
        f"{'cublas_ms':>10s} {'cublas_tf':>10s}"
    )
    print(hdr)
    results = []
    for M, K, N in SHAPES:
        r = _bench_shape(M, K, N)
        print(
            f"{r['M']:>7d} {r['K']:>7d} {r['N']:>7d} "
            f"{_fmt(r['cute_ms']):>9s} {_fmt(r['cute_tflops'], 1):>8s} "
            f"{_fmt(r['triton_ms']):>10s} {_fmt(r['triton_tflops'], 1):>10s} "
            f"{_fmt(r['cublaslt_ms']):>10s} {_fmt(r['cublaslt_tflops'], 1):>10s}"
        )
        results.append(r)

    if not args.no_plot:
        plot(results, args.out, device=device)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
