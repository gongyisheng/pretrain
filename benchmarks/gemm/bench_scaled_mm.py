"""Scaled GEMM latency across generic FP8 and INT8 scaling schemes.

Sweeps tensorwise, rowwise, blockwise-1D, and blockwise-2D configurations
across three representative transformer linear shapes (d_model=512,
intermediate=1536) at two token counts M in {4096, 16384}. Triton and cuBLASLt
run through the per-format scaled GEMM op (backend=...); native PyTorch is called
directly.
Unsupported native PyTorch contracts print `n/a`. Accuracy is relative error
against the eager backend for the same quantized operands. Prints a table and
(unless --no-plot) writes a matplotlib grouped-bar chart.

    uv run python benchmarks/gemm/bench_scaled_mm.py
    uv run python benchmarks/gemm/bench_scaled_mm.py --no-plot
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from src.kernel.selector import KernelSelectionError, dispatch
from src.kernel.utils import to_column_major, to_swizzle_32_4_4
from src.quant.quantize import quantize_operand
from src.quant.utils import scaled_mm_op

E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2
MXFP8_BLOCK = 32

# Representative transformer projection dimensions.
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

DEFAULT_OUT = "benchmarks/results/scaled_mm.png"

# Implementation colors aligned with bench_grouped_mm.py's palette.
_SCHEME_COLOR = {"triton": "#2a78d6", "pytorch": "#eb6834"}


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


def _precision_diagnostics(out, ref):
    difference = out.float() - ref.float()
    return {
        "relerr": (difference.norm() / ref.float().norm()).item(),
        "max_abs_error": difference.abs().max().item(),
        "mismatch_fraction": out.ne(ref).float().mean().item(),
    }


def _require_mxfp8_precision(label, out, ref):
    diagnostics = _precision_diagnostics(out, ref)
    reference_norm = ref.float().norm().item()
    if (
        not torch.isfinite(out).all()
        or not math.isfinite(reference_norm)
        or not math.isfinite(diagnostics["relerr"])
        or diagnostics["relerr"] > 5e-5
    ):
        raise AssertionError(
            f"{label} output exceeds the MXFP8 eager-relative error gate "
            f"(relative_error={diagnostics['relerr']}, "
            f"reference_norm={reference_norm}, "
            f"max_abs_error={diagnostics['max_abs_error']}, "
            f"mismatch_fraction={diagnostics['mismatch_fraction']})"
        )
    return diagnostics


def _make(M, K, N):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    return a, b


_FMT = {E4M3: "fp8_e4m3", E5M2: "fp8_e5m2"}


def _scaling(gran, bs=0, scale_dtype=torch.float32):
    return {
        "granularity": gran,
        "block_shape": (1, bs) if bs else (0, 0),
        "scale_dtype": scale_dtype,
    }


def _scaling_for_config(config):
    block_size = config["block_size"] or 0
    if config["label"] == "mxfp8":
        return _scaling("blockwise", block_size, torch.float8_e8m0fnu)
    if config["granularity"] == "tensorwise":
        return _scaling("tensorwise")
    if config["granularity"] == "rowwise":
        return _scaling("rowwise")
    shape = (
        (1, block_size)
        if config["granularity"] == "blockwise1d"
        else (block_size, block_size)
    )
    return {
        "granularity": "blockwise",
        "block_shape": shape,
        "scale_dtype": torch.float32,
    }


def _scaled_mm(fmt, aq, bq, sa, sb, out_dtype, block_size, scale_dtype, backend=None):
    op = scaled_mm_op(fmt, fmt, scale_dtype, (1, block_size))
    return dispatch(
        op, (aq, bq, sa, sb, out_dtype, block_size, None), {}, backend, device=aq.device
    )


def _pytorch_scaled_mm(aq, bq, sa, sb, out_dtype, block_size, scale_dtype):
    if aq.dtype == torch.int8:
        width = block_size or aq.shape[1]
        out = torch.zeros(
            (aq.shape[0], bq.shape[1]), device=aq.device, dtype=torch.float32
        )
        for start in range(0, aq.shape[1], width):
            stop = min(start + width, aq.shape[1])
            index = start // width
            out += (
                torch._int_mm(
                    aq[:, start:stop].contiguous(), bq[start:stop].contiguous()
                ).float()
                * sa[:, index : index + 1]
                * sb[index : index + 1]
            )
        return out.to(out_dtype)

    scaling_type = getattr(F, "ScalingType", None)
    if (
        scale_dtype == torch.float8_e8m0fnu
        and block_size in (32, 64, 128)
        and torch.cuda.get_device_capability(aq.device)[0] in (10, 12)
        and getattr(scaling_type, "BlockWise1x32", None) is not None
        and getattr(F, "SwizzleType", None) is not None
    ):
        repeat = block_size // 32
        groups = (aq.shape[1] + 31) // 32
        scale_a = to_swizzle_32_4_4(sa.repeat_interleave(repeat, 1)[:, :groups])
        scale_b = to_swizzle_32_4_4(sb.repeat_interleave(repeat, 0)[:groups].t())
        return F.scaled_mm(
            aq,
            to_column_major(bq),
            scale_a,
            F.ScalingType.BlockWise1x32,
            scale_b,
            F.ScalingType.BlockWise1x32,
            swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
            swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
            bias=None,
            output_dtype=out_dtype,
        )

    width = block_size or aq.shape[1]
    out = torch.zeros((aq.shape[0], bq.shape[1]), device=aq.device, dtype=torch.float32)
    for start in range(0, aq.shape[1], width):
        stop = min(start + width, aq.shape[1])
        index = start // width
        out += F.scaled_mm(
            aq[:, start:stop].contiguous(),
            to_column_major(bq[start:stop]),
            sa[:, index : index + 1].contiguous(),
            F.ScalingType.RowWise,
            sb[index : index + 1],
            F.ScalingType.RowWise,
            bias=None,
            output_dtype=torch.bfloat16,
        ).float()
    return out.to(out_dtype)


def _bench_scheme(a, b, config):
    """Benchmark one quantization contract with preparation outside timed closures."""
    result = {
        "dtype": config["dtype"],
        "granularity": config["granularity"],
        "block_size": config["block_size"],
        "scheme": config["label"],
        "triton_ms": None,
        "pytorch_ms": None,
        "triton_relerr": None,
        "pytorch_relerr": None,
        "pytorch_supported": False,
        "cublaslt_ms": None,
        "cublaslt_native_ms": None,
        "cublaslt_relerr": None,
        "cublaslt_native_relerr": None,
        "cublaslt_max_abs_error": None,
        "cublaslt_native_max_abs_error": None,
        "cublaslt_mismatch_fraction": None,
        "cublaslt_native_mismatch_fraction": None,
        "cublaslt_supported": False,
        "cublaslt_speedup": None,
        "cublaslt_native_speedup": None,
    }

    block_size = config["block_size"] or 0
    scaling = _scaling_for_config(config)
    fmt = _FMT[E4M3] if config["dtype"] == "fp8" else "int8"
    aq, sa = quantize_operand(a, -1, fmt, scaling)
    bq, sb = quantize_operand(b, -2, fmt, scaling)
    scale_dtype = scaling["scale_dtype"]

    ref = _scaled_mm(
        fmt,
        aq,
        bq,
        sa,
        sb,
        torch.bfloat16,
        block_size,
        scale_dtype,
        backend="eager",
    )

    def triton_fn():
        return _scaled_mm(
            fmt,
            aq,
            bq,
            sa,
            sb,
            torch.bfloat16,
            block_size,
            scale_dtype,
            backend="triton",
        )

    triton_out = triton_fn()
    result["triton_ms"] = _time(triton_fn)
    result["triton_relerr"] = _relerr(triton_out, ref)

    def pytorch_fn():
        return _pytorch_scaled_mm(
            aq, bq, sa, sb, torch.bfloat16, block_size, scale_dtype
        )

    try:
        pytorch_out = pytorch_fn()
    except (AttributeError, NotImplementedError, RuntimeError, ValueError):
        pass
    else:
        result["pytorch_supported"] = True
        result["pytorch_ms"] = _time(pytorch_fn)
        result["pytorch_relerr"] = _relerr(pytorch_out, ref)

    if config["label"] != "mxfp8":
        return result

    def cublaslt_fn():
        return _scaled_mm(
            fmt,
            aq,
            bq,
            sa,
            sb,
            torch.bfloat16,
            block_size,
            scale_dtype,
            backend="cublaslt",
        )

    try:
        cublaslt_out = cublaslt_fn()
    except KernelSelectionError:
        return result

    cublaslt_diagnostics = _require_mxfp8_precision(
        "cuBLASLt end-to-end", cublaslt_out, ref
    )
    result["cublaslt_supported"] = True
    result["cublaslt_ms"] = _time(cublaslt_fn)
    result["cublaslt_relerr"] = cublaslt_diagnostics["relerr"]
    result["cublaslt_max_abs_error"] = cublaslt_diagnostics["max_abs_error"]
    result["cublaslt_mismatch_fraction"] = cublaslt_diagnostics["mismatch_fraction"]
    result["cublaslt_speedup"] = result["triton_ms"] / result["cublaslt_ms"]

    native_bq = to_column_major(bq)
    native_sa = to_swizzle_32_4_4(sa)
    native_sb = to_swizzle_32_4_4(sb.t())

    def cublaslt_native_fn():
        return torch.ops.aot_kernel._scaled_mm_mxfp8_cublaslt(
            aq, native_bq, native_sa, native_sb, None
        )

    cublaslt_native_out = cublaslt_native_fn()
    cublaslt_native_diagnostics = _require_mxfp8_precision(
        "cuBLASLt native", cublaslt_native_out, ref
    )
    result["cublaslt_native_ms"] = _time(cublaslt_native_fn)
    result["cublaslt_native_relerr"] = cublaslt_native_diagnostics["relerr"]
    result["cublaslt_native_max_abs_error"] = cublaslt_native_diagnostics[
        "max_abs_error"
    ]
    result["cublaslt_native_mismatch_fraction"] = cublaslt_native_diagnostics[
        "mismatch_fraction"
    ]
    result["cublaslt_native_speedup"] = (
        result["triton_ms"] / result["cublaslt_native_ms"]
    )
    return result


def _bench_shape(name, M, K, N):
    """Return all 20 matrix rows for one (name, M, K, N) shape."""
    a, b = _make(M, K, N)
    return [
        {
            "shape": name,
            "M": M,
            "K": K,
            "N": N,
            **_bench_scheme(a, b, config),
        }
        for config in _scheme_configs()
    ]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot(results, path, device=""):
    """Write a grouped-bar chart (rows = M, cols = shape) from `results`.

    Each result: {"shape","M","K","N","scheme","triton_ms","pytorch_ms",...}.
    Import-safe without CUDA.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lut = {(r["shape"], r["M"], r["scheme"]): r for r in results}
    shape_names = [name for name, _, _ in SHAPES]
    x = np.arange(len(SCHEMES))
    bw = 0.35

    fig, axes = plt.subplots(
        len(M_LIST), len(shape_names), figsize=(13, 8), constrained_layout=True
    )
    for ri, M in enumerate(M_LIST):
        for ci, shape_name in enumerate(shape_names):
            ax = axes[ri][ci]
            for j, impl in enumerate(("triton", "pytorch")):
                key = f"{impl}_ms"
                vals = []
                for scheme in SCHEMES:
                    r = lut.get((shape_name, M, scheme))
                    value = r.get(key) if r else None
                    vals.append(value if value is not None else 0.0)
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
    fig.legend(handles, labels, loc="outside upper right", ncol=3, fontsize=10)
    sup = "Scaled GEMM latency — Triton vs native PyTorch"
    sub = "eager oracle · lower is faster · n/a = unsupported contract"
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
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.cuda.get_device_name(0)
    print(device)
    print(f"d_model={D_MODEL} intermediate={INTERMEDIATE}\n")

    hdr = (
        f"{'shape':12s} {'M':>7s} {'scheme':16s} {'triton_ms':>10s} "
        f"{'pytorch_ms':>10s} {'cublaslt':>8s} {'cublaslt_ms':>12s} {'native_ms':>10s} "
        f"{'e2e_x':>8s} {'native_x':>8s} {'triton_relerr':>14s} "
        f"{'pytorch_relerr':>14s} {'cublaslt_relerr':>16s}"
    )
    print(hdr)
    results = []
    for name, K, N in SHAPES:
        for M in M_LIST:
            rows = _bench_shape(name, M, K, N)
            for r in rows:
                print(
                    f"{r['shape']:12s} {r['M']:>7d} {r['scheme']:16s} "
                    f"{_fmt(r['triton_ms']):>10s} {_fmt(r['pytorch_ms']):>10s} "
                    f"{str(r['cublaslt_supported']):>8s} "
                    f"{_fmt(r['cublaslt_ms']):>12s} "
                    f"{_fmt(r['cublaslt_native_ms']):>10s} "
                    f"{_fmt(r['cublaslt_speedup']):>8s} "
                    f"{_fmt(r['cublaslt_native_speedup']):>8s} "
                    f"{_fmt(r['triton_relerr'], 4):>14s} "
                    f"{_fmt(r['pytorch_relerr'], 4):>14s} "
                    f"{_fmt(r['cublaslt_relerr'], 4):>16s}"
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
