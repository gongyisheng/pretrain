"""Shared quantization benchmark runner."""

import argparse
import csv
from functools import partial
import gc
from itertools import product
import json
import math
from pathlib import Path

import torch
from triton.testing import do_bench_cudagraph
import yaml

from src.kernel.ops import dequantize_dense


def run_benchmark(
    format_name, operation, granularities, block_sizes, min_arch, max_arch=None
):
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs/qwen3_51m.yaml"
    config = yaml.safe_load(config_path.read_text())
    model = config["model"]
    attention = model["attn"][0]["attn_kwargs"]
    width = model["d_model"]
    kv_width = width // attention["n_heads"] * attention["n_kv_heads"]
    intermediate = model["mlp"][0]["mlp_kwargs"]["intermediate_size"]
    weights = {
        "attn_weight": (width, width),
        "kv_weight": (kv_width, width),
        "mlp_up_weight": (intermediate, width),
        "mlp_down_weight": (width, intermediate),
    }
    activations = {"hidden": width, "kv_grad": kv_width, "mlp_hidden": intermediate}
    shape_names = list(weights) + list(activations)
    parser = argparse.ArgumentParser(
        description=f"Qwen3 51M {format_name}: torch.compile(eager) versus CUDA."
    )
    parser.add_argument("--shapes", nargs="+", choices=shape_names, default=shape_names)
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        default=[config["training"]["batch_size"] * config["max_seq_len"]],
    )
    parser.add_argument(
        "--input-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=("contiguous", "transposed"),
        default=["contiguous", "transposed"],
    )
    parser.add_argument(
        "--contract-dims", nargs="+", type=int, choices=(-1, -2), default=[-1, -2]
    )
    parser.add_argument(
        "--granularities", nargs="+", choices=granularities, default=granularities
    )
    parser.add_argument(
        "--block-sizes", nargs="+", type=int, choices=block_sizes, default=block_sizes
    )
    parser.add_argument(
        "--rounding",
        nargs="+",
        choices=("rne", "stochastic"),
        default=["rne", "stochastic"],
    )
    parser.add_argument(
        "--quantization-metrics",
        nargs="+",
        choices=("disabled", "enabled"),
        default=["disabled", "enabled"],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep-ms", type=float, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "benchmarks/results/quantize" / f"{format_name}.json",
    )
    parser.add_argument(
        "--list-cases", action="store_true", help="List cases without running GPU work."
    )
    args = parser.parse_args()
    if any(tokens <= 0 for tokens in args.tokens) or args.warmup < 1:
        parser.error("tokens and warmup must be positive")
    if not math.isfinite(args.rep_ms) or args.rep_ms <= 0:
        parser.error("rep-ms must be positive and finite")
    if args.output.suffix != ".json":
        parser.error("output must have a .json suffix; a sibling .csv is also written")

    shapes = [(name, shape) for name, shape in weights.items() if name in args.shapes]
    shapes.extend(
        (name, (tokens, channels))
        for name, channels in activations.items()
        if name in args.shapes
        for tokens in dict.fromkeys(args.tokens)
    )
    schemes = []
    for granularity in dict.fromkeys(args.granularities):
        if granularity == "tensorwise":
            schemes.append((granularity, (0, 0)))
        elif granularity == "rowwise":
            schemes.append((granularity, (1, 0)))
        else:
            for block_size in dict.fromkeys(args.block_sizes):
                outer = 1 if granularity == "blockwise1d" else block_size
                schemes.append((granularity, (outer, block_size)))
    cases = []
    for (name, shape), layout, contract_dim, (
        granularity,
        block_shape,
    ), rounding, quantization_metrics in product(
        shapes,
        dict.fromkeys(args.layouts),
        dict.fromkeys(args.contract_dims),
        schemes,
        dict.fromkeys(args.rounding),
        dict.fromkeys(args.quantization_metrics),
    ):
        input_shape = shape if layout == "contiguous" else shape[::-1]
        if format_name == "nvfp4" and input_shape[contract_dim] % 16:
            parser.error(
                f"NVFP4 contraction extent must be divisible by 16: {input_shape}, dim={contract_dim}"
            )
        cases.append(
            {
                "shape_name": name,
                "base_shape": shape,
                "input_shape": input_shape,
                "input_layout": layout,
                "contract_dim": contract_dim,
                "output_layout": "row_major" if contract_dim == -1 else "column_major",
                "granularity": granularity,
                "block_shape": block_shape,
                "rounding": rounding,
                "quantization_metrics": quantization_metrics == "enabled",
            }
        )
    if args.list_cases:
        print(
            "| Shape | Input | Layout | Axis | Output | Scheme | Block | Rounding | Metrics |"
        )
        print("|---|---|---|---:|---|---|---|---|---|")
        for case in cases:
            print(
                f"| {case['shape_name']} | {case['input_shape']} | {case['input_layout']} "
                f"| {case['contract_dim']} | {case['output_layout']} | {case['granularity']} "
                f"| {case['block_shape']} | {case['rounding']} "
                f"| {'enabled' if case['quantization_metrics'] else 'disabled'} |"
            )
        print(f"\n{len(cases)} cases; two timed implementations per case.")
        return
    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")
    capability = torch.cuda.get_device_capability()
    if capability < min_arch or (max_arch is not None and capability > max_arch):
        parser.error(
            f"{format_name} CUDA kernel does not support GPU capability {capability}"
        )

    metadata = {
        "format": format_name,
        "config": str(config_path.relative_to(root)),
        "gpu": torch.cuda.get_device_name(),
        "capability": capability,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "input_dtype": args.input_dtype,
        "seed": args.seed,
        "case_count": len(cases),
        "warmup": args.warmup,
        "rep_ms": args.rep_ms,
        "timing": "median CUDA graph replay latency; compilation and validation excluded",
        "compiled_backend": "inductor",
        "fullgraph": True,
        "operation_options": {
            key: str(value) if isinstance(value, torch.dtype) else value
            for key, value in getattr(operation, "keywords", {}).items()
        },
    }
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"metadata": metadata, "results": results}, indent=2) + "\n"
    )
    print(
        f"{format_name}: {len(cases)} cases on {metadata['gpu']}; input={args.input_dtype}",
        flush=True,
    )
    print(
        "\n| Case | Shape | Input | Layout | Axis | Scheme | Block | Rounding | Metrics | Compiled µs | CUDA µs | Speedup |"
    )
    print("|---:|---|---|---|---:|---|---|---|---|---:|---:|---:|", flush=True)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    current_shape = None
    with (
        torch.no_grad(),
        args.output.with_suffix(".csv").open("w", newline="") as csv_file,
    ):
        writer = None
        for index, case in enumerate(cases, 1):
            shape_key = (case["shape_name"], case["base_shape"])
            if shape_key != current_shape:
                base = torch.randn(
                    case["base_shape"],
                    device="cuda",
                    dtype=getattr(torch, args.input_dtype),
                    generator=generator,
                )
                current_shape = shape_key
            source = base if case["input_layout"] == "contiguous" else base.mT
            quantize = partial(
                operation,
                contract_dim=case["contract_dim"],
                block_shape=case["block_shape"],
                stochastic_rounding=case["rounding"] == "stochastic",
                output_layout=case["output_layout"],
                return_quantization_stats=case["quantization_metrics"],
            )
            torch.compiler.reset()
            reference = quantize(source, backend="eager")
            source_float = source.float()
            source_norm = source_float.norm()
            result = dict(case)
            result["format"] = format_name
            result["input_dtype"] = args.input_dtype
            result["input_strides"] = source.stride()
            result["code_dtype"] = str(reference[0].dtype)
            result["scale_dtype"] = str(reference[1].dtype)
            for backend in ("compiled_eager", "cuda"):
                function = partial(
                    quantize, backend="eager" if backend == "compiled_eager" else "cuda"
                )
                if backend == "compiled_eager":
                    function = torch.compile(
                        function, backend="inductor", fullgraph=True, dynamic=False
                    )
                for _ in range(args.warmup):
                    outputs = function(source)
                torch.cuda.synchronize()
                for actual, expected in zip(outputs, reference):
                    if expected is None or expected is False:
                        assert actual is expected
                    else:
                        assert (
                            actual.shape == expected.shape
                            and actual.dtype == expected.dtype
                        )
                decoded = dequantize_dense(
                    outputs[0],
                    outputs[1],
                    case["contract_dim"],
                    case["block_shape"],
                    outputs[2],
                    backend="eager",
                )
                relative_rmse = ((decoded - source_float).norm() / source_norm).item()
                if not math.isfinite(relative_rmse):
                    raise RuntimeError(
                        f"Nonfinite reconstruction error: {backend}, {case}"
                    )
                result[f"{backend}_relative_rmse"] = relative_rmse
                result[f"{backend}_code_mismatch_fraction"] = (
                    (outputs[0].view(torch.uint8) != reference[0].view(torch.uint8))
                    .float()
                    .mean()
                    .item()
                    if case["rounding"] == "rne"
                    else None
                )
                del outputs, decoded
                result[f"{backend}_us"] = 1000 * do_bench_cudagraph(
                    partial(function, source), rep=args.rep_ms, return_mode="median"
                )
                del function
            result["cuda_speedup"] = result["compiled_eager_us"] / result["cuda_us"]
            results.append(result)
            if writer is None:
                writer = csv.DictWriter(csv_file, fieldnames=list(result))
                writer.writeheader()
            writer.writerow(
                {
                    key: json.dumps(value)
                    if isinstance(value, (tuple, list))
                    else value
                    for key, value in result.items()
                }
            )
            csv_file.flush()
            args.output.write_text(
                json.dumps({"metadata": metadata, "results": results}, indent=2) + "\n"
            )
            print(
                f"| {index}/{len(cases)} | {case['shape_name']} | {case['input_shape']} "
                f"| {case['input_layout']} | {case['contract_dim']} | {case['granularity']} "
                f"| {case['block_shape']} | {case['rounding']} "
                f"| {'enabled' if case['quantization_metrics'] else 'disabled'} "
                f"| {result['compiled_eager_us']:.2f} | {result['cuda_us']:.2f} "
                f"| {result['cuda_speedup']:.2f}× |",
                flush=True,
            )
            del reference, source_float, source_norm
            torch.compiler.reset()
            gc.collect()
            torch.cuda.empty_cache()
    print(f"\nSaved {args.output} and {args.output.with_suffix('.csv')}")
