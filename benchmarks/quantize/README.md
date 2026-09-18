# Quantization benchmarks

These scripts measure Qwen3-51M-sized dense quantization. They compare a
compiled eager implementation (`torch.compile(..., backend="inductor")`) with
the CUDA kernel. An untimed eager result supplies the accuracy diagnostics.

| Script | Codes and scales | Granularities | Block sizes | CUDA capability |
| --- | --- | --- | --- | --- |
| `benchmark_fp8.py` | FP8 E4M3, FP32 scales | tensorwise, rowwise, blockwise 1-D/2-D | 16, 32, 64, 128 | 8.9+ |
| `benchmark_mxfp8.py` | FP8 E4M3, E8M0 scales | blockwise 1-D/2-D | 16, 32, 64, 128 | 8.9+ |
| `benchmark_int8.py` | INT8, FP32 scales | tensorwise, rowwise, blockwise 1-D/2-D | 16, 32, 64, 128 | any CUDA device |
| `benchmark_nvfp4.py` | packed E2M1, E4M3 block scales, FP32 global scale | blockwise 1-D/2-D | 16 | 10.0–12.99 |

The Qwen3-51M shapes use `d_model=512`, key/value width `256`, MLP width
`1536`, and default token count `M=16384`. They cover attention and key/value
weights, separate MLP gate/up weights (each stored as `1536×512`), MLP down
weights, hidden activations, key/value gradients, and MLP
hidden activations. The LM head is excluded by default.

| `--shapes` name | Stored shape | Representative use |
| --- | --- | --- |
| `attn_weight` | `512×512` | query/output projection weights |
| `kv_weight` | `256×512` | key/value projection weights |
| `mlp_up_weight` | `1536×512` | gate/up projection weights |
| `mlp_down_weight` | `512×1536` | down projection weights |
| `hidden` | `M×512` | hidden activations and gradients |
| `kv_grad` | `M×256` | key/value output gradients |
| `mlp_hidden` | `M×1536` | MLP activations and gradients |

All scripts take the same selectors. Defaults sweep contiguous and transposed
inputs, both contraction dimensions, RNE (nearest, ties to even) and stochastic
rounding, quantization metrics disabled and enabled, every format-supported
granularity and block size, five warmups, and 20 ms timing replicates. Shapes
and layout/axis combinations form a Cartesian sweep for kernel coverage.
Transposing a stored shape also swaps its logical dimensions. Axis `-1`
produces row-major codes and axis `-2` produces column-major codes, matching
the left/right operands in `quantized_mm`. This includes forward weight
transposes, contiguous weights for input gradients, and transposed output
gradients for weight gradients. `--tokens` changes `M` for activation shapes;
`--input-dtype` defaults to `bfloat16`.

CUDA graph replay and median GPU timing exclude compilation, validation,
Python dispatch, and host allocation overhead. Inputs are seeded unit-normal
random values, reused across the configurations for each stored shape.
Stochastic-rounding timing advances RNG state, so its codes are not compared
bitwise. Results include relative reconstruction RMSE against the input and
RNE code mismatch fraction against eager; mismatches are diagnostics, not
an assertion of bitwise equivalence for compiled arithmetic. The
`--quantization-metrics` selector controls statistics collection inside timed
quantization; reconstruction diagnostics always run outside timing. Tables
and JSON/CSV results identify the metrics setting for each case.

```bash
uv run python benchmarks/quantize/benchmark_fp8.py \
  --shapes attn_weight --layouts contiguous --contract-dims -1 --rep-ms 5

uv run python benchmarks/quantize/benchmark_fp8.py
uv run python benchmarks/quantize/benchmark_mxfp8.py
uv run python benchmarks/quantize/benchmark_int8.py
uv run python benchmarks/quantize/benchmark_nvfp4.py

uv run python benchmarks/quantize/benchmark_fp8.py \
  --granularities blockwise1d --block-sizes 32

uv run python benchmarks/quantize/benchmark_fp8.py \
  --quantization-metrics enabled
```

The first command has all ten FP8 schemes, both rounding modes, and both
quantization-metrics settings (40 cases) for one tensor layout and contraction
dimension. Full matrices can compile slowly; use `--shapes`, `--layouts`,
`--contract-dims`, `--granularities`, `--block-sizes`, and
`--quantization-metrics` to reduce them. Results default to
`benchmarks/results/quantize/<format>.json` with a CSV sibling. Latencies are in
microseconds; speedup is compiled eager latency divided by CUDA latency. Use
`--list-cases` to inspect the selected cases without running them.

Before benchmarking, choose an unused GPU and hold its lock for the command:

```bash
nvidia-smi
(
set -e
gpu=0
(set -o noclobber; : > "/tmp/gpu${gpu}.lock") || exit 1
trap 'unlink "/tmp/gpu${gpu}.lock"' EXIT
CUDA_VISIBLE_DEVICES="$gpu" uv run python benchmarks/quantize/benchmark_fp8.py
)
```
