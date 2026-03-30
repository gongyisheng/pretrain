# Triton Kernel Benchmarks

All benchmarks compare custom Triton kernels against `torch.compile` baselines (except Flash Attention which uses `F.scaled_dot_product_attention`). Measured on **NVIDIA GeForce RTX 5060 Ti**, dtype **bfloat16**, times in **milliseconds**.

## LayerNorm

**Forward** (M=4096)

| N | Triton (ms) | torch.compile (ms) | Speedup |
|---|---|---|---|
| 512 | 0.026 | 0.025 | 0.95x |
| 768 | 0.036 | 0.041 | 1.13x |
| 1024 | 0.026 | 0.051 | 1.96x |
| 2048 | 0.072 | 0.096 | 1.33x |
| 4096 | 0.176 | 0.186 | 1.05x |

**Backward** (M=4096)

| N | Triton (ms) | torch.compile (ms) | Speedup |
|---|---|---|---|
| 512 | 0.055 | 0.069 | 1.26x |
| 768 | 0.088 | 0.117 | 1.33x |
| 1024 | 0.094 | 0.137 | 1.45x |
| 2048 | 0.141 | 0.258 | 1.83x |
| 4096 | 0.353 | 0.602 | 1.70x |

## RoPE (Rotary Position Embedding)

**Forward** (B=8, n_heads=12, S=512)

| d_head | Triton (ms) | torch.compile (ms) | Speedup |
|---|---|---|---|
| 64 | 0.082 | 0.018 | 0.21x |
| 96 | 0.061 | 0.075 | 1.23x |
| 128 | 0.082 | 0.098 | 1.20x |
| 192 | 0.107 | 0.145 | 1.35x |
| 256 | 0.140 | 0.190 | 1.36x |

**Backward** (B=8, n_heads=12, S=512)

| d_head | Triton (ms) | torch.compile (ms) | Speedup |
|---|---|---|---|
| 64 | 0.083 | 0.115 | 1.38x |
| 96 | 0.083 | 0.168 | 2.02x |
| 128 | 0.076 | 0.219 | 2.89x |
| 192 | 0.106 | 0.323 | 3.06x |
| 256 | 0.141 | 0.425 | 3.02x |

## RMSNorm

**Forward** (M=4096)

| N | Triton (ms) | torch.compile (ms) | Speedup |
|---|---|---|---|
| 512 | 0.026 | 0.025 | 0.94x |
| 768 | 0.037 | 0.022 | 0.61x |
| 1024 | 0.026 | 0.045 | 1.72x |
| 2048 | 0.071 | 0.092 | 1.29x |
| 4096 | 0.175 | 0.183 | 1.04x |
| 8192 | 0.332 | 0.365 | 1.10x |
| 16384 | 0.683 | 0.757 | 1.11x |

**Backward** (M=4096)

| N | Triton (ms) | torch.compile (ms) | Speedup |
|---|---|---|---|
| 512 | 0.047 | 0.052 | 1.09x |
| 768 | 0.049 | 0.100 | 2.03x |
| 1024 | 0.065 | 0.128 | 1.96x |
| 2048 | 0.131 | 0.237 | 1.81x |
| 4096 | 0.269 | 0.580 | 2.15x |
| 8192 | 0.534 | 1.204 | 2.26x |
| 16384 | 11.284 | 2.579 | 0.23x |

## SwiGLU

**Forward** (M=4096)

| N | Triton (ms) | torch.compile (ms) | Speedup |
|---|---|---|---|
| 512 | 0.039 | 0.039 | 1.01x |
| 768 | 0.052 | 0.052 | 1.00x |
| 1024 | 0.069 | 0.069 | 1.01x |
| 2048 | 0.135 | 0.133 | 0.99x |
| 4096 | 0.261 | 0.261 | 1.00x |
| 8192 | 0.511 | 0.512 | 1.00x |
| 16384 | 1.022 | 1.018 | 1.00x |

**Backward** (M=4096)

| N | Triton (ms) | torch.compile (ms) | Speedup |
|---|---|---|---|
| 512 | 0.061 | 0.078 | 1.29x |
| 768 | 0.087 | 0.133 | 1.52x |
| 1024 | 0.114 | 0.176 | 1.55x |
| 2048 | 0.221 | 0.348 | 1.57x |
| 4096 | 0.439 | 0.689 | 1.57x |
| 8192 | 0.868 | 1.374 | 1.58x |
| 16384 | 1.731 | 2.736 | 1.58x |

## Flash Attention (Causal)

**Forward** (B=4, n_heads=8, d_head=64)

| seq_len | Triton (ms) | F.sdpa (ms) | Speedup |
|---|---|---|---|
| 128 | 0.011 | 0.012 | 1.11x |
| 256 | 0.021 | 0.025 | 1.19x |
| 512 | 0.047 | 0.054 | 1.15x |
| 1024 | 0.133 | 0.136 | 1.02x |
| 2048 | 0.443 | 0.419 | 0.95x |
| 4096 | 1.630 | 1.512 | 0.93x |

**Backward** (B=4, n_heads=8, d_head=64)

| seq_len | Triton (ms) | F.sdpa (ms) | Speedup |
|---|---|---|---|
| 128 | 0.053 | 0.046 | 0.86x |
| 256 | 0.090 | 0.088 | 0.98x |
| 512 | 0.189 | 0.182 | 0.97x |
| 1024 | 0.504 | 0.488 | 0.97x |
| 2048 | 1.665 | 1.588 | 0.95x |
| 4096 | 5.910 | 5.646 | 0.96x |

## MoE Scatter (Token ↔ Expert Dispatch)

**Scatter-In Forward** (E=64, k=2, D=512)

| T | Triton (ms) | PyTorch (ms) | Speedup |
|---|---|---|---|
| 1024 | 0.016 | 0.038 | 2.30x |
| 2048 | 0.029 | 0.068 | 2.32x |
| 4096 | 0.054 | 0.130 | 2.40x |
| 8192 | 0.103 | 0.257 | 2.49x |
| 16384 | 0.259 | 0.584 | 2.25x |

**Scatter-In Backward** (E=64, k=2, D=512)

| T | Triton (ms) | PyTorch (ms) | Speedup |
|---|---|---|---|
| 1024 | 0.018 | 0.089 | 5.01x |
| 2048 | 0.033 | 0.152 | 4.68x |
| 4096 | 0.058 | 0.257 | 4.45x |
| 8192 | 0.096 | 0.509 | 5.31x |
| 16384 | 0.205 | 1.131 | 5.53x |

**Scatter-Out Forward** (E=64, k=2, D=512)

| T | Triton (ms) | PyTorch (ms) | Speedup |
|---|---|---|---|
| 1024 | 0.020 | 0.056 | 2.82x |
| 2048 | 0.034 | 0.097 | 2.88x |
| 4096 | 0.062 | 0.169 | 2.74x |
| 8192 | 0.099 | 0.339 | 3.41x |
| 16384 | 0.215 | 0.754 | 3.51x |

**Scatter-Out Backward** (E=64, k=2, D=512)

| T | Triton (ms) | PyTorch (ms) | Speedup |
|---|---|---|---|
| 1024 | 0.017 | 0.109 | 6.45x |
| 2048 | 0.031 | 0.179 | 5.79x |
| 4096 | 0.058 | 0.292 | 5.04x |
| 8192 | 0.106 | 0.600 | 5.64x |
| 16384 | 0.258 | 1.545 | 5.98x |

## MoE Routing (Token → Expert Assignment)

**Sweep T** (E=64, k=2, capacity_factor=1.25)

| T | Triton atomic (ms) | PyTorch sort (ms) | Speedup |
|---|---|---|---|
| 1024 | 0.120 | 0.223 | 1.85x |
| 2048 | 0.115 | 0.217 | 1.88x |
| 4096 | 0.125 | 0.256 | 2.05x |
| 8192 | 0.126 | 0.269 | 2.13x |
| 16384 | 0.127 | 0.277 | 2.19x |

**Sweep E** (T=8192, k=2, capacity_factor=1.25)

| E | Triton atomic (ms) | PyTorch sort (ms) | Speedup |
|---|---|---|---|
| 4 | 0.123 | 0.269 | 2.19x |
| 8 | 0.125 | 0.281 | 2.25x |
| 16 | 0.123 | 0.263 | 2.14x |
| 32 | 0.122 | 0.265 | 2.17x |
| 64 | 0.128 | 0.266 | 2.08x |
| 128 | 0.124 | 0.274 | 2.20x |

**Sweep k** (T=8192, E=64, capacity_factor=1.25)

| k | Triton atomic (ms) | PyTorch sort (ms) | Speedup |
|---|---|---|---|
| 1 | 0.121 | 0.260 | 2.14x |
| 2 | 0.122 | 0.265 | 2.18x |
| 4 | 0.136 | 0.326 | 2.40x |
| 8 | 0.139 | 0.297 | 2.14x |

## Summary

| Kernel | Forward | Backward | Notes |
|---|---|---|---|
| LayerNorm | up to 1.96x faster | up to 1.83x faster | Consistent wins, especially at larger N |
| RoPE | up to 1.36x faster | up to 3.06x faster | Backward is the big win; forward loses at d_head=64 |
| RMSNorm | up to 1.72x faster | up to 2.26x faster | Regresses at N=16384 backward (11ms vs 2.6ms) |
| SwiGLU | ~1.00x (parity) | ~1.58x faster | Forward is memory-bound (no win); backward fuses nicely |
| Flash Attention | ~1.0-1.2x at short seq | ~0.95x (slightly slower) | Competitive with PyTorch's optimized SDPA at longer sequences |
| MoE Scatter | 2.3-3.5x faster | 4.5-6.5x faster | Biggest wins in backward; scales well with T |
| MoE Routing | 1.9-2.4x faster | N/A | Atomic counters vs sort; consistent across E and k |
