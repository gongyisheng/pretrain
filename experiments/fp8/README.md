# FP8 vs bf16 Training Ablation

Compare end-task loss and throughput between FP8 (torchao `Float8Linear`, cuBLASLt FP8 GEMM) and a bf16 baseline at two Qwen3 sizes, sweeping all three torchao FP8 recipes.

## Hypothesis

FP8 GEMM (E4M3 inputs, hardware-accelerated on Blackwell SM 12.0) should:
1. Match bf16 final loss within noise — the high-dynamic-range ops (softmax, RMSNorm, residual stream, cross-entropy) stay in bf16, so FP8 noise is bounded to the Q/K/V/O and FFN GEMMs.
2. Deliver measurable throughput gain (tokens/sec) over bf16, growing with model size (matmul share of total time increases with `d_model`).

The three recipes trade accuracy against kernel cost:
- `tensorwise` — one scale per tensor; cheapest, most aggressive quantization.
- `rowwise` — per-row (per-token / per-output-channel) scales; tighter dynamic range, slightly slower kernel.
- `rowwise_with_gw_hp` — rowwise forward/backward but keeps the **g**rad-**w**eight GEMM in **h**igh **p**recision (bf16), trading some speed to protect the weight-gradient signal.

If (1) fails at 57M with `tensorwise`, expect `rowwise` / `rowwise_with_gw_hp` to recover loss at some throughput cost.
If (2) is flat, the bottleneck is data movement / non-GEMM kernels, not the matmul itself.

## Setup

All hyperparameters are matched between bf16 and fp8 within a model size. The only independent variable is `training.fp8`. Same seed (default 42), same data order, same LR schedule.

| Config | d_model | layers | heads/kv | inter_size | fp8 | recipe | Approx params |
|---|---|---|---|---|---|---|---|
| qwen3_57m_bf16 | 512 | 8 | 8/4 | 2048 | false | — | ~57M |
| qwen3_57m_fp8_tensorwise | 512 | 8 | 8/4 | 2048 | true | tensorwise | ~57M |
| qwen3_57m_fp8_rowwise | 512 | 8 | 8/4 | 2048 | true | rowwise | ~57M |
| qwen3_57m_fp8_rowwise_with_gw_hp | 512 | 8 | 8/4 | 2048 | true | rowwise_with_gw_hp | ~57M |
| qwen3_0.5b_bf16 | 1024 | 28 | 16/8 | 4096 | false | — | ~0.5B |
| qwen3_0.5b_fp8_tensorwise | 1024 | 28 | 16/8 | 4096 | true | tensorwise | ~0.5B |
| qwen3_0.5b_fp8_rowwise | 1024 | 28 | 16/8 | 4096 | true | rowwise | ~0.5B |
| qwen3_0.5b_fp8_rowwise_with_gw_hp | 1024 | 28 | 16/8 | 4096 | true | rowwise_with_gw_hp | ~0.5B |

`fp8_exclude_lm_head: true` for all FP8 runs (lm_head stays in bf16; numerically sensitive under tied embeddings).

57M runs: seq_len=1024, batch=16, grad_accum=16 (effective batch=256, ~262K tok/step), 50K steps (~13B tokens), lr=6e-4, cosine schedule with 1500 warmup, min_lr=6e-5, OpenWebText.

0.5B runs: seq_len=1024, batch=8, grad_accum=32 (effective batch=256, ~262K tok/step), 50K steps (~13B tokens), lr=2e-4, cosine schedule with 1500 warmup, min_lr=2e-5, OpenWebText.

## What runs in FP8

For each FP8 run, only the per-layer GEMM operands inside `nn.Linear.forward` are cast to FP8 on the fly. Everything else (RoPE, RMSNorm, qk_norm, flash/flex attention itself, SwiGLU activation, residuals, embeddings, lm_head, cross-entropy, optimizer state) stays in bf16 / fp32. See `src/training/fp8.py` for the swap logic.

Hardware requirement: SM 9.0+ (Hopper or Blackwell). On the dev box (RTX PRO 6000, SM 12.0) the runs use Blackwell's FP8 path in cuBLASLt.

## Run

Each script runs the bf16 baseline followed by all three FP8 recipes for that size.

```bash
nohup bash experiments/fp8/run_57m.sh > logs/fp8_57m.log 2>&1 &
nohup bash experiments/fp8/run_0.5b.sh > logs/fp8_0.5b.log 2>&1 &
```

## Results

| Model | Precision | Recipe | Final Val Loss | Val BPB | Tokens/sec | Speedup vs bf16 |
|---|---|---|---|---|---|---|
| 57M | bf16 | — | | | | 1.00× |
| 57M | fp8 | tensorwise | | | | |
| 57M | fp8 | rowwise | | | | |
| 57M | fp8 | rowwise_with_gw_hp | | | | |
| 0.5B | bf16 | — | | | | 1.00× |
| 0.5B | fp8 | tensorwise | | | | |
| 0.5B | fp8 | rowwise | | | | |
| 0.5B | fp8 | rowwise_with_gw_hp | | | | |

## Notes

- All three recipes are swept by default; compare `tensorwise` vs `rowwise` vs `rowwise_with_gw_hp` for the loss/throughput trade-off.
- If `lm_head` ends up dominating final loss noise, flip `fp8_exclude_lm_head: false` only after the dense-only ablation is clean.
- MoE (`qwen3_moe`) is intentionally not in this experiment — FP8 there requires switching the expert path from `bmm` to `torch._grouped_mm`. Tracked separately.
- Throughput gain is expected to be larger at 0.5B than 57M; the 57M run is small enough that non-GEMM kernels are a meaningful fraction of step time.
