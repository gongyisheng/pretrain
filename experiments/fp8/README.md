# FP8 vs bf16 Training Ablation

Compare end-task loss and throughput between FP8 (torchao `Float8Linear`, cuBLASLt FP8 GEMM) and bf16 baselines at two Qwen3 sizes.

## Hypothesis

FP8 GEMM (E4M3 inputs, hardware-accelerated on Blackwell SM 12.0) should:
1. Match bf16 final loss within noise — the high-dynamic-range ops (softmax, RMSNorm, residual stream, cross-entropy) stay in bf16, so FP8 noise is bounded to the Q/K/V/O and FFN GEMMs.
2. Deliver measurable throughput gain (tokens/sec) over bf16, growing with model size (matmul share of total time increases with `d_model`).

If (1) fails at 57M, the recipe is too aggressive — switch to `rowwise` or move `lm_head` back into bf16.
If (2) is flat, the bottleneck is data movement / non-GEMM kernels, not the matmul itself.

## Setup

All hyperparameters are matched between bf16 and fp8 within a model size. The only independent variable is `training.fp8`. Same seed (default 42), same data order, same LR schedule.

| Config | d_model | layers | heads/kv | inter_size | fp8 | recipe | Approx params |
|---|---|---|---|---|---|---|---|
| qwen3_57m_bf16 | 512 | 8 | 8/4 | 2048 | false | — | ~57M |
| qwen3_57m_fp8  | 512 | 8 | 8/4 | 2048 | true  | tensorwise | ~57M |
| qwen3_0.5b_bf16 | 1024 | 28 | 16/8 | 4096 | false | — | ~0.5B |
| qwen3_0.5b_fp8  | 1024 | 28 | 16/8 | 4096 | true  | tensorwise | ~0.5B |

`fp8_exclude_lm_head: true` for all FP8 runs (lm_head stays in bf16; numerically sensitive under tied embeddings).

57M runs: seq_len=1024, batch=16, grad_accum=16 (effective batch=256, ~262K tok/step), 50K steps (~13B tokens), lr=6e-4, cosine schedule with 1500 warmup, min_lr=6e-5, OpenWebText.

0.5B runs: seq_len=1024, batch=8, grad_accum=32 (effective batch=256, ~262K tok/step), 50K steps (~13B tokens), lr=2e-4, cosine schedule with 1500 warmup, min_lr=2e-5, OpenWebText.

## What runs in FP8

For each FP8 run, only the per-layer GEMM operands inside `nn.Linear.forward` are cast to FP8 on the fly. Everything else (RoPE, RMSNorm, qk_norm, flash/flex attention itself, SwiGLU activation, residuals, embeddings, lm_head, cross-entropy, optimizer state) stays in bf16 / fp32. See `src/training/fp8.py` for the swap logic.

Hardware requirement: SM 9.0+ (Hopper or Blackwell). On the dev box (RTX PRO 6000, SM 12.0) the runs use Blackwell's FP8 path in cuBLASLt.

## Run

```bash
nohup bash experiments/fp8/run.sh > logs/fp8.log 2>&1 &
```

## Results

| Model | Precision | Final Val Loss | Val BPB | Tokens/sec | Speedup vs bf16 |
|---|---|---|---|---|---|
| 57M | bf16 | | | | 1.00× |
| 57M | fp8 | | | | |
| 0.5B | bf16 | | | | 1.00× |
| 0.5B | fp8 | | | | |

## Notes

- If the FP8 run loses loss vs bf16, try `fp8_recipe: rowwise` (tighter per-row scales, slightly slower kernel).
- If `lm_head` ends up dominating final loss noise, flip `fp8_exclude_lm_head: false` only after the dense-only ablation is clean.
- MoE (`qwen3_moe`) is intentionally not in this experiment — FP8 there requires switching the expert path from `bmm` to `torch._grouped_mm`. Tracked separately.
- Throughput gain is expected to be larger at 0.5B than 57M; the 57M run is small enough that non-GEMM kernels are a meaningful fraction of step time.
