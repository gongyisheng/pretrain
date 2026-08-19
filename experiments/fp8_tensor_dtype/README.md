# Per-Tensor FP8 Sensitivity

Measure the loss cost of tensorwise FP8 weight, activation, and `grad_out` quantization at Qwen3 51M. bf16 is the local reference.

`w`, `a`, and `g` denote weight, activation, and `grad_out`; `8` is FP8 and `16` is bf16. For example, `e4m3_w8a8_e5m2_g8` uses e4m3 for weight and activation, and e5m2 for `grad_out`.

## Hypothesis

- Weight is cheaper to quantize than activation at tensorwise granularity.
- e4m3 is best for the forward pair; e5m2 gives the best `grad_out` format through greater range.
- `w8a8` versus the individual weight and activation runs tests their interaction. Interpret apparent sub-additivity cautiously: `w8a8` fuses its forward GEMM, while the individual runs emulate it.

## Setup

All runs use Qwen3 51M (~51M parameters), OpenWebText, seq_len=1024, batch_size=16, gradient_accumulation_steps=16, 50K steps, Muon, lr=5e-4, cosine decay with 1500 warmup steps, bf16 mixed precision, seed 42, and `exclude: [lm_head]`. Checkpoints are every 5000 steps; evaluation is every 100 steps with 100 eval batches.

All FP8 configs use `scale: {granularity: tensorwise}`. Quantization applies to eligible attention and MLP linear layers; embeddings, lm_head, norms, attention, and optimizer state remain bf16/fp32.

## Run

```bash
nohup bash experiments/fp8_tensor_dtype/run.sh > logs/fp8_tensor_dtype.log 2>&1 &
```

## Results

W&B project: `pretrain-fp8-tensor-dtype`. Report mean validation loss over the final 10 evaluations, its standard deviation, and the difference from the local bf16 run. One seed per cell means gaps within that standard deviation are unresolved.

| Config | weight | act | grad_out | Val loss | σ (last 10) | Δ vs bf16 | Val BPB | Tokens/sec |
|---|---|---|---|---|---|---|---|---|
| `qwen3_51m_bf16` | — | — | — | | | 0 | | |
| `qwen3_51m_fp8_e4m3_w8a16` | e4m3 | bf16 | bf16 | | | | | — |
| `qwen3_51m_fp8_e4m3_w16a8` | bf16 | e4m3 | bf16 | | | | | — |
| `qwen3_51m_fp8_e4m3_w8a8` | e4m3 | e4m3 | bf16 | | | | | — |
| `qwen3_51m_fp8_e4m3_w8a8g8` | e4m3 | e4m3 | e4m3 | | | | | |
| `qwen3_51m_fp8_e4m3_w8a8_e5m2_g8` | e4m3 | e4m3 | e5m2 | | | | | |
| `qwen3_51m_fp8_e5m2_w8a16` | e5m2 | bf16 | bf16 | | | | | — |
| `qwen3_51m_fp8_e5m2_w16a8` | bf16 | e5m2 | bf16 | | | | | — |
| `qwen3_51m_fp8_e5m2_w8a8` | e5m2 | e5m2 | bf16 | | | | | — |
| `qwen3_51m_fp8_e5m2_w8a8g8` | e5m2 | e5m2 | e5m2 | | | | | |
| `qwen3_51m_fp8_e5m2_w8a8_e4m3_g8` | e5m2 | e5m2 | e4m3 | | | | | |

## Notes

- A scaled GEMM fuses only when both operands are FP8. The weight- or activation-only runs are not throughput-comparable with bf16; full recipes fuse all three GEMMs and can be compared with each other.
- Read `rel. quant err` from `train-quant/sqnr/<tensor>/<module>` in W&B. Use training series because validation has no backward tensors.
- `grad_out` is not tested alone: it would measure unfused emulation, not a deployable recipe.
- Related experiments: `fp8_granularity/`, `int8_tensor_dtype/`, and `int8_granularity/`.
