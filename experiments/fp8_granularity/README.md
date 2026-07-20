# FP8 Scaling Granularity Ablation

Compare end-task loss and throughput across FP8 scaling granularities (`tensorwise`, `rowwise`, `rowwise_with_gw_hp`) against a bf16 baseline at Qwen3-51M, using the in-house `src/quant/` framework (cuBLASLt FP8 GEMM via `torch._scaled_mm`). The independent variable is the scaling granularity; the bf16 run anchors the loss/throughput reference.

## Hypothesis

FP8 GEMM (E4M3 forward operands, E5M2 grad, hardware-accelerated on Blackwell SM 12.0) should:
1. Match bf16 final loss within noise — the high-dynamic-range ops (softmax, RMSNorm, residual stream, cross-entropy) stay in bf16, so FP8 noise is bounded to the Q/K/V/O and FFN GEMMs.
2. Deliver measurable throughput gain (tokens/sec) over bf16 (matmul share of step time).

The recipes trade accuracy against kernel cost:
- `tensorwise` — one scale per tensor; cheapest, most aggressive quantization.
- `rowwise` — per-row (per-token) / per-column (per-output-channel) scales; tighter dynamic range, slightly slower kernel.
- `rowwise_with_gw_hp` — rowwise, but the weight-gradient GEMM (`dW = dYᵀ@X`) runs in bf16 (`dtype.weight_grad: bf16`); forward + dgrad stay fp8. Protects the weight-gradient signal at some throughput cost.

If (1) fails with `tensorwise`, expect `rowwise` / `rowwise_with_gw_hp` to recover loss at some throughput cost.
If (2) is flat, the bottleneck is data movement / non-GEMM kernels, not the matmul itself.

## Setup

All hyperparameters are matched between bf16 and fp8. The only independent variable is the `training.quant` block. Same seed (default 42), same data order, same LR schedule.

| Config | d_model | layers | heads/kv | inter_size | quant | granularity | Approx params |
|---|---|---|---|---|---|---|---|
| qwen3_51m_bf16 | 512 | 8 | 8/4 | 2048 | off | — | ~51M |
| qwen3_51m_fp8_tensorwise | 512 | 8 | 8/4 | 2048 | fp8 | tensorwise | ~51M |
| qwen3_51m_fp8_rowwise | 512 | 8 | 8/4 | 2048 | fp8 | rowwise | ~51M |
| qwen3_51m_fp8_rowwise_with_gw_hp | 512 | 8 | 8/4 | 2048 | fp8 | rowwise + gw_hp | ~51M |

`exclude: [lm_head]` for all FP8 runs (lm_head stays in bf16; numerically sensitive under tied embeddings). fp8 uses `dtype_recipe: fp8` → weight/act `fp8_e4m3`, grad `fp8_e5m2`.

All runs: seq_len=1024, batch=16, grad_accum=16 (effective batch=256, ~262K tok/step), 50K steps (~13B tokens), Muon optimizer (`muon_adjust_lr_fn: match_rms_adamw`, momentum=0.95, nesterov), lr=5e-4, cosine schedule with 1500 warmup, min_lr=5e-5, OpenWebText.

## What runs in FP8

For each FP8 run, the converter (`src/quant/convert.py:apply_quantization`) swaps eligible `nn.Linear` modules to `QuantLinear`; only the per-layer GEMM operands are cast to FP8 on the fly (dynamic scaling). Everything else (RoPE, RMSNorm, qk_norm, flash/flex attention, SwiGLU, residuals, embeddings, lm_head, cross-entropy, optimizer state) stays in bf16 / fp32. hp master weights are preserved.

The `quant` block:
```yaml
training:
  mixed_precision: bf16
  quant:
    enabled: true
    dtype_recipe: fp8          # weight/act fp8_e4m3, grad fp8_e5m2
    scaling: {granularity: tensorwise}   # or rowwise
    exclude: [lm_head]
```

Hardware requirement: SM 8.9+ (Ada/Hopper/Blackwell). On the dev box (RTX PRO 6000, SM 12.0) the runs use Blackwell's FP8 path in cuBLASLt.

## Run

The script runs the bf16 baseline followed by both FP8 recipes.

```bash
nohup bash experiments/fp8_granularity/run.sh > logs/fp8_granularity_51m.log 2>&1 &
```

## Results

| Model | Precision | Recipe | Final Val Loss | Val BPB | Tokens/sec | Speedup vs bf16 |
|---|---|---|---|---|---|---|
| 51M | bf16 | — | | | | 1.00× |
| 51M | fp8 | tensorwise | | | | |
| 51M | fp8 | rowwise | | | | |
| 51M | fp8 | rowwise_with_gw_hp | | | | |

## Notes

- Compare `tensorwise` vs `rowwise` vs `rowwise_with_gw_hp` for the loss/throughput trade-off.
- If `lm_head` ends up dominating final loss noise, drop it from `exclude` only after the dense-only ablation is clean.
- Throughput gain at 51M is modest — the model is small enough that non-GEMM kernels are a meaningful fraction of step time.
