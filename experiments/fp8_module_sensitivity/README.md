# FP8 Module Sensitivity Ablation

Which module is most sensitive to FP8 quantization — attention, MLP, or the output head? Rank the three by **add-one-in**: quantize exactly one module group to FP8 (tensorwise) while the rest stays bf16, and measure the marginal loss each one causes. The module whose solo-quantization raises final loss the most is the most FP8-sensitive.

## Hypothesis

FP8 GEMM (E4M3 forward operands, E5M2 grad) injects rounding noise into whichever GEMMs are quantized. Different sublayers should tolerate it differently:

- **Attention** (`q/k/v/o_proj`) feeds softmax, which is scale-sensitive but also renormalizes — small operand noise may wash out.
- **MLP** (`gate_up_proj`, `down_proj`) is the widest hidden GEMM and carries most of the FLOPs; its output re-enters the residual stream directly.
- **`lm_head`** is `(d_model, vocab=50257)`, the single largest GEMM, and its output feeds straight into cross-entropy where small logit errors move the loss. Expected most sensitive under `tensorwise` (one coarse scale over a 50K-wide output).

The `fp8_all` run (all three quantized) is the additivity check: if the loss increase over bf16 is roughly the sum of the three solo increases, the module effects are independent; if it's larger, they interact.

## Untied embeddings — why 77M

`apply_quantization` **skips** a `lm_head` whose weight is tied to the embedding table (swapping it would break the tie), so an `lm_head` FP8 run is a no-op on a tied model. To make the output head a separately quantizable projection, every config here sets `tie_word_embeddings: false`. Untying adds `vocab × d_model` = 50257 × 512 ≈ 25.7M params, so the model is **~77M** (vs the tied ~51M). The untied architecture is held constant across all five runs, so the only variable is which module is FP8.

Note: `QuantConfig.exclude` defaults to `["lm_head"]`; each FP8 config here sets `exclude: []` explicitly so the `lm_head` and `all` runs actually quantize the head.

## Setup

All runs share the model, data, schedule, and optimizer; the only independent variable is the `quant` block's `include` list.

| Config | FP8 module | `quant.include` | Quantized Linears |
|---|---|---|---|
| qwen3_77m_bf16 | none (baseline) | — (quant off) | 0 |
| qwen3_77m_fp8_attn | attention | `[blocks.*.attn.*]` | 32 (4×8) |
| qwen3_77m_fp8_mlp | MLP | `[blocks.*.mlp.*]` | 16 (2×8) |
| qwen3_77m_fp8_lm_head | output head | `[lm_head]` | 1 |
| qwen3_77m_fp8_all | all three | — (nothing excluded) | 49 |

- Model: d_model=512, 8 layers, gqa 8/4, qk_norm, intermediate_size=1536, rope θ=10000, `tie_word_embeddings: false` (~77M).
- FP8: `dtype_recipe: fp8` (weight/act `fp8_e4m3`, grad `fp8_e5m2`), `scaling.granularity: tensorwise`, `exclude: []`. Only the matched GEMM operands are cast to FP8 on the fly; everything else stays bf16/fp32, hp master weights preserved.
- Optimizer: **Muon** (`MuonAdamWOptimizer`) on all runs — 2D hidden weights → Muon, embeddings/`lm_head`/1D → AdamW, `adjust_lr_fn=match_rms_adamw` (reuses AdamW-tuned lr/wd). `momentum=0.95`, `nesterov=true`.
- All runs: seq_len=1024, batch=16, grad_accum=16 (eff. batch=256, ~262K tok/step), 50K steps (~13B tokens), bf16 mixed precision, lr=5e-4, cosine with 1500 warmup, min_lr=5e-5, seed=42, OpenWebText.
- `eval_every=100`, `eval_steps=100`, `checkpoint_every=5000`.

Hardware requirement: SM 8.9+ (Ada/Hopper/Blackwell). On the dev box (RTX PRO 6000, SM 12.0) the FP8 GEMMs use Blackwell's cuBLASLt path.

## Run

Runs the bf16 baseline, then each module in FP8, then all.

```bash
nohup bash experiments/fp8_module_sensitivity/run.sh > logs/fp8_module_sensitivity.log 2>&1 &
```

## Results

Δ Loss is vs the bf16 baseline; larger Δ = more FP8-sensitive.

| FP8 module | Quantized Linears | Final Val Loss | Val BPB | Δ Loss vs bf16 | Tokens/sec |
|---|---|---|---|---|---|
| none (bf16) | 0 | TBD | TBD | — | TBD |
| attention | 32 | TBD | TBD | TBD | TBD |
| MLP | 16 | TBD | TBD | TBD | TBD |
| lm_head | 1 | TBD | TBD | TBD | TBD |
| all | 49 | TBD | TBD | TBD | TBD |

Sensitivity ranking = modules sorted by Δ Loss (descending). Compare `all` Δ against the sum of the three solo Δ's for additivity.

## Notes

- tensorwise is deliberately the noisiest recipe to amplify per-module differences. If one module dominates, rerun just that module under `rowwise` / `rowwise_with_gw_hp` (see `experiments/fp8_granularity/`) to see whether tighter scaling recovers it.
- `lm_head` quantizes a single but very large GEMM; watch tokens/sec there — one module can still be a meaningful throughput share.
- Untying changes param count and absolute loss vs the tied 51M configs, so compare Δ's *within this experiment*, not against `fp8_granularity/`.
- `optim/momentum_norm` / `optim/variance_norm` in W&B reflect only the AdamW-routed params (Muon's `momentum_buffer` is not aggregated).
