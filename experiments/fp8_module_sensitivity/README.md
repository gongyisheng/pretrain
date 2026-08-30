# FP8 Module Sensitivity Ablation

Which module is most sensitive to FP8 quantization — attention, MLP, or the output head — and how does that sensitivity change as more of the GEMM triple (weight / activation / grad) goes to FP8? Rank modules by **add-one-in**: quantize exactly one module group to FP8 while the rest stays bf16, and measure the marginal loss each one causes. The module whose solo-quantization raises final loss the most is the most FP8-sensitive. Repeat the ablation under three FP8 recipes to see whether the ranking is stable as the quantized operand set widens.

## Hypothesis

FP8 GEMM injects rounding noise into whichever GEMM operands are quantized. Different sublayers should tolerate it differently:

- **Attention** (`q/k/v/o_proj`) feeds softmax, which is scale-sensitive but also renormalizes — small operand noise may wash out.
- **MLP** (`gate_up_proj`, `down_proj`) is the widest hidden GEMM and carries most of the FLOPs; its output re-enters the residual stream directly.
- **`lm_head`** is `(d_model, vocab=50257)`, the single largest GEMM, and its output feeds straight into cross-entropy where small logit errors move the loss. Expected most sensitive under `tensorwise` (one coarse scale over a 50K-wide output).

Widening the recipe (w8a16 → w8a8 → w8a8g8) adds more FP8 operands per quantized linear, so absolute Δ should grow with each recipe; the open question is whether the *module ranking* stays the same.

## Recipes

Three FP8 recipes form the second axis. Each lives in its own subfolder and differs **only** in the `quant.dtype` block — an unset operand falls back to the bf16 compute dtype, so a recipe quantizes exactly the operands it names.

| Recipe | `quant.dtype` | FP8 operands | GEMM path |
|---|---|---|---|
| `w8a16/` | `weight: fp8_e4m3` | weight only | weight quant→dequant, matmul in bf16 (weight-only) |
| `w8a8/` | `weight, act: fp8_e4m3` | weight + act (forward) | fwd uses fp8 scaled-mm; dgrad/wgrad in bf16 (grad_out unquantized) |
| `w8a8g8/` | `weight, act: fp8_e4m3`, `grad_out: fp8_e5m2` | weight + act + grad | fwd + both backward GEMMs in fp8 (full) |

`w8a8g8` is the fully-quantized recipe (E4M3 forward operands, E5M2 grad); `w8a8` isolates forward-only error; `w8a16` isolates the weight-rounding error alone.

## Untied embeddings — why 77M

`apply_quantization` **skips** a `lm_head` whose weight is tied to the embedding table (swapping it would break the tie), so an `lm_head` FP8 run is a no-op on a tied model. To make the output head a separately quantizable projection, every config here sets `tie_word_embeddings: false`. Untying adds `vocab × d_model` = 50257 × 512 ≈ 25.7M params, so the model is **~77M** (vs the tied ~51M). The untied architecture is held constant across all runs, so the only variables are which module is FP8 and under which recipe.

Note: `QuantizationConfig.exclude` defaults to `["lm_head", "*mlp.router.gate"]`; each FP8 config sets `exclude: []` explicitly so the `lm_head` run actually quantizes the head.

## Setup

All runs share the model, data, schedule, and optimizer; the independent variables are the module (`quant.include`) and the recipe (`quant.dtype`, per subfolder). The bf16 baseline quantizes nothing, so it is **shared across all three recipes** and lives at the top level (`qwen3_77m_bf16.yaml`, `qwen3_455m_bf16.yaml`) — run once per scale, not once per recipe.

### Layout

```
fp8_module_sensitivity/
  qwen3_77m_bf16.yaml           qwen3_455m_bf16.yaml         # shared baselines
  w8a16/   qwen3_{77m,455m}_w8a16_{attn,mlp,lm_head}.yaml    # weight-only
  w8a8/    qwen3_{77m,455m}_w8a8_{attn,mlp,lm_head}.yaml     # forward fp8
  w8a8g8/  qwen3_{77m,455m}_w8a8g8_{attn,mlp,lm_head}.yaml   # full fp8
  run_77m.sh   run_455m.sh
```

### 77M scale

| Config | FP8 module | `quant.include` | Quantized Linears |
|---|---|---|---|
| `qwen3_77m_bf16` (shared) | none (baseline) | — (quant off) | 0 |
| `{recipe}/qwen3_77m_{recipe}_attn` | attention | `[blocks.*.attn.*]` | 32 (4×8) |
| `{recipe}/qwen3_77m_{recipe}_mlp` | MLP | `[blocks.*.mlp.*]` | 16 (2×8) |
| `{recipe}/qwen3_77m_{recipe}_lm_head` | output head | `[lm_head]` | 1 |

- Model: d_model=512, 8 layers, gqa 8/4, qk_norm, intermediate_size=1536, rope θ=10000, `tie_word_embeddings: false` (~77M).
- FP8: `scale.granularity: tensorwise`, `exclude: []`; the `dtype` block is per recipe (table above). Only the matched GEMM operands are cast to FP8 on the fly; everything else stays bf16/fp32, hp master weights preserved.
- Optimizer: **Muon** (`MuonAdamWOptimizer`) on all runs — 2D hidden weights → Muon, embeddings/`lm_head`/1D → AdamW, `adjust_lr_fn=match_rms_adamw` (reuses AdamW-tuned lr/wd). `momentum=0.95`, `nesterov=true`.
- All runs: seq_len=1024, batch=16, grad_accum=16 (eff. batch=256, ~262K tok/step), 50K steps (~13B tokens), bf16 mixed precision, lr=5e-4, cosine with 1500 warmup, min_lr=5e-5, seed=42, OpenWebText.
- `eval_every=100`, `eval_steps=100`, `checkpoint_every=5000`.

### 455M scale (`qwen3_455m_*`)

The same module × recipe ablation at the 404M-scale architecture (`configs/qwen3_404m.yaml`) with untied embeddings, for a cross-scale check. Untying adds `vocab × d_model` = 50257 × 1024 ≈ 51.5M, so the model is **~455M** (455,406,080 params) vs the tied ~404M (403,894,784).

| Config | FP8 module | `quant.include` | Quantized Linears |
|---|---|---|---|
| `qwen3_455m_bf16` (shared) | none (baseline) | — (quant off) | 0 |
| `{recipe}/qwen3_455m_{recipe}_attn` | attention | `[blocks.*.attn.*]` | 112 (4×28) |
| `{recipe}/qwen3_455m_{recipe}_mlp` | MLP | `[blocks.*.mlp.*]` | 56 (2×28) |
| `{recipe}/qwen3_455m_{recipe}_lm_head` | output head | `[lm_head]` | 1 |

- Model: d_model=1024, 28 layers, gqa 16/8, qk_norm, intermediate_size=3072, rope θ=10000, `tie_word_embeddings: false` (~455M).
- All runs: seq_len=1024, batch=32, grad_accum=8 (eff. batch=256), lr=2e-4, min_lr=2e-5 (lr/min_lr match `configs/qwen3_404m.yaml`; the 32×8 split uses the larger per-step batch an H200 (144GiB) affords). Everything else matches the 77M runs.

Hardware requirement: SM 8.9+ (Ada/Hopper/Blackwell). On the dev box (RTX PRO 6000, SM 12.0) the FP8 GEMMs use Blackwell's cuBLASLt path. `w8a16` needs no fp8 scaled-mm (weight-only dequant path), so it runs anywhere.

## Run

Each script runs the shared bf16 baseline once, then every module under all three recipes, at its scale.

```bash
nohup bash experiments/fp8_module_sensitivity/run_77m.sh  > logs/fp8_module_sensitivity_77m.log  2>&1 &
nohup bash experiments/fp8_module_sensitivity/run_455m.sh > logs/fp8_module_sensitivity_455m.log 2>&1 &
```

## Results

Δ Loss is vs the shared bf16 baseline; larger Δ = more FP8-sensitive. One block per recipe.

### w8a16 (weight-only)

| FP8 module | Quantized Linears | Final Val Loss | Val BPB | Δ Loss vs bf16 | Tokens/sec |
|---|---|---|---|---|---|
| none (bf16) | 0 | TBD | TBD | — | TBD |
| attention | 32 | TBD | TBD | TBD | TBD |
| MLP | 16 | TBD | TBD | TBD | TBD |
| lm_head | 1 | TBD | TBD | TBD | TBD |

### w8a8 (forward fp8)

| FP8 module | Quantized Linears | Final Val Loss | Val BPB | Δ Loss vs bf16 | Tokens/sec |
|---|---|---|---|---|---|
| attention | 32 | TBD | TBD | TBD | TBD |
| MLP | 16 | TBD | TBD | TBD | TBD |
| lm_head | 1 | TBD | TBD | TBD | TBD |

### w8a8g8 (full fp8)

| FP8 module | Quantized Linears | Final Val Loss | Val BPB | Δ Loss vs bf16 | Tokens/sec |
|---|---|---|---|---|---|
| attention | 32 | TBD | TBD | TBD | TBD |
| MLP | 16 | TBD | TBD | TBD | TBD |
| lm_head | 1 | TBD | TBD | TBD | TBD |

Sensitivity ranking = modules sorted by Δ Loss (descending), read per recipe and across recipes.

## Notes

- tensorwise is deliberately the noisiest scaling recipe to amplify per-module differences. If one module dominates, rerun just that module under rowwise or blockwise scaling (see `experiments/fp8_granularity/`) to see whether tighter scaling recovers it.
- The recipe axis separates *where* the FP8 error enters: `w8a16` is weight-rounding only, `w8a8` adds forward-activation error, `w8a8g8` adds backward (grad) error. Comparing Δ across recipes for the same module attributes the damage to a stage.
- `lm_head` quantizes a single but very large GEMM; watch tokens/sec there — one module can still be a meaningful throughput share. `w8a16` does no fp8 matmul, so its throughput ≈ bf16.
- Untying changes param count and absolute loss vs the tied 51M configs, so compare Δ's *within this experiment*, not against `fp8_granularity/`.
- `optim/momentum_norm` / `optim/variance_norm` in W&B reflect only the AdamW-routed params (Muon's `momentum_buffer` is not aggregated).
