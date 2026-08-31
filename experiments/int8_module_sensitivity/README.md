# Int8 Module Sensitivity Ablation

Which module's **weights** are most sensitive to int8 quantization — attention, MLP, or the output head? Rank the three by **add-one-in**: quantize exactly one module group's weights to int8 W8A16 (weight-only, tensorwise) while the rest stays bf16, and measure the marginal loss each one causes. The module whose solo-quantization raises final loss the most is the most int8-sensitive.

This follows the same add-one-in design as `experiments/fp8_module_sensitivity/` — same module groups and `tensorwise` scaling — but at the **404M-scale** architecture (`configs/qwen3_404m.yaml`) rather than that experiment's 51M base, and with a different number format. Note the recipes are not identical: `fp8_module_sensitivity/` quantizes both operands (W8A8), whereas this experiment is **weight-only (W8A16)** — activations stay bf16. So the int8 runs isolate weight quantization sensitivity, and the fp8 comparison is directional (different scale *and* different what's-quantized), not a controlled one-variable swap.

## Hypothesis

W8A16 leaves activations in bf16, so the activation-outlier story that drives full W8A8 does not apply here — only weights are quantized. int8 is a *uniform* format and `tensorwise` puts one scale on the whole weight matrix, so sensitivity is driven by **each matrix's weight dynamic range** and by **how directly its quantization error reaches the loss**:

- **`lm_head`** — a single 50257×1024 matrix under one tensorwise scale, and its output *is* the logits: quantization error hits the loss directly with no downstream layers to average it out. Expected most int8-sensitive.
- **MLP** (`gate_up_proj`, `down_proj`) — large matrices, but their error enters the residual stream and is diluted by every subsequent layer and the final norm. Expected middling.
- **Attention** (`q/k/v/o_proj`) — smaller per-head projections; `qk_norm` further normalizes the q/k path. Expected cheapest.

Because weight-only tensorwise int8 is generally close to lossless, all three Δ's may land near the eval noise floor. If they do, that is itself the result: module weights tolerate int8 uniformly well, and mixed-precision weight quantization is safe across the board at this scale.

## Untied embeddings — why 455M

`apply_quantization` **skips** a `lm_head` whose weight is tied to the embedding table (swapping it would break the tie), so an `lm_head` int8 run is a no-op on a tied model. To make the output head a separately quantizable projection, every config here sets `tie_word_embeddings: false`. Untying adds `vocab × d_model` = 50257 × 1024 ≈ 51.5M params, so the model is **~455M** (455,406,080 params) vs the tied ~404M (403,894,784) of `configs/qwen3_404m.yaml`. The untied architecture is held constant across all four runs, so the only variable is which module is int8.

Note: `QuantizationConfig.exclude` defaults to `["lm_head", "*mlp.router.gate"]`; each int8 config here sets `exclude: []` explicitly so the `lm_head` run actually quantizes the head.

## Setup

All runs share the model, data, schedule, and optimizer; the only independent variable is the `quant` block's `include` list.

| Config | int8 module | `quant.include` | Quantized Linears |
|---|---|---|---|
| qwen3_455m_bf16 | none (baseline) | — (quant off) | 0 |
| qwen3_455m_w8a16_attn | attention | `[blocks.*.attn.*]` | 112 (4×28) |
| qwen3_455m_w8a16_mlp | MLP | `[blocks.*.mlp.*]` | 56 (2×28) |
| qwen3_455m_w8a16_lm_head | output head | `[lm_head]` | 1 |

- Model: d_model=1024, 28 layers, gqa 16/8, qk_norm, intermediate_size=3072, rope θ=10000, `tie_word_embeddings: false` (~455M).
- int8: `dtype: {weight: int8, act: bf16, grad_out: bf16}` (**W8A16, weight-only**), `scale.granularity: tensorwise`, `exclude: []`. Only the weight operand is quantized on the fly; activations and gradients stay bf16, and hp master weights are preserved. The forward GEMM dequantizes the int8 weight against a bf16 activation.
- Optimizer: **Muon** (`MuonAdamWOptimizer`) on all runs — 2D hidden weights → Muon, embeddings/`lm_head`/1D → AdamW, `adjust_lr_fn=match_rms_adamw` (reuses AdamW-tuned lr/wd). `momentum=0.95`, `nesterov=true`.
- All runs: seq_len=1024, batch=32, grad_accum=8 (eff. batch=256, ~262K tok/step), 50K steps (~13B tokens), bf16 mixed precision, lr=2e-4, cosine with 1500 warmup, min_lr=2e-5, seed=42, OpenWebText. lr and min_lr match `configs/qwen3_404m.yaml`; the 32×8 split (vs that config's 8×32) keeps the same effective batch while using the larger per-step batch an H200 (144GiB) affords.
- `eval_every=100`, `eval_steps=100`, `checkpoint_every=5000`.

The bf16 baseline shares this experiment's 455M untied architecture; there is no byte-identical baseline elsewhere to reuse (the existing untied-1024 config in `experiments/tie_word_embd/` uses AdamW and the default MLP width), so run `qwen3_455m_bf16` fresh. Because W8A16 touches only weights, the baseline and the int8 runs share an identical architecture and forward math outside the quantized weight matrices — every Δ is pure weight-quantization cost.

## Run

Each script runs the bf16 baseline, then each module in int8, at its scale. `run_455m.sh` is the primary 404M-scale sweep this README documents; `run_77m.sh` runs the smaller untied-51M-base variant (`qwen3_77m_*` configs) for a cheaper cross-scale check.

```bash
nohup bash experiments/int8_module_sensitivity/run_455m.sh > logs/int8_module_sensitivity_455m.log 2>&1 &
nohup bash experiments/int8_module_sensitivity/run_77m.sh  > logs/int8_module_sensitivity_77m.log  2>&1 &
```

## Results

Δ Loss is vs the bf16 baseline; larger Δ = more int8-sensitive. Weight SQNR is the per-site weight quantization SNR from `src/metrics/quant.py`, averaged over the module group — it explains *why* a module ranks where it does.

| int8 module | Quantized Linears | Final Val Loss | Val BPB | Δ Loss vs bf16 | Weight SQNR (dB) | Tokens/sec |
|---|---|---|---|---|---|---|
| none (bf16) | 0 | TBD | TBD | — | — | TBD |
| attention | 112 | TBD | TBD | TBD | TBD | TBD |
| MLP | 56 | TBD | TBD | TBD | TBD | TBD |
| lm_head | 1 | TBD | TBD | TBD | TBD | TBD |

Sensitivity ranking = modules sorted by Δ Loss (descending). Compare this ranking against `experiments/fp8_module_sensitivity/`, keeping in mind that experiment is W8A8 (activations quantized too) at a smaller scale — a reordering is suggestive, not a controlled result.

## Notes

- Compare each int8 run with this experiment's bf16 baseline using the mean validation loss over the final 10 evaluations.
- W8A16 (weight-only) tensorwise int8 is deliberately near-lossless; the Δ's may sit close to the eval noise floor, and if they do the takeaway is that weights tolerate int8 uniformly well at this scale. `rowwise` weight-only int8 (see `experiments/int8_granularity/`) would be even more lossless and is the natural follow-up if tensorwise already separates the modules; `tensorwise` is chosen here to give the modules the best chance to separate and to match the fp8 experiment's knob.
- A tensorwise-int8 module may destabilize rather than merely degrade, most likely `lm_head`. A diverged run gives a floor, not a Δ — if that happens, rerun that module under `rowwise` (see `experiments/int8_granularity/`) and mark the row as rowwise in the table rather than dropping it.
- Untying changes param count and absolute loss vs the tied 404M config, and this experiment's 455M scale differs from `fp8_module_sensitivity/`'s 77M, so compare Δ's *within this experiment*; treat the fp8 comparison as ranking-only (cross-scale and cross-recipe), not against `int8_granularity/` or `int8_tensor_dtype/`.
- `optim/momentum_norm` / `optim/variance_norm` in W&B reflect only the AdamW-routed params (Muon's `momentum_buffer` is not aggregated).
