# Int8 Module Sensitivity Ablation

Which module is most sensitive to int8 quantization — attention, MLP, or the output head? Rank the three by **add-one-in**: quantize exactly one module group to int8 W8A8 (tensorwise) while the rest stays bf16, and measure the marginal loss each one causes. The module whose solo-quantization raises final loss the most is the most int8-sensitive.

This mirrors `experiments/fp8_module_sensitivity/` — same architecture, schedule, optimizer, seed, module groups, and `tensorwise` scaling. Two things change: the number format, and an `activation_limit` of 127 (the int8 code max) on the int8 runs, so the Δ tables compare with that caveat.

## Hypothesis

int8 is a *uniform* format; fp8-E4M3 has an exponent field. A single large activation outlier costs int8 most of its resolution for the bulk of the tensor, but costs E4M3 little. So the int8 ranking should be driven by **where activation outliers live**, not by GEMM size or position in the network as with fp8:

- **MLP** (`gate_up_proj`, `down_proj`) — `down_proj`'s input is the SwiGLU product, the canonical outlier-heavy activation. Expected most int8-sensitive, and a module fp8 barely notices. The `activation_limit` of 127 caps that outlier tail, so if it binds, the MLP's Δ lands lower than the unbounded prediction.
- **`lm_head`** — its input is the final hidden state, which carries massive-activation channels, and its output is 50257-wide under a single tensorwise scale. Was the fp8 favorite; expected still bad.
- **Attention** (`q/k/v/o_proj`) — `q/k/v` consume a post-RMSNorm activation, which is well-conditioned; `o_proj`'s input can carry attention-sink outliers. Expected cheapest.

If the ranking differs from the fp8 experiment's, the takeaway is that module sensitivity is **format-dependent** and a mixed-precision recipe tuned for fp8 does not transfer to int8.

## Untied embeddings — why 77M

`apply_quantization` **skips** a `lm_head` whose weight is tied to the embedding table (swapping it would break the tie), so an `lm_head` int8 run is a no-op on a tied model. To make the output head a separately quantizable projection, every config here sets `tie_word_embeddings: false`. Untying adds `vocab × d_model` = 50257 × 512 ≈ 25.7M params, so the model is **~77M** (vs the tied ~51M). The untied architecture is held constant across all four runs, so the only variable is which module is int8.

Note: `QuantizationConfig.exclude` defaults to `["lm_head", "*mlp.router.gate"]`; each int8 config here sets `exclude: []` explicitly so the `lm_head` run actually quantizes the head.

## Setup

All runs share the model, data, schedule, and optimizer; the only independent variable is the `quant` block's `include` list.

| Config | int8 module | `quant.include` | Quantized Linears | `activation_limit` |
|---|---|---|---|---|
| qwen3_77m_bf16 | none (baseline) | — (quant off) | 0 | — |
| qwen3_77m_int8_attn | attention | `[blocks.*.attn.*]` | 32 (4×8) | 127 |
| qwen3_77m_int8_mlp | MLP | `[blocks.*.mlp.*]` | 16 (2×8) | 127 |
| qwen3_77m_int8_lm_head | output head | `[lm_head]` | 1 | 127 |

- Model: d_model=512, 8 layers, gqa 8/4, qk_norm, intermediate_size=1536, rope θ=10000, `tie_word_embeddings: false` (~77M).
- All three int8 runs set `mlp_kwargs.activation_limit: 127.0`, the int8 code max, matching the W8A8 cells in `int8_tensor_dtype/`. It bounds the dense MLP's `gate_up_proj` output before SwiGLU, so it keeps `down_proj`'s input — the outlier-heavy activation the hypothesis turns on — inside a fixed range. The limit is held constant across the three int8 runs so the only variable stays the `include` list.
- int8: `dtype: {weight: int8, act: int8, grad_out: bf16}` (W8A8), `scale.granularity: tensorwise`, `exclude: []`. `grad_out` stays bf16 — matching `int8_granularity/` and `int8_tensor_dtype/`, which hold gradients out of the integer path. Only the matched GEMM operands are quantized on the fly; everything else stays bf16/fp32, hp master weights preserved.
- Optimizer: **Muon** (`MuonAdamWOptimizer`) on all runs — 2D hidden weights → Muon, embeddings/`lm_head`/1D → AdamW, `adjust_lr_fn=match_rms_adamw` (reuses AdamW-tuned lr/wd). `momentum=0.95`, `nesterov=true`.
- All runs: seq_len=1024, batch=16, grad_accum=16 (eff. batch=256, ~262K tok/step), 50K steps (~13B tokens), bf16 mixed precision, lr=5e-4, cosine with 1500 warmup, min_lr=5e-5, seed=42, OpenWebText.
- `eval_every=100`, `eval_steps=100`, `checkpoint_every=5000`.

The bf16 baseline is byte-identical to `experiments/fp8_module_sensitivity/qwen3_77m_bf16.yaml` apart from its checkpoint dir and W&B project. If that run already completed, its final loss can be reused and this baseline skipped.

## Run

Runs the bf16 baseline, then each module in int8.

```bash
nohup bash experiments/int8_module_sensitivity/run.sh > logs/int8_module_sensitivity.log 2>&1 &
```

## Results

Δ Loss is vs the bf16 baseline; larger Δ = more int8-sensitive. Act SQNR is the per-site activation quantization SNR from `src/metrics/quant.py`, averaged over the module group — it explains *why* a module ranks where it does.

| int8 module | Quantized Linears | Final Val Loss | Val BPB | Δ Loss vs bf16 | Act SQNR (dB) | Tokens/sec |
|---|---|---|---|---|---|---|
| none (bf16) | 0 | TBD | TBD | — | — | TBD |
| attention | 32 | TBD | TBD | TBD | TBD | TBD |
| MLP | 16 | TBD | TBD | TBD | TBD | TBD |
| lm_head | 1 | TBD | TBD | TBD | TBD | TBD |

Sensitivity ranking = modules sorted by Δ Loss (descending). Compare this ranking against `experiments/fp8_module_sensitivity/` — a reordering is the headline result.

## Notes

- Compare each int8 run with this experiment's bf16 baseline using the mean validation loss over the final 10 evaluations.
- The bf16 baseline has no `activation_limit`, so every Δ includes the limit's own full-precision cost on top of the quantization cost. That cost is shared by all three int8 runs, so the *ranking* is unaffected; only the absolute Δ's are inflated. [`activation_limit`](../activation_limit/README.md) measures the bf16-only cost if you want to subtract it.
- The limit applies to the MLP in all three runs, including `int8_attn` and `int8_lm_head` where the MLP is not quantized — there it is pure cost with no matching quantization benefit. This is the price of holding the architecture constant across the ablation; it biases the two non-MLP Δ's upward relative to the MLP's.
- tensorwise W8A8 is deliberately the noisiest int8 recipe, chosen to amplify per-module differences and to match the fp8 experiment's knob. Weight-only (W8A16) rowwise int8 is close to lossless and would put all three Δ's under the eval noise floor.
- A tensorwise-int8 module may destabilize rather than merely degrade, most likely `lm_head` or the MLP. A diverged run gives a floor, not a Δ — if that happens, rerun that module under `rowwise` (see `experiments/int8_granularity/`) and mark the row as rowwise in the table rather than dropping it.
- Untying changes param count and absolute loss vs the tied 51M configs, so compare Δ's *within this experiment* (and against `fp8_module_sensitivity/`, which shares this architecture), not against `int8_granularity/` or `int8_tensor_dtype/`.
- `optim/momentum_norm` / `optim/variance_norm` in W&B reflect only the AdamW-routed params (Muon's `momentum_buffer` is not aggregated).
