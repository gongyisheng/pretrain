# Activation Limit under Int8 W8A8

Sweep the MLP activation limit (`mlp_kwargs.activation_kwargs.act_limit`) at Qwen3-51M under int8 W8A8 blockwise-2D 32x32 quantization, against a bf16 baseline. `act_limit` bounds the `gate_proj` / `up_proj` outputs before SwiGLU, as in gpt-oss's `swiglu_limit` and DeepSeek V4. It is a per-side mapping — `{gate: {min, max}, up: {min, max}}`, every key optional (missing = unbounded on that side). Each `limit L` row uses the DeepSeek asymmetric form: `gate: {max: L}` (SiLU is already bounded below) and `up: {min: -L, max: L}`.

> Semantics note: earlier runs clamped both gate and up symmetrically to `[-L, L]`. The current schema clamps gate and up separately, so tight-limit results are not directly comparable to any pre-migration numbers.

## Hypothesis

A 32x32 scale block already contains outliers locally: one large element only stretches the scale of its own tile, so the bulk of the distribution keeps its int8 codes. That leaves the limit a narrower job — shrinking the range inside the few tiles that hold an outlier — so the expected gain is smaller than under tensorwise scaling, and the limit may be pure signal loss. This run tests whether clamping still buys anything once the granularity is fine.

The limit also costs loss in full precision on its own. [`activation_limit`](../activation_limit/README.md) runs the same ladder in bf16; subtract its deltas to isolate the quantization benefit.

## Setup

7 runs: bf16 baseline, int8 W8A16 (weight-only) reference, unbounded int8 W8A8, and four int8 W8A8 limits. Limits halve each step down to 3, so the range shrinks geometrically.

| Config | Weight | Activation | grad_out | `act_limit` |
|---|---|---|---|---|
| qwen3_51m_bf16 | bf16 | bf16 | bf16 | — |
| qwen3_51m_int8_w8a16 | int8 | bf16 | bf16 | — |
| qwen3_51m_int8_w8a8 | int8 | int8 | bf16 | — |
| qwen3_51m_int8_w8a8_act_limit31 | int8 | int8 | bf16 | 31 |
| qwen3_51m_int8_w8a8_act_limit15 | int8 | int8 | bf16 | 15 |
| qwen3_51m_int8_w8a8_act_limit7 | int8 | int8 | bf16 | 7 |
| qwen3_51m_int8_w8a8_act_limit3 | int8 | int8 | bf16 | 3 |

All runs: Qwen3 51M (d_model=512, 8 layers, GQA 8/4 with qk_norm, dense SwiGLU MLP intermediate_size=1536, ~50.9M params), seq_len=1024, batch_size=32, grad_accum=8 (effective batch=256), 50K steps, Muon (`match_rms_adamw`, momentum=0.95, nesterov), lr=5e-4, cosine schedule with 1500 warmup steps and min_lr=5e-5, bf16 mixed precision, OpenWebText, seed 42, `eval_every=100`, `eval_steps=100`. Int8 runs use blockwise-2D scaling with a 32x32 tile and `lm_head` excluded; `grad_out` stays bf16. The W8A16 run quantizes weights only, so it isolates the weight-quantization share of the W8A8 gap and is unaffected by the limit.

## Run

```bash
nohup bash experiments/int8_activation_limit/run.sh > logs/int8_activation_limit.log 2>&1 &
```

## Results

W&B project: `pretrain-int8-activation-limit`.

| Config | Precision | `act_limit` | Final Val Loss | Δ vs bf16 | Δ vs unbounded int8 |
|---|---|---|---|---|---|
| qwen3_51m_bf16 | bf16 | — | | 0 | |
| qwen3_51m_int8_w8a16 | int8 W8A16 | — | | | |
| qwen3_51m_int8_w8a8 | int8 W8A8 | — | | | 0 |
| qwen3_51m_int8_w8a8_act_limit31 | int8 W8A8 | 31 | | | |
| qwen3_51m_int8_w8a8_act_limit15 | int8 W8A8 | 15 | | | |
| qwen3_51m_int8_w8a8_act_limit7 | int8 W8A8 | 7 | | | |
| qwen3_51m_int8_w8a8_act_limit3 | int8 W8A8 | 3 | | | |

## Notes

- Compare runs using the mean validation loss over the final 10 evaluations.
- The limit only touches the dense MLP's pre-activation. Attention projections and `lm_head` are unaffected, so it bounds the `down_proj` input but not every quantized activation.
- Limits are absolute pre-activation magnitudes, not int8 codes; 31/15/7/3 are a geometric ladder, not a bit-width mapping.
- A limit helps quantization only if its int8 gain exceeds the matching bf16 loss from `activation_limit`.
