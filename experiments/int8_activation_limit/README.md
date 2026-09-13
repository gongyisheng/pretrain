# Activation Limit under Int8 W8A8

Sweep the MLP activation limit (`mlp_kwargs.activation_kwargs.act_limit`) at Qwen3-51M under int8 W8A8, crossed with the activation scaling granularity — tensorwise vs blockwise-1D 32 vs blockwise-2D 32x32 — against a bf16 baseline. `act_limit` bounds the `gate_proj` / `up_proj` outputs before SwiGLU, as in gpt-oss's `swiglu_limit` and DeepSeek V4. It is a per-side mapping — `{gate: {min, max}, up: {min, max}}`, every key optional (missing = unbounded on that side). Each `limit L` row uses the DeepSeek asymmetric form: `gate: {max: L}` (SiLU is already bounded below) and `up: {min: -L, max: L}`.

> Semantics note: earlier runs clamped both gate and up symmetrically to `[-L, L]`. The current schema clamps gate and up separately, so tight-limit results are not directly comparable to any pre-migration numbers.

## Hypothesis

The limit's value should depend on scaling granularity. Under **tensorwise** scaling a single outlier stretches the scale of the whole tensor, so every element loses int8 codes to it — clamping the outlier away should recover real precision for the bulk of the distribution. Under **blockwise-2D 32x32** a 32x32 tile already contains its outliers locally: one large element only stretches the scale of its own tile, so the bulk keeps its codes regardless. That leaves the limit a narrower job — shrinking the range inside the few tiles that hold an outlier — so the expected gain is smaller, and the limit may be pure signal loss.

**Blockwise-1D 32** sits between the two, and is the interesting middle because activation outliers are channel-structured. A 1x32 block spans 32 consecutive channels of one token, so it isolates an outlier to 32 elements — a tighter containment than the 1024-element 2D tile — but a persistently large channel still stretches its block in every token. Whether the limit buys anything here is the question: if outliers are confined to a few channels, 1D containment is already near-complete and the ladder should look like the 2D one; if they spread across many channels, the limit should still help.

Prediction: the tensorwise ladder shows a loss minimum at some finite limit, the blockwise-2D ladder is flat-to-monotonically-worse as the limit tightens, and blockwise-1D falls between — closer to 2D if the containment argument holds.

The limit also costs loss in full precision on its own. [`activation_limit`](../activation_limit/README.md) runs the same ladder in bf16; subtract its deltas to isolate the quantization benefit.

## Setup

19 runs: bf16 baseline, plus a 3x5 grid of granularity x limit under W8A8 (unbounded, 31, 15, 7, 3), plus one int8 W8A16 weight-only reference per granularity. Limits halve each step down to 3, so the range shrinks geometrically.

| Config | Weight | Activation | grad_out | Granularity | `act_limit` |
|---|---|---|---|---|---|
| qwen3_51m_bf16 | bf16 | bf16 | bf16 | — | — |
| qwen3_51m_int8_w8a16_tensorwise | int8 | bf16 | bf16 | tensorwise | — |
| qwen3_51m_int8_w8a8_tensorwise | int8 | int8 | bf16 | tensorwise | — |
| qwen3_51m_int8_w8a8_tensorwise_act_limit31 | int8 | int8 | bf16 | tensorwise | 31 |
| qwen3_51m_int8_w8a8_tensorwise_act_limit15 | int8 | int8 | bf16 | tensorwise | 15 |
| qwen3_51m_int8_w8a8_tensorwise_act_limit7 | int8 | int8 | bf16 | tensorwise | 7 |
| qwen3_51m_int8_w8a8_tensorwise_act_limit3 | int8 | int8 | bf16 | tensorwise | 3 |
| qwen3_51m_int8_w8a16_blockwise1d_32 | int8 | bf16 | bf16 | blockwise1d 32 | — |
| qwen3_51m_int8_w8a8_blockwise1d_32 | int8 | int8 | bf16 | blockwise1d 32 | — |
| qwen3_51m_int8_w8a8_blockwise1d_32_act_limit31 | int8 | int8 | bf16 | blockwise1d 32 | 31 |
| qwen3_51m_int8_w8a8_blockwise1d_32_act_limit15 | int8 | int8 | bf16 | blockwise1d 32 | 15 |
| qwen3_51m_int8_w8a8_blockwise1d_32_act_limit7 | int8 | int8 | bf16 | blockwise1d 32 | 7 |
| qwen3_51m_int8_w8a8_blockwise1d_32_act_limit3 | int8 | int8 | bf16 | blockwise1d 32 | 3 |
| qwen3_51m_int8_w8a16_blockwise2d_32 | int8 | bf16 | bf16 | blockwise2d 32x32 | — |
| qwen3_51m_int8_w8a8_blockwise2d_32 | int8 | int8 | bf16 | blockwise2d 32x32 | — |
| qwen3_51m_int8_w8a8_blockwise2d_32_act_limit31 | int8 | int8 | bf16 | blockwise2d 32x32 | 31 |
| qwen3_51m_int8_w8a8_blockwise2d_32_act_limit15 | int8 | int8 | bf16 | blockwise2d 32x32 | 15 |
| qwen3_51m_int8_w8a8_blockwise2d_32_act_limit7 | int8 | int8 | bf16 | blockwise2d 32x32 | 7 |
| qwen3_51m_int8_w8a8_blockwise2d_32_act_limit3 | int8 | int8 | bf16 | blockwise2d 32x32 | 3 |

All runs: Qwen3 51M (d_model=512, 8 layers, GQA 8/4 with qk_norm, dense SwiGLU MLP intermediate_size=1536, ~50.9M params), seq_len=1024, batch_size=32, grad_accum=8 (effective batch=256), 50K steps, Muon (`match_rms_adamw`, momentum=0.95, nesterov), lr=5e-4, cosine schedule with 1500 warmup steps and min_lr=5e-5, bf16 mixed precision, OpenWebText, seed 42, `eval_every=100`, `eval_steps=100`. Int8 runs exclude `lm_head` and keep `grad_out` in bf16; granularity applies to both weight and activation scales. The W8A16 runs quantize weights only, so each isolates the weight-quantization share of its granularity's W8A8 gap and is unaffected by the limit.

## Run

```bash
nohup bash experiments/int8_activation_limit/run.sh > logs/int8_activation_limit.log 2>&1 &
```

## Results

W&B project: `pretrain-int8-activation-limit`.

| Config | Precision | Granularity | `act_limit` | Final Val Loss | Δ vs bf16 | Δ vs unbounded int8 (same granularity) |
|---|---|---|---|---|---|---|
| qwen3_51m_bf16 | bf16 | — | — | | 0 | |
| qwen3_51m_int8_w8a16_tensorwise | int8 W8A16 | tensorwise | — | | | |
| qwen3_51m_int8_w8a8_tensorwise | int8 W8A8 | tensorwise | — | | | 0 |
| qwen3_51m_int8_w8a8_tensorwise_act_limit31 | int8 W8A8 | tensorwise | 31 | | | |
| qwen3_51m_int8_w8a8_tensorwise_act_limit15 | int8 W8A8 | tensorwise | 15 | | | |
| qwen3_51m_int8_w8a8_tensorwise_act_limit7 | int8 W8A8 | tensorwise | 7 | | | |
| qwen3_51m_int8_w8a8_tensorwise_act_limit3 | int8 W8A8 | tensorwise | 3 | | | |
| qwen3_51m_int8_w8a16_blockwise1d_32 | int8 W8A16 | blockwise1d 32 | — | | | |
| qwen3_51m_int8_w8a8_blockwise1d_32 | int8 W8A8 | blockwise1d 32 | — | | | 0 |
| qwen3_51m_int8_w8a8_blockwise1d_32_act_limit31 | int8 W8A8 | blockwise1d 32 | 31 | | | |
| qwen3_51m_int8_w8a8_blockwise1d_32_act_limit15 | int8 W8A8 | blockwise1d 32 | 15 | | | |
| qwen3_51m_int8_w8a8_blockwise1d_32_act_limit7 | int8 W8A8 | blockwise1d 32 | 7 | | | |
| qwen3_51m_int8_w8a8_blockwise1d_32_act_limit3 | int8 W8A8 | blockwise1d 32 | 3 | | | |
| qwen3_51m_int8_w8a16_blockwise2d_32 | int8 W8A16 | blockwise2d 32x32 | — | | | |
| qwen3_51m_int8_w8a8_blockwise2d_32 | int8 W8A8 | blockwise2d 32x32 | — | | | 0 |
| qwen3_51m_int8_w8a8_blockwise2d_32_act_limit31 | int8 W8A8 | blockwise2d 32x32 | 31 | | | |
| qwen3_51m_int8_w8a8_blockwise2d_32_act_limit15 | int8 W8A8 | blockwise2d 32x32 | 15 | | | |
| qwen3_51m_int8_w8a8_blockwise2d_32_act_limit7 | int8 W8A8 | blockwise2d 32x32 | 7 | | | |
| qwen3_51m_int8_w8a8_blockwise2d_32_act_limit3 | int8 W8A8 | blockwise2d 32x32 | 3 | | | |

## Notes

- Compare runs using the mean validation loss over the final 10 evaluations.
- The headline number is the interaction: how much more the limit buys as granularity coarsens (blockwise2d 32x32 → blockwise1d 32 → tensorwise). Compare each ladder against its own unbounded W8A8 row, not across granularities.
- The limit only touches the dense MLP's pre-activation. Attention projections and `lm_head` are unaffected, so it bounds the `down_proj` input but not every quantized activation.
- Limits are absolute pre-activation magnitudes, not int8 codes; 31/15/7/3 are a geometric ladder, not a bit-width mapping.
- A limit helps quantization only if its int8 gain exceeds the matching bf16 loss from `activation_limit`.
