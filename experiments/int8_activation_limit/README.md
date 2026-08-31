# Activation Limit under Int8 W8A8

Sweep the MLP activation limit (`mlp_kwargs.activation_limit`) at Qwen3-51M under int8 W8A8 tensorwise quantization, against a bf16 baseline. `activation_limit` bounds the `gate_up_proj` output to `[-L, L]` before SwiGLU, as in gpt-oss's `swiglu_limit` and DeepSeek V4.

## Hypothesis

Tensorwise activation scaling is set by the single largest element, so a few MLP outliers stretch the scale and waste int8 codes on the bulk of the distribution. Bounding the pre-activation should shrink the dynamic range, cut int8 quantization error, and close part of the W8A8 gap to bf16 — up to the point where the limit starts destroying signal.

The limit also costs loss in full precision on its own. [`activation_limit`](../activation_limit/README.md) runs the same ladder in bf16; subtract its deltas to isolate the quantization benefit.

## Setup

9 runs: bf16 baseline, int8 W8A16 (weight-only) reference, unbounded int8 W8A8, and six int8 W8A8 limits. Limits halve each step so the range shrinks geometrically.

| Config | Weight | Activation | grad_out | `activation_limit` |
|---|---|---|---|---|
| qwen3_51m_bf16 | bf16 | bf16 | bf16 | — |
| qwen3_51m_int8_w8a16 | int8 | bf16 | bf16 | — |
| qwen3_51m_int8_w8a8 | int8 | int8 | bf16 | — |
| qwen3_51m_int8_w8a8_act_limit127 | int8 | int8 | bf16 | 127 |
| qwen3_51m_int8_w8a8_act_limit63 | int8 | int8 | bf16 | 63 |
| qwen3_51m_int8_w8a8_act_limit31 | int8 | int8 | bf16 | 31 |
| qwen3_51m_int8_w8a8_act_limit15 | int8 | int8 | bf16 | 15 |
| qwen3_51m_int8_w8a8_act_limit7 | int8 | int8 | bf16 | 7 |
| qwen3_51m_int8_w8a8_act_limit3 | int8 | int8 | bf16 | 3 |

All runs: Qwen3 51M (d_model=512, 8 layers, GQA 8/4 with qk_norm, dense SwiGLU MLP intermediate_size=1536, ~50.9M params), seq_len=1024, batch_size=16, grad_accum=16 (effective batch=256), 50K steps, Muon (`match_rms_adamw`, momentum=0.95, nesterov), lr=5e-4, cosine schedule with 1500 warmup steps and min_lr=5e-5, bf16 mixed precision, OpenWebText, seed 42, `eval_every=100`, `eval_steps=100`. Int8 runs use tensorwise scaling with `lm_head` excluded; `grad_out` stays bf16. The W8A16 run quantizes weights only, so it isolates the weight-quantization share of the W8A8 gap and is unaffected by the limit.

## Run

```bash
nohup bash experiments/int8_activation_limit/run.sh > logs/int8_activation_limit.log 2>&1 &
```

## Results

W&B project: `pretrain-int8-activation-limit`.

| Config | Precision | `activation_limit` | Final Val Loss | Δ vs bf16 | Δ vs unbounded int8 |
|---|---|---|---|---|---|
| qwen3_51m_bf16 | bf16 | — | | 0 | |
| qwen3_51m_int8_w8a16 | int8 W8A16 | — | | | |
| qwen3_51m_int8_w8a8 | int8 W8A8 | — | | | 0 |
| qwen3_51m_int8_w8a8_act_limit127 | int8 W8A8 | 127 | | | |
| qwen3_51m_int8_w8a8_act_limit63 | int8 W8A8 | 63 | | | |
| qwen3_51m_int8_w8a8_act_limit31 | int8 W8A8 | 31 | | | |
| qwen3_51m_int8_w8a8_act_limit15 | int8 W8A8 | 15 | | | |
| qwen3_51m_int8_w8a8_act_limit7 | int8 W8A8 | 7 | | | |
| qwen3_51m_int8_w8a8_act_limit3 | int8 W8A8 | 3 | | | |

## Notes

- Compare runs using the mean validation loss over the final 10 evaluations.
- The limit only touches the dense MLP's pre-activation. Attention projections and `lm_head` are unaffected, so it bounds the `down_proj` input but not every quantized activation.
- Limits are absolute pre-activation magnitudes, not int8 codes; 127/63/31 are a geometric ladder, not a bit-width mapping.
- A limit helps quantization only if its int8 gain exceeds the matching bf16 loss from `activation_limit`.
