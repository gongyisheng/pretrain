# MLP Activation Limit

Sweep the MLP activation limit (`mlp_kwargs.activation_limit`) at Qwen3-51M in bf16. `activation_limit` bounds the `gate_up_proj` output to `[-L, L]` before SwiGLU, as in gpt-oss's `swiglu_limit` and DeepSeek V4.

## Hypothesis

The limit is a capacity constraint in full precision: it removes the tail of the pre-activation distribution and blocks gradient through clamped elements. A loose limit should be a near no-op; a tight one should cost loss. This run establishes where that cost begins, independent of quantization — see [`int8_activation_limit`](../int8_activation_limit/README.md) for the same ladder under int8 W8A8.

## Setup

6 runs. Limits halve each step so the range shrinks geometrically.

| Config | `activation_limit` |
|---|---|
| qwen3_51m_bf16 | — |
| qwen3_51m_act_limit127 | 127 |
| qwen3_51m_act_limit63 | 63 |
| qwen3_51m_act_limit31 | 31 |
| qwen3_51m_act_limit15 | 15 |
| qwen3_51m_act_limit7 | 7 |

All runs: Qwen3 51M (d_model=512, 8 layers, GQA 8/4 with qk_norm, dense SwiGLU MLP intermediate_size=1536, ~50.9M params), seq_len=1024, batch_size=16, grad_accum=16 (effective batch=256), 50K steps, Muon (`match_rms_adamw`, momentum=0.95, nesterov), lr=5e-4, cosine schedule with 1500 warmup steps and min_lr=5e-5, bf16 mixed precision, OpenWebText, seed 42, `eval_every=100`, `eval_steps=25`.

## Run

```bash
nohup bash experiments/activation_limit/run.sh > logs/activation_limit.log 2>&1 &
```

## Results

W&B project: `pretrain-activation-limit`.

| Config | `activation_limit` | Final Val Loss | Δ vs unbounded |
|---|---|---|---|
| qwen3_51m_bf16 | — | | 0 |
| qwen3_51m_act_limit127 | 127 | | |
| qwen3_51m_act_limit63 | 63 | | |
| qwen3_51m_act_limit31 | 31 | | |
| qwen3_51m_act_limit15 | 15 | | |
| qwen3_51m_act_limit7 | 7 | | |

## Notes

- Compare runs using the mean validation loss over the final 10 evaluations.
- The limit only touches the dense MLP's pre-activation; attention projections and `lm_head` are unaffected.
- These bf16 numbers are the control arm for the int8 sweep: subtract them from the int8 deltas to isolate the quantization effect from the plain capacity cost.
