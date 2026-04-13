# Bias Ablation

Test whether adding bias terms to Qwen3 linear layers improves pretraining quality. Qwen3 (like LLaMA) defaults to no bias everywhere. This is a reverse ablation: baseline is no-bias, variants add bias.

## Hypothesis

Bias terms add a constant offset to linear projections. Modern architectures remove them without quality loss. At 57M scale, adding bias should be neutral or slightly negative — the extra parameters don't contribute meaningful expressivity, and the optimization landscape may be slightly harder with the additional degrees of freedom.

## Setup

| Config | attn_bias | mlp_bias | lm_head_bias | Description | Approx params |
|---|---|---|---|---|---|
| qwen3_57m_bias_none | false | false | false | Baseline (Qwen3 default) | ~57M |
| qwen3_57m_bias_all | true | true | false | All hidden-layer bias added | ~57M |
| qwen3_57m_bias_attn | true | false | false | Attention bias only (Q/K/V/O) | ~57M |
| qwen3_57m_bias_mlp | false | true | false | MLP bias only (gate/up/down) | ~57M |
| qwen3_57m_bias_lm_head | false | false | true | LM head bias only | ~57M |

All runs share: Qwen3 57M (d_model=512, layers=8, heads=8, kv_heads=4), seq_len=1024, batch_size=16, grad_accum=16 (effective batch=256, ~262K tokens/step), 5K steps (~1.3B tokens, ~23x params), lr=6e-4, cosine schedule with 200 warmup steps (~4%), min_lr=6e-5, bf16, OpenWebText.

Note: RMSNorm has no bias parameter, so norm layers are unaffected by this ablation.

## Run

```bash
nohup bash experiments/bias/run.sh > logs/bias.log 2>&1 &
```

## Results

| Variant | Final Val Loss |
|---|---|
| none (baseline) | |
| all | |
| attn-only | |
| mlp-only | |
| lm-head-only | |

## Notes
