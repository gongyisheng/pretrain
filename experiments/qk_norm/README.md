# QK Norm Ablation

Test the effect of QK normalization (pre-RoPE query/key normalization) on pretraining quality using the Qwen3 architecture.

## Hypothesis

QK norm stabilizes attention logit magnitudes and prevents entropy collapse, especially in deeper models or longer training runs. At 57M scale the effect may be small, but it should improve training stability (fewer gradient spikes) even if final loss is similar.

## Setup

| Config | qk_norm | Description | Approx params |
|---|---|---|---|
| qwen3_57m_qk_norm_on | true | QK norm enabled (Qwen3 default) | ~57M |
| qwen3_57m_qk_norm_off | false | QK norm disabled | ~57M |

All runs share: Qwen3 57M (d_model=512, layers=8, heads=8, kv_heads=4), seq_len=1024, batch_size=16, grad_accum=16 (effective batch=256, ~262K tokens/step), 5K steps (~1.3B tokens, ~23x params), lr=6e-4, cosine schedule with 200 warmup steps (~4%), min_lr=6e-5, bf16, OpenWebText.

## Run

```bash
nohup bash experiments/qk_norm/run.sh > logs/qk_norm.log 2>&1 &
```

## Results

| qk_norm | Final Val Loss |
|---|---|
| on | |
| off | |

## Notes
