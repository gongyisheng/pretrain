# QK Norm Ablation

Test the effect of QK normalization (pre-RoPE query/key normalization) on pretraining quality using the Qwen3 architecture.

## Hypothesis

QK norm stabilizes attention logit magnitudes and prevents entropy collapse, especially in deeper models or longer training runs. At 57M scale the effect may be small, but it should improve training stability (fewer gradient spikes) even if final loss is similar.

## Setup

| Config | qk_norm | Description | Approx params |
|---|---|---|---|
| qwen3_57m_qk_norm_on | true | QK norm enabled (Qwen3 default) | ~57M |
| qwen3_57m_qk_norm_off | false | QK norm disabled | ~57M |
| qwen3_0.5b_qk_norm_on | true | QK norm enabled (Qwen3 default) | ~0.5B |
| qwen3_0.5b_qk_norm_off | false | QK norm disabled | ~0.5B |

57M runs: Qwen3 57M (d_model=512, layers=8, heads=8, kv_heads=4), seq_len=1024, batch_size=16, grad_accum=16 (effective batch=256, ~262K tokens/step), 50K steps (~13B tokens), lr=6e-4, cosine schedule with 1500 warmup steps (~3%), min_lr=6e-5, bf16, OpenWebText.

0.5B runs: Qwen3 0.5B (d_model=1024, layers=28, heads=16, kv_heads=8), seq_len=1024, batch_size=8, grad_accum=4 (effective batch=32, ~32K tokens/step), 50K steps (~1.6B tokens), lr=2e-4, cosine schedule with 1500 warmup steps (~3%), min_lr=2e-5, bf16, OpenWebText.

## Run

```bash
nohup bash experiments/qk_norm/run.sh > logs/qk_norm.log 2>&1 &
```

## Results

| Model | qk_norm | Final Val Loss |
|---|---|---|
| 57M | on | |
| 57M | off | |
| 0.5B | on | |
| 0.5B | off | |

## Notes
