# MLP Gating Ablation

Test the effect of MLP gating (SwiGLU-style 3-matrix gated FFN vs. plain 2-matrix FFN) on pretraining quality using the Qwen3 architecture.

## Hypothesis

A gated FFN (SwiGLU) improves expressivity and final loss at fixed parameter count by letting one branch modulate the other. Effect should be visible at 57M, more pronounced at 0.5B.

## Setup

To isolate the gating mechanism, the activation is held fixed at `silu` and `intermediate_size` is adjusted to match FFN parameter counts across variants:

- Gated FFN params = `3 * d_model * intermediate_size`
- Ungated FFN params = `2 * d_model * intermediate_size`

Setting `intermediate_size = 4 * d_model` for gated and `6 * d_model` for ungated yields identical FFN params (`12 * d_model^2`), so the only difference is the gating multiply.

| Config | mlp_gated | intermediate_size | Description | Approx params |
|---|---|---|---|---|
| qwen3_57m_mlp_gated_on | true | 2048 (4·d) | SwiGLU (Qwen3 default) | ~57M |
| qwen3_57m_mlp_gated_off | false | 3072 (6·d) | Plain SiLU MLP, params matched | ~57M |
| qwen3_0.5b_mlp_gated_on | true | 4096 (4·d) | SwiGLU (Qwen3 default) | ~0.5B |
| qwen3_0.5b_mlp_gated_off | false | 6144 (6·d) | Plain SiLU MLP, params matched | ~0.5B |

57M runs: Qwen3 57M (d_model=512, layers=8, heads=8, kv_heads=4), seq_len=1024, batch_size=16, grad_accum=16 (effective batch=256, ~262K tokens/step), 50K steps (~13B tokens), lr=6e-4, cosine schedule with 1500 warmup steps (~3%), min_lr=6e-5, bf16, OpenWebText.

0.5B runs: Qwen3 0.5B (d_model=1024, layers=28, heads=16, kv_heads=8), seq_len=1024, batch_size=8, grad_accum=32 (effective batch=256, ~262K tokens/step), 50K steps (~13B tokens), lr=2e-4, cosine schedule with 1500 warmup steps (~3%), min_lr=2e-5, bf16, OpenWebText.

## Run

```bash
nohup bash experiments/mlp_gated/run.sh > logs/mlp_gated.log 2>&1 &
```

## Results

| Model | mlp_gated | Final Val Loss |
|---|---|---|
| 57M | on | |
| 57M | off | |
| 0.5B | on | |
| 0.5B | off | |

## Notes
