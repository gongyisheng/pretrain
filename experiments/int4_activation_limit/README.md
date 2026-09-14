# Activation Limit under Int4 W4A16

Sweep MLP activation limits 31, 15, 7, and 3 at Qwen3-51M with int4 weights, bf16 activations and output gradients, blockwise 1D `(1, 16)` scaling, fp32 block scales, and global scaling disabled. Include bf16 and unbounded W4A16 controls.

## Hypothesis

Clipping large MLP activations may reduce the effect of int4 weight reconstruction error on the MLP output and improve validation loss. Tight limits may instead discard useful signal. Compare each limit with the unbounded W4A16 control to measure the net effect.

Activations stay in bf16, so this sweep tests clipping during weight-only quantized training. It does not change activation quantization precision.

## Setup

Six runs, all approximately 50.9M parameters:

| Config | Weight / activation / grad_out | Block shape | Block scale / global scale | Activation limit | Params |
|---|---|---|---|---|---|
| [qwen3_51m_bf16.yaml](qwen3_51m_bf16.yaml) | bf16 / bf16 / bf16 | — | — | Unbounded | ~50.9M |
| [qwen3_51m_int4_w4a16_blockwise1d_16.yaml](qwen3_51m_int4_w4a16_blockwise1d_16.yaml) | int4 / bf16 / bf16 | (1, 16) | fp32 / disabled | Unbounded | ~50.9M |
| [qwen3_51m_int4_w4a16_blockwise1d_16_act_limit31.yaml](qwen3_51m_int4_w4a16_blockwise1d_16_act_limit31.yaml) | int4 / bf16 / bf16 | (1, 16) | fp32 / disabled | 31 | ~50.9M |
| [qwen3_51m_int4_w4a16_blockwise1d_16_act_limit15.yaml](qwen3_51m_int4_w4a16_blockwise1d_16_act_limit15.yaml) | int4 / bf16 / bf16 | (1, 16) | fp32 / disabled | 15 | ~50.9M |
| [qwen3_51m_int4_w4a16_blockwise1d_16_act_limit7.yaml](qwen3_51m_int4_w4a16_blockwise1d_16_act_limit7.yaml) | int4 / bf16 / bf16 | (1, 16) | fp32 / disabled | 7 | ~50.9M |
| [qwen3_51m_int4_w4a16_blockwise1d_16_act_limit3.yaml](qwen3_51m_int4_w4a16_blockwise1d_16_act_limit3.yaml) | int4 / bf16 / bf16 | (1, 16) | fp32 / disabled | 3 | ~50.9M |

Shared setup: d_model=512, 8 layers, GQA 8/4 with QK norm, dense SwiGLU with intermediate_size=1536, RMSNorm, RoPE, and zero dropout. OpenWebText uses `tokenizers/custom_bpe_50k`, sequence length 1024, and a 1% validation split. Training uses batch_size=128, gradient_accumulation_steps=2 (effective batch 256 sequences, 262,144 tokens), 50,000 steps, bf16 mixed precision, and seed 42.

Optimizer: Muon with `match_rms_adamw`, momentum 0.95, Nesterov enabled, lr=5e-4, and weight decay 0.1. The cosine schedule uses 1,500 warmup steps and min_lr=5e-5. Gradient clipping is 1.0; checkpoints save every 5,000 steps; evaluation runs every 100 steps for 100 batches.

All quantized runs use RNE weight rounding, exclude `lm_head`, and enable quantization metrics. Checkpoints go to `checkpoints/int4_activation_limit/<config_name>/`; W&B run names are the config stems with underscores replaced by hyphens.

For each limit `L`, every dense MLP uses:

```yaml
activation_kwargs:
  act_limit:
    gate:
      max: L
    up:
      min: -L
      max: L
```

These bounds apply to the gate and up projection outputs before SwiGLU. The gate has no lower clamp.

## Run

From the repository root, inspect GPU usage and replace `<free_idx>` with a free device:

```bash
nvidia-smi
mkdir -p logs
CUDA_VISIBLE_DEVICES=<free_idx> nohup bash experiments/int4_activation_limit/run.sh > logs/int4_activation_limit.log 2>&1 &
```

The runner executes the bf16 control, the unbounded W4A16 control, then limits 31, 15, 7, and 3 sequentially on the selected device. Extra arguments are forwarded to `scripts/train.py`.

## Results

W&B project: `pretrain-int4-activation-limit`. Training has not been run for this experiment. Report the mean validation loss over the final 10 evaluations.

| Config | Activation limit | Mean final val loss | Δ vs bf16 | Δ vs unbounded W4A16 |
|---|---|---|---|---|
| qwen3_51m_bf16 | Unbounded | Pending | — | — |
| qwen3_51m_int4_w4a16_blockwise1d_16 | Unbounded | Pending | Pending | — |
| qwen3_51m_int4_w4a16_blockwise1d_16_act_limit31 | 31 | Pending | Pending | Pending |
| qwen3_51m_int4_w4a16_blockwise1d_16_act_limit15 | 15 | Pending | Pending | Pending |
| qwen3_51m_int4_w4a16_blockwise1d_16_act_limit7 | 7 | Pending | Pending | Pending |
| qwen3_51m_int4_w4a16_blockwise1d_16_act_limit3 | 3 | Pending | Pending | Pending |

## Notes

- Limits are absolute pre-activation magnitudes, independent of the int4 code range. Clipping affects MLP gate/up outputs; attention projections are not clipped.
- Compare against this experiment's controls. The sweep measures clipping's net effect under W4A16; attributing that effect specifically to quantization would require matching clipped bf16 controls, as explored in [activation_limit](../activation_limit/README.md).
- `w4a16` follows the existing int4 experiment naming convention. Int4 codes are stored in int8 containers, so these runs measure the accuracy effect of a 4-bit code range and do not benchmark packed-int4 throughput.
- The int4 runs share all settings except the activation bounds and their run/checkpoint identifiers.
