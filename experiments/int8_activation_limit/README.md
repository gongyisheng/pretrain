# Int8 Activation Limit

Sweep pre-SwiGLU activation limits for Qwen3-51M with int8 quantization. The bf16 run is the baseline; each W8A16, W8A8, and W8A8G8 arm includes limits 15, 7, 3, and an unbounded control.

## Hypothesis

Clipping large MLP gate and up-projection values may make int8 activation quantization more stable. Tight limits may instead remove useful signal. The unbounded control isolates the effect of the limit in each quantization arm.

## Setup

All 13 runs use Qwen3-51M (50,931,200 parameters), sequence length 1024, OpenWebText, Muon, and bf16 mixed precision. They train for 50,000 steps with batch_size=16 and gradient_accumulation_steps=16 (effective batch 256), checkpoint every 5,000 steps, and evaluate every 100 steps for 100 batches.

| Config | Weight | Activation | Grad out | Limit | Params |
|---|---|---|---|---|---|
| [qwen3_51m_bf16.yaml](qwen3_51m_bf16.yaml) | bf16 | bf16 | bf16 | — | ~51M |
| [qwen3_51m_int8_w8a16_act_limit15.yaml](qwen3_51m_int8_w8a16_act_limit15.yaml) | int8 blockwise (32, 32) | bf16 | bf16 | 15 | ~51M |
| [qwen3_51m_int8_w8a16_act_limit7.yaml](qwen3_51m_int8_w8a16_act_limit7.yaml) | int8 blockwise (32, 32) | bf16 | bf16 | 7 | ~51M |
| [qwen3_51m_int8_w8a16_act_limit3.yaml](qwen3_51m_int8_w8a16_act_limit3.yaml) | int8 blockwise (32, 32) | bf16 | bf16 | 3 | ~51M |
| [qwen3_51m_int8_w8a16_act_limitnone.yaml](qwen3_51m_int8_w8a16_act_limitnone.yaml) | int8 blockwise (32, 32) | bf16 | bf16 | unbounded | ~51M |
| [qwen3_51m_int8_w8a8_act_limit15.yaml](qwen3_51m_int8_w8a8_act_limit15.yaml) | int8 blockwise (32, 32) | int8 blockwise (1, 32) | bf16 | 15 | ~51M |
| [qwen3_51m_int8_w8a8_act_limit7.yaml](qwen3_51m_int8_w8a8_act_limit7.yaml) | int8 blockwise (32, 32) | int8 blockwise (1, 32) | bf16 | 7 | ~51M |
| [qwen3_51m_int8_w8a8_act_limit3.yaml](qwen3_51m_int8_w8a8_act_limit3.yaml) | int8 blockwise (32, 32) | int8 blockwise (1, 32) | bf16 | 3 | ~51M |
| [qwen3_51m_int8_w8a8_act_limitnone.yaml](qwen3_51m_int8_w8a8_act_limitnone.yaml) | int8 blockwise (32, 32) | int8 blockwise (1, 32) | bf16 | unbounded | ~51M |
| [qwen3_51m_int8_w8a8g8_act_limit15.yaml](qwen3_51m_int8_w8a8g8_act_limit15.yaml) | int8 blockwise (32, 32) | int8 blockwise (1, 32) | int8 blockwise (1, 32) | 15 | ~51M |
| [qwen3_51m_int8_w8a8g8_act_limit7.yaml](qwen3_51m_int8_w8a8g8_act_limit7.yaml) | int8 blockwise (32, 32) | int8 blockwise (1, 32) | int8 blockwise (1, 32) | 7 | ~51M |
| [qwen3_51m_int8_w8a8g8_act_limit3.yaml](qwen3_51m_int8_w8a8g8_act_limit3.yaml) | int8 blockwise (32, 32) | int8 blockwise (1, 32) | int8 blockwise (1, 32) | 3 | ~51M |
| [qwen3_51m_int8_w8a8g8_act_limitnone.yaml](qwen3_51m_int8_w8a8g8_act_limitnone.yaml) | int8 blockwise (32, 32) | int8 blockwise (1, 32) | int8 blockwise (1, 32) | unbounded | ~51M |

W8A16 and W8A8 use bf16 output gradients. W8A8G8 uses int8 output gradients with blockwise `(1, 32)` scaling. All quantized runs use fp32 scales, RNE rounding, and exclude `lm_head`.

For a finite limit `L`, each dense MLP sets:

```yaml
activation_kwargs:
  act_limit:
    gate:
      max: L
    up:
      min: -L
      max: L
```

## Run

```bash
nvidia-smi
mkdir -p logs
export CUDA_VISIBLE_DEVICES=0  # Replace 0 with a free GPU from nvidia-smi.
nohup bash experiments/int8_activation_limit/run.sh > logs/int8_activation_limit.log 2>&1 &
```

## Results

W&B project: `pretrain-int8-activation-limit`. Training has not been run.

| Arm | Limit | Final validation loss |
|---|---:|---|
| bf16 | — | Pending |
| W8A16 | 15 | Pending |
| W8A16 | 7 | Pending |
| W8A16 | 3 | Pending |
| W8A16 | unbounded | Pending |
| W8A8 | 15 | Pending |
| W8A8 | 7 | Pending |
| W8A8 | 3 | Pending |
| W8A8 | unbounded | Pending |
| W8A8G8 | 15 | Pending |
| W8A8G8 | 7 | Pending |
| W8A8G8 | 3 | Pending |
| W8A8G8 | unbounded | Pending |

## Notes

- Limits are applied to MLP gate and up values before SwiGLU. The gate is capped only at `max: L`; the up path is clamped to `[-L, L]`.
- Evaluation bypasses quantization, so evaluation loss measures the reconstructed bf16 model rather than quantized evaluation kernels.
- The sweep excludes `lm_head`; it does not test activation limits in the language-model head.
