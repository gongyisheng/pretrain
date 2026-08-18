# Int8 Weight-Only Scale-Granularity Sweep

Sweep quantization scale granularity at Qwen3-51M, comparing int8 weights with a bf16 baseline. Activations and gradients remain bf16; `lm_head` is excluded from quantization.

## Hypothesis

Smaller scale blocks reduce weight quantization error and improve validation loss, at the cost of more fp32 scale storage.

## Setup

| Config | Granularity | `block_shape` | Effective bits/weight |
|---|---|---|---|
| qwen3_51m_bf16 | bf16 | — | 16 |
| qwen3_51m_int8_tensorwise | tensorwise | — | 8.00 |
| qwen3_51m_int8_rowwise | rowwise | — | 8.06 |
| qwen3_51m_int8_blockwise1d_128 | blockwise1D | (1, 128) | 8.25 |
| qwen3_51m_int8_blockwise1d_64 | blockwise1D | (1, 64) | 8.50 |
| qwen3_51m_int8_blockwise1d_32 | blockwise1D | (1, 32) | 9.00 |
| qwen3_51m_int8_blockwise1d_16 | blockwise1D | (1, 16) | 10.00 |
| qwen3_51m_int8_blockwise2d_128 | blockwise2D | (128, 128) | 8.002 |
| qwen3_51m_int8_blockwise2d_64 | blockwise2D | (64, 64) | 8.008 |
| qwen3_51m_int8_blockwise2d_32 | blockwise2D | (32, 32) | 8.031 |
| qwen3_51m_int8_blockwise2d_16 | blockwise2D | (16, 16) | 8.125 |

All runs: ~51M params, seq_len=1024, batch=16, grad_accum=16, 50K steps, Muon (`match_rms_adamw`, momentum=0.95, nesterov), lr=5e-4, cosine schedule with 1500 warmup steps, min_lr=5e-5, OpenWebText, bf16 mixed precision, seed 42, and `eval_steps: 100`.

## Run

```bash
nohup bash experiments/int8_granularity/run.sh > logs/int8_granularity.log 2>&1 &
```

## Results

W&B project: `pretrain-int8-granularity`.

| Model | Granularity | `block_shape` | Effective bits/weight | Val Loss | Δ vs bf16 | Val BPB | Weight rel. err |
|---|---|---|---|---|---|---|---|
| 51M | bf16 | — | 16 | | 0 | | — |
| 51M | tensorwise | — | 8.00 | | | | |
| 51M | rowwise | — | 8.06 | | | | |
| 51M | blockwise-1D | (1, 128) | 8.25 | | | | |
| 51M | blockwise-1D | (1, 64) | 8.50 | | | | |
| 51M | blockwise-1D | (1, 32) | 9.00 | | | | |
| 51M | blockwise-1D | (1, 16) | 10.00 | | | | |
| 51M | blockwise-2D | (128, 128) | 8.002 | | | | |
| 51M | blockwise-2D | (64, 64) | 8.008 | | | | |
| 51M | blockwise-2D | (32, 32) | 8.031 | | | | |
| 51M | blockwise-2D | (16, 16) | 8.125 | | | | |

## Notes

- Compare each int8 run with this experiment's bf16 baseline using the mean validation loss over the final 10 evaluations.
- `weight rel. err` should fall as blocks shrink; compare it with validation loss to isolate quantization error.
