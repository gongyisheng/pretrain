# Int4 Weight Late Start

Test whether beginning W4A16 weight quantization after an initial bf16 training period improves the final training outcome. The sweep compares a bf16 baseline with W4A16 runs that start quantization after 0, 500, 1,000, 1,500, 2,000, 2,500, 3,000, 5,000, or 10,000 optimizer updates.

## Hypothesis

The earliest optimizer updates establish the model's initial representations while gradients and weights are changing most rapidly. Training those updates in bf16 may make the subsequent transition to int4 weights less disruptive, producing a better validation loss than quantizing from the first update. Starts at 500 and 1,000 occur during the fixed 1,500-step warmup, 1,500 coincides with its end, and later starts occur during cosine decay.

## Setup

| Config | Weights / activations | Int4 start | Weight scale | Global scale | Approx. params |
|---|---|---:|---|---|---:|
| [qwen3_51m_bf16.yaml](qwen3_51m_bf16.yaml) | bf16 / bf16 | — | — | — | 50,931,200 |
| [qwen3_51m_int4_w4a16_start_0.yaml](qwen3_51m_int4_w4a16_start_0.yaml) | int4 / bf16 | 0 | blockwise 2D `(16, 16)`, fp8_e4m3 | enabled | 50,931,200 |
| [qwen3_51m_int4_w4a16_start_500.yaml](qwen3_51m_int4_w4a16_start_500.yaml) | int4 / bf16 | 500 | blockwise 2D `(16, 16)`, fp8_e4m3 | enabled | 50,931,200 |
| [qwen3_51m_int4_w4a16_start_1000.yaml](qwen3_51m_int4_w4a16_start_1000.yaml) | int4 / bf16 | 1,000 | blockwise 2D `(16, 16)`, fp8_e4m3 | enabled | 50,931,200 |
| [qwen3_51m_int4_w4a16_start_1500.yaml](qwen3_51m_int4_w4a16_start_1500.yaml) | int4 / bf16 | 1,500 | blockwise 2D `(16, 16)`, fp8_e4m3 | enabled | 50,931,200 |
| [qwen3_51m_int4_w4a16_start_2000.yaml](qwen3_51m_int4_w4a16_start_2000.yaml) | int4 / bf16 | 2,000 | blockwise 2D `(16, 16)`, fp8_e4m3 | enabled | 50,931,200 |
| [qwen3_51m_int4_w4a16_start_2500.yaml](qwen3_51m_int4_w4a16_start_2500.yaml) | int4 / bf16 | 2,500 | blockwise 2D `(16, 16)`, fp8_e4m3 | enabled | 50,931,200 |
| [qwen3_51m_int4_w4a16_start_3000.yaml](qwen3_51m_int4_w4a16_start_3000.yaml) | int4 / bf16 | 3,000 | blockwise 2D `(16, 16)`, fp8_e4m3 | enabled | 50,931,200 |
| [qwen3_51m_int4_w4a16_start_5000.yaml](qwen3_51m_int4_w4a16_start_5000.yaml) | int4 / bf16 | 5,000 | blockwise 2D `(16, 16)`, fp8_e4m3 | enabled | 50,931,200 |
| [qwen3_51m_int4_w4a16_start_10000.yaml](qwen3_51m_int4_w4a16_start_10000.yaml) | int4 / bf16 | 10,000 | blockwise 2D `(16, 16)`, fp8_e4m3 | enabled | 50,931,200 |

All runs use Qwen3-51M (`d_model=512`, 8 layers, GQA 8/4 heads, `intermediate_size=1536`), OpenWebText, sequence length 1024, batch size 16, gradient accumulation 16, effective batch 256, seed 42, bf16 mixed precision, MuonAdam, and a fixed 50,000 optimizer updates. The cosine scheduler has a fixed 1,500-step warmup and does not restart when quantization begins. W4A16 uses RNE, blockwise 16×16 fp8_e4m3 weight scales, and a global scale. Every W4A16 arm uses randomized Hadamard rotation with block size 16 on both weight axes for fwd and dgrad; the rotation is enabled when quantization starts. It quantizes all selected linear layers except `lm_head`.

## Run

```bash
nvidia-smi
mkdir -p logs
CUDA_VISIBLE_DEVICES=1 nohup bash experiments/int4_weight_late_start/run.sh > logs/int4_weight_late_start.log 2>&1 &
```

Replace `1` with a free GPU index. Extra arguments are forwarded to each run.

## Results

| Model | Recipe | Int4 start | Steps | Val Loss | Val BPB | Tokens/s |
|---|---|---:|---:|---|---|---|
| 50,931,200 | bf16 | — | Not run | Not run | Not run | Not run |
| 50,931,200 | W4A16 | 0 | Not run | Not run | Not run | Not run |
| 50,931,200 | W4A16 | 500 | Not run | Not run | Not run | Not run |
| 50,931,200 | W4A16 | 1,000 | Not run | Not run | Not run | Not run |
| 50,931,200 | W4A16 | 1,500 | Not run | Not run | Not run | Not run |
| 50,931,200 | W4A16 | 2,000 | Not run | Not run | Not run | Not run |
| 50,931,200 | W4A16 | 2,500 | Not run | Not run | Not run | Not run |
| 50,931,200 | W4A16 | 3,000 | Not run | Not run | Not run | Not run |
| 50,931,200 | W4A16 | 5,000 | Not run | Not run | Not run | Not run |
| 50,931,200 | W4A16 | 10,000 | Not run | Not run | Not run | Not run |

## Notes

- A threshold of N means the first N optimizer updates use bf16 weights; W4A16 begins on update N+1. Threshold 0 therefore enables W4A16 on the first update.
- Quantization is used only during training. Evaluation bypasses quantization and runs the bf16 model path.
- Checkpoints are isolated under `checkpoints/int4_weight_late_start/`; W&B runs use project `pretrain-int4-weight-late-start`.
