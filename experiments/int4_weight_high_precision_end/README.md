# Int4 Weight High-Precision End

Test whether ending W4A16 weight quantization before training completes improves the final training outcome. The sweep compares a bf16 baseline with W4A16 runs that switch to bf16 for the final 500, 1,000, 2,500, 5,000, or 10,000 optimizer updates. Each cutoff has a standard arm and an arm that keeps blocks 0 and 7 in bf16 throughout.

## Hypothesis

The final optimizer updates refine weights at low learning rates. Finishing these updates in bf16 may recover accuracy lost to int4 weight quantization while retaining W4A16 for most of training.

## Setup

| Config | Weights / activations | W4A16 updates | Final bf16 updates | Skipped blocks | Weight scale | Global scale | Approx. params |
|---|---|---:|---:|---|---|---|---:|
| [qwen3_51m_bf16.yaml](qwen3_51m_bf16.yaml) | bf16 / bf16 | — | 50,000 | — | — | — | 50,931,200 |
| [qwen3_51m_int4_w4a16_end_49500.yaml](qwen3_51m_int4_w4a16_end_49500.yaml) | int4 / bf16 | 49,500 | 500 | — | blockwise 2D `(16, 16)`, fp8_e4m3 | enabled | 50,931,200 |
| [qwen3_51m_int4_w4a16_end_49000.yaml](qwen3_51m_int4_w4a16_end_49000.yaml) | int4 / bf16 | 49,000 | 1,000 | — | blockwise 2D `(16, 16)`, fp8_e4m3 | enabled | 50,931,200 |
| [qwen3_51m_int4_w4a16_end_47500.yaml](qwen3_51m_int4_w4a16_end_47500.yaml) | int4 / bf16 | 47,500 | 2,500 | — | blockwise 2D `(16, 16)`, fp8_e4m3 | enabled | 50,931,200 |
| [qwen3_51m_int4_w4a16_end_45000.yaml](qwen3_51m_int4_w4a16_end_45000.yaml) | int4 / bf16 | 45,000 | 5,000 | — | blockwise 2D `(16, 16)`, fp8_e4m3 | enabled | 50,931,200 |
| [qwen3_51m_int4_w4a16_end_40000.yaml](qwen3_51m_int4_w4a16_end_40000.yaml) | int4 / bf16 | 40,000 | 10,000 | — | blockwise 2D `(16, 16)`, fp8_e4m3 | enabled | 50,931,200 |
| [qwen3_51m_int4_w4a16_skip_0_7_end_49500.yaml](qwen3_51m_int4_w4a16_skip_0_7_end_49500.yaml) | int4 / bf16 | 49,500 | 500 | 0, 7 | blockwise 2D `(16, 16)`, fp8_e4m3 | enabled | 50,931,200 |
| [qwen3_51m_int4_w4a16_skip_0_7_end_49000.yaml](qwen3_51m_int4_w4a16_skip_0_7_end_49000.yaml) | int4 / bf16 | 49,000 | 1,000 | 0, 7 | blockwise 2D `(16, 16)`, fp8_e4m3 | enabled | 50,931,200 |
| [qwen3_51m_int4_w4a16_skip_0_7_end_47500.yaml](qwen3_51m_int4_w4a16_skip_0_7_end_47500.yaml) | int4 / bf16 | 47,500 | 2,500 | 0, 7 | blockwise 2D `(16, 16)`, fp8_e4m3 | enabled | 50,931,200 |
| [qwen3_51m_int4_w4a16_skip_0_7_end_45000.yaml](qwen3_51m_int4_w4a16_skip_0_7_end_45000.yaml) | int4 / bf16 | 45,000 | 5,000 | 0, 7 | blockwise 2D `(16, 16)`, fp8_e4m3 | enabled | 50,931,200 |
| [qwen3_51m_int4_w4a16_skip_0_7_end_40000.yaml](qwen3_51m_int4_w4a16_skip_0_7_end_40000.yaml) | int4 / bf16 | 40,000 | 10,000 | 0, 7 | blockwise 2D `(16, 16)`, fp8_e4m3 | enabled | 50,931,200 |

All runs use Qwen3-51M (`d_model=512`, 8 layers, GQA 8/4 heads, `intermediate_size=1536`), OpenWebText, sequence length 1024, batch size 16, gradient accumulation 16, effective batch 256, seed 42, bf16 mixed precision, MuonAdam, and 50,000 optimizer updates. W4A16 uses RNE, blockwise 16×16 fp8_e4m3 weight scales, and a global scale. Every W4A16 arm uses randomized Hadamard rotation with block size 16 on both weight axes for fwd and dgrad. It always excludes `lm_head`; the skip arms also exclude blocks 0 and 7.

## Run

```bash
nvidia-smi
mkdir -p logs
CUDA_VISIBLE_DEVICES=1 nohup bash experiments/int4_weight_high_precision_end/run.sh > logs/int4_weight_high_precision_end.log 2>&1 &
```

Replace `1` with a free GPU index. The launcher runs 11 configurations: the bf16 baseline, five standard W4A16 cutoffs, and five skip-0/7 cutoffs. Extra arguments are forwarded to each run.

## Results

| Model | Recipe | Skipped blocks | Final bf16 updates | Steps | Val Loss | Val BPB | Tokens/s |
|---|---|---|---:|---|---|---|---|
| 50,931,200 | bf16 | — | 50,000 | Not run | Not run | Not run | Not run |
| 50,931,200 | W4A16 | — | 500 | Not run | Not run | Not run | Not run |
| 50,931,200 | W4A16 | — | 1,000 | Not run | Not run | Not run | Not run |
| 50,931,200 | W4A16 | — | 2,500 | Not run | Not run | Not run | Not run |
| 50,931,200 | W4A16 | — | 5,000 | Not run | Not run | Not run | Not run |
| 50,931,200 | W4A16 | — | 10,000 | Not run | Not run | Not run | Not run |
| 50,931,200 | W4A16 | 0, 7 | 500 | Not run | Not run | Not run | Not run |
| 50,931,200 | W4A16 | 0, 7 | 1,000 | Not run | Not run | Not run | Not run |
| 50,931,200 | W4A16 | 0, 7 | 2,500 | Not run | Not run | Not run | Not run |
| 50,931,200 | W4A16 | 0, 7 | 5,000 | Not run | Not run | Not run | Not run |
| 50,931,200 | W4A16 | 0, 7 | 10,000 | Not run | Not run | Not run | Not run |

## Notes

- An `end_N` arm uses W4A16 for the first N optimizer updates, then bf16 for the remaining updates.
- In skip-0/7 arms, zero-based blocks 0 and 7 stay bf16 throughout; blocks 1–6 use W4A16 until the cutoff.
- Evaluation runs the bf16 model path. Rotation is bypassed after the cutoff with quantization.
- The 50,000-step cosine schedule and its 1,500-step warmup are unchanged when W4A16 is disabled.
- Checkpoints are isolated under `checkpoints/int4_weight_high_precision_end/`; W&B runs use project `pretrain-int4-weight-high-precision-end`.
