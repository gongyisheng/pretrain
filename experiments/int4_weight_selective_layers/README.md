# Int4 Weight Selective Layers

Test whether retaining selected transformer blocks in bf16 improves W4A16 pretraining quality. The sweep holds the Qwen3-51M recipe fixed and compares quantizing every block with retaining the final block, the final two blocks, the first and final blocks, or the first and final two blocks in bf16.

## Hypothesis

Final transformer blocks may be more sensitive to weight quantization than earlier blocks. Keeping them in bf16 may recover validation quality; `skip_0_7` tests whether protecting the first block helps more than protecting the penultimate block, while `skip_0_6_7` measures the incremental effect of also protecting the first block when the final two are already bf16. This adapts the final-block sensitivity observed in the [NVFP4 paper](https://arxiv.org/html/2509.25149v2#A5.SS2) to INT4 weight-only training.

## Setup

| Config | INT4 blocks | BF16 transformer blocks | Weight scale | Approx. parameters |
|---|---|---|---|---:|
| [qwen3_51m_bf16.yaml](qwen3_51m_bf16.yaml) | — | 0–7 | — | 50,931,200 |
| [qwen3_51m_int4_w4a16_no_skip.yaml](qwen3_51m_int4_w4a16_no_skip.yaml) | 0–7 | — | 16×16, fp8_e4m3, global | 50,931,200 |
| [qwen3_51m_int4_w4a16_skip_7.yaml](qwen3_51m_int4_w4a16_skip_7.yaml) | 0–6 | 7 | 16×16, fp8_e4m3, global | 50,931,200 |
| [qwen3_51m_int4_w4a16_skip_6_7.yaml](qwen3_51m_int4_w4a16_skip_6_7.yaml) | 0–5 | 6–7 | 16×16, fp8_e4m3, global | 50,931,200 |
| [qwen3_51m_int4_w4a16_skip_0_7.yaml](qwen3_51m_int4_w4a16_skip_0_7.yaml) | 1–6 | 0, 7 | 16×16, fp8_e4m3, global | 50,931,200 |
| [qwen3_51m_int4_w4a16_skip_0_6_7.yaml](qwen3_51m_int4_w4a16_skip_0_6_7.yaml) | 1–5 | 0, 6–7 | 16×16, fp8_e4m3, global | 50,931,200 |

All runs use Qwen3-51M: width 512, eight GQA blocks with 8 query and 4 KV heads, QK normalization, and a SwiGLU MLP with intermediate size 1536. They train on OpenWebText with `tokenizers/custom_bpe_50k`, sequence length 1024, batch size 16, gradient accumulation 16, seed 42, 50,000 optimizer steps, MuonAdam with `match_rms_adamw`, learning rate 5e-4, 1,500 warmup steps, and bf16 mixed precision. Checkpoints run every 5,000 steps; evaluation runs every 100 steps for 100 batches.

W4A16 quantizes weights only: activations and output gradients remain bf16, and all three use RNE. Quantized weights use 16×16 blockwise fp8_e4m3 scales with a global scale. A randomized Hadamard transform of size 16 rotates both weight axes for forward and dgrad. Every W4A16 configuration excludes `lm_head`.

## Run

Check GPU usage, choose a free device, then run:

```bash
nvidia-smi
mkdir -p logs
CUDA_VISIBLE_DEVICES=1 nohup bash experiments/int4_weight_selective_layers/run.sh > logs/int4_weight_selective_layers.log 2>&1 &
```

Replace `1` with a free GPU index. Extra arguments are forwarded to each training command.

## Results

| Config | Steps | Validation loss | Validation BPB | Tokens/s |
|---|---:|---|---|---|
| bf16 | Not run | Not run | Not run | Not run |
| W4A16, no skip | Not run | Not run | Not run | Not run |
| W4A16, skip 7 | Not run | Not run | Not run | Not run |
| W4A16, skip 6–7 | Not run | Not run | Not run | Not run |
| W4A16, skip 0 and 7 | Not run | Not run | Not run | Not run |
| W4A16, skip 0, 6–7 | Not run | Not run | Not run | Not run |

## Notes

- Block numbers are zero-based. A skipped block stays bf16 for all of its linear projections; `lm_head` stays bf16 in every W4A16 arm.
- The only difference among the W4A16 configurations is the block-exclusion list.
- Quantization is active from the first training update. Evaluation bypasses quantization and uses the bf16 model path.
- Checkpoints are isolated under `checkpoints/int4_weight_selective_layers/`; W&B runs use project `pretrain-int4-weight-selective-layers`.
- This is INT4 weight-only training, not NVIDIA NVFP4: activations and output gradients are bf16, and the arithmetic/rounding recipe differs.
