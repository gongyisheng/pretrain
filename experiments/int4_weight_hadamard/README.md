# Int4 Weight Hadamard

Test whether randomized Hadamard transforms improve int4 weight-only training. The weight scales use 16×16 tiles; the transform independently mixes vectors along each GEMM contraction dimension. Five runs compare bf16, unrotated W4A16, and W4A16 with Hadamard block sizes 4, 16, and 128. Quantization is limited to MLP linears (`include: ['*mlp.*']`); `lm_head` is excluded.

## Hypothesis

The transform mixes a contraction block before it is quantized, reducing the influence of individual weight outliers within each 16×16 scale tile. Block size 4 mixes within a tile, size 16 matches a tile along the contraction dimension, and size 128 spans eight tiles along that dimension. At the same int4 code range and scale storage, the Hadamard runs should reduce weight quantization error and improve validation loss relative to the unrotated W4A16 control.

## Setup

| Config | Weights / activations | Weight scale | Global scale | Hadamard | Approx. params |
|---|---|---|---|---|---:|
| [qwen3_51m_bf16.yaml](qwen3_51m_bf16.yaml) | bf16 / bf16 | — | — | — | ~50.93M |
| [qwen3_51m_int4_w4a16.yaml](qwen3_51m_int4_w4a16.yaml) | int4 / bf16 | blockwise 2D `(16, 16)`, fp8_e4m3 | fp32, enabled | disabled | ~50.93M |
| [qwen3_51m_int4_w4a16_hadamard_4.yaml](qwen3_51m_int4_w4a16_hadamard_4.yaml) | int4 / bf16 | blockwise 2D `(16, 16)`, fp8_e4m3 | fp32, enabled | size 4 | ~50.93M |
| [qwen3_51m_int4_w4a16_hadamard_16.yaml](qwen3_51m_int4_w4a16_hadamard_16.yaml) | int4 / bf16 | blockwise 2D `(16, 16)`, fp8_e4m3 | fp32, enabled | size 16 | ~50.93M |
| [qwen3_51m_int4_w4a16_hadamard_128.yaml](qwen3_51m_int4_w4a16_hadamard_128.yaml) | int4 / bf16 | blockwise 2D `(16, 16)`, fp8_e4m3 | fp32, enabled | size 128 | ~50.93M |

All runs use Qwen3-51M (`d_model=512`, 8 layers, GQA 8/4 heads, `intermediate_size=1536`), OpenWebText, sequence length 1024, batch size 16, gradient accumulation 16, effective batch 256, 50K steps, seed 42, bf16 mixed precision, Muon, and cosine decay with 1500 warmup steps. Checkpoints are written under `checkpoints/int4_weight_hadamard/`; W&B project: `pretrain-int4-weight-hadamard`.

## Run

```bash
nvidia-smi
mkdir -p logs
CUDA_VISIBLE_DEVICES=1 nohup bash experiments/int4_weight_hadamard/run.sh > logs/int4_weight_hadamard.log 2>&1 &
```

Replace `1` with a free GPU index. Extra arguments are forwarded to each run.

## Results

| Model | Recipe | Steps | Val Loss | Val BPB | Tokens/s |
|---|---|---|---|---|---|
| 50.93M | bf16 | Not run | Not run | Not run | Not run |
| 50.93M | int4 W4A16, 16×16 | Not run | Not run | Not run | Not run |
| 50.93M | int4 W4A16, 16×16, Hadamard-4 | Not run | Not run | Not run | Not run |
| 50.93M | int4 W4A16, 16×16, Hadamard-16 | Not run | Not run | Not run | Not run |
| 50.93M | int4 W4A16, 16×16, Hadamard-128 | Not run | Not run | Not run | Not run |

## Notes

- `random_sign: true` draws one sign vector whose length matches each block size (4, 16, or 128) from seed 42. It is stored with the rotation and shared by every selected MLP module for the complete run; it is not sampled per operation.
- fp8_e4m3 block scales are paired with one fp32 global factor per quantized tensor. This scale dtype is required because fp32 block scales disable the global factor. The global factor is recomputed from the current weight tensor when it is quantized.
- With W4A16, forward and dgrad quantize a weight operand. Wgrad has no quantized operand, so the rotated config declares only `gemms: [fwd, dgrad]`.
- Evaluation uses the high-precision bf16 model path: quantization, including int4 weight quantization and rotation, is bypassed while the model is in evaluation mode.
- Validation: all five configs load with 50,931,200 parameters and identical settings outside quantization and run names. GPU forward/backward smoke checks passed at batch size 1 and sequence length 32, with finite losses and gradients; all four W4A16 runs convert 24 MLP projections. Full training and full-batch memory use remain untested.
