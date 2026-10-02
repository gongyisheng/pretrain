# FP8 Gradient Granularity

Measure how output-gradient scale granularity affects FP8 W8A8G8 pretraining. All quantized runs use FP8 E4M3 weights with 32×32 blocks and FP8 E4M3 activations with 1×32 blocks. The E4M3 gradient sweep changes only `training.quantization.scale.grad_out`; one 1×32 E5M2 run tests gradient format range.

## Hypothesis

Smaller output-gradient scale groups should reduce outlier-driven quantization error and improve training stability. E5M2 at 1×32 may trade precision for range and help if gradient outliers dominate.

## Setup

Eight runs: a BF16 baseline, a W8A8 control with BF16 `grad_out`, and six W8A8G8 configurations. Quantized runs use FP32 scales, RNE rounding, and exclude `lm_head`. No activation clipping or Hadamard rotation is enabled.

| Config (`.yaml`) | Weight | Activation | Output gradient |
|---|---|---|---|
| `qwen3_51m_bf16` | BF16 | BF16 | BF16 |
| `qwen3_51m_fp8_w8a8` | FP8 E4M3 (32, 32) | FP8 E4M3 (1, 32) | BF16 |
| `qwen3_51m_fp8_w8a8g8_grad_blockwise1d_16` | FP8 E4M3 (32, 32) | FP8 E4M3 (1, 32) | FP8 E4M3 (1, 16) |
| `qwen3_51m_fp8_w8a8g8_grad_blockwise1d_32` | FP8 E4M3 (32, 32) | FP8 E4M3 (1, 32) | FP8 E4M3 (1, 32) |
| `qwen3_51m_fp8_w8a8g8_grad_blockwise1d_64` | FP8 E4M3 (32, 32) | FP8 E4M3 (1, 32) | FP8 E4M3 (1, 64) |
| `qwen3_51m_fp8_w8a8g8_grad_blockwise1d_128` | FP8 E4M3 (32, 32) | FP8 E4M3 (1, 32) | FP8 E4M3 (1, 128) |
| `qwen3_51m_fp8_w8a8g8_grad_blockwise2d_32` | FP8 E4M3 (32, 32) | FP8 E4M3 (1, 32) | FP8 E4M3 (32, 32) |
| `qwen3_51m_fp8_w8a8g8_grad_blockwise1d_32_e5m2` | FP8 E4M3 (32, 32) | FP8 E4M3 (1, 32) | FP8 E5M2 (1, 32) |

| Shared parameter | Value |
|---|---|
| Model | Qwen3-style dense Transformer, 50,931,200 parameters (approximately 51M) |
| Dimensions | `d_model=512`, 8 layers, 8 Q / 4 KV heads, QK norm, SwiGLU intermediate size 1,536 |
| Data | OpenWebText, `tokenizers/custom_bpe_50k`, validation split 0.01 |
| Sequence / batch | 1,024 tokens, batch 16, accumulation 16; 262,144 tokens per optimizer step |
| Budget | 50,000 steps; 13.1072B training tokens per run |
| Optimizer | Muon, momentum 0.95, Nesterov, `match_rms_adamw`, weight decay 0.1 |
| Learning rate | 5e-4, cosine decay to 5e-5, 1,500 warmup steps |
| Precision / clipping | BF16 mixed precision, gradient norm clip 1.0 |
| Seed | 42 for initialization and data ordering |
| Evaluation | Every 100 steps, 100 batches, evaluation batch size 16 |
| Checkpoints / logging | Every 5,000 / 10 steps; quantization metrics enabled |

## Run

From the repository root, inspect GPU usage and select a free device. Prepare the tokenizer and OpenWebText data as described in the repository README.

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=0  # Replace 0 with a free GPU from the output above.
mkdir -p logs
nohup bash experiments/fp8_gradient_granularity/run.sh > logs/fp8_gradient_granularity.log 2>&1 &
```

The launcher runs all eight configurations sequentially on the selected device. W&B project: `pretrain-fp8-gradient-granularity`. Checkpoints: `checkpoints/fp8_gradient_granularity/<config>/`.

## Results

Primary metric: mean `val/loss` over the final ten scheduled evaluations, at steps 49,100–50,000. Compare runs at equal training tokens. Record validation BPB, per-module gradient SQNR and underflow, and any nonfinite loss or divergence step. Results are pending.

| Config suffix | Mean val loss | Δ vs BF16 | Δ vs W8A8 | Val BPB | Status |
|---|---|---|---|---|---|
| `bf16` | — | 0 | — | — | Pending |
| `fp8_w8a8` | — | — | 0 | — | Pending |
| `fp8_w8a8g8_grad_blockwise1d_16` | — | — | — | — | Pending |
| `fp8_w8a8g8_grad_blockwise1d_32` | — | — | — | — | Pending |
| `fp8_w8a8g8_grad_blockwise1d_64` | — | — | — | — | Pending |
| `fp8_w8a8g8_grad_blockwise1d_128` | — | — | — | — | Pending |
| `fp8_w8a8g8_grad_blockwise2d_32` | — | — | — | — | Pending |
| `fp8_w8a8g8_grad_blockwise1d_32_e5m2` | — | — | — | — | Pending |

Report `loss(1D, N) - loss(1D, 32)` for each 1D width and `loss(2D, 32) - loss(1D, 32)`; positive values favor 1D 32. Compare the E5M2 run with E4M3 1D 32 to measure the output-gradient format effect at fixed geometry.

## Notes

- Validation bypasses quantization in `QuantizedLinear.eval()`. Validation loss measures the effect of quantized training on the learned model, evaluated in BF16.
- Output gradients are the left operand in dgrad (`g @ W`) and wgrad (`g.T @ X`). In dgrad, 1D `(1, N)` groups N output channels for one token and 2D `(32, 32)` groups 32 tokens × 32 output channels. In wgrad, 1D groups N tokens for one output channel and 2D groups 32 output channels × 32 tokens.
- Width-16, width-64, and width-128 gradients have a block width that differs from the 32-wide weight and activation blocks. They use the BF16 backward fallback; width-32 1D and 2D configurations can use scaled FP8 kernels where available. Do not use this experiment for timing comparisons across granularity cells.
- Fixed weight and activation policies keep forward quantization identical across the sweep. The W8A8 control measures the added cost of output-gradient quantization.
- Seed 42 is a screening sweep. Repeat runs with additional seeds before drawing conclusions from small differences.
- Check each run reaches 50,000 steps: `scripts/train.py` catches training exceptions without returning a failing exit status, so launcher completion alone does not prove successful training.
