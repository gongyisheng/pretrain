# INT8 Output-Head Sensitivity

Measure output-head sensitivity to INT8 weight, activation, and output-gradient quantization during pretraining. Compare an untied 77M BF16 baseline with head-only W8A16, W8A8, and W8A8G8 block-size sweeps.

## Hypothesis

Output-head quantization error perturbs the logits and their backward signal. Smaller scale groups may preserve accuracy better. Sweep weight blocks with BF16 activations and gradients, then activation blocks with fixed `(32, 32)` weights, then gradient blocks with fixed `(32, 32)` weights and `(1, 32)` activations. Compare each sweep with its corresponding higher-precision baseline.

## Setup

| Config | Head weights | Activations | Output gradients |
|---|---|---|---|
| `qwen3_77m_bf16` | BF16 | BF16 | BF16 |
| `qwen3_77m_int8_w8a16_blockwise1d_32` | INT8 `(1, 32)` | BF16 | BF16 |
| `qwen3_77m_int8_w8a16_blockwise2d_16` | INT8 `(16, 16)` | BF16 | BF16 |
| `qwen3_77m_int8_w8a16_blockwise2d_32` | INT8 `(32, 32)` | BF16 | BF16 |
| `qwen3_77m_int8_w8a16_blockwise2d_64` | INT8 `(64, 64)` | BF16 | BF16 |
| `qwen3_77m_int8_w8a16_blockwise2d_128` | INT8 `(128, 128)` | BF16 | BF16 |
| `qwen3_77m_int8_w8a8_act_blockwise1d_16` | INT8 `(32, 32)` | INT8 `(1, 16)` | BF16 |
| `qwen3_77m_int8_w8a8_act_blockwise1d_32` | INT8 `(32, 32)` | INT8 `(1, 32)` | BF16 |
| `qwen3_77m_int8_w8a8_act_blockwise1d_64` | INT8 `(32, 32)` | INT8 `(1, 64)` | BF16 |
| `qwen3_77m_int8_w8a8_act_blockwise1d_128` | INT8 `(32, 32)` | INT8 `(1, 128)` | BF16 |
| `qwen3_77m_int8_w8a8_act_blockwise2d_32` | INT8 `(32, 32)` | INT8 `(32, 32)` | BF16 |
| `qwen3_77m_int8_w8a8g8_grad_blockwise1d_16` | INT8 `(32, 32)` | INT8 `(1, 32)` | INT8 `(1, 16)` |
| `qwen3_77m_int8_w8a8g8_grad_blockwise1d_32` | INT8 `(32, 32)` | INT8 `(1, 32)` | INT8 `(1, 32)` |
| `qwen3_77m_int8_w8a8g8_grad_blockwise1d_64` | INT8 `(32, 32)` | INT8 `(1, 32)` | INT8 `(1, 64)` |
| `qwen3_77m_int8_w8a8g8_grad_blockwise1d_128` | INT8 `(32, 32)` | INT8 `(1, 32)` | INT8 `(1, 128)` |
| `qwen3_77m_int8_w8a8g8_grad_blockwise2d_32` | INT8 `(32, 32)` | INT8 `(1, 32)` | INT8 `(32, 32)` |

| Shared parameter | Value |
|---|---|
| Model | Qwen3-style dense Transformer; 76,686,848 parameters (77M); untied embeddings |
| Architecture | Width 512; 8 layers; GQA 8/4; QK norm; MLP width 1,536; RoPE theta 10,000 |
| Head | 50,304 padded vocabulary entries × 512; 25,755,648 weights |
| Quantization | Head-only INT8; FP32 scales; RNE; no global scale; `include: [lm_head]`; `exclude: []` |
| Data | OpenWebText with `tokenizers/custom_bpe_50k`; validation split 0.01 |
| Training | Sequence length 1,024; batch 16; accumulation 16; 262,144 tokens per step; 50,000 steps / 13.1072B tokens |
| Optimizer | Muon; momentum 0.95; Nesterov; `match_rms_adamw`; weight decay 0.1; head routed to AdamW |
| Schedule | 5e-4 cosine to 5e-5; 1,500 warmup steps |
| Precision | BF16 mixed precision |
| Evaluation / checkpoints | 100 batches every 100 steps; checkpoint every 5,000 steps |
| Seed / logging | 42; W&B project `pretrain-int8-lm-head` |

All 16 runs share the same untied architecture and training settings. Each sweep varies only the indicated tensor's quantization and run identifiers. Every INT8 run quantizes one linear, `lm_head`; attention, MLP, and embeddings remain unquantized.

## Run

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=0  # Replace 0 with a free device from nvidia-smi.
mkdir -p logs
nohup bash experiments/int8_lm_head/run.sh > logs/int8_lm_head.log 2>&1 &
```

Prepare OpenWebText and `tokenizers/custom_bpe_50k` as described in the repository README. The script runs all 16 configs sequentially: BF16, five W8A16 weight cases, five W8A8 activation cases, and five W8A8G8 gradient cases. Checkpoints are stored under `checkpoints/int8_lm_head/<config>/`.

## Results

Compare mean validation loss and BPB over the final ten evaluations, steps 49,100–50,000, at equal training tokens. Compute each loss delta against BF16. Also compare W8A8 with W8A16 `blockwise2d_32` to isolate activation quantization, and W8A8G8 with W8A8 `act_blockwise1d_32` to isolate gradient quantization. Results are pending. Config names below omit the shared `qwen3_77m_` prefix.

| Config | Mean final-10 val loss | Mean final-10 val BPB | Delta loss vs BF16 | Status |
|---|---:|---:|---:|---|
| `bf16` | — | — | 0 | Pending |
| `int8_w8a16_blockwise1d_32` | — | — | — | Pending |
| `int8_w8a16_blockwise2d_16` | — | — | — | Pending |
| `int8_w8a16_blockwise2d_32` | — | — | — | Pending |
| `int8_w8a16_blockwise2d_64` | — | — | — | Pending |
| `int8_w8a16_blockwise2d_128` | — | — | — | Pending |
| `int8_w8a8_act_blockwise1d_16` | — | — | — | Pending |
| `int8_w8a8_act_blockwise1d_32` | — | — | — | Pending |
| `int8_w8a8_act_blockwise1d_64` | — | — | — | Pending |
| `int8_w8a8_act_blockwise1d_128` | — | — | — | Pending |
| `int8_w8a8_act_blockwise2d_32` | — | — | — | Pending |
| `int8_w8a8g8_grad_blockwise1d_16` | — | — | — | Pending |
| `int8_w8a8g8_grad_blockwise1d_32` | — | — | — | Pending |
| `int8_w8a8g8_grad_blockwise1d_64` | — | — | — | Pending |
| `int8_w8a8g8_grad_blockwise1d_128` | — | — | — | Pending |
| `int8_w8a8g8_grad_blockwise2d_32` | — | — | — | Pending |

## Notes

- Untied embeddings allow `apply_quantization` to convert `lm_head`; a head tied to the embedding table is skipped. `exclude: []` overrides the default head exclusion.
- Weight quantization applies to forward and input-gradient GEMMs; activation quantization applies to forward and weight-gradient GEMMs; output-gradient quantization applies to both backward GEMMs. Master weights remain in their original storage dtype.
- Gradient `(1, N)` blocks group N output channels per token in the input-gradient GEMM (`g @ W`), and N tokens per output channel in the weight-gradient GEMM (`g.T @ X`). Gradient `(32, 32)` blocks group both dimensions.
- Weight-only INT8 uses quantize/dequantize emulation with BF16 matrix multiplication. Mismatched operand block widths also use this fallback: W8A8 activation widths 16, 64, and 128 in forward, and W8A8G8 gradient widths 16, 64, and 128 in backward. Matching-width cases use INT8 kernels where available; these sweeps measure accuracy rather than equal kernel throughput.
- Validation bypasses quantization in `QuantizedLinear.eval()`, so these metrics measure the effect of quantization during training, not quantized inference accuracy.
