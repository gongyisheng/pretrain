# FP8 Output-Head Sensitivity

Measure output-head sensitivity to FP8 E4M3 weight, activation, and output-gradient quantization during pretraining. Compare an untied 77M BF16 baseline with head-only W8A16, W8A8, and W8A8G8 block-size sweeps.

## Hypothesis

E4M3 quantization error in the output head perturbs logits and their backward signal. Smaller scale groups may preserve training quality better. Sweep weights with BF16 activations and gradients, then activations with fixed `(32, 32)` weights, then output gradients with fixed `(32, 32)` weights and `(1, 32)` activations.

## Setup

| Config | Head weights | Activations | Output gradients |
|---|---|---|---|
| `qwen3_77m_bf16` | BF16 | BF16 | BF16 |
| `qwen3_77m_fp8_w8a16_blockwise1d_32` | FP8 E4M3 `(1, 32)` | BF16 | BF16 |
| `qwen3_77m_fp8_w8a16_blockwise2d_16` | FP8 E4M3 `(16, 16)` | BF16 | BF16 |
| `qwen3_77m_fp8_w8a16_blockwise2d_32` | FP8 E4M3 `(32, 32)` | BF16 | BF16 |
| `qwen3_77m_fp8_w8a16_blockwise2d_64` | FP8 E4M3 `(64, 64)` | BF16 | BF16 |
| `qwen3_77m_fp8_w8a16_blockwise2d_128` | FP8 E4M3 `(128, 128)` | BF16 | BF16 |
| `qwen3_77m_fp8_w8a8_act_blockwise1d_16` | FP8 E4M3 `(32, 32)` | FP8 E4M3 `(1, 16)` | BF16 |
| `qwen3_77m_fp8_w8a8_act_blockwise1d_32` | FP8 E4M3 `(32, 32)` | FP8 E4M3 `(1, 32)` | BF16 |
| `qwen3_77m_fp8_w8a8_act_blockwise1d_64` | FP8 E4M3 `(32, 32)` | FP8 E4M3 `(1, 64)` | BF16 |
| `qwen3_77m_fp8_w8a8_act_blockwise1d_128` | FP8 E4M3 `(32, 32)` | FP8 E4M3 `(1, 128)` | BF16 |
| `qwen3_77m_fp8_w8a8_act_blockwise2d_32` | FP8 E4M3 `(32, 32)` | FP8 E4M3 `(32, 32)` | BF16 |
| `qwen3_77m_fp8_w8a8g8_grad_blockwise1d_16` | FP8 E4M3 `(32, 32)` | FP8 E4M3 `(1, 32)` | FP8 E4M3 `(1, 16)` |
| `qwen3_77m_fp8_w8a8g8_grad_blockwise1d_32` | FP8 E4M3 `(32, 32)` | FP8 E4M3 `(1, 32)` | FP8 E4M3 `(1, 32)` |
| `qwen3_77m_fp8_w8a8g8_grad_blockwise1d_64` | FP8 E4M3 `(32, 32)` | FP8 E4M3 `(1, 32)` | FP8 E4M3 `(1, 64)` |
| `qwen3_77m_fp8_w8a8g8_grad_blockwise1d_128` | FP8 E4M3 `(32, 32)` | FP8 E4M3 `(1, 32)` | FP8 E4M3 `(1, 128)` |
| `qwen3_77m_fp8_w8a8g8_grad_blockwise2d_32` | FP8 E4M3 `(32, 32)` | FP8 E4M3 `(1, 32)` | FP8 E4M3 `(32, 32)` |

| Shared parameter | Value |
|---|---|
| Model | Qwen3-style dense Transformer; 76,686,848 parameters (77M); untied embeddings |
| Architecture | Width 512; 8 layers; GQA 8/4; QK norm; MLP width 1,536; RoPE theta 10,000 |
| Head | 50,304 padded vocabulary entries × 512; 25,755,648 weights |
| Quantization | Head-only FP8 E4M3; FP32 scales; RNE; no global scale; `include: [lm_head]`; `exclude: []` |
| Data | OpenWebText with `tokenizers/custom_bpe_50k`; validation split 0.01 |
| Training | Sequence length 1,024; batch 16; accumulation 16; 262,144 tokens per step; 50,000 steps / 13.1072B tokens |
| Optimizer | Muon; momentum 0.95; Nesterov; `match_rms_adamw`; weight decay 0.1; head routed to AdamW |
| Schedule | 5e-4 cosine to 5e-5; 1,500 warmup steps |
| Precision | BF16 mixed precision |
| Evaluation / checkpoints | 100 batches every 100 steps; checkpoint every 5,000 steps |
| Seed / logging | 42; W&B project `pretrain-fp8-lm-head` |

All 16 runs share the same untied architecture and training settings. Each FP8 run quantizes only `lm_head`; attention, MLP, and embeddings remain unquantized.

## Run

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=0  # Replace 0 with a free device from nvidia-smi.
mkdir -p logs
nohup bash experiments/fp8_lm_head/run.sh > logs/fp8_lm_head.log 2>&1 &
```

Prepare OpenWebText and `tokenizers/custom_bpe_50k` as described in the repository README. The script runs BF16, five W8A16 weight cases, five W8A8 activation cases, and five W8A8G8 gradient cases. Checkpoints are stored under `checkpoints/fp8_lm_head/<config>/`.

## Results

Compare mean validation loss and BPB over the final ten evaluations, steps 49,100–50,000, at equal training tokens. Compute each loss delta against BF16. Compare W8A8 with W8A16 `blockwise2d_32` to isolate activation quantization, then W8A8G8 with W8A8 `act_blockwise1d_32` to isolate output-gradient quantization.

| Config | Mean final-10 val loss | Mean final-10 val BPB | Delta loss vs BF16 | Status |
|---|---:|---:|---:|---|
| `bf16` | — | — | 0 | Pending |
| `fp8_w8a16_blockwise1d_32` | — | — | — | Pending |
| `fp8_w8a16_blockwise2d_16` | — | — | — | Pending |
| `fp8_w8a16_blockwise2d_32` | — | — | — | Pending |
| `fp8_w8a16_blockwise2d_64` | — | — | — | Pending |
| `fp8_w8a16_blockwise2d_128` | — | — | — | Pending |
| `fp8_w8a8_act_blockwise1d_16` | — | — | — | Pending |
| `fp8_w8a8_act_blockwise1d_32` | — | — | — | Pending |
| `fp8_w8a8_act_blockwise1d_64` | — | — | — | Pending |
| `fp8_w8a8_act_blockwise1d_128` | — | — | — | Pending |
| `fp8_w8a8_act_blockwise2d_32` | — | — | — | Pending |
| `fp8_w8a8g8_grad_blockwise1d_16` | — | — | — | Pending |
| `fp8_w8a8g8_grad_blockwise1d_32` | — | — | — | Pending |
| `fp8_w8a8g8_grad_blockwise1d_64` | — | — | — | Pending |
| `fp8_w8a8g8_grad_blockwise1d_128` | — | — | — | Pending |
| `fp8_w8a8g8_grad_blockwise2d_32` | — | — | — | Pending |

## Notes

- Untied embeddings allow `apply_quantization` to convert `lm_head`; a tied head is skipped. `exclude: []` overrides the default head exclusion.
- Weight quantization applies to forward and input-gradient GEMMs; activation quantization applies to forward and weight-gradient GEMMs; output-gradient quantization applies to both backward GEMMs. Master weights remain in their original storage dtype.
- A scaled FP8 GEMM requires quantized operands with matching contracted scale extents. W8A16 always dequantizes to the BF16 matmul path; W8A8 activation widths 16, 64, and 128 fall back in forward, and W8A8G8 gradient widths 16, 64, and 128 fall back in backward. Matching-width cases use the scaled FP8 path.
- `QuantizedLinear.eval()` bypasses quantization, so validation measures the effect of FP8 training rather than quantized inference accuracy.
