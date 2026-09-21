# INT8 Output-Head Sensitivity

Test whether the output head is sensitive to weight-only INT8 quantization during pretraining. Compare an untied 77M BF16 baseline with head-only W8A16 using 1D and 2D weight blocks of width 32. Activations and output gradients remain BF16.

## Hypothesis

Output-head weight error directly perturbs the logits. Smaller scale groups in 1D `(1, 32)` blocks may preserve accuracy better than 2D `(32, 32)` blocks. Compare both with BF16 to measure the training cost of quantizing only the head's weights.

## Setup

| Config | Head weights | Activations / output gradients | Quantized linears |
|---|---|---|---:|
| `qwen3_77m_bf16` | BF16 | BF16 / BF16 | 0 |
| `qwen3_77m_int8_w8a16_blockwise1d_32` | INT8, `(1, 32)` | BF16 / BF16 | 1 (`lm_head`) |
| `qwen3_77m_int8_w8a16_blockwise2d_32` | INT8, `(32, 32)` | BF16 / BF16 | 1 (`lm_head`) |

| Shared parameter | Value |
|---|---|
| Model | Qwen3-style dense Transformer; 76,686,848 parameters (77M); untied embeddings |
| Architecture | Width 512; 8 layers; GQA 8/4; QK norm; MLP width 1,536; RoPE theta 10,000 |
| Head | 50,304 padded vocabulary entries × 512; 25,755,648 weights |
| Quantization | Weight-only INT8; FP32 scales; RNE; `include: [lm_head]`; `exclude: []` |
| Data | OpenWebText with `tokenizers/custom_bpe_50k`; validation split 0.01 |
| Training | Sequence length 1,024; batch 16; accumulation 16; 262,144 tokens per step; 50,000 steps / 13.1072B tokens |
| Optimizer | Muon; momentum 0.95; Nesterov; `match_rms_adamw`; weight decay 0.1; head routed to AdamW |
| Schedule | 5e-4 cosine to 5e-5; 1,500 warmup steps |
| Precision | BF16 mixed precision |
| Evaluation / checkpoints | 100 batches every 100 steps; checkpoint every 5,000 steps |
| Seed / logging | 42; W&B project `pretrain-int8-lm-head` |

All runs share the same untied architecture and training settings. The two INT8 configs differ only in weight block shape and run identifiers. Attention, MLP, and embeddings remain unquantized.

## Run

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=0  # Replace 0 with a free device from nvidia-smi.
mkdir -p logs
nohup bash experiments/int8_lm_head/run.sh > logs/int8_lm_head.log 2>&1 &
```

Prepare OpenWebText and `tokenizers/custom_bpe_50k` as described in the repository README. The script runs BF16, then the 1D and 2D ablations. Checkpoints are stored under `checkpoints/int8_lm_head/<config>/`.

## Results

Compare mean validation loss and BPB over the final ten evaluations, steps 49,100–50,000, at equal training tokens. Compute each loss delta against this experiment's BF16 baseline. Results are pending.

| Head weights | Mean final-10 val loss | Mean final-10 val BPB | Delta loss vs BF16 | Status |
|---|---:|---:|---:|---|
| BF16 | — | — | 0 | Pending |
| INT8 `(1, 32)` | — | — | — | Pending |
| INT8 `(32, 32)` | — | — | — | Pending |

## Notes

- Untied embeddings allow `apply_quantization` to convert `lm_head`; a head tied to the embedding table is skipped. `exclude: []` overrides the default head exclusion.
- Weight quantization applies to forward and input-gradient GEMMs. The weight-gradient GEMM uses BF16 activations and output gradients. Master weights remain in their original storage dtype.
- Weight-only INT8 uses quantize/dequantize emulation with BF16 matrix multiplication; this experiment measures accuracy rather than fused INT8 throughput.
- Validation bypasses quantization in `QuantizedLinear.eval()`, so these metrics measure the effect of quantization during training, not quantized inference accuracy.
