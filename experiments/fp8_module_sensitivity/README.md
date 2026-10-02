# FP8 Module Sensitivity

Measure how attention, MLP, and output-head projections respond to FP8 E4M3 quantization in an untied 77M model. Quantize one module group per run under BF16 mixed precision, leave other modules unquantized, and progress from W8A16 to W8A8 to W8A8G8.

## Hypothesis

Weight, activation, and output-gradient quantization introduce distinct errors in attention and MLP projections. Comparing each recipe against its preceding recipe identifies the cost of adding activation quantization and then output-gradient quantization for each module group.

## Setup

| Config | FP8 module | Recipe | `training.quantization.include` | Quantized linears |
|---|---|---|---|---:|
| `qwen3_77m_bf16` | None | BF16 | Quantization disabled | 0 |
| `qwen3_77m_fp8_w8a16_attn` | Attention | W8A16 | `[blocks.*.attn.*]` | 32 |
| `qwen3_77m_fp8_w8a16_mlp` | MLP | W8A16 | `[blocks.*.mlp.*]` | 24 |
| `qwen3_77m_fp8_w8a16_lm_head` | Output head | W8A16 | `[lm_head]` | 1 |
| `qwen3_77m_fp8_w8a8_attn` | Attention | W8A8 | `[blocks.*.attn.*]` | 32 |
| `qwen3_77m_fp8_w8a8_mlp` | MLP | W8A8 | `[blocks.*.mlp.*]` | 24 |
| `qwen3_77m_fp8_w8a8_lm_head` | Output head | W8A8 | `[lm_head]` | 1 |
| `qwen3_77m_fp8_w8a8g8_attn` | Attention | W8A8G8 | `[blocks.*.attn.*]` | 32 |
| `qwen3_77m_fp8_w8a8g8_mlp` | MLP | W8A8G8 | `[blocks.*.mlp.*]` | 24 |
| `qwen3_77m_fp8_w8a8g8_lm_head` | Output head | W8A8G8 | `[lm_head]` | 1 |

`attn` uses `[blocks.*.attn.*]`, `mlp` uses `[blocks.*.mlp.*]`, and `lm_head` uses `[lm_head]`. Attention and MLP configurations exclude `lm_head`; head configurations use `exclude: []`.

| Shared parameter | Value |
|---|---|
| Model | Qwen3-style dense Transformer; 76,686,848 parameters (77M); untied embeddings |
| Architecture | Width 512; 8 layers; GQA 8/4; QK norm; MLP width 1,536; RoPE theta 10,000 |
| Quantization | FP8 E4M3; weight blocks `(32, 32)`; activation and output-gradient blocks `(1, 32)`; FP32 scales; RNE; no global scale |
| Data | OpenWebText with `tokenizers/custom_bpe_50k`; validation split 0.01 |
| Training | Sequence length 1,024; batch 16; accumulation 16; 262,144 tokens per step; 50,000 steps / 13.1072B tokens |
| Optimizer | Muon; momentum 0.95; Nesterov; `match_rms_adamw`; weight decay 0.1 |
| Schedule | 5e-4 cosine to 5e-5; 1,500 warmup steps |
| Precision | BF16 mixed precision |
| Evaluation / checkpoints | 100 batches every 100 steps; checkpoint every 5,000 steps |
| Seed / logging | 42; W&B project `pretrain-fp8-module-sensitivity` |

The ten runs differ only in recipe, module selection, exclusions, checkpoint path, and run identifier.

## Run

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=0  # Replace 0 with a free device from nvidia-smi.
mkdir -p logs
nohup bash experiments/fp8_module_sensitivity/run.sh > logs/fp8_module_sensitivity.log 2>&1 &
```

Prepare OpenWebText and `tokenizers/custom_bpe_50k` as described in the repository README. The script runs BF16, then each module group for W8A16, W8A8, and W8A8G8. Each config writes to `checkpoints/fp8_module_sensitivity/<config>/`.

## Results

Compare mean validation loss and BPB over the final ten evaluations, steps 49,100–50,000. First compare every run with BF16, then compare W8A8 with W8A16 and W8A8G8 with W8A8 within each module group.

| Recipe | Module | Quantized linears | Mean final-10 val loss | Mean final-10 val BPB | Delta loss vs BF16 | Status |
|---|---|---:|---:|---:|---:|---|
| BF16 | None | 0 | — | — | 0 | Pending |
| W8A16 | Attention | 32 | — | — | — | Pending |
| W8A16 | MLP | 24 | — | — | — | Pending |
| W8A16 | Output head | 1 | — | — | — | Pending |
| W8A8 | Attention | 32 | — | — | — | Pending |
| W8A8 | MLP | 24 | — | — | — | Pending |
| W8A8 | Output head | 1 | — | — | — | Pending |
| W8A8G8 | Attention | 32 | — | — | — | Pending |
| W8A8G8 | MLP | 24 | — | — | — | Pending |
| W8A8G8 | Output head | 1 | — | — | — | Pending |

## Notes

- `apply_quantization` skips a head tied to the embedding table. All runs use untied embeddings so the head can be quantized while holding architecture constant; untying adds 25,755,648 parameters (50,304 padded vocabulary entries × 512).
- W8A16 quantizes weights only, W8A8 adds activations, and W8A8G8 adds output gradients. Quantized values are FP8 E4M3; master weights remain in their original storage dtype.
- `QuantizedLinear.eval()` bypasses quantization, so validation measures the effect of FP8 training.
- Interpret sensitivity at the module-group level because groups contain different numbers and sizes of projections.
