# INT8 Module Sensitivity

Measure how attention, MLP, and output-head projections respond to INT8 W8A8G8 quantization in an untied 77M model. Quantize one module group per run, keeping other modules in BF16, and compare against a shared BF16 baseline.

## Hypothesis

Quantizing weights, activations, and output gradients introduces different errors in attention and MLP projections. The module group with the larger validation-loss increase is more sensitive under this blockwise recipe.

## Setup

| Config | INT8 module | `training.quantization.include` | Quantized linears |
|---|---|---|---:|
| `qwen3_77m_bf16` | None (baseline) | Quantization disabled | 0 |
| `qwen3_77m_int8_w8a8g8_attn` | Attention | `[blocks.*.attn.*]` | 32 (4 × 8) |
| `qwen3_77m_int8_w8a8g8_mlp` | MLP | `[blocks.*.mlp.*]` | 24 (3 × 8) |
| `qwen3_77m_int8_w8a8g8_lm_head` | Output head | `[lm_head]` | 1 |

| Shared parameter | Value |
|---|---|
| Model | Qwen3-style dense Transformer; 76,686,848 parameters (77M); untied embeddings |
| Architecture | Width 512; 8 layers; GQA 8/4; QK norm; MLP width 1,536; RoPE theta 10,000 |
| Quantization | INT8 weights, activations, and output gradients; weight blocks `(32, 32)`; activation and gradient blocks `(1, 32)`; FP32 scales; RNE |
| Data | OpenWebText with `tokenizers/custom_bpe_50k`; validation split 0.01 |
| Training | Sequence length 1,024; batch 16; accumulation 16; 262,144 tokens per step; 50,000 steps / 13.1072B tokens |
| Optimizer | Muon; momentum 0.95; Nesterov; `match_rms_adamw`; weight decay 0.1 |
| Schedule | 5e-4 cosine to 5e-5; 1,500 warmup steps |
| Precision | BF16 mixed precision; `lm_head` excluded in attention/MLP runs; `exclude: []` in the head run |
| Evaluation / checkpoints | 100 batches every 100 steps; checkpoint every 5,000 steps |
| Seed / logging | 42; W&B project `pretrain-int8-module-sensitivity` |

All four runs share the same untied architecture and training settings. The three INT8 runs differ only in their module selection, exclusions, and run identifiers.

## Run

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=0  # Replace 0 with a free device from nvidia-smi.
mkdir -p logs
nohup bash experiments/int8_module_sensitivity/run.sh > logs/int8_module_sensitivity.log 2>&1 &
```

Prepare OpenWebText and `tokenizers/custom_bpe_50k` as described in the repository README. The script runs the BF16 baseline, then attention, MLP, and head ablations. Each config writes to its own `checkpoints/int8_module_sensitivity/<config>/` directory.

## Results

Compare mean validation loss and BPB over the final ten evaluations, steps 49,100–50,000. Compute every delta against the untied 77M BF16 baseline. A larger loss increase indicates greater module-group sensitivity. Results are pending.

| INT8 module | Quantized linears | Mean final-10 val loss | Mean final-10 val BPB | Delta loss vs BF16 | Status |
|---|---:|---:|---:|---:|---|
| None (77M BF16) | 0 | — | — | 0 | Pending |
| Attention (77M) | 32 | — | — | — | Pending |
| MLP (77M) | 24 | — | — | — | Pending |
| Output head (77M) | 1 | — | — | — | Pending |

## Notes

- `apply_quantization` skips a head tied to the embedding table. All runs use untied embeddings so the head can be quantized while holding the architecture constant; untying adds 25,755,648 parameters (50,304 padded vocabulary entries × 512).
- Quantized modules use W8A8G8 for forward and backward matrix multiplications. Other modules remain in BF16; master weights are not stored as INT8.
- Validation runs in BF16 because `QuantizedLinear.eval()` bypasses quantization, so validation measures the effect of quantization during training.
- Interpret sensitivity at the module-group level: the groups contain different numbers and sizes of projections.
