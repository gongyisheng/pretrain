# BF16 vs NVIDIA NVFP4

## Hypothesis

NVIDIA's NVFP4 training recipe can keep validation loss close to BF16 on Qwen3-51M. Compare loss at equal optimizer steps and tokens, and measure throughput separately.

## Setup

| Config | Training GEMMs | Weight blocks | Activation / gradient blocks | Optimizer | Approx. parameters |
|---|---|---|---|---|---|
| [qwen3_51m_bf16.yaml](qwen3_51m_bf16.yaml) | BF16 | — | — | AdamW | 50.93M |
| [qwen3_51m_nvfp4.yaml](qwen3_51m_nvfp4.yaml) | NVFP4 in blocks 0–6 | 16×16 | 1×16 | AdamW | 50.93M |

Both runs use 8 layers, width 512, GQA with 8 query / 4 KV heads, QK normalization, and an MLP intermediate size of 1536. Data is OpenWebText with `tokenizers/custom_bpe_50k` and the same 1% validation split.

Shared settings: sequence length 1024, batch size 16, accumulation 16 (262,144 tokens per optimizer step), 50,000 steps (13.11B tokens), seed 42, BF16 mixed precision, and whole-model compilation. AdamW uses LR 5e-4, weight decay 0.1, betas (0.9, 0.95), and epsilon 1e-8. The cosine schedule warms up for 1,500 steps and ends at 5e-5. Evaluate every 100 steps for 100 batches; checkpoint every 5,000 steps.

The NVFP4 config follows the numerical ingredients in [NVIDIA's Transformer Engine recipe](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html): E2M1 operands, E4M3 block scales with an FP32 tensor scale, 2D weight scaling, stochastic rounding for output gradients, and a shared random Hadamard transform of size 16 applied only to the two weight-gradient GEMM operands. Weights and activations use round-to-nearest-even.

The final transformer block and `lm_head` stay in BF16. Keeping one of eight blocks in higher precision is this experiment's small-model choice, inspired by the selective high-precision layers in [NVIDIA's pretraining paper](https://arxiv.org/abs/2509.25149); it is not a reproduction of that paper's architecture or layer allocation. Embeddings, normalization, attention softmax, nonlinearities, and optimizer state follow the existing BF16/FP32 training path.

## Run

Prepare the data and tokenizer using the repository [setup instructions](../../README.md). Check GPU usage and select a free Blackwell GPU:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=1 bash experiments/nvfp4/run.sh
```

Replace `1` with the free GPU's index. The runner executes BF16 followed by NVFP4. Extra training arguments are forwarded to both runs, for example `--no-wandb --training.early_stop=1000`. Checkpoints go to separate `checkpoints/nvfp4/` directories; W&B runs share project `pretrain-nvfp4`.

## Results

| Precision | Steps | Validation loss | Validation BPB | Tokens/s |
|---|---|---|---|---|
| BF16 | Not run | — | — | — |
| NVFP4 | Not run | — | — | — |

## Notes

- Compare `val/loss` and `val/bpb` at equal token counts. Evaluation uses the learned high-precision weights without operand quantization in both runs.
- This uses the repository's NVFP4 implementation, not Transformer Engine. Hadamard intermediates use FP32 and stochastic rounding uses software RNG, so numerical and speed parity with NVIDIA's implementation is not assumed.
- `dtype: {recipe: nvfp4}` alone supplies the format and 1D scaling. This experiment explicitly adds `scale.block_shape.weight: [16, 16]`, rounding, rotation, and layer exclusions.
- Use `perf/tokens_per_sec` after compilation warmup, with identical hardware and logging settings. This small model may spend more time quantizing than it saves in GEMMs. Full training results remain unmeasured.
- Validation: both configs load with identical non-quantization settings and 50,931,200 parameters; the NVFP4 scope contains 49 linear projections. Compiled GPU smoke runs passed on an RTX 5060 Ti using batch size 1, accumulation 1, sequence length 128, and disabled SVD/quantization metrics. These runs do not validate full-batch memory use or convergence.
