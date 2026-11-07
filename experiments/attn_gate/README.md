# Attention gating

## Hypothesis

Headwise or elementwise sigmoid gating of attention outputs may improve pretraining loss. Hold the MLP width at 1536 for both gated models. Compare headwise against the 51M ungated baseline, and elementwise against the parameter-matched ungated control at 53M to assess gating effects at similar capacity. Compare attention sinks enabled and disabled for each gated model and ungated control.

## Setup

| Config | Attention gate | Attention sink | MLP width | Exact parameters | Approx. parameters |
|---|---|---|---:|---:|---:|
| `qwen3_51m_ungated.yaml` | `null` | Off | 1536 | 50,931,200 | 50.93M |
| `qwen3_51m_headwise.yaml` | `headwise` | Off | 1536 | 50,963,968 | 50.96M |
| `qwen3_53m_ungated.yaml` | `null` | Off | 1712 | 53,093,888 | 53.09M |
| `qwen3_53m_elementwise.yaml` | `elementwise` | Off | 1536 | 53,028,352 | 53.03M |
| `qwen3_51m_ungated_sink_on.yaml` | `null` | On | 1536 | 50,931,264 | 50.93M |
| `qwen3_51m_headwise_sink_on.yaml` | `headwise` | On | 1536 | 50,964,032 | 50.96M |
| `qwen3_53m_ungated_sink_on.yaml` | `null` | On | 1712 | 53,093,952 | 53.09M |
| `qwen3_53m_elementwise_sink_on.yaml` | `elementwise` | On | 1536 | 53,028,416 | 53.03M |

All configs share 8 layers, d_model=512, 8 query / 4 KV heads, QK normalization, RMSNorm, and RoPE. The sweep varies the attention gate, attention sink setting, and MLP width as listed above, with separate output identifiers for every run. Parameter counts are verified through `build_model` and `TransformerLM.compute_parameters`.

Shared training settings: OpenWebText with `tokenizers/custom_bpe_50k`, sequence length 1024, batch size 16, gradient accumulation 16, bf16 mixed precision, 50,000 steps, and seed 42. MuonAdam uses lr=5e-4, weight decay=0.1, AdamW betas=(0.9, 0.99), and Muon momentum=0.95. The cosine schedule uses 1,500 warmup steps and min lr=5e-5. Checkpoints are saved every 5,000 steps; evaluation runs every 100 steps for 100 batches.

## Run

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=<free_idx> bash experiments/attn_gate/run.sh
```

The runner executes 51M ungated, 51M headwise, 53M ungated, then 53M elementwise with sinks off, followed by the same order with sinks on. It forwards trainer arguments to every run:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=<free_idx> bash experiments/attn_gate/run.sh --no-wandb --training.early_stop=100
```

## Results

| Config | Final validation loss | Throughput (tokens/s) | Peak GPU memory | Notes |
|---|---:|---:|---:|---|
| `qwen3_51m_ungated` | Pending | Pending | Pending | Sinks off; ungated baseline |
| `qwen3_51m_headwise` | Pending | Pending | Pending | Sinks off; MLP width 1536 |
| `qwen3_53m_ungated` | Pending | Pending | Pending | Sinks off; parameter-matched ungated control |
| `qwen3_53m_elementwise` | Pending | Pending | Pending | Sinks off; MLP width 1536 |
| `qwen3_51m_ungated_sink_on` | Pending | Pending | Pending | Sinks on; ungated baseline |
| `qwen3_51m_headwise_sink_on` | Pending | Pending | Pending | Sinks on; MLP width 1536 |
| `qwen3_53m_ungated_sink_on` | Pending | Pending | Pending | Sinks on; parameter-matched ungated control |
| `qwen3_53m_elementwise_sink_on` | Pending | Pending | Pending | Sinks on; MLP width 1536 |

## Notes

Attention sinks add `8 * 8 = 64` learned parameters. Compare each gated model with its ungated control under the same sink setting.

Elementwise gating adds `8 * 512 * 512 = 2,097,152` parameters; headwise gating adds `8 * 512 * 8 = 32,768`. Both gated models retain MLP width 1536. Widening each SwiGLU MLP in the 53M ungated control from 1536 to 1712 adds `8 * 3 * 512 * (1712 - 1536) = 2,162,688` parameters. The 53M ungated model has 65,536 more parameters than elementwise (+0.1236%). Their capacity is approximately matched; their MLP widths differ, so equal parameter counts would not imply equal throughput.

The code imposes no BF16 or Muon width-alignment requirement. [cuBLAS guidance](https://docs.nvidia.com/cuda/cublas/index.html#tensor-core-usage) recommends 16-byte alignment for optimal Tensor Core use, corresponding to 8 BF16 elements. Larger multiples such as 16 or 64 are performance choices, rather than universal hardware requirements; [NVIDIA's execution-efficiency guidance](https://docs.nvidia.com/deeplearning/performance/dl-performance-getting-started/index.html#choosing-parameters-to-maximize-execution-efficiency) discusses larger powers of two for efficient tiling.

An isolated `DenseMLPBlock` microbenchmark on an RTX 5060 Ti used batch size 16, sequence length 1024, d_model=512, BF16 autocast, CUDA events, and interleaved rounds. These are compiled forward-plus-backward timings without an optimizer or full model:

| MLP width | Compiled time (ms) |
|---:|---:|
| 1704 | 7.096 |
| 1707 | 7.207 |
| 1712 | 6.673 |
| 1728 | 6.672 |

Width 1712 is divisible by 16 and performed essentially the same as the multiple-of-64 width 1728 while matching elementwise capacity more closely. This local measurement motivates the widened ungated control's width only; both gated models use 1536. It does not establish a universal fastest width. Microbenchmarks and short trainer checks are separate from the pending experiment results.

W&B project: `pretrain-attn-gate`; run names are the config stems with underscores replaced by hyphens. Checkpoints are isolated under `checkpoints/attn_gate/<stem>/`. Full training results are single-seed and pending.
