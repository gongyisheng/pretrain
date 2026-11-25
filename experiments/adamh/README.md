# AdamH Learning-Rate Sweep

## Hypothesis

AdamH keeps hidden weight matrices at their initial Frobenius norms and applies
updates relative to those norms. A peak LR in this sweep may give lower
validation loss than the fixed AdamW and AdamC references at the same step.

## Setup

All runs use 50,931,264-parameter Qwen3 with width 512, eight layers, GQA with
8 query and 4 KV heads, QK normalization, attention sinks, SwiGLU width 1536,
RMSNorm, RoPE, and tied embeddings. Training uses OpenWebText with custom 50K
BPE, a 1% validation split, sequence length 1024, and seed 42.

| Config | Optimizer | Peak LR | Weight decay | Initialization | Approx. parameters |
|---|---|---:|---:|---|---:|
| `qwen3_51m_adamw` | AdamW | 1e-2 | 0.1 | LeCun normal | 50.93M |
| `qwen3_51m_adamc` | AdamC | 1e-2 | 0.2 | LeCun normal | 50.93M |
| `qwen3_51m_adamh_lr1e-3` | AdamH | 1e-3 | 0.1 | LeCun normal | 50.93M |
| `qwen3_51m_adamh_lr2e-3` | AdamH | 2e-3 | 0.1 | LeCun normal | 50.93M |
| `qwen3_51m_adamh_lr5e-3` | AdamH | 5e-3 | 0.1 | LeCun normal | 50.93M |
| `qwen3_51m_adamh_lr1e-2` | AdamH | 1e-2 | 0.1 | LeCun normal | 50.93M |
| `qwen3_51m_adamh_lr2e-2` | AdamH | 2e-2 | 0.1 | LeCun normal | 50.93M |
| `qwen3_51m_adamh_lr5e-2` | AdamH | 5e-2 | 0.1 | LeCun normal | 50.93M |

All optimizers use betas `(0.9, 0.99)`, epsilon `1e-8`, and no
parameter-specific LR multipliers. The cosine schedule warms up for
1,500 steps and decays to `1e-5`. Runs use CUDA, bf16, whole-model compilation,
gradient clipping at 1.0, and 50,000 optimizer steps at batch size 16 with
16 accumulation steps. Checkpoints save every 5,000 steps, evaluation runs
every 100 steps for 100 batches, and metrics log every 10 steps.

## Run

From the repository root, check GPU usage and set `gpu` to a free device index.

```bash
nvidia-smi
gpu=0
CUDA_VISIBLE_DEVICES="$gpu" bash experiments/adamh/run.sh
```

The launcher runs AdamW, AdamC, and the six AdamH rates in ascending order.
It forwards CLI overrides to each run. Checkpoints go to
`checkpoints/adamh/<config>/`, and W&B runs appear in `pretrain-adamh`.

## Results

| Config | Final validation loss | Final validation BPB | Status |
|---|---:|---:|---|
| `qwen3_51m_adamw` | — | — | Pending |
| `qwen3_51m_adamc` | — | — | Pending |
| `qwen3_51m_adamh_lr1e-3` | — | — | Pending |
| `qwen3_51m_adamh_lr2e-3` | — | — | Pending |
| `qwen3_51m_adamh_lr5e-3` | — | — | Pending |
| `qwen3_51m_adamh_lr1e-2` | — | — | Pending |
| `qwen3_51m_adamh_lr2e-2` | — | — | Pending |
| `qwen3_51m_adamh_lr5e-2` | — | — | Pending |

## Notes

- Compare validation loss and BPB at equal optimizer steps. The sweep uses one seed.
- All runs use LeCun normal for hidden linears and fixed normal with standard
  deviation `0.02` for tied embeddings.
- AdamH constrains hidden matrices to their initial norms. The `0.1` decay
  applies to fallback weights eligible for weight decay, including the tied
  embedding.
