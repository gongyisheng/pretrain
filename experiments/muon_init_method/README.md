# Muon Initialization Sweep

## Hypothesis

Initialization scale and fan scaling affect early training stability and validation loss for Qwen3 51M with hybrid Muon. Use fixed standard deviation 0.02 as the control and compare each case with its AdamW counterpart.

## Setup

| Config stem | Method | `init_std` | Approx. parameters |
|---|---|---:|---:|
| `qwen3_51m_muon_fixed_std_0.001` | Fixed | 0.001 | 50.93M |
| `qwen3_51m_muon_fixed_std_0.002` | Fixed | 0.002 | 50.93M |
| `qwen3_51m_muon_fixed_std_0.005` | Fixed | 0.005 | 50.93M |
| `qwen3_51m_muon_fixed_std_0.01` | Fixed | 0.01 | 50.93M |
| `qwen3_51m_muon_fixed_std_0.02` | Fixed | 0.02 | 50.93M |
| `qwen3_51m_muon_fixed_std_0.05` | Fixed | 0.05 | 50.93M |
| `qwen3_51m_muon_lecun` | LeCun | 0.02 | 50.93M |
| `qwen3_51m_muon_xavier` | Xavier | 0.02 | 50.93M |
| `qwen3_51m_muon_kaiming` | Kaiming | 0.02 | 50.93M |

All cases use normal initialization. Fixed applies `init_std` to attention and MLP projections, embeddings, and the LM head. LeCun, Xavier, and Kaiming use fan-based standard deviations for the projections and 0.02 for embeddings and the LM head.

The hybrid optimizer uses Muon momentum 0.99 with Nesterov and `match_rms_adamw` learning-rate adjustment. The AdamW companion uses betas (0.9, 0.99). Both use weight decay 0.1 on decayed groups and epsilon 1e-8.

Runs match the AdamW sweep on OpenWebText with the custom 50K BPE tokenizer and a 1% validation split. Each uses seed 42, sequence length 1024, batch size 16 with 16 accumulation steps, bf16, and gradient clipping at 1.0. The 50,000-step cosine schedule uses peak learning rate 1e-2, minimum 1e-5, and 1,500 warmup steps. Evaluation runs every 100 steps for 100 batches. Checkpoints are saved every 5,000 steps.

## Run

Use `nvidia-smi` to identify a free GPU and set `gpu_index` to its index. From the repository root, run the full sweep or an individual config.

```bash
CUDA_VISIBLE_DEVICES="$gpu_index" bash experiments/muon_init_method/run.sh

CUDA_VISIBLE_DEVICES="$gpu_index" uv run python scripts/train.py \
  --config experiments/muon_init_method/qwen3_51m_muon_fixed_std_0.02.yaml
```

The launcher runs each config sequentially and forwards CLI overrides to every run. Checkpoints go under `checkpoints/muon_init_method/<config-stem>/`. The W&B project is `pretrain-muon-init-method`, and each run uses its config stem as its name.

## Results

| Method | `init_std` | Final validation loss | Status |
|---|---:|---:|---|
| Fixed | 0.001 | — | Pending |
| Fixed | 0.002 | — | Pending |
| Fixed | 0.005 | — | Pending |
| Fixed | 0.01 | — | Pending |
| Fixed (control) | 0.02 | — | Pending |
| Fixed | 0.05 | — | Pending |
| LeCun | 0.02 | — | Pending |
| Xavier | 0.02 | — | Pending |
| Kaiming | 0.02 | — | Pending |

## Notes

Use training loss and gradient norms to assess early stability. Compare validation loss at matched optimizer steps.
