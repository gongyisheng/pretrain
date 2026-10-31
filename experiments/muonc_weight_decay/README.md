# MuonC Weight-Decay Sweep

## Hypothesis

MuonC's best weight-decay coefficient may differ from `0.1` for Qwen3 51M at 200k steps with attention sinks and an effective batch size of 256.

## Setup

| Config | Weight decay | Approx. parameters |
|---|---:|---:|
| `qwen3_51m_muonc_wd_0` | 0 | 50.93M |
| `qwen3_51m_muonc_wd_0.01` | 0.01 | 50.93M |
| `qwen3_51m_muonc_wd_0.02` | 0.02 | 50.93M |
| `qwen3_51m_muonc_wd_0.05` | 0.05 | 50.93M |
| `qwen3_51m_muonc_wd_0.1` | 0.1 | 50.93M |
| `qwen3_51m_muonc_wd_0.2` | 0.2 | 50.93M |
| `qwen3_51m_muonc_wd_0.5` | 0.5 | 50.93M |
| `qwen3_51m_muonc_wd_1` | 1 | 50.93M |

Model, data, training, scheduler, logging cadence, and seed match the [AdamC weight-decay sweep](../adamc_weight_decay/), with separate output identifiers. The model uses dense Qwen3 51M, GQA (8 query heads, 4 KV heads), QK normalization, and explicit attention sinks (`attn_sink: true`). Data is OpenWebText with the custom 50K BPE tokenizer and a 1% validation split; seed is 42.

The `muonadam` optimizer combines MuonC for hidden matrices with AdamW for remaining parameters. MuonC uses momentum `0.95`, Nesterov updates, `match_rms_adamw` learning-rate adjustment, and ε `1e-8`. AdamW uses β₁/β₂ `(0.9, 0.99)` and ε `1e-8`; β₂ belongs to the AdamW companion, not MuonC.

All configurations use sequence length 1024, batch size 16, gradient accumulation 16, effective batch size 256, 262,144 tokens per update, 200,000 updates, and 52.4288B tokens total. Training uses CUDA, bf16, whole-model compilation, gradient clipping 1.0, and cosine scheduling from `2.5e-4` to `2.5e-5` with 6,000 warmup steps. Evaluation runs every 100 steps for 100 batches, checkpoints every 20,000 steps, and logging every 50 steps.

## Run

The launcher runs all eight coefficients sequentially and forwards CLI arguments to each run. Checkpoints go under `checkpoints/muonc_weight_decay/`; W&B uses project `pretrain-muonc-weight-decay` and run names `qwen3-51m-muonc-wd-<value>`.

Check GPU usage and choose a free device; these examples use device 0:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 bash experiments/muonc_weight_decay/run.sh
```

For a 10k-step screen retaining the 200k-step cosine schedule:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 bash experiments/muonc_weight_decay/run.sh --training.early_stop=10000
```

Run an individual configuration from scratch:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
  --config experiments/muonc_weight_decay/qwen3_51m_muonc_wd_0.1.yaml
```

## Results

| Weight decay | Final validation loss | Final validation BPB | Status |
|---:|---:|---:|---|
| 0 | — | — | Pending |
| 0.01 | — | — | Pending |
| 0.02 | — | — | Pending |
| 0.05 | — | — | Pending |
| 0.1 | — | — | Pending |
| 0.2 | — | — | Pending |
| 0.5 | — | — | Pending |
| 1 | — | — | Pending |

## Notes

- The shared weight-decay coefficient applies to both decayed optimizer branches. Only MuonC hidden matrices receive corrected decay scaled by `lr / max_lr`; the AdamW companion uses ordinary AdamW decay. No-decay parameter groups remain exempt.
- Attention sinks, accumulation 16, and AdamW β₂ `0.99` intentionally match the AdamC sweep; the [standard 200k MuonC experiment](../muonc/qwen3_51m_muonc_steps_200k.yaml) uses accumulation 4 and AdamW β₂ `0.95`.
- Checkpoints every 20,000 steps intentionally match the 200k-step regime instead of the general experiment default of 5,000.
- Compare validation loss and BPB at equal token counts. A 10k-step screen follows the prefix of the full 200k-step schedule; train selected finalists from scratch to 200k steps.
