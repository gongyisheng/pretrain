# Muon vs MuonC

Compare Muon with MuonC on Qwen3 51M / OpenWebText. Both use AdamW for parameters outside Muon's hidden matrices.

## Hypothesis

MuonC corrects decoupled weight decay during the cosine schedule for Muon-managed hidden matrices. Compared with Muon, this may improve validation loss without changing the model, data, AdamW branch, or learning-rate schedule.

## Setup

All runs use the same dense Qwen3 51M model: width 512, 8 layers, GQA with 8 query and 4 KV heads, QK normalization, and MLP intermediate size 1536. It has 50,931,200 parameters (50.93M). Data is OpenWebText with `tokenizers/custom_bpe_50k` and a 1% validation split.

| Config | Muon optimizer | Adam companion | Regime | Approx. parameters |
|---|---|---|---|---:|
| `qwen3_51m_muon_steps_50k` | Muon | AdamW | 50k | 50.93M |
| `qwen3_51m_muonc_steps_50k` | MuonC | AdamW | 50k | 50.93M |
| `qwen3_51m_muon_steps_200k` | Muon | AdamW | 200k | 50.93M |
| `qwen3_51m_muonc_steps_200k` | MuonC | AdamW | 200k | 50.93M |
| `qwen3_51m_muon_steps_800k` | Muon | AdamW | 800k | 50.93M |
| `qwen3_51m_muonc_steps_800k` | MuonC | AdamW | 800k | 50.93M |
| `qwen3_51m_muon_steps_200k_bs_256` | Muon | AdamW | 200k, batch 256 | 50.93M |
| `qwen3_51m_muonc_steps_200k_bs_256` | MuonC | AdamW | 200k, batch 256 | 50.93M |

Muon uses momentum 0.95, Nesterov, `match_rms_adamw`, and epsilon `1e-8`. The AdamW branch uses betas `(0.9, 0.95)` and epsilon `1e-8`. All runs use weight decay 0.1, bf16 mixed precision, gradient clipping 1.0, and cosine decay.

| Regime | Batch size | Gradient accumulation | Tokens/step | Steps | Total tokens | Peak/min LR | Warmup | Eval/checkpoint/log every |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50k | 16 | 16 | 262,144 | 50,000 | 13.1072B | `5e-4` / `5e-5` | 1,500 | 100 / 5,000 / 10 steps |
| 200k | 16 | 4 | 65,536 | 200,000 | 13.1072B | `2.5e-4` / `2.5e-5` | 6,000 | 100 / 20,000 / 50 steps |
| 800k | 16 | 1 | 16,384 | 800,000 | 13.1072B | `1.25e-4` / `1.25e-5` | 24,000 | 100 / 80,000 / 100 steps |
| 200k_bs_256 | 16 | 16 | 262,144 | 200,000 | 52.4288B | `5e-4` / `5e-5` | 6,000 | 100 / 20,000 / 50 steps |

The 200k regime intentionally uses accumulation 4 and checkpointing every 20,000 steps. The 800k regime intentionally uses accumulation 1 and checkpointing every 80,000 steps. The 200k_bs_256 regime keeps accumulation 16 for an optimizer batch of 256 and runs four times as many tokens as the other regimes. Compare Muon and MuonC only within the same regime.

## Run

Prepare the dataset and tokenizer using the repository setup instructions. Check GPU usage, then select a free GPU:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 bash experiments/muonc/run.sh
```

Replace `0` with a free GPU index. The runner executes Muon then MuonC for each regime and forwards extra arguments to every run. Each run writes to its own `checkpoints/muonc/` directory and logs to the `pretrain-muonc` W&B project.

## Results

| Regime | Optimizer | Final validation loss | Final validation BPB | Status |
|---|---|---:|---:|---|
| 50k | Muon | — | — | Not run |
| 50k | MuonC | — | — | Not run |
| 200k | Muon | — | — | Not run |
| 200k | MuonC | — | — | Not run |
| 800k | Muon | — | — | Not run |
| 800k | MuonC | — | — | Not run |
| 200k_bs_256 | Muon | — | — | Not run |
| 200k_bs_256 | MuonC | — | — | Not run |

## Notes

- The Adam companion is fixed to AdamW in every run.
- Only MuonC-managed hidden matrices use corrected decay, multiplying weight decay by `lr / max_lr` before the update.
