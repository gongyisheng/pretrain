# Muon Momentum Sweep

## Hypothesis

Muon momentum may improve validation loss and BPB relative to the baseline `0.95` when varied across `0.8`, `0.9`, `0.95`, and `0.99`, with AdamW companion β₁/β₂ fixed at `(0.9, 0.99)`.

## Setup

| Config | Muon momentum | Approx. parameters |
|---|---:|---:|
| `qwen3_51m_muon_momentum_0.8` | 0.8 | 50.93M |
| `qwen3_51m_muon_momentum_0.9` | 0.9 | 50.93M |
| `qwen3_51m_muon_momentum_0.95` | 0.95 | 50.93M |
| `qwen3_51m_muon_momentum_0.99` | 0.99 | 50.93M |

The parameter count is 50,931,264, computed with `TransformerLM.compute_parameters`. All runs use the dense Qwen3 51M model from `configs/qwen3_51m.yaml`, OpenWebText, the custom 50K BPE tokenizer, and a 1% validation split.

The hybrid `muonadam` optimizer uses Muon with the listed momentum, Nesterov enabled, `match_rms_adamw` learning-rate adjustment, and ε `1e-8`. Its AdamW companion uses fixed β₁/β₂ `(0.9, 0.99)` and ε `1e-8`; weight decay is fixed at `0.1`.

All other settings match the base configuration: sequence length 1024, effective batch size 256 (batch size 16 × accumulation 16), 50,000 steps, 13.1072B tokens, bf16, gradient clipping 1.0, and a cosine schedule from `5e-4` to `5e-5` with 1,500 warmup steps. Evaluation runs every 100 steps for 100 batches; checkpoints run every 5,000 steps.

## Run

The runner executes all four momenta sequentially and forwards CLI arguments to each run. Checkpoints go under `checkpoints/muon_momentum/<config stem>/`; W&B uses project `pretrain-muon-momentum` and run names `qwen3-51m-muon-momentum-<value>`.

Check GPU usage, then replace `<free-index>` with a free device index:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=<free-index> bash experiments/muon_momentum/run.sh
```

For a 10k-step screen that keeps the 50k-step cosine schedule:

```bash
CUDA_VISIBLE_DEVICES=<free-index> bash experiments/muon_momentum/run.sh --training.early_stop=10000
```

Run an individual configuration directly:

```bash
CUDA_VISIBLE_DEVICES=<free-index> uv run python scripts/train.py \
  --config experiments/muon_momentum/qwen3_51m_muon_momentum_0.95.yaml
```

## Results

| Muon momentum | Final validation loss | Final validation BPB | Status |
|---:|---:|---:|---|
| 0.8 | — | — | Pending |
| 0.9 | — | — | Pending |
| 0.95 | — | — | Pending |
| 0.99 | — | — | Pending |

## Notes

- Compare validation loss and BPB at equal token counts.
- A screening run stops at 10k optimizer steps while retaining the original 50k-step scheduler, so its learning-rate trajectory matches the prefix of a full run.
- Retrain selected finalists from scratch to 50k steps for the final comparison.
- The selected momentum is conditional on the fixed AdamW companion betas and other optimizer settings.
