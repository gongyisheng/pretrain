# MuonH Learning-Rate Sweep

## Hypothesis

MuonH keeps selected hidden weight matrices at their initial Frobenius norms
and scales updates by those norms. A peak LR in this sweep may yield lower
validation loss than the fixed MuonW and MuonC references at the same step.

## Setup

All runs use 50,931,264-parameter Qwen3 with width 512, eight layers, GQA with
8 query and 4 KV heads, QK normalization, attention sinks, SwiGLU width 1536,
RMSNorm, RoPE, and tied embeddings. Training uses OpenWebText with custom 50K
BPE, a 1% validation split, sequence length 1024, and seed 42.

| Config | Muon variant | Peak LR | Muon decay | AdamW decay | Initialization | Approx. parameters |
|---|---|---:|---:|---:|---|---:|
| `qwen3_51m_muonw` | MuonW | 1e-2 | 0.1 | 0.1 | LeCun normal | 50.93M |
| `qwen3_51m_muonc` | MuonC | 1e-2 | 0.2 | 0.2 | LeCun normal | 50.93M |
| `qwen3_51m_muonh_lr1e-3` | MuonH | 1e-3 | 0 | 0.1 | LeCun normal | 50.93M |
| `qwen3_51m_muonh_lr2e-3` | MuonH | 2e-3 | 0 | 0.1 | LeCun normal | 50.93M |
| `qwen3_51m_muonh_lr5e-3` | MuonH | 5e-3 | 0 | 0.1 | LeCun normal | 50.93M |
| `qwen3_51m_muonh_lr1e-2` | MuonH | 1e-2 | 0 | 0.1 | LeCun normal | 50.93M |
| `qwen3_51m_muonh_lr2e-2` | MuonH | 2e-2 | 0 | 0.1 | LeCun normal | 50.93M |
| `qwen3_51m_muonh_lr5e-2` | MuonH | 5e-2 | 0 | 0.1 | LeCun normal | 50.93M |

All runs use `muonadam`, with Muon on selected hidden matrices and AdamW on
the remaining parameters. Muon uses momentum `0.99` and Nesterov updates.
AdamW uses betas `(0.9, 0.99)`. Both use epsilon `1e-8`.

The cosine schedule warms up for 1,500 steps and decays to `1e-5`. Runs use
CUDA, bf16, whole-model compilation, gradient clipping at 1.0, and 50,000
optimizer steps at batch size 16 with 16 accumulation steps. Checkpoints save
every 5,000 steps, evaluation runs every 100 steps for 100 batches, and metrics
log every 10 steps.

## Run

From the repository root, check GPU usage and set `gpu` to a free device index.

```bash
nvidia-smi
gpu=0
mkdir -p logs
CUDA_VISIBLE_DEVICES="$gpu" nohup bash experiments/muonh/run.sh > logs/muonh_experiment.log 2>&1 &
```

The launcher runs MuonW, MuonC, and the six MuonH rates in ascending order.
It forwards CLI overrides to each run. Checkpoints go to
`checkpoints/muonh/<config>/`, and W&B runs appear in `pretrain-muonh`.

## Results

| Config | Final validation loss | Final validation BPB | Status |
|---|---:|---:|---|
| `qwen3_51m_muonw` | — | — | Pending |
| `qwen3_51m_muonc` | — | — | Pending |
| `qwen3_51m_muonh_lr1e-3` | — | — | Pending |
| `qwen3_51m_muonh_lr2e-3` | — | — | Pending |
| `qwen3_51m_muonh_lr5e-3` | — | — | Pending |
| `qwen3_51m_muonh_lr1e-2` | — | — | Pending |
| `qwen3_51m_muonh_lr2e-2` | — | — | Pending |
| `qwen3_51m_muonh_lr5e-2` | — | — | Pending |

## Notes

- Compare validation loss and BPB at equal optimizer steps. All runs use one
  seed, with MuonW and MuonC at fixed reference rates.
- All runs use LeCun normal for hidden linears and fixed normal with standard
  deviation `0.02` for tied embeddings.
- MuonH fixes the selected matrix norms with zero Muon decay. Its AdamW
  companion applies `0.1` decay to eligible fallback weights, including
  the tied embedding.
- The global schedule sets the same group LR for Muon and AdamW, with
  `lr_mult: {}`. MuonW and MuonC use the internal matrix-shape adjustment
  `match_rms_adamw` when computing Muon updates. MuonH disables that adjustment
  with `adjust_lr_fn: null`.
