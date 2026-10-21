# Muon vs MuonS

## Hypothesis

Applying Muon's update to every matrix may change optimization relative to the regular hidden-matrix-only Muon routing at the same learning rate.

## Setup

All runs use the same 50,931,264-parameter (50.93M) dense Qwen3 model: width 512, 8 layers, GQA with 8 query and 4 KV heads, QK normalization, and MLP intermediate size 1536. Word embeddings are tied, RoPE is used, and attention sink is enabled. Data is OpenWebText with `tokenizers/custom_bpe_50k` and a 1% validation split.

| Config | Optimizer | Peak LR | Min LR | Approx. parameters |
|---|---|---:|---:|---:|
| `qwen3_51m_muon` | Muon | `5e-4` | `5e-5` | 50.93M |
| `qwen3_51m_muons` | MuonS | `5e-4` | `5e-5` | 50.93M |

Every run uses batch size 16, gradient accumulation 16, sequence length 1024, 50,000 steps, and 1,500 warmup steps. Both use `match_rms_adamw`, a shared peak LR of `5e-4`, and a minimum LR of `5e-5`. That is 262,144 tokens per optimizer step and 13.1072B tokens total. Evaluation runs every 100 steps for 100 steps; checkpoints are written every 5,000 steps.

## Run

Check GPU usage, select a free device, then run both configurations:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 bash experiments/muons/run.sh
```

Replace `0` with the selected GPU index. Extra arguments are forwarded to every training command.

## Results

| Config | Final validation loss | Final validation BPB | Status |
|---|---:|---:|---|
| `qwen3_51m_muon` | — | — | Not run |
| `qwen3_51m_muons` | — | — | Not run |

## Notes

- Regular Muon updates hidden matrices and uses AdamW for the remaining parameters. MuonS updates every matrix, including the tied vocabulary embedding/output weight, and leaves vectors to AdamW.
- Each configuration shares its `5e-4` learning rate between its Muon/MuonS and AdamW groups.
- This compares the repository defaults and is not a reproduction of the MuonS paper recipe.
