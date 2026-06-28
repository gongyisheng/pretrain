# MoE Batch-Size Sweep

## Hypothesis

For the Qwen3 MoE 183M testbed there is an effective batch size that best trades
gradient-noise reduction against per-token efficiency. Sweep gradient
accumulation (effective batch) at a fixed token budget and compare final val
loss to find it.

## Setup

Single config type: `qwen3_183m_a51m` (top-8, ~51M active; per-expert
`intermediate_size=192`, 64 experts → active `k·is = 1536`). Fixed
`batch_size=8`, `max_seq_len=1024`, cosine LR `5e-4 → 5e-5`, AdamW, bf16.
Effective batch = `batch_size × grad_accu × max_seq_len`. `max_steps=50000` is
fixed across all runs, so the token budget scales with the effective batch (the
larger batches train on proportionally more tokens, from 13.1B up to 104.9B).

Configs are named by **effective batch size in sequences** (`bs` =
`batch_size × grad_accu` = 8 × grad_accu).

| Config | eff. batch (seq) | grad_accu | Effective batch (tokens) | max_steps | warmup | total tokens |
|--------|------------------|-----------|--------------------------|-----------|--------|--------------|
| `qwen3_183m_a51m_bs256`  | 256  | 32  | 262K  | 50000 | 1000 | 13.1B  |
| `qwen3_183m_a51m_bs512`  | 512  | 64  | 524K  | 50000 | 500  | 26.2B  |
| `qwen3_183m_a51m_bs1024` | 1024 | 128 | 1.05M | 50000 | 250  | 52.4B  |
| `qwen3_183m_a51m_bs2048` | 2048 | 256 | 2.10M | 50000 | 125  | 104.9B |

`bs256` is the established baseline.

## Running

```bash
nohup bash experiments/moe_batch_size/run.sh > logs/moe_batch_size.log 2>&1 &

# Single config:
uv run python scripts/train.py --config experiments/moe_batch_size/qwen3_183m_a51m_bs512.yaml
```

## Results

| Config | eff. batch (seq) | Final val loss | Val PPL | Notes |
|--------|------------------|----------------|---------|-------|
| `qwen3_183m_a51m_bs256`  | 256  | | | |
| `qwen3_183m_a51m_bs512`  | 512  | | | |
| `qwen3_183m_a51m_bs1024` | 1024 | | | |
| `qwen3_183m_a51m_bs2048` | 2048 | | | |

## Notes

- LR is held fixed across batch sizes. If the larger batches underperform, the
  cause may be a too-small LR rather than the batch size itself — a follow-up
  sweep would scale LR with effective batch (√ or linear rule).
