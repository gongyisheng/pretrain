# MoE Batch-Size Sweep

## Hypothesis

For the `qwen3_133m_a35m` MoE (133M total, ~35M active, 64 experts top-2), there is
an effective batch size that best trades gradient-noise reduction against
per-token efficiency. Sweep gradient accumulation (effective batch) at a fixed
token budget and compare final val loss to find it.

## Setup

Fixed `batch_size=8`, `max_seq_len=1024`, lr `5e-4 → 5e-5` cosine, AdamW, bf16.
Effective batch = `batch_size × grad_accu × max_seq_len`. `max_steps` and
`warmup_steps` scale inversely with grad_accu so every run consumes the same
~13.1B tokens (~99× the 133M total params).

| Config | grad_accu | Effective batch (tokens) | max_steps | warmup |
|--------|-----------|--------------------------|-----------|--------|
| `qwen3_133m_a35m_ga32`  | 32  | 262K  | 50000 | 1000 |
| `qwen3_133m_a35m_ga64`  | 64  | 524K  | 25000 | 500  |
| `qwen3_133m_a35m_ga128` | 128 | 1.05M | 12500 | 250  |
| `qwen3_133m_a35m_ga256` | 256 | 2.10M | 6250  | 125  |

`ga32` is the established baseline. All other settings match the canonical
`configs/qwen3_133m_a35m.yaml`.

## Running

```bash
# All configs sequentially:
nohup bash experiments/moe_batch_size/run.sh > logs/moe_batch_size.log 2>&1 &

# Single config:
uv run python scripts/train.py --config experiments/moe_batch_size/qwen3_133m_a35m_ga64.yaml
```

## Results

| Config | grad_accu | Final val loss | Val PPL | Notes |
|--------|-----------|----------------|---------|-------|
| `qwen3_133m_a35m_ga32`  | 32  | | | |
| `qwen3_133m_a35m_ga64`  | 64  | | | |
| `qwen3_133m_a35m_ga128` | 128 | | | |
| `qwen3_133m_a35m_ga256` | 256 | | | |

## Notes

- LR is held fixed across batch sizes. If the larger batches underperform, the
  cause may be a too-small LR rather than the batch size itself — a follow-up
  sweep would scale LR with effective batch (√ or linear rule).
