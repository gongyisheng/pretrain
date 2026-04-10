# Batch Size

Measure how effective batch size (tokens per gradient step) affects final validation loss, holding all other hyperparameters fixed.

## Hypothesis

Larger batch sizes reduce gradient noise, leading to faster initial convergence but potentially worse generalization. With a fixed learning rate and step budget, smaller batches see more parameter updates over the same token budget, which may yield lower final loss.

## Setup

Fixed: Qwen3 57M architecture, `seq_len=1024`, `lr=6e-4`, 50K steps.

The hardware batch size stays at 8 across all runs — only `gradient_accumulation_steps` varies to control the effective batch size.

| Config | batch_size | grad_accum | Tokens/step |
|---|---|---|---|
| qwen3_57m_bs_16k | 8 | 2 | ~16K |
| qwen3_57m_bs_64k | 8 | 8 | ~64K |
| qwen3_57m_bs_256k | 8 | 32 | ~256K |
| qwen3_57m_bs_1m | 8 | 128 | ~1M |

Note: total tokens seen = `tokens/step × max_steps`. Larger batches therefore process far more tokens total, but take the same number of optimizer steps.

## Run

```bash
nohup bash experiments/batch_size/run.sh > logs/batch_size.log 2>&1 &
```

Or a single run:

```bash
uv run python scripts/train.py --config experiments/batch_size/qwen3_57m_bs_64k.yaml
```

## W&B

Project: `pretrain-batch-size`. Compare runs by `val/loss` vs `train/step`.

## Results

TODO

## Notes

### Effective batch size

Effective batch size (in tokens) = `batch_size × gradient_accumulation_steps × seq_len`. This is the number of tokens the model sees per optimizer step. In this experiment, we fix `batch_size=8` and `seq_len=1024`, then vary `gradient_accumulation_steps` to sweep over effective batch sizes from ~16K to ~1M tokens/step.

Recommended effective batch size for our experiments: **512K tokens/step** (`batch_size=8, grad_accum=64`). Effective batch sizes smaller than this show training instability. Sources of instability include:

1. **Dataset** — a single bad batch has outsized impact when the effective batch size is small, since each batch contributes a larger fraction of the gradient.
2. **cuBLAS non-determinism** — cuBLAS uses non-deterministic algorithms by default, introducing small numerical differences across runs. At small batch sizes these fluctuations are not averaged out. Enabling deterministic training (`os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'`) resolves this.

### How other LLMs choose their batch sizes

**GPT-2** (Radford et al., 2019) trained GPT-2 124M with a batch size of 512 and sequence length 1024, giving an effective batch size of ~512K tokens/step. Larger GPT-2 variants (345M, 762M, 1.5B) all used the same batch size.

**GPT-3** (Brown et al., 2020) scaled effective batch size with model size:

| Model | Params | Tokens/step |
|---|---|---|
| Small | 125M | 0.5M |
| Medium | 350M | 0.5M |
| Large | 760M | 0.5M |
| XL | 1.3B | 1M |
| 2.7B | 2.7B | 1M |
| 6.7B | 6.7B | 2M |
| 13B | 13B | 2M |
| 175B | 175B | 3.2M |
