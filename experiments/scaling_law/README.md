# Scaling Law

Reproduce IsoLoss curves from Kaplan et al. — plot training loss vs compute (FLOPs) across model sizes to find the compute-optimal frontier.

## Hypothesis

For a fixed compute budget, there exists an optimal (model size, tokens) pair. Larger models are more sample-efficient but cost more FLOPs per token.

## Models

| Config | d_model | layers | heads | ~Params | LR |
|---|---|---|---|---|---|
| gpt2_16m | 256 | 4 | 4 | ~16M | 1e-3 |
| gpt2_30m | 384 | 6 | 6 | ~30M | 8e-4 |
| gpt2_55m | 512 | 8 | 8 | ~51M | 6e-4 |
| gpt2_124m | 768 | 12 | 12 | ~124M | 6e-4 |

All share: same tokenizer (50K BPE), same data (OpenWebText), same seq_len (1024).

## FLOPs

Logged to W&B as `train/flops` using `C = 6 * N_non_embedding * tokens`.

## Run

```bash
uv run python scripts/train.py --config experiments/scaling_law/gpt2_16m.yaml
uv run python scripts/train.py --config experiments/scaling_law/gpt2_30m.yaml
uv run python scripts/train.py --config experiments/scaling_law/gpt2_55m.yaml
uv run python scripts/train.py --config experiments/scaling_law/gpt2_124m.yaml
```

## Plot

In W&B: custom chart with X=`train/flops` (log scale), Y=`train/loss` (log scale), grouped by run.

## Results

TODO
