# Pretrain

A research playground for LLM pretraining and model architecture exploration.
Pure PyTorch (>=12GB VRAM).

## Install
Requires python >=3.12
Notable pins:
- `torch>=2.10`
- `triton>=3.6`
- `tokenizers>=0.22`
- `datasets>=4.5`
- `wandb>=0.26`

```bash
uv sync
wandb login   # required unless you pass --no-wandb to train.py
```

## Data & tokenizer

Option A — download prebuilt tokenizer and tokenized `.bin` files from [gongyisheng/openwebtext-exp](https://huggingface.co/datasets/gongyisheng/openwebtext-exp) (recommended, skips ~hours of preprocessing):

```bash
uv run hf download gongyisheng/openwebtext-exp --repo-type dataset --local-dir .
```

Option B — build from scratch:

```bash
uv run python scripts/train_tokenizer.py --config configs/gpt2_124m.yaml
uv run python scripts/preprocess_data.py --config configs/gpt2_124m.yaml
```

## Train

Pick a config from `configs/` and launch training:

```bash
uv run python scripts/train.py --config configs/gpt2_124m.yaml        # GPT-2 124M (MHA baseline)
uv run python scripts/train.py --config configs/qwen3_57m.yaml        # Qwen3 57M (GQA + RoPE + RMSNorm + SwiGLU)
uv run python scripts/train.py --config configs/qwen3_moe_133m.yaml   # Qwen3 MoE 133M
```

Override config values from the CLI, disable W&B, or resume from a checkpoint:

```bash
uv run python scripts/train.py --config configs/gpt2_124m.yaml --no-wandb
uv run python scripts/train.py --config configs/gpt2_124m.yaml --optimizer.lr=1e-4 --training.backend=triton
uv run python scripts/train.py --config configs/gpt2_124m.yaml --resume checkpoints/step_1000.pt
```

Architecture-specific experiments (attention variants, MoE, scaling laws, etc.) live in `experiments/` with their own configs and write-ups.
