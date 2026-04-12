# Tokenizer Vocab Size Sweep

Test how tokenizer vocabulary size affects pretraining quality on Qwen3 57M. A larger vocab means longer subword merges (higher compression, fewer tokens per document) but a larger embedding matrix and sparser gradient updates.

## Hypothesis

A moderate vocab size (~50k) sits at a sweet spot: large enough to avoid excessive fragmentation of common words, small enough that the embedding matrix is well-trained within a fixed token budget. Very small vocabs (10k) will hurt due to over-fragmentation; very large vocabs (200k) may hurt because many rare tokens receive insufficient gradient signal.

## Tokenizer Training

All tokenizers are trained using BPE on OpenWebText (1M samples, same dataset used for pretraining). The 50k tokenizer (`tokenizers/custom_bpe_50k`) is already trained and reused.

```bash
# Step 1: train tokenizers (10k, 20k, 100k, 200k) + preprocess data (all 5 sizes)
nohup bash experiments/tokenizer_vocab/run_tokenizer.sh > logs/tokenizer_vocab_prep.log 2>&1 &

# Step 2: train models
nohup bash experiments/tokenizer_vocab/run_train.sh > logs/tokenizer_vocab.log 2>&1 &
```

Each vocab size requires its own preprocessed data because token IDs differ across tokenizers.

**Note:** Vocab sizes > 65535 (100k, 200k) use `uint32` token storage instead of `uint16`, doubling disk usage for those data splits (~40GB vs ~20GB for the full OpenWebText).

## Setup

| Config | vocab_size | tokenizer_path | data_dir | Embedding params |
|---|---|---|---|---|
| qwen3_57m_vocab10k | 10000 | tokenizers/custom_bpe_10k | data/vocab_10k/ | ~5M |
| qwen3_57m_vocab20k | 20000 | tokenizers/custom_bpe_20k | data/vocab_20k/ | ~10M |
| qwen3_57m_vocab50k | 50257 | tokenizers/custom_bpe_50k | data/vocab_50k/ | ~26M |
| qwen3_57m_vocab100k | 100000 | tokenizers/custom_bpe_100k | data/vocab_100k/ | ~51M |
| qwen3_57m_vocab200k | 200000 | tokenizers/custom_bpe_200k | data/vocab_200k/ | ~103M |

All runs share: Qwen3 57M backbone (d_model=512, layers=8, heads=8, kv_heads=4, qk_norm=true), seq_len=1024, batch_size=16, grad_accum=16, 5K steps, lr=6e-4, warmup=200 steps, min_lr=6e-5, bf16, OpenWebText.

Note: total parameter counts vary significantly due to embedding size. The 200k model has ~103M embedding params on top of ~31M non-embedding params.

## Results

| vocab_size | Final Val Loss | Tokens/doc (avg) |
|---|---|---|
| 10k | | |
| 20k | | |
| 50k | | |
| 100k | | |
| 200k | | |

## Notes

