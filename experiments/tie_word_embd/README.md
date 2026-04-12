# Tie Word Embeddings Ablation

Test the effect of tying `lm_head.weight` to `token_emb.weight` on pretraining quality using the Qwen3 architecture at two scales.

## Hypothesis

Weight tying reduces parameter count and enforces that input and output token representations share the same space, acting as a regularizer. However, it also constrains the model — the input embedding must simultaneously serve as a good token lookup table and a good output projection. Untied weights give the model more expressivity at the cost of extra parameters. The effect is expected to be more visible at larger scale where the model has capacity to benefit from separate representations.

## Setup

| Config | tie_word_embeddings | Approx params |
|---|---|---|
| qwen3_57m_tied | true | ~57M |
| qwen3_57m_untied | false | ~83M (+26M for separate lm_head) |
| qwen3_0.5b_tied | true | ~0.5B |
| qwen3_0.5b_untied | false | ~0.55B (+51M for separate lm_head) |

**57M runs**: Qwen3 (d_model=512, layers=8, heads=8, kv_heads=4, qk_norm=true), seq_len=1024, batch_size=16, grad_accum=16, 5K steps (~1.3B tokens), lr=6e-4, warmup=200 steps, min_lr=6e-5, bf16, OpenWebText.

**0.5B runs**: Qwen3 (d_model=1024, layers=28, heads=16, kv_heads=8, qk_norm=true), seq_len=1024, batch_size=16, grad_accum=16, 50K steps (~13B tokens), lr=2e-4, warmup=1500 steps, min_lr=2e-5, bf16, OpenWebText.

## Run

```bash
# All runs (57M + 0.5B)
nohup bash experiments/tie_word_embd/run.sh > logs/tie_word_embd.log 2>&1 &

# 57M only
uv run python scripts/train.py --config experiments/tie_word_embd/qwen3_57m_tied.yaml
uv run python scripts/train.py --config experiments/tie_word_embd/qwen3_57m_untied.yaml

# 0.5B only
uv run python scripts/train.py --config experiments/tie_word_embd/qwen3_0.5b_tied.yaml
uv run python scripts/train.py --config experiments/tie_word_embd/qwen3_0.5b_untied.yaml
```

## Results

| Model | tie_word_embeddings | Final Val Loss |
|---|---|---|
| qwen3_57m | true (tied) | |
| qwen3_57m | false (untied) | |
| qwen3_0.5b | true (tied) | |
| qwen3_0.5b | false (untied) | |

## Notes

