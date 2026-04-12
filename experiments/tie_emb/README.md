# Tie Word Embeddings Ablation

Test the effect of tying `lm_head.weight` to `token_emb.weight` on pretraining quality using the Qwen3 architecture.

## Hypothesis

Weight tying reduces parameter count and enforces that input and output token representations share the same space, acting as a regularizer. However, it also constrains the model — the input embedding must simultaneously serve as a good token lookup table and a good output projection. Untied weights give the model more expressivity at the cost of ~26M extra parameters.

## Setup

| Config | tie_word_embeddings | Approx params |
|---|---|---|
| qwen3_57m_tied | true | ~57M |
| qwen3_57m_untied | false | ~83M (+26M for separate lm_head) |

All runs share: Qwen3 57M (d_model=512, layers=8, heads=8, kv_heads=4, qk_norm=true), seq_len=1024, batch_size=16, grad_accum=16 (effective batch=256, ~262K tokens/step), 5K steps (~1.3B tokens, ~23x params for tied), lr=6e-4, cosine schedule with 200 warmup steps (~4%), min_lr=6e-5, bf16, OpenWebText.

## Run

```bash
nohup bash experiments/tie_emb/run.sh > logs/tie_emb.log 2>&1 &
```

## Results

| tie_word_embeddings | Final Val Loss |
|---|---|
| true (tied) | |
| false (untied) | |

## Notes

