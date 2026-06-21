# FP8 `lm_head` Inclusion Ablation

Sister experiment to `experiments/fp8/`. Same 57M Qwen3 setup, but the `lm_head`
projection is **included** in the FP8 swap (`fp8_exclude_lm_head: false`) for every
recipe. The question: does quantizing the output projection to FP8 hurt final loss,
or is the extra GEMM throughput free?

## Hypothesis

`lm_head` is `(d_model, vocab=50257)` — the single largest GEMM in the model, and
its output feeds directly into cross-entropy where small logit errors can shift the
loss. With tied embeddings it also shares weights with the input embedding table, so
FP8 rounding here perturbs both read and write paths.

1. If loss matches the `fp8_exclude_lm_head: true` runs (in `experiments/fp8/`),
   `lm_head` is FP8-safe and should stay in FP8 for the throughput win.
2. If loss degrades, the output projection needs bf16 — keep `fp8_exclude_lm_head: true`.

Expectation: `tensorwise` is most at risk (single coarse scale over a 50K-wide
output); `rowwise` / `rowwise_with_gw_hp` should be more robust if there's any
degradation at all.

## Setup

Only independent variable vs `experiments/fp8/` is `fp8_exclude_lm_head` (false here,
true there). All other hyperparameters identical: seq_len=1024, batch=16,
grad_accum=16, lr=6e-4, cosine with 1500 warmup, min_lr=6e-5, OpenWebText, seed 42.

This is a fast divergence check: `early_stop: 20000` halts each run at 20K steps
(~5.2B tokens) while the cosine schedule stays shaped for `max_steps: 50000`, so at
the stopping point the LR is still well above min and the runs are compared apples-to-apples.
If `lm_head` FP8 hurts, the gap shows up well before 20K steps.

| Config | fp8 | recipe | exclude_lm_head | Approx params |
|---|---|---|---|---|
| qwen3_57m_bf16 | false | — | — | ~57M |
| qwen3_57m_fp8_tensorwise | true | tensorwise | false | ~57M |
| qwen3_57m_fp8_rowwise | true | rowwise | false | ~57M |
| qwen3_57m_fp8_rowwise_with_gw_hp | true | rowwise_with_gw_hp | false | ~57M |

## Run

Runs the bf16 baseline followed by all three FP8 recipes.

```bash
nohup bash experiments/fp8_lm_head/run_57m.sh > logs/fp8_lm_head_57m.log 2>&1 &
```

## Results

Compare the fp8 rows against the matching `fp8_exclude_lm_head: true` runs in
`experiments/fp8/` (same `wandb_project: pretrain-fp8`, run names suffixed `-lmhead`).

| Precision | Recipe | Final Val Loss | Val BPB | Tokens/sec | Δ Loss vs exclude=true | Speedup vs bf16 |
|---|---|---|---|---|---|---|
| bf16 | — | | | | — | 1.00× |
| fp8 | tensorwise | | | | | |
| fp8 | rowwise | | | | | |
| fp8 | rowwise_with_gw_hp | | | | | |

## Notes

- Tied embeddings (`tie_word_embeddings`) mean the `lm_head` weight is the embedding
  table; the FP8 swap only affects the GEMM cast, not stored weights.
- If degradation appears only with `tensorwise`, the fix is the recipe, not the flag.
- 0.5B is intentionally out of scope here — settle the `lm_head` decision at 57M first.
