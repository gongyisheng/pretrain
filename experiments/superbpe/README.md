# SuperBPE Reproduction (arXiv:2503.13423)

Reproduce the paper's encoding-efficiency claim: at fixed vocab size T=200k, a SuperBPE tokenizer with transition point t=80k uses ~33% fewer tokens than vanilla BPE to encode the same text.

## Hypothesis

A two-stage BPE curriculum — subwords until vocab=t, then superwords (no whitespace pretokenization) until vocab=T — produces a more efficient encoding than single-stage BPE at the same vocab size. The mechanism is that high-frequency multi-word expressions (`" of the "`, `" by the way"`) become single tokens, absorbing token budget that single-stage BPE wastes on rarely-merged subword fragments.

## Tokenizer Training

All tokenizers are trained on the same 100k-sample slice of OpenWebText. The `bpe_200k` baseline uses the SuperBPE trainer with `t == T` so stage 2 is a no-op — this gives a BPE baseline whose stage-1 pretokenizer (regex Split + ByteLevel) matches SuperBPE, eliminating the confound between pretokenizer choice and the two-stage curriculum.

```bash
# Step 1: train all 4 tokenizers (sequential; each takes minutes to hours)
nohup bash experiments/superbpe/run_train_tokenizers.sh > logs/superbpe_all.log 2>&1 &

# Step 2: evaluate bytes/token on a held-out OpenWebText slice
bash experiments/superbpe/run_eval.sh
```

W&B logs the efficiency curve (vocab_size on x-axis, bytes_per_token on y-axis) for each SuperBPE run. The transition point appears as a visible kink.

## Setup

| Config | T | t | method | max_superword_words | Notes |
|---|---|---|---|---|---|
| `bpe_200k` | 200000 | 200000 | superbpe | 4 | Baseline; stage 2 is a no-op (t==T). Stage-1 pretokenizer matches SuperBPE for a fair head-to-head. |
| `superbpe_200k_t80k` | 200000 | 80000 | superbpe | 4 | Paper's "best encoding efficiency" config. |
| `superbpe_200k_t160k` | 200000 | 160000 | superbpe | 4 | Mid transition. |
| `superbpe_200k_t180k` | 200000 | 180000 | superbpe | 4 | Paper's "best downstream LM" config. |

All runs share: `--num_samples 100000` of OpenWebText, special_tokens=`["<|endoftext|>"]`.

## Results

Fill in after running. `bytes_per_token` is higher = better.

| Tokenizer | bytes/token (OpenWebText, 10k docs) | Ratio vs bpe_200k | Stage-2 merges accepted | Wall time |
|---|---|---|---|---|
| bpe_200k | | 1.000 | — | |
| superbpe_200k_t80k | | | | |
| superbpe_200k_t160k | | | | |
| superbpe_200k_t180k | | | | |

## Notes

- Paper claims ~33% fewer tokens (i.e., `bytes_per_token` ratio ~1.33) at `t=80k`. We expect a similar number; small deviation is fine since we train on 100k OpenWebText docs vs. the paper's 10 GB of olmo-mix.
- The W&B efficiency curve for each SuperBPE run shows a kink at `vocab_size = t`. If the kink is in the wrong direction (efficiency drops after the transition), something is wrong with stage 2.
- See `docs/superpowers/specs/2026-05-16-superbpe-design.md` for the full design + Risk #6 caveat.
