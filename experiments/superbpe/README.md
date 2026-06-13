# SuperBPE Grid Sweep (arXiv:2503.13423)

Reproduce the paper's encoding-efficiency claim and probe the joint
sensitivity to **vocab size (V)**, **transition point (t)**, and **max
superword words (m)**.

## Hypothesis

At fixed V, a two-stage curriculum — subwords until vocab = t, then
superwords (no whitespace pretokenization) until vocab = V — produces a
more efficient encoding than single-stage BPE. The paper reports a kink
at t ≈ 0.4·V for V = 200k; we test whether that ratio holds across
V ∈ {50k, 100k, 150k, 200k} and how it interacts with m.

## Grid

| Axis | Values |
|---|---|
| `vocab_size` (V) | 50k, 100k, 150k, 200k |
| `transition_size` (t) | 0, then every 20k up to V (strict) |
| `max_superword_words` (m) | 2, 3, 4, 5, 6 |

Total: **130 SuperBPE configs + 4 BPE baselines = 134 configs**, grouped
by V:

```
v50k/   16 yamls  (1 bpe + 3 t × 5 m)
v100k/  26 yamls  (1 bpe + 5 t × 5 m)
v150k/  41 yamls  (1 bpe + 8 t × 5 m)
v200k/  51 yamls  (1 bpe + 10 t × 5 m)
```

Names use round-number axes (e.g. `superbpe_v200k_t80k_m4`); YAML values
carry a +257 offset (256-byte alphabet + 1 special token), so the
example above lists `vocab_size: 200257`, `transition_size: 80257`.
The `t0k` configs (`transition_size: 257`) skip stage 1 entirely —
all merges happen in stage 2 with no whitespace pretokenizer.

## Pipeline

```bash
# 1. Generate configs (idempotent; re-run after editing the script).
uv run python experiments/superbpe/generate_configs.py

# 2. Train all 114 tokenizers, smallest V first. Per-run logs land in
#    logs/superbpe/train/<name>.log.
nohup bash experiments/superbpe/run_train.sh > logs/superbpe_train.log 2>&1 &

# 3. Evaluate bytes/token on 10k OpenWebText held-out docs. Per-run
#    JSON output captured to logs/superbpe/eval/<name>.log.
bash experiments/superbpe/run_eval.sh

# 4. Aggregate eval logs into a single CSV.
uv run python experiments/superbpe/collect_results.py
# → experiments/superbpe/results.csv
```

All runs share `--num_samples 100000` of OpenWebText for training and
the same `<|endoftext|>` special token.

## Results

Aggregated row-per-run output lives in `results.csv` with the columns
below. (Markdown can't hold 114 rows readably; load the CSV in pandas
or open in a spreadsheet to slice/pivot.)

| Column | Source |
|---|---|
| `name` | Filename stem / wandb run name |
| `V`, `t`, `m` | Parsed from `name` |
| `method` | `bpe` or `superbpe` |
| `bytes_per_token` | `TokenizerTrainer.evaluate` output |
| `stage2_merges_accepted` | (TODO — parsed from train logs) |
| `wall_seconds` | (TODO — captured from train script timing) |

Example slice (first row per V at m = 4 once runs land):

| name | bytes/token | ratio vs bpe_v200k |
|---|---|---|
| bpe_v200k | _TBD_ | 1.000 |
| superbpe_v200k_t80k_m4 | _TBD_ | _TBD_ |
| superbpe_v150k_t60k_m4 | _TBD_ | _TBD_ |
| superbpe_v100k_t40k_m4 | _TBD_ | _TBD_ |
| superbpe_v50k_t20k_m4 | _TBD_ | _TBD_ |

(Fill in after the sweep; full table in `results.csv`.)

## Notes

- The paper reports ~33% fewer tokens (`bytes/token` ratio ≈ 1.33) at
  V = 200k, t = 80k. We expect a similar number for our equivalent
  config; small deviation is fine since we train on 100k OpenWebText
  docs vs. the paper's 10 GB of olmo-mix.
- For each SuperBPE run, the W&B efficiency curve should kink at
  `vocab_size = transition_size`. A kink in the wrong direction
  (efficiency drops post-transition) means stage 2 is broken.
- See `docs/superpowers/specs/2026-05-16-superbpe-design.md` for the
  original design + Risk #6 caveat, and
  `docs/superpowers/specs/2026-05-18-superbpe-grid-design.md` for the
  grid-sweep design.
