# Intermediate Size LR Sweep (Qwen3 57M)

Sweep dense FFN expansion ratio `intermediate_size / d_model` against learning rate to find the optimal ratio at fixed depth (8 layers) and width (d_model=512).

## Hypothesis

Standard transformers use a 4x FFN expansion ratio; production Qwen3 dense models use ~3x. This experiment tests whether wider FFN keeps paying off (sub-linear loss gain) or plateaus, and whether the optimal LR shifts systematically with width.

## Setup

8 widths x 4 LRs = 32 runs. All runs share the qwen3_57m baseline except for `intermediate_size`, `lr`, `min_lr`, `debug.max_steps`, and W&B/checkpoint paths.

| Mult | intermediate_size | ~Params | LRs tested |
|---|---|---|---|
| 1 | 512 | ~38M | 1e-4, 2e-4, 5e-4, 1e-3 |
| 2 | 1024 | ~45M | 1e-4, 2e-4, 5e-4, 1e-3 |
| 3 | 1536 | ~51M | 1e-4, 2e-4, 5e-4, 1e-3 |
| 4 | 2048 | ~57M (baseline) | 1e-4, 2e-4, 5e-4, 1e-3 |
| 6 | 3072 | ~70M | 1e-4, 2e-4, 5e-4, 1e-3 |
| 8 | 4096 | ~82M | 1e-4, 2e-4, 5e-4, 1e-3 |
| 12 | 6144 | ~108M | 1e-4, 2e-4, 5e-4, 1e-3 |
| 16 | 8192 | ~133M | 1e-4, 2e-4, 5e-4, 1e-3 |

`min_lr = lr / 10`. Param count = ~32M (embedding + attention + norms) + 12.3K x intermediate_size.

**Fixed across all runs:** Qwen3 (8 layers, 8 heads, 4 kv_heads, d_model=512, qk_norm=true, rope_theta=10000), seq_len=1024, batch_size=8, grad_accum=32 (effective batch=256, ~262K tok/step), warmup_steps=1500, cosine schedule, bf16, OpenWebText.

**Early stop:** `debug.max_steps: 12000` halts training at step 12000 while keeping the LR schedule shaped for 50000 steps. This is ~3.14B tokens per run (~24x params at mult=16, well above Chinchilla 20x). At step 12000 the LR has decayed only ~9% from peak, so each run is compared near its peak LR — the right regime for picking an LR winner per width.

## Run

```bash
nohup bash experiments/intermediate_size/lr_sweep/run.sh > logs/intermediate_size_lr_sweep.log 2>&1 &
```

Runs sequentially in (mult ascending, lr ascending) order so the cheapest configs surface failures first.

## Results

Best validation loss per (mult, lr) cell (filled in after running):

| Mult | lr=1e-4 | lr=2e-4 | lr=5e-4 | lr=1e-3 | Best LR |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 6 | | | | | |
| 8 | | | | | |
| 12 | | | | | |
| 16 | | | | | |

Headline plot: best-of-LR loss vs intermediate_size (log-log).

## W&B

Project: `pretrain-intermediate-size-lr-sweep`. Group: `mult{M}` (collapses the 4-LR sweep into one curve per width).

Suggested charts:
- Best-of-LR loss vs intermediate_size (headline plot, log-log)
- Loss vs LR per width (validate optimal LR per width)
- Loss vs FLOPs (param count varies, FLOP-normalising helps cross-width comparison)

## Notes

### Caveats

- **mult=1 is unusual.** intermediate_size = d_model means the FFN is no wider than the attention output — far outside standard transformer ratios. Expected to underperform; included to anchor the lower end of the curve.

- **Early-stop loss values are not comparable to full-schedule runs** in `pretrain-scaling-law`. Schedule is shaped for 50K but training stops at 12K.

- **Boundary effects in the LR grid.** For mult=1 (38M), 1e-3 is likely optimal (per the size table in CLAUDE.md), so a winner of 1e-3 might still leave room above the grid. If 1e-3 wins at the small widths, consider a small follow-up at 1.5e-3 to confirm.

- **OOM fallback at mult=16.** mult=16 (intermediate_size=8192) with batch_size=8, seq_len=1024 produces ~268 MB FFN activation per layer x 8 layers. Should fit on 24GB+ GPUs; if OOM, set `activation_checkpointing: true` for that config.

### Optional follow-up

Once the best LR per width is identified, a follow-up sweep (8 runs, no LR ablation) at full 50K steps gives publication-quality loss numbers. Out of scope for this experiment.
