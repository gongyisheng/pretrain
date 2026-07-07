# MoE Router Score Function: softmax vs sigmoid

## Hypothesis

The router turns gate logits into per-expert scores before top-k selection. Two
choices:

- **softmax** (Switch/GShard, DeepSeek-V2): scores compete — they sum to 1 across
  experts, so raising one expert's score lowers the rest. Selection and combine
  weights come from the same normalized distribution.
- **sigmoid** (DeepSeek-V3): each expert scored independently in [0, 1], no
  cross-expert competition. Top-k weights are then renormalized to sum to 1.

Sigmoid decouples experts, which can ease optimization and load balancing at large
expert counts. The catch for a fair comparison: the two score fns feed
different-magnitude scores into the Switch aux-loss formula (softmax probs sum to 1;
sigmoid scores don't), so a single `aux_loss_coef` does **not** apply equal balancing
pressure to both. We therefore sweep the coef per score fn and compare each at its own
best-balanced setting. The question: at this scale (64 experts, top-8), does the score
function move final val loss, and which balances better once the coef is tuned?

## Setup

Fixed testbed: `qwen3_183m_a51m` (183M total, ~51M active), 64 experts, top-8,
per-expert `intermediate_size` 192 (active `8·192 = 1536 = 3·d_model`), no expert
capacity limit. Same testbed as `../moe_aux_loss/` and `../moe_expert_bias/`.
Balancing is aux-loss only (`expert_bias: false`); top-k weights renormalized
(`normalize` default) in both. Two axes vary: `router_score_fn ∈ {softmax, sigmoid}`
and `aux_loss_coef ∈ {1e-2, 3e-3, 1e-3, 1e-4, 3e-5, 0}` — a 2×6 grid (12 runs).
`coef=0` keeps `aux_loss` on (so MaxVio is still logged) but applies no balancing
pressure — the unbalanced floor for each score fn.

| Param | Value |
|-------|-------|
| d_model / n_layers | 512 / 8 |
| n_experts / top-k | 64 / 8 |
| intermediate_size (per expert) | 192 |
| batch × grad_accum × seq | 64 × 4 × 1024 (≈0.26M tok/step) |
| max_steps | 50000 (≈13B tokens) |
| lr / min_lr / warmup | 1e-3 / 1e-4 / 1500 |

Configs: `qwen3_183m_a51m_{softmax,sigmoid}_aux{coef}` for each coef in
`{1e-2, 3e-3, 1e-3, 1e-4, 3e-5, 0}`.

## Run

```bash
nohup bash experiments/moe_router_score_fn/run.sh > logs/moe_router_score_fn.log 2>&1 &
# single:
uv run python scripts/train.py --config experiments/moe_router_score_fn/qwen3_183m_a51m_sigmoid_aux1e-3.yaml
```

W&B project: `pretrain-moe-router-score-fn`.

## Results

Fill val_loss and MaxVio (`train-moe/maxvio_batch/mean`) per cell; pick each score fn's
best-balanced coef, then compare the two winners.

| aux_loss_coef | softmax val_loss | softmax maxvio | sigmoid val_loss | sigmoid maxvio |
|---------------|:----------------:|:--------------:|:----------------:|:--------------:|
| 1e-2 | | | | |
| 3e-3 | | | | |
| 1e-3 | | | | |
| 1e-4 | | | | |
| 3e-5 | | | | |
| 0    | | | | |

Best coef → softmax: _TBD_, sigmoid: _TBD_. Winner: _TBD_.

## Notes

- The coef is swept because softmax and sigmoid scores enter the Switch aux-loss at
  different magnitudes, so a shared coef would over/under-regularize one of them.
  Compare each score fn at its own best-balanced coef, not at a matched coef.
- `coef=0` is the no-balancing floor (aux_loss computed but not added to the loss).
  Expect the worst MaxVio here; how much val loss degrades vs the tuned coef shows how
  much each score fn relies on the aux term.
- With sigmoid, gate scores don't sum to 1; the top-k renormalization keeps combine
  weights comparable to softmax. Judge balance by `train-moe/maxvio_batch/mean`, not
  the raw aux-loss value (not comparable across score fns).
