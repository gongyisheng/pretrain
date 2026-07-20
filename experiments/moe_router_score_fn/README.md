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
expert counts. We hold `aux_loss_coef = 1e-3` fixed and compare the two score fns
head-to-head. The question: at this scale (64 routed experts, top-6 + 2 shared),
does the score function move final val loss, and which balances better?

Caveat: the two score fns feed different-magnitude scores into the Switch aux-loss
formula (softmax probs sum to 1; sigmoid scores don't), so the same `aux_loss_coef`
does not apply identical balancing pressure to both — read MaxVio, not the raw aux
value, when judging balance.

## Setup

Fixed testbed: `qwen3_188m_a51m` (188M total, ~51M active), 64 routed experts + 2
always-on shared experts, top-6 routed, per-expert `intermediate_size` 192 (active
`(2+6)·192 = 1536 = 3·d_model`), no expert capacity limit. Same testbed as
`../moe_expert_bias/`. The score fn and aux loss balance only the 6 routed experts.
Balancing is aux-loss only (`expert_bias: false`); top-k weights renormalized
(`normalize` default) in both. Two runs, both at `aux_loss_coef = 1e-3`: one
**softmax**, one **sigmoid**.

| Param | Value |
|-------|-------|
| d_model / n_layers | 512 / 8 |
| n_routed_experts / top-k | 64 / 6 |
| n_shared_experts | 2 |
| intermediate_size (per expert) | 192 |
| batch × grad_accum × seq | 64 × 4 × 1024 (≈0.26M tok/step) |
| max_steps | 50000 (≈13B tokens) |
| lr / min_lr / warmup | 1e-3 / 1e-4 / 1500 |

Configs: `qwen3_188m_a51m_softmax_aux1e-3` and `qwen3_188m_a51m_sigmoid_aux1e-3`.

## Run

```bash
nohup bash experiments/moe_router_score_fn/run.sh > logs/moe_router_score_fn.log 2>&1 &
# single:
uv run python scripts/train.py --config experiments/moe_router_score_fn/qwen3_188m_a51m_sigmoid_aux1e-3.yaml
```

W&B project: `pretrain-moe-router-score-fn`.

## Results

Fill val_loss and MaxVio (`train-moe/maxvio_batch/mean`) per cell.

| score fn | aux_loss_coef | val_loss | maxvio |
|----------|---------------|:--------:|:------:|
| softmax | 1e-3 | | |
| sigmoid | 1e-3 | | |

softmax vs sigmoid: _TBD_.

## Notes

- softmax and sigmoid scores enter the Switch aux-loss at different magnitudes, so the
  shared `1e-3` coef does not apply identical balancing pressure. Judge balance by
  MaxVio, not the raw aux-loss value (not comparable across score fns).
- With sigmoid, gate scores don't sum to 1; the top-k renormalization keeps combine
  weights comparable to softmax.
