# MoE Layer-0 Routing Balance

## Hypothesis

MoE routing imbalance tends to be worst at the shallowest layer: layer-0
hidden states are the least differentiated (closest to the raw token
embedding), so the router has the least signal to separate tokens into
experts, and a few experts absorb a disproportionate share of load. Once the
imbalance sets in early it can compound in later layers via load-balancing
bias/aux-loss dynamics. Two remedies:

- **`dense_first`**: replace layer 0's MoE with a dense SwiGLU MLP sized to
  match the MoE layers' active FFN width. No routing happens at layer 0 at
  all, so there's nothing to imbalance — trades this layer's conditional
  compute (a routing decision) for unconditional compute of the same size.
- **`shallow_bias_boost`**: keep layer 0 as MoE but give it a 4x
  `expert_bias_update_rate` (DeepSeek loss-free balancing, arXiv:2408.15664),
  so its per-expert bias corrects load imbalance faster than the other
  layers.

Both are compared against **`baseline`**: uniform MoE with the same
`expert_bias_update_rate` (1e-3) at every layer.

## Setup

Fixed testbed: `qwen3_188m_a51m` architecture (same as
`../moe_expert_bias/qwen3_188m_a51m_sigmoid_expert_bias_rate1e-3.yaml`) — 512
d_model, 8 layers, 64 routed experts + 2 always-on shared experts, top-6
routed, no expert capacity limit, `expert_bias: true` / `aux_loss: false`,
sigmoid router, muon lr 1e-3, warmup 1500. Active MoE FFN width
`(k + s)·intermediate_size = (6+2)·192 = 1536`; `dense_first`'s layer-0 dense
MLP uses `intermediate_size: 1536, activation: silu, gated: true` to match it
(SwiGLU, compute-comparable to a routed+shared MoE layer).

| Param | Value |
|-------|-------|
| d_model / n_layers | 512 / 8 |
| n_routed_experts / top-k | 64 / 6 |
| n_shared_experts | 2 |
| intermediate_size (per expert) | 192 |
| batch × grad_accum × seq | 64 × 4 × 1024 (≈0.26M tok/step) |
| max_steps | 50000 (≈13B tokens) |
| lr / min_lr / warmup | 1e-3 / 1e-4 / 1500 |

| Config | Layer 0 | Layers 1-7 | Layer 0 bias rate | Total params | Active params |
|--------|---------|------------|--------------------|--------------|----------------|
| `baseline` | MoE | MoE | 1e-3 | 188.03M | 51.19M |
| `dense_first` | Dense SwiGLU (is=1536) | MoE | n/a | 170.89M | 51.16M |
| `shallow_bias_boost` | MoE | MoE | 4e-3 | 188.03M | 51.19M |

Param counts computed via:
```bash
uv run python -c "from src.utils.config import load_config; from src.utils.metric_utils import count_parameters; c=load_config('experiments/moe_layer0_balance/baseline.yaml'); print(count_parameters(c))"
```
`{'total': 188032512, 'non_emb': 162276864, 'active_non_emb': 25437696}` for
`baseline`/`shallow_bias_boost` (identical param shapes — only the bias update
rate differs) and `{'total': 170894848, 'non_emb': 145139200,
'active_non_emb': 25404928}` for `dense_first`. `dense_first`'s total is lower
because a dense SwiGLU layer at `is=1536` has fewer parameters than a
64-routed-expert MoE layer at `is=192` (only the top-6 + 2 shared experts are
ever active per token, but all 64 routed experts' weights are resident).
Active params (embeddings + active_non_emb) are ~51.2M / ~51.16M across all
three, so the arms are compute-comparable per forward pass.

## Run

```bash
nohup bash experiments/moe_layer0_balance/run.sh > logs/moe_layer0_balance.log 2>&1 &
# single:
uv run python scripts/train.py --config experiments/moe_layer0_balance/baseline.yaml
```

## Results

| arm | val_loss | layer-0 balance (MaxVio) | notes |
|-----|----------|--------------------------|-------|
| `baseline` | | | |
| `dense_first` | | | |
| `shallow_bias_boost` | | | |

## Notes

- Layer-0 balance is read from the existing per-layer MoE MaxVio metrics
  (`src/utils/metric_utils.py::compute_moe_maxvio` /
  `compute_moe_global_maxvio`, logged per layer during training) — compare
  `layer_0` MaxVio across `baseline` and `shallow_bias_boost` directly.
  `dense_first` has no layer-0 MoE metric (layer 0 is dense), so its "balance"
  is by construction 0 there; compare its layers 1-7 MaxVio against the
  corresponding layers in `baseline` to see whether removing layer-0 routing
  changes downstream balance.
- Compare `val_loss` across all three to see whether either remedy costs (or
  gains) quality relative to `baseline`, since both trade off something:
  `dense_first` removes a routing decision (fixed compute, no adaptivity),
  `shallow_bias_boost` only changes the balancing speed (no compute change).
- Same testbed as `../moe_expert_bias/`, so `baseline` here is expected to
  reproduce `qwen3_188m_a51m_sigmoid_expert_bias_rate1e-3` if the two ever
  need to be cross-checked (config-diff, not just result-diff).
