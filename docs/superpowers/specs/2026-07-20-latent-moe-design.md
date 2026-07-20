# LatentMoE — design

Support LatentMoE (arXiv:2601.18089, "LatentMoE: Toward Optimal Accuracy per FLOP
and Parameter in Mixture of Experts") as an optional mode of the existing
`SparseMoEBlock`.

## Background

Standard MoE routed experts operate directly in the hidden dimension `d`.
LatentMoE inserts a shared **down-projection** `W↓: d→ℓ` before the routed experts,
runs the routed experts entirely in the latent space `ℓ`, then applies a shared
**up-projection** `W↑: ℓ→d` after aggregation. The router still scores the original
token at dim `d`, and shared experts still operate at dim `d` — only routed-expert
dispatch and compute are latent.

Crucially, the FFN intermediate width `m` (`intermediate_size`) is **held constant**;
only the expert I/O rank shrinks `d→ℓ`. This preserves the nonlinear budget (∝ K·m)
while cutting the expert I/O rank, which is what lets the savings (factor α = d/ℓ) be
reinvested into more experts / more active experts at matched FLOPs.

Routed expert path:
`x (d) → down_proj → ℓ → [gate_up: ℓ→2m] → act → m → [down: m→ℓ] → ℓ → up_proj → d`.

| | expert I/O width | intermediate | expert weight shapes |
|---|---|---|---|
| standard MoE | `d_model` | `intermediate_size` (m) | gate_up `(E, 2m, d)`, down `(E, d, m)` |
| LatentMoE | `latent_dim` (ℓ) | `intermediate_size` (m) — unchanged | gate_up `(E, 2m, ℓ)`, down `(E, ℓ, m)` |

## Decisions

- **Integration:** extend the existing `moe` block (`SparseMoEBlock`), not a new
  registry class. Reuses all routing, dispatch, aux-loss, expert-bias, load
  tracking, and FLOP/param code; one code path to maintain. `latent_moe=False`
  is byte-for-byte today's block.
- **α-scaling is manual.** The block takes `n_routed_experts` (= αN) and
  `n_routed_experts_per_token` (= αK) as-is; the eff/acc recipe is purely a
  YAML-authoring choice, not runtime logic.
- **Scope:** architecture + tests + config docs. No example YAML config and no
  experiment folder in this change.

## 1. `SparseMoEBlock` (src/layers/mlp.py)

New constructor params: `latent_moe: bool = False`, `latent_dim: int | None = None`.
Let `expert_dim = latent_dim if latent_moe else d_model`.

- **Expert weights** size their `d_model` axis to `expert_dim`:
  `expert_gate_up` → `(E, 2·I, expert_dim)`, `expert_down` → `(E, expert_dim, I)`
  (bias shapes follow; `expert_down_bias` becomes `(E, expert_dim)`). When
  `latent_moe=False` this is identical to today.
- **Projections** (built only when `latent_moe`), both `bias=False`, model dtype:
  - `latent_down_proj = nn.Linear(d_model, latent_dim, bias=False)` — W↓
  - `latent_up_proj  = nn.Linear(latent_dim, d_model, bias=False)` — W↑
- **Router** and **shared experts** are untouched — both operate on `x` at dim `d`.

**Forward** — three surgical edits:
1. Router runs on `x_flat` (dim `d`) as today → `top_indices`, `top_weights`,
   `router_probs`.
2. `tokens = self.latent_down_proj(x_flat) if latent_moe else x_flat`; dispatch
   gathers from `tokens`, so `x_sorted` is `(R, expert_dim)`. The down-projection
   runs **once per token** before dispatch (matches the paper's "dispatch in latent
   space"; cheaper than projecting each of the k routed copies).
3. Accumulator becomes `x_flat.new_zeros(BS, expert_dim)`; after `index_add_`,
   `output = self.latent_up_proj(output)` when `latent_moe` → back to `(BS, d)`.
   Reshape to `(B, S, d)` + shared-expert add unchanged.

`grouped_mlp` is called unchanged — it simply sees `expert_dim` as its in/out width.
Aux-loss, expert-bias, load tracking, dropout: unchanged.

## 2. Config (src/utils/config.py)

Inside `ModelConfig.__post_init__`, in the `if self.mlp_cls == "moe":` block:

```python
self.mlp_kwargs.setdefault("latent_moe", False)
if self.mlp_kwargs["latent_moe"]:
    latent_dim = self.mlp_kwargs.get("latent_dim")
    if not isinstance(latent_dim, int) or latent_dim <= 0:
        raise ValueError(
            f"latent_dim must be a positive int when latent_moe=True; got {latent_dim!r}"
        )
```

- `latent_moe` defaults to `False` → every existing config unchanged.
- No default for `latent_dim`: α = d_model / latent_dim is always set deliberately.
  Required and asserted `> 0` only when `latent_moe=True`; ignored otherwise.
- No `transformer.py` change: both kwargs are self-contained in `mlp_kwargs` and
  forwarded directly to the constructor (unlike MLA, which reads back derived dims).

## 3. FLOP / param accounting

`compute_flops` / `compute_parameters` gain `latent_moe=False, latent_dim=None`
kwargs. Let `edim = latent_dim if latent_moe else d_model`.

- **Experts** use `edim` in place of `d_model`:
  - gated: `expert = 6·edim·d_ff` (flops) / `3·d_ff·edim` (params)
  - ungated: `expert = 4·edim·d_ff` / `2·d_ff·edim`
  - bias down-width becomes `edim`.
- **Projections** (added only when `latent_moe`):
  - params `2·d_model·latent_dim` (W↓ `ℓ×d` + W↑ `d×ℓ`)
  - flops `4·d_model·latent_dim` per token (both run once/token, **outside** the
    `×k` factor)
  - counted in both `active` and total (always run).
- Router and shared-expert terms unchanged.

## 4. Tests (tests/fast/, numerical-parity style)

- **Off-path regression:** `latent_moe=False` block is bit-identical to the current
  `SparseMoEBlock` (same seed → same output and same param count).
- **Reference parity:** a plain-Python LatentMoE reference (explicit per-token
  `latent_down_proj` → loop over the selected experts in `ℓ` → weighted sum →
  `latent_up_proj`) matches the grouped-GEMM forward within tolerance, in fp32 and
  bf16.
- **Shapes / params:** output is `(B, S, d)`; expert weights carry the `ℓ` axis;
  `compute_parameters` equals `sum(p.numel())` for both variants (guards §3).
- **Config validation:** `latent_moe=True` with missing / non-int / ≤0 `latent_dim`
  raises; a valid config round-trips into the expected `mlp_kwargs`.

## Out of scope

- Auto-derivation of N/K from α (`eff`/`acc` variants) — manual in YAML.
- Example training config and experiment folder (separate follow-up).
- Distributed dispatch / all-to-all savings (single-GPU codebase; the latent
  projection still reduces per-expert compute and weight memory).
