# LatentMoE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LatentMoE (arXiv:2601.18089) as an optional mode of the existing `SparseMoEBlock`: routed experts run in a compressed latent space ℓ, with a shared down-projection `d→ℓ` before dispatch and a shared up-projection `ℓ→d` after aggregation.

**Architecture:** Extend `SparseMoEBlock` in place with `latent_moe`/`latent_dim` kwargs. `expert_dim = latent_dim if latent_moe else d_model` sizes the expert weights; two bias-free `nn.Linear` projections (`latent_down_proj`, `latent_up_proj`) wrap the routed-expert path. Router and shared experts stay at dim `d`. When `latent_moe=False`, the block is unchanged.

**Tech Stack:** Python, PyTorch, pytest, `uv`. Dropless MoE via `torch._grouped_mm`.

## Global Constraints

- Design/spec: `docs/superpowers/specs/2026-07-20-latent-moe-design.md`.
- Config defaults & validation live in `ModelConfig.__post_init__` (src/utils/config.py) — NOT component constructors. Components take resolved values as explicit args.
- Fused/expert ops must support float32, float16, bfloat16. Never hardcode or cast to a dtype; preserve the caller's dtype.
- α-scaling is manual: `n_routed_experts` (=αN) and `n_routed_experts_per_token` (=αK) are set in YAML, not derived in code.
- Tests are numerical-parity against explicit references (`tests/fast/layers/_refs.py`), per repo convention. `COMPOUND_DTYPES = [(float32,1e-5),(float16,1e-2),(bfloat16,8e-2)]`.
- Run tests with `uv run pytest`. Lint with `uv run ruff check src/ tests/` and `uv run ruff format --check src/ tests/`.
- Commit message prefixes: `[feat]`, `[test]`, `[docs]`, etc.

---

### Task 1: Config validation for `latent_moe` / `latent_dim`

**Files:**
- Modify: `src/utils/config.py` (the `if self.mlp_cls == "moe":` block in `ModelConfig.__post_init__`, currently ending at line 98)
- Test: `tests/fast/utils/test_config.py`

**Interfaces:**
- Produces: `mlp_kwargs["latent_moe"]` (bool, default `False`) and, when on, a validated `mlp_kwargs["latent_dim"]` (positive int). No `latent_dim` key is injected when `latent_moe` is off.

- [ ] **Step 1: Write the failing tests**

Append to `tests/fast/utils/test_config.py` (module already imports `pytest` and `ModelConfig`):

```python
def test_modelconfig_latent_moe_default_off():
    cfg = ModelConfig(
        d_model=64,
        mlp_cls="moe",
        mlp_kwargs={"n_routed_experts": 4, "aux_loss": True},
    )
    assert cfg.mlp_kwargs["latent_moe"] is False
    assert "latent_dim" not in cfg.mlp_kwargs


@pytest.mark.parametrize("latent_dim", [None, 0, -4, 3.5])
def test_modelconfig_latent_moe_requires_positive_latent_dim(latent_dim):
    kwargs = {"n_routed_experts": 4, "aux_loss": True, "latent_moe": True}
    if latent_dim is not None:
        kwargs["latent_dim"] = latent_dim
    with pytest.raises(ValueError, match="latent_dim must be a positive int"):
        ModelConfig(d_model=64, mlp_cls="moe", mlp_kwargs=kwargs)


def test_modelconfig_latent_moe_valid():
    cfg = ModelConfig(
        d_model=64,
        mlp_cls="moe",
        mlp_kwargs={
            "n_routed_experts": 4,
            "aux_loss": True,
            "latent_moe": True,
            "latent_dim": 16,
        },
    )
    assert cfg.mlp_kwargs["latent_moe"] is True
    assert cfg.mlp_kwargs["latent_dim"] == 16
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/fast/utils/test_config.py -k latent_moe -v`
Expected: FAIL — `latent_moe` default test fails on `KeyError`/`assert`, and the raise tests fail because no `ValueError` is raised.

- [ ] **Step 3: Add validation in config.py**

In `src/utils/config.py`, inside the `if self.mlp_cls == "moe":` block, immediately after the existing `aux_loss` handling (after the line `self.mlp_kwargs.setdefault("aux_loss_coef", 0.001)`), add:

```python
            self.mlp_kwargs.setdefault("latent_moe", False)
            if self.mlp_kwargs["latent_moe"]:
                latent_dim = self.mlp_kwargs.get("latent_dim")
                if not isinstance(latent_dim, int) or isinstance(latent_dim, bool) or latent_dim <= 0:
                    raise ValueError(
                        f"latent_dim must be a positive int when latent_moe=True; "
                        f"got {latent_dim!r}"
                    )
```

(The `isinstance(..., bool)` guard rejects `True`/`False`, since `bool` is a subclass of `int`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/fast/utils/test_config.py -k latent_moe -v`
Expected: PASS (all parametrizations).

- [ ] **Step 5: Commit**

```bash
git add src/utils/config.py tests/fast/utils/test_config.py
git commit -m "[feat] config: validate latent_moe/latent_dim for MoE block"
```

---

### Task 2: `SparseMoEBlock` latent architecture (weights, projections, forward) + reference

**Files:**
- Modify: `tests/fast/layers/_refs.py` (extend `sparse_moe_block_ref`)
- Modify: `src/layers/mlp.py` (`SparseMoEBlock.__init__` and `forward`)
- Test: `tests/fast/layers/test_mlp.py`

**Interfaces:**
- Consumes: `latent_moe`/`latent_dim` kwargs (Task 1 forwards them via `mlp_kwargs`).
- Produces:
  - `SparseMoEBlock(..., latent_moe: bool = False, latent_dim: int | None = None)`.
  - Attributes `latent_down_proj` / `latent_up_proj` — `nn.Linear(bias=False)` when `latent_moe`, else `None`.
  - Expert params carry axis `expert_dim = latent_dim if latent_moe else d_model`: `expert_gate_up (E, 2·I, expert_dim)`, `expert_up (E, I, expert_dim)`, `expert_down (E, expert_dim, I)`, `expert_down_bias (E, expert_dim)`.
  - `forward(x) -> (out (B,S,d), aux_loss)` unchanged in signature.
  - `sparse_moe_block_ref(..., latent_down_weight=None, latent_up_weight=None)` — new trailing kwargs.

- [ ] **Step 1: Extend the reference `sparse_moe_block_ref`**

In `tests/fast/layers/_refs.py`, replace the whole `sparse_moe_block_ref` function with this version (adds two trailing kwargs and a latent path; routing still uses the original `x` at dim `d`):

```python
def sparse_moe_block_ref(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    expert_down: torch.Tensor,
    n_routed_experts_per_token: int,
    activation: str = "silu",
    expert_gate_up: torch.Tensor | None = None,
    expert_up: torch.Tensor | None = None,
    normalize: bool = True,
    expert_gate_up_bias: torch.Tensor | None = None,
    expert_up_bias: torch.Tensor | None = None,
    expert_down_bias: torch.Tensor | None = None,
    latent_down_weight: torch.Tensor | None = None,
    latent_up_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager sparse MoE: naive per-(token, slot) expert dispatch.

    Mirrors MLP's gated/ungated split:
      - gated   (expert_gate_up given): hidden = act(gate)*up via GATED_ACTIVATIONS_REFS
      - ungated (expert_up given):      hidden = act(up)      via UNGATED_ACTIVATIONS_REFS
    Optional bias tensors are applied per-expert when provided.

    LatentMoE (latent_down_weight (ℓ,d) and latent_up_weight (d,ℓ) given): experts run
    in latent space ℓ. Routing still uses the original x at dim d; the per-token expert
    input is x @ W↓ᵀ, expert outputs are accumulated in ℓ, then projected back via W↑.
    Returns (output (B,S,D), aux_loss scalar) — Switch Transformer load-balancing loss.
    """
    if (expert_gate_up is None) == (expert_up is None):
        raise ValueError(
            "Exactly one of expert_gate_up (gated) or expert_up (ungated) must be given"
        )
    B, S, D = x.shape
    T = B * S
    E = gate_weight.shape[0]
    x_flat = x.view(T, D)

    top_indices, top_weights, router_probs = moe_router_ref(
        x_flat, gate_weight, n_routed_experts_per_token, normalize=normalize
    )

    # Routed-expert input: latent (x @ W↓ᵀ) or original x.
    expert_in = x_flat @ latent_down_weight.T if latent_down_weight is not None else x_flat
    output = torch.zeros(
        T, expert_in.shape[-1], dtype=x_flat.dtype, device=x_flat.device
    )
    for t in range(T):
        for slot in range(n_routed_experts_per_token):
            e = top_indices[t, slot].item()
            w = top_weights[t, slot]
            if expert_gate_up is not None:
                gate_up = expert_in[t] @ expert_gate_up[e].T
                if expert_gate_up_bias is not None:
                    gate_up = gate_up + expert_gate_up_bias[e]
                g, u = gate_up.chunk(2, dim=-1)
                hidden = GATED_ACTIVATIONS_REFS[activation](g, u)
            else:
                up = expert_in[t] @ expert_up[e].T
                if expert_up_bias is not None:
                    up = up + expert_up_bias[e]
                hidden = UNGATED_ACTIVATIONS_REFS[activation](up)
            out_t = hidden @ expert_down[e].T
            if expert_down_bias is not None:
                out_t = out_t + expert_down_bias[e]
            output[t] = output[t] + w * out_t

    if latent_up_weight is not None:
        output = output @ latent_up_weight.T  # ℓ → d

    # Switch Transformer aux loss: E * sum_i (f_i * P_i), f_i = #tokens routed to i / T.
    expert_counts = torch.zeros(E, dtype=torch.long, device=x.device)
    ones = torch.ones_like(top_indices.flatten())
    expert_counts.scatter_add_(0, top_indices.flatten(), ones)
    f = expert_counts.to(x.dtype) / T
    P = router_probs.mean(0)
    aux_loss = E * (f * P).sum()

    return output.view(B, S, D), aux_loss
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/fast/layers/test_mlp.py` (module already imports `SparseMoEBlock`, `sparse_moe_block_ref`, `COMPOUND_DTYPES`, `torch`, `pytest`):

```python
def test_sparse_moe_latent_off_by_default():
    block = SparseMoEBlock(
        d_model=64,
        intermediate_size=32,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
    )
    assert block.latent_moe is False
    assert block.latent_down_proj is None
    assert block.latent_up_proj is None
    # expert I/O axis is d_model when latent is off
    assert block.expert_down.shape == (4, 64, 32)
    assert block.expert_gate_up.shape == (4, 2 * 32, 64)


def test_sparse_moe_latent_weight_shapes():
    d_model, inter, E, k, ell = 64, 32, 4, 2, 16
    block = SparseMoEBlock(
        d_model=d_model,
        intermediate_size=inter,
        n_routed_experts=E,
        n_routed_experts_per_token=k,
        latent_moe=True,
        latent_dim=ell,
    )
    assert block.latent_down_proj.weight.shape == (ell, d_model)
    assert block.latent_up_proj.weight.shape == (d_model, ell)
    assert block.expert_gate_up.shape == (E, 2 * inter, ell)
    assert block.expert_down.shape == (E, ell, inter)
    # router still scores at d_model
    assert block.router.gate.weight.shape == (E, d_model)


def test_sparse_moe_latent_output_shape():
    d_model, ell = 64, 16
    block = SparseMoEBlock(
        d_model=d_model,
        intermediate_size=32,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
        latent_moe=True,
        latent_dim=ell,
    )
    block.eval()
    x = torch.randn(2, 8, d_model)
    out, aux = block(x)
    assert out.shape == (2, 8, d_model)
    assert aux.ndim == 0


@pytest.mark.parametrize("gated,activation", [(True, "silu"), (False, "gelu")])
@pytest.mark.parametrize("dtype,atol", COMPOUND_DTYPES)
def test_sparse_moe_latent_matches_ref(gated, activation, dtype, atol):
    torch.manual_seed(0)
    d_model, inter, E, k, ell = 64, 32, 4, 2, 16
    block = SparseMoEBlock(
        d_model=d_model,
        intermediate_size=inter,
        n_routed_experts=E,
        n_routed_experts_per_token=k,
        dropout=0.0,
        gated=gated,
        activation=activation,
        router_score_fn="softmax",
        latent_moe=True,
        latent_dim=ell,
    )
    with torch.no_grad():
        w1 = block.expert_gate_up if gated else block.expert_up
        torch.nn.init.normal_(w1, std=0.02)
        torch.nn.init.normal_(block.expert_down, std=0.02)
        torch.nn.init.normal_(block.latent_down_proj.weight, std=0.02)
        torch.nn.init.normal_(block.latent_up_proj.weight, std=0.02)
    block.to(dtype)
    block.eval()

    x = torch.randn(2, 8, d_model, dtype=dtype)
    out, aux = block(x)
    out_ref, aux_ref = sparse_moe_block_ref(
        x,
        block.router.gate.weight,
        block.expert_down,
        n_routed_experts_per_token=k,
        activation=activation,
        normalize=True,
        expert_gate_up=block.expert_gate_up if gated else None,
        expert_up=None if gated else block.expert_up,
        latent_down_weight=block.latent_down_proj.weight,
        latent_up_weight=block.latent_up_proj.weight,
    )
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)
    assert torch.allclose(aux, aux_ref, atol=atol)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/fast/layers/test_mlp.py -k "latent" -v`
Expected: FAIL — `SparseMoEBlock.__init__` rejects unexpected kwarg `latent_moe` (`TypeError`).

- [ ] **Step 4: Update `SparseMoEBlock.__init__`**

In `src/layers/mlp.py`, add `latent_moe`/`latent_dim` to the signature (after `dropout: float = 0.0,`):

```python
        latent_moe: bool = False,
        latent_dim: int | None = None,
```

After the existing attribute assignments (right after `self.act_fn = registry[activation]`), add:

```python
        self.latent_moe = latent_moe
        self.latent_dim = latent_dim
        expert_dim = latent_dim if latent_moe else d_model
```

Replace the expert weight allocations so their `d_model` axis uses `expert_dim`:

```python
        if gated:
            self.expert_gate_up = nn.Parameter(
                torch.empty(n_routed_experts, 2 * intermediate_size, expert_dim)
            )
            self.expert_gate_up_bias = (
                nn.Parameter(torch.zeros(n_routed_experts, 2 * intermediate_size))
                if bias
                else None
            )
        else:
            self.expert_up = nn.Parameter(
                torch.empty(n_routed_experts, intermediate_size, expert_dim)
            )
            self.expert_up_bias = (
                nn.Parameter(torch.zeros(n_routed_experts, intermediate_size))
                if bias
                else None
            )
        self.expert_down = nn.Parameter(
            torch.empty(n_routed_experts, expert_dim, intermediate_size)
        )
        self.expert_down_bias = (
            nn.Parameter(torch.zeros(n_routed_experts, expert_dim)) if bias else None
        )
```

Then, right after `self.expert_dropout = nn.Dropout(dropout)`, add the projections:

```python
        # LatentMoE (arXiv:2601.18089): shared down/up projections wrap the routed
        # experts, which run in latent space ℓ. Router + shared experts stay at d.
        if latent_moe:
            self.latent_down_proj = nn.Linear(d_model, latent_dim, bias=False)
            self.latent_up_proj = nn.Linear(latent_dim, d_model, bias=False)
        else:
            self.latent_down_proj = None
            self.latent_up_proj = None
```

- [ ] **Step 5: Update `SparseMoEBlock.forward`**

In `src/layers/mlp.py` `forward`, after the router call (`top_indices, top_weights, router_probs = self.router(x_flat)`), add:

```python
        tokens = self.latent_down_proj(x_flat) if self.latent_moe else x_flat
```

Change the dispatch gather from `x_flat` to `tokens`:

```python
        x_sorted = tokens[token_ids_sorted]
```

Replace the aggregation block (`output = torch.zeros_like(x_flat)` … `output = output.view(B, S, D)`) with:

```python
        output = x_flat.new_zeros(BS, expert_out.shape[-1])
        output.index_add_(0, token_ids_sorted, expert_out)

        aux_loss = None
        if self.aux_loss:
            with torch.no_grad():
                f = expert_counts.to(x.dtype) / BS  # (E,)
            probs = router_probs
            if self.router_score_fn == "sigmoid":
                probs = probs / (probs.sum(-1, keepdim=True) + 1e-9)
            P = probs.mean(0)  # (E,)
            aux_loss = E * (f * P).sum()

        if self.latent_moe:
            output = self.latent_up_proj(output)
        output = output.view(B, S, D)
        if self.shared_expert is not None:
            shared_out, _ = self.shared_expert(x)
            output = output + shared_out
        return output, aux_loss
```

(This preserves the existing aux-loss computation and shared-expert add; the only real changes are the accumulator width `expert_out.shape[-1]` and the `latent_up_proj` before `view`.)

- [ ] **Step 6: Run the latent tests**

Run: `uv run pytest tests/fast/layers/test_mlp.py -k "latent" -v`
Expected: PASS (all parametrizations).

- [ ] **Step 7: Run the full MLP suite to confirm the non-latent path is unchanged**

Run: `uv run pytest tests/fast/layers/test_mlp.py -v`
Expected: PASS — including all existing `test_sparse_moe_block_matches_ref` / HF-parity / param / flop tests (guards the off-path).

- [ ] **Step 8: Commit**

```bash
git add src/layers/mlp.py tests/fast/layers/_refs.py tests/fast/layers/test_mlp.py
git commit -m "[feat] mlp: LatentMoE mode for SparseMoEBlock (latent-space routed experts)"
```

---

### Task 3: FLOP / parameter accounting for latent mode

**Files:**
- Modify: `src/layers/mlp.py` (`SparseMoEBlock.compute_flops`, `SparseMoEBlock.compute_parameters`)
- Test: `tests/fast/layers/test_mlp.py`

**Interfaces:**
- Consumes: expert-weight layout and projections from Task 2.
- Produces: `compute_flops(...)` / `compute_parameters(...)` accept `latent_moe=False, latent_dim=None`; use `edim = latent_dim if latent_moe else d_model` for the expert term and add projection cost (`2·d_model·latent_dim` params, `4·d_model·latent_dim` flops/token) when `latent_moe`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/fast/layers/test_mlp.py`:

```python
def test_sparse_moe_latent_param_count_matches_module():
    d_model, inter, E, k, ell = 64, 32, 4, 2, 16
    block = SparseMoEBlock(
        d_model=d_model,
        intermediate_size=inter,
        n_routed_experts=E,
        n_routed_experts_per_token=k,
        latent_moe=True,
        latent_dim=ell,
    )
    counted = SparseMoEBlock.compute_parameters(
        d_model,
        intermediate_size=inter,
        n_routed_experts=E,
        n_routed_experts_per_token=k,
        latent_moe=True,
        latent_dim=ell,
    )
    actual = sum(p.numel() for p in block.parameters())
    assert counted == actual


def test_sparse_moe_latent_compute_flops_gated():
    d_model, inter, E, k, ell = 64, 32, 8, 2, 16
    f = SparseMoEBlock.compute_flops(
        d_model,
        intermediate_size=inter,
        n_routed_experts=E,
        n_routed_experts_per_token=k,
        latent_moe=True,
        latent_dim=ell,
    )
    router = 2 * d_model * E
    expert = 6 * ell * inter  # expert I/O axis is ℓ, not d_model
    proj = 4 * d_model * ell  # W↓ + W↑, once per token
    assert f == router + k * expert + proj


def test_sparse_moe_latent_flops_less_than_dense_moe():
    kwargs = dict(intermediate_size=128, n_routed_experts=8, n_routed_experts_per_token=2)
    dense = SparseMoEBlock.compute_flops(256, **kwargs)
    latent = SparseMoEBlock.compute_flops(256, latent_moe=True, latent_dim=64, **kwargs)
    assert latent < dense
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/fast/layers/test_mlp.py -k "latent_param_count or latent_compute_flops or latent_flops_less" -v`
Expected: FAIL — `compute_flops`/`compute_parameters` ignore latent (param count mismatch; flops use `d_model`, no proj term).

- [ ] **Step 3: Update `compute_flops`**

In `src/layers/mlp.py`, add `latent_moe=False, latent_dim=None` to the `compute_flops` signature (before `**_`) and rewrite its body:

```python
        d_ff = intermediate_size
        edim = latent_dim if latent_moe else d_model
        router = MoERouter.compute_flops(
            d_model, n_routed_experts, expert_bias=expert_bias
        )
        if gated:
            expert = 6 * edim * d_ff
            b = (2 * d_ff + edim) if bias else 0
        else:
            expert = 4 * edim * d_ff
            b = (d_ff + edim) if bias else 0
        proj = 4 * d_model * latent_dim if latent_moe else 0
        shared = (
            DenseMLPBlock.compute_flops(
                d_model,
                intermediate_size=n_shared_experts * d_ff,
                gated=gated,
                bias=bias,
            )
            if n_shared_experts > 0
            else 0
        )
        return router + n_routed_experts_per_token * (expert + b) + proj + shared
```

- [ ] **Step 4: Update `compute_parameters`**

Add `latent_moe=False, latent_dim=None` to the `compute_parameters` signature (before `active=False`) and rewrite its body:

```python
        d_ff = intermediate_size
        edim = latent_dim if latent_moe else d_model
        router = MoERouter.compute_parameters(
            d_model, n_routed_experts, expert_bias=expert_bias
        )
        n = n_routed_experts_per_token if active else n_routed_experts
        if gated:
            expert = 3 * d_ff * edim
            b = (2 * d_ff + edim) if bias else 0
        else:
            expert = 2 * d_ff * edim
            b = (d_ff + edim) if bias else 0
        proj = 2 * d_model * latent_dim if latent_moe else 0
        shared = (
            DenseMLPBlock.compute_parameters(
                d_model,
                intermediate_size=n_shared_experts * d_ff,
                gated=gated,
                bias=bias,
            )
            if n_shared_experts > 0
            else 0
        )
        return router + n * (expert + b) + proj + shared
```

(`proj` sits outside the `n ×` factor — the projections always run, so they count in both `active` and total.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/fast/layers/test_mlp.py -k "latent" -v`
Expected: PASS (all latent tests, including Task 2's).

- [ ] **Step 6: Commit**

```bash
git add src/layers/mlp.py tests/fast/layers/test_mlp.py
git commit -m "[feat] mlp: FLOP/param accounting for LatentMoE (edim + projections)"
```

---

### Task 4: Document LatentMoE in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (Architecture → "Layers vs. models" MoE sentence, and the MoE config note under Experiments)

**Interfaces:** none (docs only).

- [ ] **Step 1: Add the architecture note**

In `CLAUDE.md`, find the sentence in the `mlp.py` description that reads `SparseMoEBlock` … `SwiGLU = DenseMLPBlock(activation="silu", gated=True)`. Immediately after the `SparseMoEBlock` mention, add:

```
`SparseMoEBlock` also supports LatentMoE (arXiv:2601.18089) via `latent_moe: true` + `latent_dim: ℓ`: routed experts run in a compressed latent space ℓ (shared `latent_down_proj` d→ℓ before dispatch, `latent_up_proj` ℓ→d after aggregation); the router and shared experts stay at `d`, and `intermediate_size` is unchanged. Reinvest the α = d/ℓ savings manually by setting `n_routed_experts` (=αN) and `n_routed_experts_per_token` (=αK).
```

- [ ] **Step 2: Add to the MoE config example note**

In `CLAUDE.md`, find the line under Experiments: `MoE replaces \`mlp_cls: dense\` with \`mlp_cls: moe\` and adds \`mlp_kwargs: {n_routed_experts: N, n_routed_experts_per_token: K, n_shared_experts: 0, ...}\`.` Append after it:

```
For LatentMoE add `latent_moe: true, latent_dim: <ℓ>` to `mlp_kwargs` (ℓ < d_model; `latent_dim` is required when `latent_moe` is on).
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "[docs] CLAUDE.md: document LatentMoE (latent_moe/latent_dim)"
```

---

## Final verification

- [ ] Run the full fast suite: `uv run pytest tests/fast/ -q` → all pass.
- [ ] Lint: `uv run ruff check src/ tests/` and `uv run ruff format --check src/ tests/` → clean.
