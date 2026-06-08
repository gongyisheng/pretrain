# AdamW + weight decay spikes in grokking — and why label smoothing fixes them

In `experiments/grokking/spike/`, a tiny Qwen3 (~1M params) trained on `(a - b) mod 97` with `AdamW + wd=0.1 + cross_entropy_fp64` still produces large loss/grad spikes long after the model has memorized the training set. Switching to `cross_entropy_fp64` (which Liu et al. 2025 propose as the cure for fp32 softmax-saturation slingshots) does **not** remove these spikes. But two other interventions do:

1. switching the optimizer to **Lion** with the same `wd`;
2. adding **label smoothing** as small as `ε = 1e-5` to AdamW with the same `wd`.

This doc explains why. The TL;DR:

> The grokking spike is **not** fp32 softmax saturation (Liu et al. 2025).
> It's an Adam-specific *timescale mismatch*: after memorization, the
> classifier's gradient `g_W` collapses to ~0, so Adam's `v = EMA(g²)` decays
> at `β₂` (halflife ~35 steps), while weight decay shrinks `‖W‖` at the much
> slower rate `lr·wd` (halflife ~7000 steps). Eventually `v` is many orders
> of magnitude smaller than the gradient that re-emerges when wd has eroded
> enough confidence, and the effective step `lr·m / (√v + ε)` slingshots.
> fp64 CE preserves *direction* of the recovered gradient, but the slingshot
> is driven by `1/√v`, not by gradient absorption — so fp64 doesn't help.
> Lion has no `v` (its step is bounded by `lr` per coord); label smoothing
> pins logits to a finite optimum, so `g_W` never collapses in the first
> place and `v` stays healthy.

This directory contains two minimal repros on a mod-`p=23` addition task — `repro_wd_spike.py` (Demo 1) and `repro_ls_fix.py` (Demo 2) — and the figures they produce.

---

## Background — Liu et al. 2025 and why it isn't the whole story

[Liu et al. 2025, *Mitigating Loss Spikes in LLM Pretraining*](https://arxiv.org/abs/2605.06152) identify a precision failure mode for cross-entropy: in high-confidence regimes (`p_correct ≈ 1` to within fp32 precision, ~1e-7), the softmax rounds the correct-class probability to exactly 1.0, so the correct-class gradient `(p_correct − 1)` rounds to **exactly zero**. The other-class gradients keep their nonzero values, breaking the class-grad zero-sum identity `Σ_k g_k = 0` and producing a directional bias that drives parameter drift ("slingshot").

The fix Liu et al. propose: compute softmax and CE in fp64. That gives an absolute precision of ~1e-16, so `1 − p_correct` doesn't round to zero unless logit gaps exceed roughly 35 (since `log(1e-16) ≈ −37`). We did adopt that fix — `src/training/loss.py` exposes `cross_entropy_fp64`.

But on the grokking spike sweep, **fp32 CE and fp64 CE produce the same spike pattern**. That tells us the trigger isn't fp32 saturation. The trigger has to be something Adam-specific that survives in fp64 — and that something is the `v` collapse described below.

## The mechanism

The model has two reasons to change its parameters: the **data gradient** and **weight decay**.

| | rate | timescale (for `lr=1e-3`, `β₂=0.98`, `wd=0.1`) |
|---|---|---|
| data gradient when memorized | ‖g‖ shrinks toward 0 | falls to noise once `p_correct ≈ 1` |
| AdamW's `v = EMA(g²)` | decays at `β₂` | half-life `ln(2)/(1-β₂) ≈ 35` steps |
| `‖θ‖` under wd alone | decays at `1 − lr·wd` | half-life `ln(2)/(lr·wd) ≈ 7000` steps |

After the model memorizes the training set:

1. `g_W ≈ 0` for the classifier head W (and similar for token embeddings on memorized examples).
2. With `g_W ≈ 0`, AdamW's `v_W = β₂·v_W + (1-β₂)·g_W²` collapses geometrically with halflife `~35` steps. After a few hundred steps `v_W` can be 5–10 orders of magnitude smaller than its memorization-time value.
3. WD doesn't see `g_W` at all (it's decoupled). It keeps shrinking `W` at `lr·wd·W` per step. After ~7000 steps `‖W‖` has roughly halved.
4. Smaller `W` → smaller logit gap → eventually `p_correct` falls enough that `g_W` re-emerges.
5. The re-emerged gradient lands on coordinates where `v` is tiny → effective step `lr · m / (√v + ε)` is approximately `lr · m / ε`, three to five orders of magnitude larger than during memorization.
6. The single huge step overshoots, train loss spikes, the network reorganizes, and the cycle restarts.

**This is fully Adam-specific.** It needs the second-moment normalization `1/√v` and it needs `v` to have a much faster decay timescale than `θ`. Lion replaces `m / (√v + ε)` with `sign(β₁·m + (1−β₁)·g)`, which is bounded by 1 per coord. Step size is bounded by `lr` regardless of how small the data signal has gotten. So the same `wd` drift just smoothly shrinks `W` without ever generating a slingshot step. (Lion does have its own pathologies — sign updates are coarse and unbounded under sustained gradient direction — but it cannot manufacture an overshoot from a *small* signal the way AdamW can.)

**Why fp64 doesn't fix this.** Liu et al.'s correction restores the **direction** of `g_W` when `p_correct` is close to 1. But our spike fires from `1/√v`, not from a wrong gradient direction. fp64 changes `g_W` from "rounded to zero" to "correctly tiny" — `~1e-12` instead of `0`. Either way, `v ≈ EMA(g²)` is going to be tiny enough that when wd drift eventually re-energizes `g_W`, the ratio `m/√v` blows up. The fp64 fix is necessary for one failure mode and orthogonal to this one.

## Why label smoothing fixes the spike

Standard CE with one-hot targets has its minimum at `p_correct → 1`, which is reached only in the limit `z_correct − z_other → ∞`. So the optimization keeps pushing logits apart indefinitely, and the only thing balancing wd's pull is "memorization is good enough" — i.e. `g_W ≈ 0` once `p_correct` is essentially 1 in floating-point. That's exactly the regime where `v` collapses.

Label smoothing with parameter `ε` replaces the one-hot target with `(1 − ε)·one_hot + (ε/K)·uniform`. The minimum of the smoothed CE now sits at a **finite** softmax distribution: `p_correct* = 1 − ε + ε/K`, `p_other* = ε/K`. That corresponds to a finite logit gap

```
z_correct* − z_other* = log( (1 − ε + ε/K) / (ε/K) )  ≈  log(K/ε)   for small ε.
```

For our subtraction task with `K = vocab_size ≈ 100` and `ε = 1e-5`, that's `z_gap* ≈ log(1e7) ≈ 16`. As long as `wd` pushes `W` toward zero, CE actively pushes it back to maintain that finite gap. The result:

- `g_W` does not collapse to zero — it settles at the equilibrium where wd pull equals CE pushback.
- `v_W = EMA(g_W²)` settles at a healthy nonzero value.
- The effective step `lr · m / (√v + ε)` stays in the `O(lr)` regime forever.
- No slingshot.

In the toy here `K=23`, `ε=1e-3` is needed to see the effect clearly (the logit gap target drops from ~`∞` to about 5). At the real `K=100` scale used in the grokking experiments, `ε=1e-5` already gives a target gap of ~16, which is comfortably below the gap the model otherwise drives logits to before wd erosion catches up.

## What the repros show

Both scripts are tiny self-contained experiments on `c = (a + b) mod 23`, training an `Embedding(p) × 2 → Linear(d → p)` model on a 30% subset (memorization regime) with `wd=0.3` for 40k steps (`wd` chosen so the model memorizes within 5k steps and then drifts).

- `repro_wd_spike.py` sweeps the **optimizer** at fixed loss (Demo 1 below): AdamW+CE(fp32), AdamW+CE_fp64, Lion+CE(fp32).
- `repro_ls_fix.py` sweeps **label smoothing** at fixed optimizer (Demo 2 below): AdamW+CE_fp64 with `ls ∈ {0, 1e-5, 1e-3}`.

Each script logs at every step:

| metric | what to watch |
|---|---|
| train loss | the would-be spike location |
| `‖g_W‖` | data signal on the head — collapses after memorization in plain CE, holds at a nonzero floor with label smoothing |
| `‖v_W‖` (AdamW only) | the slingshot precondition — collapses with plain CE, stays healthy with label smoothing |
| `max |lr·m/(√v + ε)|` | what each step actually does to the largest-affected coord |
| `‖W‖` | wd's steady erosion |
| `p_correct`, `logit_gap` | confidence; label smoothing pins these to finite equilibria |

### Demo 1 — `wd_spike.png`

Variants: `AdamW + CE (fp32)`, `AdamW + CE_fp64`, `Lion + CE (fp32)`, all at `wd=0.3`.

What you see in the `||v|| of W` panel (top-right): the two AdamW lines (orange/blue) sit on top of each other and crater from `~1e-5` to `~2e-14` over the first 12k steps. **fp32 vs fp64 CE makes no visible difference to the v trajectory.** Lion has no `v` at all (green is NaN).

The `logit_gap` panel (bottom-left) shows that AdamW with one-hot CE pushes the gap to ~15 and holds it there — exactly the regime where `g_W ≈ 0` and `v` decays freely. Lion (with the same wd) holds a gap of ~5 because its bounded sign steps can't push logits arbitrarily far in finite time.

In this toy, the eps floor (1e-8) doesn't fire a *full* slingshot inside 40k steps — the toy is too clean and `‖W‖` doesn't erode far enough for `g_W` to re-emerge. The real grokking run does fire it because (a) `wd=0.1` runs for hundreds of thousands of steps in a much larger logit space and (b) other parameter blocks (embeddings, attention weights) all participate in the same v-collapse. But the **precondition is in the same panel of this plot**: `‖v‖` falling many orders of magnitude below its memorization-time value, with fp64 CE making no difference.

### Demo 2 — `ls_fix.png`

Variants: `AdamW + CE_fp64` at `ls ∈ {0, 1e-5, 1e-3}`, same `wd=0.3`.

In the `||v|| of W` panel: `ls=0` (orange) and `ls=1e-5` (blue) both collapse to `~2e-14` — `ε=1e-5` is too weak at `K=23` to move the target logit gap meaningfully. But `ls=1e-3` (green) **holds v at `~3e-11`**, three orders of magnitude higher.

The `logit_gap` panel makes the why obvious: `ls=1e-3` pins the gap at `log(K/ε) = log(23000) ≈ 10`, while one-hot CE drives the gap to 15+. With a finite target gap, the head sees a steady-state nonzero `g_W`, `v_W` never gets the chance to crater, and the slingshot path is closed.

For the real grokking run (`K ≈ 100`), `ls=1e-5` is already enough because `log(100/1e-5) = log(1e7) ≈ 16` — close enough to the un-smoothed steady gap that the head doesn't get pushed into the v-collapse regime in the first place.

### Demo 3 — `softmax_collapse.png` — the paper's own slingshot, at `wd=0`

`repro_softmax_collapse.py` is a direct reproduction of the **Softmax Collapse → Numerical Feature Inflation → Slingshot** chain that Liu et al. 2025 establish in [arXiv:2605.06152](https://arxiv.org/abs/2605.06152). It exists because Demos 1–2 isolate a *different* failure mode (the AdamW + wd `v`-collapse slingshot) and a reader could reasonably ask "what's the slingshot Liu et al. were originally talking about, then?". The answer is below, and it's at `wd = 0`.

Setup mirrors Appendix D.1: modular **division** `c = (a · b⁻¹) mod 97`, 50/50 train/val split, Adam (not AdamW) with `lr=1e-3, betas=(0.9, 0.999), wd=0`, LN disabled, no label smoothing. To keep the toy under a minute, we use a 2-hidden-layer ReLU MLP head (in place of the paper's LN-disabled 2-layer transformer; the SC signature comes from the classifier W, which is identical in both) and 20k steps at full batch (vs the paper's 100k at batch=512). Two variants run on the same seed and same model: **Adam + CE in fp32**, and **Adam + CE in fp64** (fp64 cast on the logits only, just for the loss — the exact mitigation from Sec 4.2 / Figure 1a).

What the panels show:

| panel | what it shows |
|---|---|
| `train loss (fp32-evaluated)` | fp32 develops late-phase slingshot spikes (peak ~2.7 in our run); fp64 stays at floor ~0 |
| `SC fraction` | fraction of train samples with `z_correct − max_other > 23 ln 2 ≈ 15.94`. Climbs to ~1.0 in both — SC is the *precondition*, not the spike itself |
| `‖W_G‖`, `‖μ_G‖` | the two quantities Theorem 3.7 predicts grow exponentially under NFI. In our run fp32 reaches `‖μ_G‖ ≈ 564` while fp64 stays at ~83 |
| `cos(W_G, μ_G)` | the NFI signature: fp32 settles near `−0.92` (anti-parallel); fp64 stays near `0` |
| `max sample margin` | sanity check that we cross the fp32 SC threshold |

The script also prints a synthetic one-step example with `K=97` and a logit margin of 25 that makes the mechanism falsifiably concrete:

- **fp32**: `p_correct = 1.0` exactly, so `g_correct = ŷ − 1 = 0` *exactly*. The K−1 incorrect-class gradients survive at `~1.4 × 10⁻¹¹` each, so `Σ_k g_k ≈ +1.3 × 10⁻⁹ ≠ 0` — the per-sample zero-sum (Eq. 4 of the paper) is broken, and `W_G` drifts.
- **fp64**: same logits, `p_correct = 0.9999…9986667`, so `g_correct = −1.3 × 10⁻⁹` (correctly tiny). `Σ_k g_k ≈ −1.6 × 10⁻¹⁵` — zero to machine epsilon, no drift.

Two important caveats for this toy:

1. **Val accuracy is ~1% in both runs at 20k steps.** Grokking on mod-97 needs ~10⁵ steps even in the paper's transformer; our 20k MLP run only finishes memorization. The phenomenon being reproduced is SC/NFI (a *precision* artifact that happens to coincide with grokking in the original setting), not the grokking transition itself.
2. **The fp32 run reaches `SC fraction = 1` and the NFI signature (`cos → −1`, `‖μ_G‖` inflation) but the loss spikes here are smaller and rarer than the paper's `~100`-scale spikes** — the MLP head and the smaller step budget both reduce the gradient re-emergence probability per step. The diagnostic panels match the paper's mechanism even when the loss-scale spike is softer.

## How to run

```bash
uv run python docs/grokking_wd_spike/repro_wd_spike.py            # produces wd_spike.png
uv run python docs/grokking_wd_spike/repro_ls_fix.py              # produces ls_fix.png
uv run python docs/grokking_wd_spike/repro_softmax_collapse.py    # produces softmax_collapse.png
```

Demos 1–2 take ~30s on CPU. Demo 3 takes ~1 min on a CUDA box, longer on CPU. Each prints a numeric summary per variant.

## Files in this directory

- `README.md` — this file.
- `repro_wd_spike.py` — Demo 1: AdamW(fp32) vs AdamW(fp64) vs Lion at fixed wd.
- `repro_ls_fix.py` — Demo 2: AdamW+CE_fp64 with `ls ∈ {0, 1e-5, 1e-3}` at fixed wd.
- `repro_softmax_collapse.py` — Demo 3: Adam+CE(fp32) vs Adam+CE(fp64) at `wd=0`, reproducing Liu et al. 2025's SC → NFI → Slingshot chain.
- `wd_spike.png` — Demo 1 plot: fp64 CE doesn't fix the v-collapse; Lion has no v.
- `ls_fix.png` — Demo 2 plot: label smoothing breaks the v-collapse precondition.
- `softmax_collapse.png` — Demo 3 plot: fp32 NFI (W_G, μ_G inflate; cosine → −1) vs the fp64-CE fix.

## Related references

- Liu et al. 2025, *Mitigating Loss Spikes in LLM Pretraining* — [arXiv:2605.06152](https://arxiv.org/abs/2605.06152). The fp32 → fp64 CE fix used in our `cross_entropy_fp64` loss.
- Chen et al. 2023, *Symbolic Discovery of Optimization Algorithms* (Lion) — [arXiv:2302.06675](https://arxiv.org/abs/2302.06675). The optimizer that, by construction, can't slingshot.
- Power et al. 2022, *Grokking* — [arXiv:2201.02177](https://arxiv.org/abs/2201.02177). The original setting where this spike pattern shows up under strong weight decay.
- `experiments/grokking/spike/` — the source configs the toy is mirroring, and the W&B logs that motivated this writeup.
