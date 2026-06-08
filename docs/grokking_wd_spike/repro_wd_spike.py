"""Demo 1: why AdamW + wd slingshots when v collapses.

The grokking spike in `experiments/grokking/spike/` is hard to faithfully
reproduce in a tiny training toy because the AdamW + wd slingshot needs
several things to line up: full memorization (g_W -> exactly 0), wd erosion
slow enough that v has time to collapse to underflow, and a gradient
re-emergence pattern where m accumulates direction while v stays starved.

So instead of fighting the toy, this script isolates the mechanism in three
controlled views. All three use the SAME `lr=1e-3`, `wd=0.1`,
`betas=(0.9, 0.98)` to match the real configs in `experiments/grokking/spike/`.

  Panel A: v collapse trajectory.
      We feed Adam a long string of zero gradients and watch v decay
      geometrically at the beta2 halflife (~35 steps). Meanwhile weight
      decay shrinks theta at the much slower lr*wd halflife (~7000 steps).
      Two timescales separate.

  Panel B: AdamW's effective step as a function of v.
      Hold (m, g) fixed at typical post-memorization values. Sweep v from
      its memorization-time value (~1e-4) down to underflow (1e-30). The
      AdamW step lr * m / (sqrt(v) + eps) goes from O(lr) to O(lr/eps).
      Plot it for eps in {1e-8, 1e-12, 1e-15}. Lion's step is plotted as a
      flat line at lr -- bounded by construction, no matter what.

  Panel C: a worked numerical example.
      Show one AdamW update at memorization-time v vs collapsed-v, and the
      same step for Lion, side by side as a bar chart. This is the slingshot
      in arithmetic form.

Outputs `wd_spike.png` next to this file and prints the numeric summary.

Run:
    uv run python docs/grokking_wd_spike/repro_wd_spike.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------- shared constants (mirror experiments/grokking/spike/ configs) ---- #
LR    = 1e-3
WD    = 0.1
BETA1 = 0.9
BETA2 = 0.98


# ============ Panel A: v collapse vs theta decay ========================== #

def trace_v_and_theta_under_zero_grad(n_steps: int, v0: float, theta0: float,
                                       eps: float = 1e-8):
    """Run Adam's state with g_t = 0 for n_steps. Returns (v_traj, theta_traj)."""
    v     = torch.tensor([v0],     device=DEVICE)
    m     = torch.zeros(1,         device=DEVICE)
    theta = torch.tensor([theta0], device=DEVICE)
    v_traj, theta_traj = [], []
    for _ in range(n_steps):
        # m, v updates with g = 0
        m = BETA1 * m
        v = BETA2 * v
        step = LR * m / (v.sqrt() + eps)
        theta = theta - step - LR * WD * theta
        v_traj.append(v.clone())
        theta_traj.append(theta.clone())
    return (torch.stack(v_traj).squeeze().cpu().numpy(),
            torch.stack(theta_traj).squeeze().cpu().numpy())


# ============ Panel B: effective step vs v ================================ #

def adamw_step_size(v_arr, m: float, eps: float) -> torch.Tensor:
    """|AdamW update| = lr * |m| / (sqrt(v) + eps),  ignoring bias correction
    (it's ~1 once t >> 1/(1-beta2))."""
    return LR * abs(m) / (v_arr.sqrt() + eps)


# ============ Panel C: one-step worked example ============================ #

def one_step_examples():
    """Numerical example. m is held at the post-memorization residual value;
    we compare what AdamW does at memorization-time v (everything fine) vs
    collapsed v (slingshot)."""
    m_typical    = 1e-3
    v_memorized  = 1e-4    # typical Adam v at end of memorization
    v_collapsed  = 1e-14   # after a few thousand zero-gradient steps
    eps          = 1e-8
    adamw_normal    = LR * abs(m_typical) / (v_memorized**0.5 + eps)
    adamw_collapsed = LR * abs(m_typical) / (v_collapsed**0.5 + eps)
    lion            = LR   # bounded by lr per coord, regardless of m or v
    return {
        "AdamW @ v=1e-4 (memorized)":  adamw_normal,
        "AdamW @ v=1e-14 (collapsed)": adamw_collapsed,
        "Lion (any v)":                lion,
    }


# ============ main / plotting ============================================= #

def main():
    # --- Panel A data ---
    v_traj, theta_traj = trace_v_and_theta_under_zero_grad(
        n_steps=8000, v0=1e-4, theta0=1.0, eps=1e-8,
    )

    # v halflife = ln(2) / (1 - beta2) = ln(2) / 0.02 ~ 35 steps
    # theta halflife = ln(2) / (lr * wd) = ln(2) / 1e-4 ~ 6930 steps
    v_halflife     = torch.log(torch.tensor(2.0)).item() / (1 - BETA2)
    theta_halflife = torch.log(torch.tensor(2.0)).item() / (LR * WD)

    # --- Panel B data ---
    vs = torch.logspace(-30, -2, 200, device=DEVICE)
    epss = [1e-8, 1e-12, 1e-15]
    adamw_step_curves = {
        f"AdamW eps={e:.0e}": adamw_step_size(vs, m=1e-3, eps=e).cpu().numpy()
        for e in epss
    }

    # --- Panel C data ---
    one_step = one_step_examples()

    # --- Plot --- #
    fig = plt.figure(figsize=(13, 9))
    axA1 = plt.subplot2grid((2, 2), (0, 0))
    axA2 = axA1.twinx()
    axB  = plt.subplot2grid((2, 2), (0, 1))
    axC  = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    # Panel A.
    line_v, = axA1.plot(v_traj, label="v   (Adam, halflife ~{:.0f} steps)".format(v_halflife),
                         color="C0", lw=1.4)
    axA1.set_yscale("log")
    axA1.set_ylabel("v", color="C0")
    axA1.tick_params(axis="y", labelcolor="C0")
    axA1.set_xlabel("step (gradient = 0 throughout)")
    line_th, = axA2.plot(theta_traj,
                         label="theta  (wd, halflife ~{:.0f} steps)".format(theta_halflife),
                         color="C1", lw=1.4)
    axA2.set_ylabel("theta", color="C1")
    axA2.tick_params(axis="y", labelcolor="C1")
    axA1.set_title(
        f"A.  v collapses ~200x faster than theta\n"
        f"    (beta2 timescale vs lr*wd timescale)"
    )
    axA1.legend(handles=[line_v, line_th], fontsize=8, loc="lower left")
    axA1.grid(alpha=0.3)

    # Panel B.
    vs_np = vs.cpu().numpy()
    for name, curve in adamw_step_curves.items():
        axB.plot(vs_np, curve, label=name, lw=1.4)
    axB.axhline(LR, color="C3", lw=1.4, ls="--",
                label="Lion: lr (bounded by construction)")
    axB.axvline(1e-4, color="gray", lw=0.6, ls=":", label="v at memorization")
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.invert_xaxis()                # smaller v -> further right (later in time)
    axB.set_xlabel("v   (decreases as quiet plateau lengthens ->)")
    axB.set_ylabel("|effective step|")
    axB.set_title("B.  AdamW step blows up as v collapses; Lion is flat at lr")
    axB.legend(fontsize=8, loc="upper left")
    axB.grid(alpha=0.3, which="both")

    # Panel C.
    names  = list(one_step.keys())
    values = [one_step[k] for k in names]
    bars = axC.bar(names, values, color=["C0", "C0", "C3"])
    axC.set_yscale("log")
    axC.set_ylabel("|effective step| at one update (m=1e-3, lr=1e-3, eps=1e-8)")
    axC.set_title("C.  One update.  AdamW slingshots ~10000x when v has collapsed.  Lion does not.")
    axC.grid(alpha=0.3, axis="y", which="both")
    for bar, v in zip(bars, values):
        axC.text(bar.get_x() + bar.get_width() / 2, v * 1.5,
                 f"{v:.2e}", ha="center", fontsize=10)

    fig.suptitle(
        f"Demo 1: the AdamW + wd slingshot, in three views (lr={LR}, wd={WD}, betas={BETA1, BETA2})",
        fontsize=12,
    )
    fig.tight_layout()
    out = Path(__file__).with_name("wd_spike.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}\n")

    # Numeric summary.
    print("Panel A — timescales:")
    print(f"  v halflife     = ln(2) / (1 - beta2) = {v_halflife:.1f} steps")
    print(f"  theta halflife = ln(2) / (lr * wd)   = {theta_halflife:.1f} steps")
    print(f"  ratio          = {theta_halflife / v_halflife:.1f}x slower")
    print()
    print("Panel B — AdamW step at v = 1e-14, m = 1e-3:")
    for e in epss:
        s = LR * 1e-3 / (1e-14**0.5 + e)
        print(f"  eps = {e:.0e}: step = {s:.3e}   ({s / LR:.1e} x lr)")
    print(f"  Lion (any v):    step = {LR:.3e}   (= lr)")
    print()
    print("Panel C — one-step example (m=1e-3, lr=1e-3, eps=1e-8):")
    for k, v in one_step.items():
        print(f"  {k:35s}: {v:.3e}   ({v / LR:.1e} x lr)")


if __name__ == "__main__":
    main()
