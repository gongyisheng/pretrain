"""Demo 2: label smoothing breaks the v-collapse precondition.

Same mod-p=23 memorization toy as `repro_wd_spike.py`, but here we hold the
optimizer fixed (AdamW + CE_fp64 + wd=0.3) and sweep label smoothing
ls in {0, 1e-5, 1e-3}.

What you should see:
  - ls=0: gradient on head g_W collapses, ||v_W|| craters to ~1e-14.
  - ls=1e-5: same shape at K=23 (eps too weak to move the target logit gap).
  - ls=1e-3: g_W floors at a steady-state nonzero value, ||v_W|| stays
             several orders of magnitude higher than the un-smoothed runs.

Outputs `ls_fix.png` next to this file and prints a per-variant summary.

Run:
    uv run python docs/grokking_wd_spike/repro_ls_fix.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


P = 23
D = 32
TRAIN_FRAC = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_data():
    pairs = [(a, b, (a + b) % P) for a in range(P) for b in range(P)]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(pairs), generator=g).tolist()
    n_train = int(TRAIN_FRAC * len(pairs))
    train = [pairs[i] for i in perm[:n_train]]
    a_idx = torch.tensor([t[0] for t in train], device=DEVICE)
    b_idx = torch.tensor([t[1] for t in train], device=DEVICE)
    y = torch.tensor([t[2] for t in train], device=DEVICE)
    return a_idx, b_idx, y


A_IDX, B_IDX, Y = make_data()


class TinyMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.Ea = nn.Embedding(P, D)
        self.Eb = nn.Embedding(P, D)
        self.W = nn.Linear(D, P, bias=False)

    def forward(self, a, b):
        return self.W(self.Ea(a) + self.Eb(b))


def run(wd: float, n_steps: int, label_smoothing: float, eps: float = 1e-8):
    torch.manual_seed(42)
    model = TinyMod().to(DEVICE)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3, weight_decay=wd, betas=(0.9, 0.98), eps=eps,
        fused=(DEVICE.type == "cuda"),
    )

    # Keep diagnostics as 0-d GPU tensors to avoid per-step CPU sync; convert
    # all at once at the end of training.
    keys = ["loss", "g_norm_W", "v_norm_W", "m_norm_W",
            "W_norm", "p_correct", "eff_step_W", "logit_gap"]
    buf = {k: [] for k in keys}
    idx = torch.arange(len(Y), device=DEVICE)
    eye_mask = torch.zeros((len(Y), P), dtype=torch.bool, device=DEVICE)
    eye_mask[idx, Y] = True

    for _ in range(n_steps):
        opt.zero_grad()
        logits = model(A_IDX, B_IDX)
        loss = F.cross_entropy(logits.double(), Y, label_smoothing=label_smoothing)
        loss.backward()

        with torch.no_grad():
            Wgrad = model.W.weight.grad
            buf["loss"].append(loss.detach())
            buf["g_norm_W"].append(Wgrad.norm())
            buf["W_norm"].append(model.W.weight.detach().norm())
            probs = F.softmax(logits.detach().float(), dim=-1)
            buf["p_correct"].append(probs[idx, Y].mean())
            z = logits.detach().float()
            z_correct = z[idx, Y]
            z_other = z.masked_fill(eye_mask, float("-inf")).max(dim=-1).values
            buf["logit_gap"].append((z_correct - z_other).mean())

            st = opt.state.get(model.W.weight, {})
            v, m = st["exp_avg_sq"], st["exp_avg"]
            lr = opt.param_groups[0]["lr"]
            buf["v_norm_W"].append(v.norm())
            buf["m_norm_W"].append(m.norm())
            buf["eff_step_W"].append(
                (lr * m.abs() / (v.sqrt() + eps)).max()
            )

        opt.step()

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return {k: torch.stack(v).float().cpu().tolist() for k, v in buf.items()}


PANELS = [
    ("loss",       "train loss",                       True),
    ("g_norm_W",   "||grad W||  (data signal on head)", True),
    ("v_norm_W",   "||v|| of W",                        True),
    ("eff_step_W", "max |effective step| on W coord",  True),
    ("logit_gap",  "mean (z_correct - max z_other)",   False),
    ("p_correct",  "mean p(correct class)",            False),
]


def main():
    WD = 0.3
    STEPS = 40000

    runs = {
        "AdamW + CE_fp64, ls=0":    run(wd=WD, n_steps=STEPS, label_smoothing=0.0),
        "AdamW + CE_fp64, ls=1e-5": run(wd=WD, n_steps=STEPS, label_smoothing=1e-5),
        "AdamW + CE_fp64, ls=1e-3": run(wd=WD, n_steps=STEPS, label_smoothing=1e-3),
    }

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    for ax, (key, label, logy) in zip(axes.flat, PANELS):
        for name, h in runs.items():
            ax.plot(h[key], label=name, lw=1.0)
        ax.set_title(label)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(
        f"Demo 2: label smoothing fixes the spike at fixed wd={WD}\n"
        "ls > 0 -> finite optimal logit gap -> steady-state ||g_W|| -> v never collapses",
        fontsize=12,
    )
    fig.tight_layout()
    out = Path(__file__).with_name("ls_fix.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}\n")

    warmup = 500
    for name, h in runs.items():
        peak_loss = max(h["loss"][warmup:])
        peak_g = max(h["g_norm_W"][warmup:])
        eff = [e for e in h["eff_step_W"][warmup:] if e == e]
        peak_eff = max(eff) if eff else float("nan")
        vs = [v for v in h["v_norm_W"][warmup:] if v == v]
        min_v = min(vs) if vs else float("nan")
        print(
            f"  {name:30s} | peak loss (post-warmup) = {peak_loss:8.4f} | "
            f"peak ||g_W|| = {peak_g:7.4f} | min ||v_W|| = {min_v:.2e} | "
            f"peak |eff step| = {peak_eff:.2e}"
        )


if __name__ == "__main__":
    main()
