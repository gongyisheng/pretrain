"""Per-parameter optimizer state analysis for debugging loss spikes.

Loads a single checkpoint and shows per-parameter Adam statistics:
  - grad_norm   : ||m_hat|| = ||m / (1 - beta1^t)||   — debiased gradient signal
  - noise       : mean(sqrt(v_hat)) = mean(sqrt(v / (1-beta2^t))) — gradient noise scale
  - snr         : mean(m_hat^2 / (v_hat + eps))        — signal-to-noise ratio (low = noisy gradient)
  - eff_step    : ||m_hat / (sqrt(v_hat) + eps)||       — effective Adam update norm (proxy for param change magnitude)

Usage:
    python scripts/debug_optim.py --ckpt <checkpoint.pt>
    python scripts/debug_optim.py --ckpt checkpoints/attn_res/gpt2_d512_l12/step_15000.pt
    python scripts/debug_optim.py --ckpt checkpoints/attn_res/gpt2_d512_l12/step_15000.pt --sort eff_step --top 20
"""
import argparse
import math
import torch


def load(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def param_group(name: str) -> str:
    if "emb" in name:
        return "embed"
    if any(t in name for t in ("ln", "norm", "layernorm", "rmsnorm")):
        return "norm/scale"
    if any(t in name for t in ("attn_res_proj", "mlp_res_proj")):
        return "attn_res_proj"
    if any(t in name for t in ("q_proj", "k_proj", "v_proj", "out_proj", "attn")):
        return "attention"
    if any(t in name for t in ("fc", "ffn", "mlp", "gate_proj", "up_proj", "down_proj")):
        return "ffn"
    if "lm_head" in name:
        return "lm_head"
    return "other"


def optim_stats(state: dict, beta1: float = 0.9, beta2: float = 0.95, eps: float = 1e-8) -> dict:
    """Compute per-parameter Adam statistics from optimizer state."""
    if "exp_avg" not in state or "exp_avg_sq" not in state:
        return None

    m = state["exp_avg"].float().flatten()
    v = state["exp_avg_sq"].float().flatten()
    t = state.get("step", None)
    if isinstance(t, torch.Tensor):
        t = t.item()

    # Bias correction
    if t is not None and t > 0:
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t
    else:
        bc1 = bc2 = 1.0

    m_hat = m / bc1
    v_hat = v / bc2

    grad_norm = m_hat.norm().item()
    noise     = v_hat.sqrt().mean().item()
    snr       = (m_hat ** 2 / (v_hat + eps)).mean().item()
    eff_step  = (m_hat / (v_hat.sqrt() + eps)).norm().item()

    return {
        "grad_norm": grad_norm,
        "noise":     noise,
        "snr":       snr,
        "eff_step":  eff_step,
        "step":      t,
    }


def main():
    parser = argparse.ArgumentParser(description="Per-parameter Adam optimizer state analysis")
    parser.add_argument("--ckpt",  required=True, help="Checkpoint file (.pt)")
    parser.add_argument("--sort",  default="eff_step",
                        choices=["grad_norm", "noise", "snr", "eff_step", "name"],
                        help="Sort by this stat (default: eff_step)")
    parser.add_argument("--top",   type=int, default=0,
                        help="Show only top N rows (default: 0 = all)")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps",   type=float, default=1e-8)
    args = parser.parse_args()

    print(f"Loading {args.ckpt} ...")
    ckpt = load(args.ckpt)
    global_step = ckpt.get("step", "?")

    model_sd  = ckpt.get("model", {})
    opt_sd    = ckpt.get("optimizer", {})

    if "state" not in opt_sd:
        print("[!] No optimizer state found in checkpoint.")
        return

    param_names = list(model_sd.keys())
    opt_state   = opt_sd["state"]

    rows = []
    for idx, param_state in opt_state.items():
        name = param_names[idx] if idx < len(param_names) else f"param_{idx}"
        s = optim_stats(param_state, beta1=args.beta1, beta2=args.beta2, eps=args.eps)
        if s is None:
            continue
        rows.append((name, s))

    if args.sort == "name":
        rows.sort(key=lambda r: r[0])
    else:
        rows.sort(key=lambda r: r[1][args.sort], reverse=(args.sort != "snr" or True))
        # always descending: high eff_step/grad_norm/noise = interesting; high snr = good (show low first)
        if args.sort == "snr":
            rows.sort(key=lambda r: r[1]["snr"])  # ascending: worst SNR first

    display = rows[:args.top] if args.top > 0 else rows

    t_sample = rows[0][1]["step"] if rows else "?"
    print(f"\nGlobal step: {global_step}  |  optimizer step: {t_sample}  |  {len(rows)} parameters")
    print(f"beta1={args.beta1}  beta2={args.beta2}  eps={args.eps}")
    print(f"Sorted by: {args.sort}{'  (ascending — worst SNR first)' if args.sort == 'snr' else '  (descending)'}\n")

    hdr = (f"  {'parameter':<55} {'grad_norm':>10} {'noise':>10} {'snr':>10} {'eff_step':>10}  group")
    print(hdr)
    print("  " + "-" * 115)
    for name, s in display:
        g = param_group(name)
        print(f"  {name:<55} {s['grad_norm']:10.5f} {s['noise']:10.5f} {s['snr']:10.5f} {s['eff_step']:10.5f}  {g}")

    if args.top > 0 and args.top < len(rows):
        print(f"\n  ... ({len(rows) - args.top} more rows hidden, use --top 0 to show all)")

    # Summary: global effective grad norm (approximate)
    total_eff_sq = sum(r[1]["eff_step"] ** 2 for r in rows)
    print(f"\n  Global eff_step norm (sqrt sum of squares): {math.sqrt(total_eff_sq):.5f}")
    print()


if __name__ == "__main__":
    main()
