"""Per-weight statistics for a single checkpoint.

Usage:
    uv run python scripts/inspect_weights.py --ckpt <checkpoint.pt>
    uv run python scripts/inspect_weights.py --ckpt <checkpoint.pt> --mode svd --sort srank
    uv run python scripts/inspect_weights.py --ckpt checkpoints/grokking/qwen3_1m_sub_wd0.5_ce_fp64_lion_ls1e-5/step_30000.pt
    uv run python scripts/inspect_weights.py --ckpt checkpoints/gqa/qwen3_57m_kv8/step_5000.pt --mode svd

Modes:
    basic  max / min / mean / std / rms / p1 / p10 / p50 / p90 / p99 for every float tensor.
    svd    smax / smin / srank / pr / erank / enrg90 for every 2D float tensor.
"""
import argparse
import math
import torch


PCTL_TENSOR = torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99])
PCTL_KEYS = ["p1", "p10", "p50", "p90", "p99"]
BASIC_KEYS = ["min"] + PCTL_KEYS + ["max", "mean", "std", "rms"]
SVD_KEYS = ["smax", "smin", "srank", "pr", "erank", "enrg90", "entropy"]


def load(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def basic_stats(t: torch.Tensor) -> dict:
    f = t.float().flatten()
    nan_mask = torch.isnan(f)
    valid = f[~nan_mask]
    nan_count = nan_mask.sum().item()
    shape_str = "x".join(str(d) for d in t.shape)
    nan_flag = " !" if nan_count > 0 else ""
    out = {"nans": nan_count, "trailing": f"{nan_count:>5} {shape_str:>11}{nan_flag}"}
    if valid.numel() == 0:
        for k in BASIC_KEYS:
            out[k] = float("nan")
        return out
    out["max"] = valid.max().item()
    out["min"] = valid.min().item()
    out["mean"] = valid.mean().item()
    out["std"] = valid.std().item()
    out["rms"] = valid.pow(2).mean().sqrt().item()
    q = torch.quantile(valid, PCTL_TENSOR.to(valid.dtype))
    for k, v in zip(PCTL_KEYS, q.tolist()):
        out[k] = v
    return out


def svd_stats(t: torch.Tensor) -> dict:
    s = torch.linalg.svdvals(t.float())  # descending singular values
    s = s[s > 0]
    shape_str = "x".join(str(d) for d in t.shape)
    out = {"trailing": f"{shape_str:>11}"}
    if s.numel() == 0:
        for k in SVD_KEYS:
            out[k] = float("nan")
        return out
    energy = s.pow(2)
    total = energy.sum()
    p = energy / total
    out["smax"] = s[0].item()
    out["smin"] = s[-1].item()
    entropy = -(p * p.log()).sum()                             # Shannon entropy of σ² spectrum (nats)
    n = s.numel()
    out["srank"] = (total / energy[0]).item()                  # stable rank (top-heavy, rank-1 canary)
    out["pr"] = (total.pow(2) / energy.pow(2).sum()).item()    # participation ratio = (Σσ²)²/Σσ⁴ (bulk effective dim)
    out["erank"] = torch.exp(entropy).item()                   # entropy-based effective rank = exp(entropy)
    out["enrg90"] = int((energy.cumsum(0) / total < 0.90).sum().item()) + 1
    out["entropy"] = (entropy.item() / math.log(n)) if n > 1 else 0.0  # normalized to [0,1]
    return out


def fmt_num(x: float) -> str:
    a = abs(x)
    if a != 0 and (a < 1e-1 or a >= 1e5):  # avoid long leading-zero / huge fixed strings
        return f"{x:.2e}"
    return f"{x:.5g}"  # 5 significant figures


def main():
    parser = argparse.ArgumentParser(description="Per-weight statistics for a checkpoint")
    parser.add_argument("--ckpt", required=True, help="Checkpoint file (.pt)")
    parser.add_argument("--mode", default="basic", choices=["basic", "svd"],
                        help="basic: element-wise stats; svd: spectral stats on 2D weights (default: basic)")
    parser.add_argument("--sort", default="name",
                        help="Sort by this stat or 'name' (default: name)")
    parser.add_argument("--top",  type=int, default=0,
                        help="Show only top N rows (default: 0 = all)")
    parser.add_argument("--filter", default="",
                        help="Keep only parameters whose name contains this substring")
    args = parser.parse_args()

    if args.mode == "svd":
        keys, stat_fn, val_fmt = SVD_KEYS, svd_stats, fmt_num  # cond/smin span many orders
        trailing_hdr = f"{'shape':>11}"
        qualifies = lambda name, t: t.is_floating_point() and t.ndim == 2 and not name.startswith("rope.")
    else:
        keys, stat_fn, val_fmt = BASIC_KEYS, basic_stats, lambda x: f"{x:.5f}"
        trailing_hdr = f"{'nans':>5} {'shape':>11}"
        qualifies = lambda name, t: t.is_floating_point()
    if args.sort not in keys + ["nans", "name"]:
        parser.error(f"--sort must be one of {keys + ['nans', 'name']} for mode {args.mode}")

    print(f"Loading {args.ckpt} ...")
    ckpt = load(args.ckpt)
    step = ckpt.get("step", "?")
    model_sd = ckpt.get("model", ckpt)  # support bare state_dict too

    rows = []
    for name, tensor in model_sd.items():
        name = name.removeprefix("_orig_mod.")  # strip torch.compile wrapper prefix
        if not isinstance(tensor, torch.Tensor) or not qualifies(name, tensor):
            continue
        if args.filter and args.filter not in name:
            continue
        rows.append((name, stat_fn(tensor)))

    if args.sort == "name":
        rows.sort(key=lambda r: r[0])
    else:
        rows.sort(key=lambda r: r[1][args.sort], reverse=(args.sort != "min"))

    display = rows[:args.top] if args.top > 0 else rows

    print(f"\nStep: {step}  |  {len(rows)} tensors  |  mode {args.mode}  |  sorted by {args.sort}\n")
    hdr_stats = " ".join(f"{c:>9}" for c in keys)
    hdr = f"  {'parameter':<50} {hdr_stats} {trailing_hdr}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, s in display:
        stat_vals = " ".join(f"{val_fmt(s[c]):>9}" for c in keys)
        print(f"  {name:<50} {stat_vals} {s['trailing']}")

    if args.top > 0 and args.top < len(rows):
        print(f"\n  ... ({len(rows) - args.top} more rows hidden, use --top 0 to show all)")

    print()


if __name__ == "__main__":
    main()
