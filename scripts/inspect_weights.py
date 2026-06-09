"""Per-weight statistics for a single checkpoint.

Usage:
    uv run python scripts/inspect_weights.py --ckpt <checkpoint.pt>
    uv run python scripts/inspect_weights.py --ckpt <checkpoint.pt> --mode svd --sort srank
    uv run python scripts/inspect_weights.py --ckpt checkpoints/grokking/qwen3_1m_sub_wd0.5_ce_fp64_lion_ls1e-5/step_30000.pt

Modes:
    basic  max / min / mean / std / rms / p1 / p10 / p50 / p90 / p99 for every float tensor.
    svd    smax / smin / cond / srank / erank / enrg90 for every 2D float tensor.
"""
import argparse
import torch


PCTL_TENSOR = torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99])
PCTL_KEYS = ["p1", "p10", "p50", "p90", "p99"]
BASIC_KEYS = ["min"] + PCTL_KEYS + ["max", "mean", "std", "rms"]
SVD_KEYS = ["smax", "smin", "cond", "srank", "erank", "enrg90"]


def load(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def basic_stats(t: torch.Tensor) -> dict:
    f = t.float().flatten()
    nan_mask = torch.isnan(f)
    valid = f[~nan_mask]
    nan_count = nan_mask.sum().item()
    out = {"nans": nan_count, "numel": f.numel()}
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
    out = {"nans": 0, "numel": t.numel()}
    if s.numel() == 0:
        for k in SVD_KEYS:
            out[k] = float("nan")
        return out
    energy = s.pow(2)
    total = energy.sum()
    p = energy / total
    out["smax"] = s[0].item()
    out["smin"] = s[-1].item()
    out["cond"] = (s[0] / s[-1]).item()
    out["srank"] = (total / energy[0]).item()                  # stable rank
    out["erank"] = torch.exp(-(p * p.log()).sum()).item()      # entropy-based effective rank
    out["enrg90"] = int((energy.cumsum(0) / total < 0.90).sum().item()) + 1
    return out


def main():
    parser = argparse.ArgumentParser(description="Per-weight statistics for a checkpoint")
    parser.add_argument("--ckpt", required=True, help="Checkpoint file (.pt)")
    parser.add_argument("--mode", default="basic", choices=["basic", "svd"],
                        help="basic: element-wise stats; svd: spectral stats on 2D weights (default: basic)")
    parser.add_argument("--sort", default="name",
                        help="Sort by this stat or 'name' (default: name)")
    parser.add_argument("--top",  type=int, default=0,
                        help="Show only top N rows (default: 0 = all)")
    args = parser.parse_args()

    if args.mode == "svd":
        keys, stat_fn = SVD_KEYS, svd_stats
        qualifies = lambda t: t.is_floating_point() and t.ndim == 2
    else:
        keys, stat_fn = BASIC_KEYS, basic_stats
        qualifies = lambda t: t.is_floating_point()
    if args.sort not in keys + ["nans", "name"]:
        parser.error(f"--sort must be one of {keys + ['nans', 'name']} for mode {args.mode}")

    print(f"Loading {args.ckpt} ...")
    ckpt = load(args.ckpt)
    step = ckpt.get("step", "?")
    model_sd = ckpt.get("model", ckpt)  # support bare state_dict too

    rows = []
    for name, tensor in model_sd.items():
        if not isinstance(tensor, torch.Tensor) or not qualifies(tensor):
            continue
        name = name.removeprefix("_orig_mod.")  # strip torch.compile wrapper prefix
        rows.append((name, stat_fn(tensor)))

    if args.sort == "name":
        rows.sort(key=lambda r: r[0])
    else:
        rows.sort(key=lambda r: r[1][args.sort], reverse=(args.sort != "min"))

    display = rows[:args.top] if args.top > 0 else rows

    print(f"\nStep: {step}  |  {len(rows)} tensors  |  mode {args.mode}  |  sorted by {args.sort}\n")
    hdr_stats = " ".join(f"{c:>9}" for c in keys)
    hdr = f"  {'parameter':<50} {hdr_stats}  {'nans':>5}  {'numel':>8}"
    print(hdr)
    print("  " + "-" * (50 + len(keys) * 10 + 18))
    for name, s in display:
        nan_flag = "  !" if s["nans"] > 0 else ""
        stat_vals = " ".join(f"{s[c]:9.5f}" for c in keys)
        print(f"  {name:<50} {stat_vals}  {s['nans']:>5}{nan_flag}  {s['numel']:>8,}")

    if args.top > 0 and args.top < len(rows):
        print(f"\n  ... ({len(rows) - args.top} more rows hidden, use --top 0 to show all)")

    print()


if __name__ == "__main__":
    main()
