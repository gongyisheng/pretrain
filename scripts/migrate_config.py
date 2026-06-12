"""One-time migration script: rewrite YAML model: blocks from the old flat schema
to the new cls+kwargs schema. Run on each config file you want to migrate.

Usage: python scripts/migrate_config.py <path> [<path>...]
"""
import sys
import yaml


def migrate_model_block(old: dict) -> dict:
    arch = old.get("arch", "gpt2")
    is_moe = arch == "qwen3_moe" or old.get("moe_n_experts", 0) > 0

    # Infer cls values from arch
    attn_cls = "mha" if arch == "gpt2" else "gqa"
    norm_cls = "layernorm" if arch == "gpt2" else "rmsnorm"
    pos_emb_cls = "learned" if arch == "gpt2" else "rope"
    mlp_cls = "moe" if is_moe else "dense"

    new = {}

    # Scalar top-level fields (only emit if present in source)
    for key in ("d_model", "n_layers", "vocab_size", "dropout_embd", "tie_word_embeddings", "lm_head_bias"):
        if key in old:
            new[key] = old[key]

    # cls fields (always emit — inferred from arch)
    new["attn_cls"] = attn_cls
    new["mlp_cls"] = mlp_cls
    new["norm_cls"] = norm_cls
    new["pos_emb_cls"] = pos_emb_cls

    # attn_kwargs
    attn_kwargs = {}
    if "n_heads" in old:
        attn_kwargs["n_heads"] = old["n_heads"]
    if "n_kv_heads" in old:
        attn_kwargs["n_kv_heads"] = old["n_kv_heads"]
    if "qk_norm" in old:
        attn_kwargs["qk_norm"] = old["qk_norm"]
    if "dropout_attn" in old:
        attn_kwargs["dropout"] = old["dropout_attn"]
    if "attn_bias" in old:
        attn_kwargs["bias"] = old["attn_bias"]
    if "attn_implementation" in old:
        attn_kwargs["attn_implementation"] = old["attn_implementation"]
    new["attn_kwargs"] = attn_kwargs

    # mlp_kwargs
    mlp_kwargs = {}
    if is_moe:
        # intermediate_size: prefer moe_intermediate_size, fall back to intermediate_size
        if "moe_intermediate_size" in old:
            mlp_kwargs["intermediate_size"] = old["moe_intermediate_size"]
        elif "intermediate_size" in old:
            mlp_kwargs["intermediate_size"] = old["intermediate_size"]
        if "moe_n_experts" in old:
            mlp_kwargs["n_experts"] = old["moe_n_experts"]
        if "moe_n_experts_per_token" in old:
            mlp_kwargs["n_experts_per_token"] = old["moe_n_experts_per_token"]
        if "moe_aux_loss_coef" in old:
            mlp_kwargs["aux_loss_coef"] = old["moe_aux_loss_coef"]
        if "moe_expert_capacity_factor" in old:
            mlp_kwargs["expert_capacity_factor"] = old["moe_expert_capacity_factor"]
    else:
        if "intermediate_size" in old:
            mlp_kwargs["intermediate_size"] = old["intermediate_size"]

    # Dense fields shared between dense and moe
    if "mlp_activation" in old:
        mlp_kwargs["activation"] = old["mlp_activation"]
    if "mlp_gated" in old:
        mlp_kwargs["gated"] = old["mlp_gated"]
    if "mlp_bias" in old:
        mlp_kwargs["bias"] = old["mlp_bias"]
    if "dropout_ffn" in old:
        mlp_kwargs["dropout"] = old["dropout_ffn"]
    new["mlp_kwargs"] = mlp_kwargs

    # norm_kwargs: always empty; omit to keep diffs minimal
    # (the dataclass default is {})

    # pos_emb_kwargs
    if pos_emb_cls == "rope" and "rope_theta" in old:
        new["pos_emb_kwargs"] = {"rope_theta": old["rope_theta"]}

    # residual_cls / residual_kwargs: only emit if present in source
    if "residual_cls" in old:
        new["residual_cls"] = old["residual_cls"]
    if "residual_kwargs" in old:
        new["residual_kwargs"] = old["residual_kwargs"]

    return new


def migrate_file(path: str) -> None:
    with open(path) as f:
        doc = yaml.safe_load(f)

    model_block = doc.get("model", {})

    # Idempotency check: already migrated if attn_cls present and arch absent
    if "attn_cls" in model_block and "arch" not in model_block:
        print(f"already migrated: {path}")
        return

    doc["model"] = migrate_model_block(model_block)

    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)

    print(f"migrated: {path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/migrate_config.py <path> [<path>...]", file=sys.stderr)
        sys.exit(1)

    for path in sys.argv[1:]:
        migrate_file(path)


if __name__ == "__main__":
    main()
