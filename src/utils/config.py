from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import yaml

# Valid activation names, mirroring the registries in src/layers/activation.py.
# Duplicated here (as plain literals) so config validation stays import-light —
# config.py never imports torch/layer modules. test_config guards them in sync.
_UNGATED_ACTIVATIONS = frozenset(
    {"relu", "gelu", "silu", "leaky_relu", "relu2", "gelu2", "silu2", "leaky_relu2"}
)
_GATED_ACTIVATIONS = _UNGATED_ACTIVATIONS | {"bilinear", "bilinear2", "powlu"}


@dataclass
class ModelConfig:
    d_model: int = 768
    n_layers: int = 12
    vocab_size: int = 50257

    attn_cls: str = "gqa"
    attn_kwargs: dict = field(default_factory=dict)
    mlp_cls: str = "dense"
    mlp_kwargs: dict = field(default_factory=dict)
    norm_cls: str = "rmsnorm"
    norm_kwargs: dict = field(default_factory=dict)
    pos_emb_cls: str = "rope"
    pos_emb_kwargs: dict = field(default_factory=dict)
    residual_cls: str = "standard"
    residual_kwargs: dict = field(default_factory=dict)

    dropout_embd: float = 0.0
    tie_word_embeddings: bool = True
    lm_head_bias: bool = False

    def __post_init__(self):
        # config.py is the single place for component defaults + validation, so
        # layer/model code can assume fully-resolved, valid kwargs and call sites
        # use plain `[key]` access (no `.get(key, default)`).

        # --- Attention: implementation default, dim validation, GQA kv default ---
        self.attn_kwargs.setdefault("attn_implementation", "flex_attention")
        n_heads = self.attn_kwargs.get("n_heads")
        if n_heads is not None:
            if self.d_model % n_heads != 0:
                raise ValueError(
                    f"d_model ({self.d_model}) must be divisible by n_heads ({n_heads})"
                )
            if self.attn_cls == "gqa":
                n_kv = self.attn_kwargs.setdefault("n_kv_heads", n_heads)
                if n_heads % n_kv != 0:
                    raise ValueError(
                        f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv})"
                    )

        # --- MLP: intermediate_size default, activation validation ---
        self.mlp_kwargs.setdefault("intermediate_size", 4 * self.d_model)
        gated = self.mlp_kwargs.get("gated", True)
        activation = self.mlp_kwargs.get("activation", "silu")
        valid = _GATED_ACTIVATIONS if gated else _UNGATED_ACTIVATIONS
        if activation not in valid:
            raise ValueError(
                f"Unknown activation: {activation!r}; expected one of {sorted(valid)}"
            )
        if self.mlp_cls == "moe":
            self.mlp_kwargs.setdefault("n_experts_per_token", 2)
            self.mlp_kwargs.setdefault("aux_loss_coef", 0.01)
            self.mlp_kwargs.setdefault("expert_capacity_factor", None)
            self.mlp_kwargs.setdefault("bias", False)


@dataclass
class DataConfig:
    dataset: str = "openwebtext"
    data_dir: str = "data/"
    val_split: float = 0.01
    num_workers: int = 4
    packing: bool = True
    tokenizer_path: str = "tokenizers/custom_bpe"


@dataclass
class TokenizerTrainingConfig:
    method: str = "bpe"  # "bpe" | "superbpe"
    method_kwargs: dict = field(default_factory=dict)
    num_samples: int = 1_000_000
    checkpoint_dir: str = (
        "tokenizers/custom_bpe"  # where the trained tokenizer is saved
    )
    checkpoint_every: int = 5000  # interval (in merges) at which to save the tokenizer
    # SuperBPE-only: interval (in merges) at which to log/evaluate curve points.
    eval_every: int = 5000

    def __post_init__(self):
        # Defaults for method_kwargs read by TokenizerTrainer (so it uses plain
        # `[key]` access). `transition_size` is required for superbpe (no default).
        self.method_kwargs.setdefault("eval_num_docs", 1000)
        if self.method == "superbpe":
            self.method_kwargs.setdefault("max_superword_words", 4)


@dataclass
class TrainingConfig:
    batch_size: int = 16
    gradient_accumulation_steps: int = 16
    max_steps: int = 50000
    early_stop: int = 0  # if > 0, stop at this step (overrides max_steps without affecting the LR schedule)
    mixed_precision: str = "bf16"
    loss_fn: str = "cross_entropy"
    label_smoothing: float = 0.0  # for CE loss only
    activation_checkpointing: bool = False
    use_deterministic_algo: bool = False
    seed: int = 42
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints/"
    checkpoint_every: int = 5000
    eval_every: int = 100
    eval_steps: int = 25
    eval_train: bool = (
        False  # run eval pass over train set (logs val/train_acc for SFT)
    )
    intra_doc_masking: bool = True


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 6e-4
    lr_mult: Dict[str, float] = field(default_factory=lambda: {"lm_head": 1.0})
    weight_decay: float = 0.1
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 100
    min_lr: float = 6e-5


@dataclass
class LoggingConfig:
    wandb_project: str = "pretrain"
    wandb_run_name: str = ""
    wandb_group: str = ""
    log_every: int = 10
    log_layer_grad_norms: bool = True  # log per-layer gradient norms to W&B
    log_optimizer_step_norms: bool = True  # log ||Δθ|| and ||m||; extra 1x param memory


@dataclass
class TrainConfig:
    task: str = "pretrain"  # "pretrain" | "sft"
    max_seq_len: int = 1024
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tokenizer_training: TokenizerTrainingConfig = field(
        default_factory=TokenizerTrainingConfig
    )
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self):
        return asdict(self)


def _apply_overrides(config: TrainConfig, overrides: List[str]):
    for override in overrides:
        key, value = override.split("=", 1)
        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
        field_name = parts[-1]
        current = (
            obj.get(field_name) if isinstance(obj, dict) else getattr(obj, field_name)
        )
        if isinstance(current, bool):
            value = value.lower() in ("true", "1", "yes")
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        elif current is None:
            # Optional field: try float, then int, then leave as string
            try:
                value = float(value)
            except ValueError:
                try:
                    value = int(value)
                except ValueError:
                    pass
        if isinstance(obj, dict):
            obj[field_name] = value
        else:
            setattr(obj, field_name, value)


def _coerce_types(dc_class, raw_dict: dict) -> dict:
    """Coerce raw YAML values to match dataclass field types.

    PyYAML safe_load treats scientific notation (e.g. 6e-4) as strings.
    This converts them to the correct type based on the dataclass annotation.
    """
    import dataclasses

    field_types = {f.name: f.type for f in dataclasses.fields(dc_class)}
    coerced = {}
    for k, v in raw_dict.items():
        if k not in field_types:
            continue  # silently ignore unknown/deprecated YAML fields
        expected = field_types.get(k)
        if expected is float and isinstance(v, str):
            v = float(v)
        elif expected is int and isinstance(v, str):
            v = int(v)
        elif expected is bool and isinstance(v, str):
            v = v.lower() in ("true", "1", "yes")
        coerced[k] = v
    return coerced


def _coerce_kwargs(d: dict) -> None:
    for k, v in d.items():
        if isinstance(v, str):
            try:
                d[k] = int(v)
            except ValueError:
                try:
                    d[k] = float(v)
                except ValueError:
                    pass


def load_config(path: str, overrides: Optional[List[str]] = None) -> TrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    config = TrainConfig(
        task=raw.get("task", "pretrain"),
        max_seq_len=raw.get("max_seq_len", 1024),
        model=ModelConfig(**_coerce_types(ModelConfig, raw.get("model", {}))),
        data=DataConfig(**_coerce_types(DataConfig, raw.get("data", {}))),
        tokenizer_training=TokenizerTrainingConfig(
            **_coerce_types(TokenizerTrainingConfig, raw.get("tokenizer_training", {}))
        ),
        training=TrainingConfig(
            **_coerce_types(TrainingConfig, raw.get("training", {}))
        ),
        optimizer=OptimizerConfig(
            **_coerce_types(OptimizerConfig, raw.get("optimizer", {}))
        ),
        scheduler=SchedulerConfig(
            **_coerce_types(SchedulerConfig, raw.get("scheduler", {}))
        ),
        logging=LoggingConfig(**_coerce_types(LoggingConfig, raw.get("logging", {}))),
    )

    for kw in (
        config.model.attn_kwargs,
        config.model.mlp_kwargs,
        config.model.norm_kwargs,
        config.model.pos_emb_kwargs,
        config.model.residual_kwargs,
        config.tokenizer_training.method_kwargs,
    ):
        _coerce_kwargs(kw)

    if overrides:
        _apply_overrides(config, overrides)

    return config
