from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union
import yaml

from src.layers.activation import GATED_ACTIVATIONS, UNGATED_ACTIVATIONS
from src.layers.attention import ATTN_REGISTRY
from src.layers.mlp import MLP_REGISTRY, MOE_ROUTER_SCORE_FNS
from src.layers.pos_emb import POS_EMB_REGISTRY
from src.quant import (
    QUANT_FORMATS,
    QUANT_GRANULARITY,
    QUANT_DTYPE_RECIPES,
    QUANT_OPERANDS,
)
from src.training.loss import LOSS_REGISTRY
from src.training.optimizer import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY

_MIXED_PRECISION = frozenset({"no", "bf16", "fp16"})
_DEVICES = frozenset({"auto", "cuda", "cpu"})


@dataclass
class ModelConfig:
    d_model: int = 768
    n_layers: int = 12
    vocab_size: int = 50257

    attn: list = field(default_factory=lambda: [{"attn_cls": "gqa", "attn_kwargs": {}}])
    mlp: list = field(default_factory=lambda: [{"mlp_cls": "dense", "mlp_kwargs": {}}])
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
        self._post_init_attn()
        self._post_init_mlp()

    def _set_default_attn_kwargs(self, attn_cls: str, kwargs: dict) -> None:
        """Fill defaults/validation for one attn item's kwargs, keyed on attn_cls."""
        kwargs.setdefault("attn_implementation", "flex_attention")
        n_heads = kwargs.get("n_heads")
        if n_heads is not None:
            # MLA sets its head dims explicitly, so d_model need not divide n_heads.
            if attn_cls in ("mha", "gqa") and self.d_model % n_heads != 0:
                raise ValueError(
                    f"d_model ({self.d_model}) must be divisible by n_heads ({n_heads})"
                )
            if attn_cls == "gqa":
                n_kv = kwargs.setdefault("n_kv_heads", n_heads)
                if n_heads % n_kv != 0:
                    raise ValueError(
                        f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv})"
                    )
            elif attn_cls == "mla":
                # Decoupled-RoPE head dims default off d_model // n_heads;
                # q_lora_rank=0 disables query compression.
                head_dim = self.d_model // n_heads
                kwargs.setdefault("qk_nope_head_dim", head_dim)
                kwargs.setdefault("qk_rope_head_dim", max(head_dim // 2, 1))
                kwargs.setdefault("v_head_dim", head_dim)
                kwargs.setdefault("kv_lora_rank", 4 * head_dim)
                kwargs.setdefault("q_lora_rank", 0)

    def _post_init_attn(self) -> None:
        if not self.attn:
            raise ValueError("model.attn must have at least one item")
        for item in self.attn:
            if "attn_cls" not in item:
                raise ValueError("each model.attn item requires 'attn_cls'")
            if item["attn_cls"] not in ATTN_REGISTRY:
                raise ValueError(
                    f"unknown attn_cls: {item['attn_cls']!r}; "
                    f"expected one of {sorted(ATTN_REGISTRY)}"
                )
            item.setdefault("attn_kwargs", {})
            self._set_default_attn_kwargs(item["attn_cls"], item["attn_kwargs"])

        n = self.n_layers
        claimed: dict[int, int] = {}
        bare: list[int] = []
        for idx, item in enumerate(self.attn):
            layer_idx = item.get("layer_idx")
            if layer_idx is None:
                bare.append(idx)
                continue
            seen = set()
            for layer in layer_idx:
                if layer in seen:
                    raise ValueError(
                        f"layer {layer} listed more than once in one attn item's layer_idx"
                    )
                seen.add(layer)
                if not (0 <= layer < n):
                    raise ValueError(f"layer_idx {layer} out of range [0, {n})")
                if layer in claimed:
                    raise ValueError(f"layer {layer} claimed by multiple attn items")
                claimed[layer] = idx
        if len(bare) > 1:
            raise ValueError("at most one model.attn item may omit layer_idx")
        if bare:
            remaining = [layer for layer in range(n) if layer not in claimed]
            self.attn[bare[0]]["layer_idx"] = remaining
            for layer in remaining:
                claimed[layer] = bare[0]
        missing = [layer for layer in range(n) if layer not in claimed]
        if missing:
            raise ValueError(
                f"layers {missing} have no attn item; add a fallback item without layer_idx"
            )

        self._layer_attn = [
            (
                self.attn[claimed[layer]]["attn_cls"],
                self.attn[claimed[layer]]["attn_kwargs"],
            )
            for layer in range(n)
        ]

        impls = {kw["attn_implementation"] for _, kw in self._layer_attn}
        if len(impls) > 1:
            raise ValueError(
                "all attn layers must share the same attn_implementation (the "
                f"trainer builds one attention mask shared across layers); got {sorted(impls)}"
            )

        if POS_EMB_REGISTRY[self.pos_emb_cls].rotary:
            dims = set()
            for attn_cls, attn_kwargs in self._layer_attn:
                if attn_cls == "mla":
                    dim = attn_kwargs.get("qk_rope_head_dim")
                elif attn_kwargs.get("n_heads") is not None:
                    dim = self.d_model // attn_kwargs["n_heads"]
                else:
                    dim = None
                if dim is not None:
                    dims.add(dim)
            if len(dims) > 1:
                raise ValueError(
                    "rotary pos_emb requires a single rope head-dim across layers; "
                    f"got {sorted(dims)}. Make qk_rope_head_dim / (d_model // "
                    "n_heads) match."
                )

    def resolve_attn(self, layer_idx: int) -> tuple[str, dict]:
        return self._layer_attn[layer_idx]

    @property
    def attn_implementation(self) -> str:
        """Shared attn_implementation across all layers (validated in _post_init_attn)."""
        return self._layer_attn[0][1]["attn_implementation"]

    def _set_default_mlp_kwargs(self, mlp_cls: str, kwargs: dict) -> None:
        """Fill defaults/validation for one MLP item's kwargs, keyed on mlp_cls."""
        kwargs.setdefault("intermediate_size", 4 * self.d_model)
        gated = kwargs.get("gated", True)
        activation = kwargs.get("activation", "silu")
        valid = GATED_ACTIVATIONS if gated else UNGATED_ACTIVATIONS
        if activation not in valid:
            raise ValueError(
                f"Unknown activation: {activation!r}; expected one of {sorted(valid)}"
            )
        if mlp_cls == "moe":
            kwargs.setdefault("n_shared_experts", 0)
            kwargs.setdefault("bias", False)
            kwargs.setdefault("router_score_fn", "sigmoid")
            if kwargs["router_score_fn"] not in MOE_ROUTER_SCORE_FNS:
                raise ValueError(
                    f"router_score_fn must be one of {sorted(MOE_ROUTER_SCORE_FNS)}; "
                    f"got {kwargs['router_score_fn']!r}"
                )
            # aux_loss (Switch) and expert_bias (arXiv:2408.15664) are mutually
            # exclusive balancing strategies; exactly one must be enabled.
            kwargs.setdefault("expert_bias", False)
            kwargs.setdefault("aux_loss", False)
            if not kwargs["aux_loss"] and not kwargs["expert_bias"]:
                raise ValueError(
                    "exactly one of aux_loss / expert_bias must be enabled; both are off"
                )
            if kwargs["aux_loss"] and kwargs["expert_bias"]:
                raise ValueError(
                    "aux_loss and expert_bias are mutually exclusive; both are on"
                )
            if kwargs["expert_bias"]:
                kwargs.setdefault("expert_bias_update_rate", 0.001)
            if kwargs["aux_loss"]:
                kwargs.setdefault("aux_loss_coef", 0.001)
            kwargs.setdefault("latent_moe", False)
            if kwargs["latent_moe"]:
                latent_dim = kwargs.get("latent_dim")
                if (
                    not isinstance(latent_dim, int)
                    or isinstance(latent_dim, bool)
                    or latent_dim <= 0
                ):
                    raise ValueError(
                        f"latent_dim must be a positive int when latent_moe=True; "
                        f"got {latent_dim!r}"
                    )

    def _post_init_mlp(self) -> None:
        if not self.mlp:
            raise ValueError("model.mlp must have at least one item")
        for item in self.mlp:
            if "mlp_cls" not in item:
                raise ValueError("each model.mlp item requires 'mlp_cls'")
            if item["mlp_cls"] not in MLP_REGISTRY:
                raise ValueError(
                    f"unknown mlp_cls: {item['mlp_cls']!r}; "
                    f"expected one of {sorted(MLP_REGISTRY)}"
                )
            item.setdefault("mlp_kwargs", {})
            self._set_default_mlp_kwargs(item["mlp_cls"], item["mlp_kwargs"])

        n = self.n_layers
        claimed: dict[int, int] = {}  # layer -> item index
        bare: list[int] = []
        for idx, item in enumerate(self.mlp):
            layer_idx = item.get("layer_idx")
            if layer_idx is None:
                bare.append(idx)
                continue
            if len(set(layer_idx)) != len(layer_idx):
                dupe = next(layer for layer in layer_idx if layer_idx.count(layer) > 1)
                raise ValueError(
                    f"layer {dupe} listed more than once in one mlp item's layer_idx"
                )
            for layer in layer_idx:
                if not (0 <= layer < n):
                    raise ValueError(f"layer_idx {layer} out of range [0, {n})")
                if layer in claimed:
                    raise ValueError(f"layer {layer} claimed by multiple mlp items")
                claimed[layer] = idx
        if len(bare) > 1:
            raise ValueError("at most one model.mlp item may omit layer_idx")
        if bare:
            remaining = [layer for layer in range(n) if layer not in claimed]
            self.mlp[bare[0]]["layer_idx"] = remaining
            for layer in remaining:
                claimed[layer] = bare[0]
        missing = [layer for layer in range(n) if layer not in claimed]
        if missing:
            raise ValueError(
                f"layers {missing} have no mlp item; add a fallback item without layer_idx"
            )

        # per-layer (cls, kwargs) map; plain attribute, not a dataclass field,
        # so asdict()/to_dict() serialize only the `mlp` list.
        self._layer_mlp = [
            (
                self.mlp[claimed[layer]]["mlp_cls"],
                self.mlp[claimed[layer]]["mlp_kwargs"],
            )
            for layer in range(n)
        ]

    def resolve_mlp(self, layer_idx: int) -> tuple[str, dict]:
        return self._layer_mlp[layer_idx]

    @property
    def is_moe(self) -> bool:
        return any(cls == "moe" for cls, _ in self._layer_mlp)


@dataclass
class DataConfig:
    dataset: str = "openwebtext"
    data_dir: str = "data/"
    val_split: float = 0.01
    num_workers: int = 4
    prefetch_factor: int = 4
    packing: bool = True
    tokenizer_path: str = "tokenizers/custom_bpe"


@dataclass
class TokenizerTrainingConfig:
    method: str = "bpe"  # "bpe" | "superbpe"
    method_kwargs: dict = field(default_factory=dict)
    num_samples: int = 1_000_000
    checkpoint_dir: str = "tokenizers/custom_bpe"
    checkpoint_every: int = 5000
    eval_every: int = 5000

    def __post_init__(self):
        self.method_kwargs.setdefault("eval_num_docs", 1000)
        if self.method == "superbpe":
            self.method_kwargs.setdefault("max_superword_words", 4)


@dataclass
class QuantConfig:
    enabled: bool = False
    dtype_recipe: Optional[str] = None
    dtype: dict = field(default_factory=dict)  # {weight/act/grad: fmt}
    scaling: dict = field(default_factory=dict)  # {granularity, block_size}
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=lambda: ["lm_head"])

    def __post_init__(self):
        if not self.enabled:
            return

        if self.dtype_recipe is not None:
            if self.dtype_recipe not in QUANT_DTYPE_RECIPES:
                raise ValueError(
                    f"unknown quant dtype_recipe: {self.dtype_recipe!r}; "
                    f"expected one of {sorted(QUANT_DTYPE_RECIPES)}"
                )
            for operand, fmt in QUANT_DTYPE_RECIPES[self.dtype_recipe].items():
                self.dtype.setdefault(operand, fmt)  # explicit dtype wins

        self.scaling.setdefault("granularity", "tensorwise")
        granularity = self.scaling["granularity"]
        if granularity not in QUANT_GRANULARITY:
            raise ValueError(
                f"unknown quant granularity: {granularity!r}; "
                f"expected one of {sorted(QUANT_GRANULARITY)}"
            )

        for operand, fmt in self.dtype.items():
            if operand not in QUANT_OPERANDS:
                raise ValueError(
                    f"unknown quant dtype key: {operand!r}; "
                    f"expected one of {sorted(QUANT_OPERANDS)}"
                )
            if fmt not in QUANT_FORMATS:
                raise ValueError(
                    f"unknown quant fmt for {operand}: {fmt!r}; "
                    f"expected one of {sorted(QUANT_FORMATS)}"
                )


@dataclass
class TrainingConfig:
    batch_size: int = 16
    gradient_accumulation_steps: int = 16
    max_steps: int = 50000
    early_stop: int = 0
    device: str = "auto"  # "auto" (cuda if available else cpu) | "cuda" | "cpu"
    mixed_precision: str = "bf16"
    loss_fn: str = "cross_entropy"
    label_smoothing: float = 0.0  # for CE loss only
    enable_torch_compile: bool = True
    use_deterministic_algo: bool = False
    seed: int = 42
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints/"
    checkpoint_every: int = 5000
    eval_every: int = 100
    eval_steps: int = 25
    eval_batch_size: int = 16
    eval_train: bool = False  # for SFT
    intra_doc_masking: bool = True
    quant: Union[QuantConfig, dict, list] = field(default_factory=QuantConfig)

    def __post_init__(self):
        if self.device not in _DEVICES:
            raise ValueError(
                f"unknown device: {self.device!r}; expected one of {sorted(_DEVICES)}"
            )
        if self.mixed_precision not in _MIXED_PRECISION:
            raise ValueError(
                f"unknown mixed_precision: {self.mixed_precision!r}; "
                f"expected one of {sorted(_MIXED_PRECISION)}"
            )
        if self.loss_fn not in LOSS_REGISTRY:
            raise ValueError(
                f"unknown loss_fn: {self.loss_fn!r}; "
                f"expected one of {sorted(LOSS_REGISTRY)}"
            )
        rules = self.quant if isinstance(self.quant, list) else [self.quant]
        amp_dtype = "fp32" if self.mixed_precision == "no" else self.mixed_precision
        normalized = []
        for rule in rules:
            if isinstance(rule, dict):
                rule = QuantConfig(**rule)
            if rule.enabled:
                for operand in QUANT_OPERANDS:
                    rule.dtype.setdefault(operand, amp_dtype)
            normalized.append(rule)
        self.quant = normalized


@dataclass
class OptimizerConfig:
    name: str = "adamw"  # "adamw" | "lion" | "muon"
    lr: float = 6e-4
    lr_mult: Dict[str, float] = field(default_factory=lambda: {"lm_head": 1.0})
    weight_decay: float = 0.1
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-8
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_coefficients: List[float] = field(
        default_factory=lambda: [3.4445, -4.775, 2.0315]
    )
    muon_ns_max_batch_elems: int = 8_000_000  # tuned on 5090/6000 pro blackwell
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str = "match_rms_adamw"

    def __post_init__(self):
        if self.name not in OPTIMIZER_REGISTRY:
            raise ValueError(
                f"unknown optimizer: {self.name!r}; "
                f"expected one of {sorted(OPTIMIZER_REGISTRY)}"
            )


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 100
    min_lr: float = 6e-5

    def __post_init__(self):
        if self.name not in SCHEDULER_REGISTRY:
            raise ValueError(
                f"unknown scheduler: {self.name!r}; "
                f"expected one of {sorted(SCHEDULER_REGISTRY)}"
            )


@dataclass
class LoggingConfig:
    wandb_project: str = "pretrain"
    wandb_run_name: str = ""
    wandb_group: str = ""
    log_every: int = 10
    log_layer_grad_norms: bool = True  # log per-layer gradient norms to W&B
    log_optimizer_step_norms: bool = True  # log ||Δθ|| and ||m||; extra 1x param memory
    log_optimizer_svd_metrics: bool = (
        True  # log per-2D-weight srank/pr; costly, SVD per weight
    )


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

    def __post_init__(self):
        self._validate_moe_compile_precision()

    def _validate_moe_compile_precision(self):
        m = self.model
        if m.is_moe and self.training.mixed_precision != "bf16":
            raise ValueError(
                "dropless MoE requires "
                f"training.mixed_precision='bf16'; got {self.training.mixed_precision!r}. "
                "torch._grouped_mm is bf16-only under torch.compile."
            )

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
        config.model.norm_kwargs,
        config.model.pos_emb_kwargs,
        config.model.residual_kwargs,
        config.tokenizer_training.method_kwargs,
    ):
        _coerce_kwargs(kw)
    for item in config.model.attn:
        _coerce_kwargs(item["attn_kwargs"])
    for item in config.model.mlp:
        _coerce_kwargs(item["mlp_kwargs"])
    for rule in config.training.quant:
        _coerce_kwargs(rule.scaling)

    if overrides:
        _apply_overrides(config, overrides)

    return config
