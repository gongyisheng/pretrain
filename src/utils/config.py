from dataclasses import dataclass, field, asdict
from typing import List, Optional
import yaml


@dataclass
class ModelConfig:
    arch: str = "gpt2"
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    intermediate_size: int = 0  # 0 means 4 * d_model, set in post_init
    vocab_size: int = 50257
    dropout: float = 0.0
    attn_res: bool = False        # enable Block Attention Residuals (works with any arch)
    attn_res_block_size: int = 2  # number of full layers per block for AttnRes
    attn_res_norm: str = "rmsnorm"  # norm for attn_res keys: "rmsnorm" or "layernorm"
    n_kv_heads: int = 0           # 0 means same as n_heads (MHA); set >0 for GQA
    rope_theta: float = 10000.0   # RoPE base frequency; only used by qwen3, L_max ~62800
    qk_norm: bool = False         # apply RMSNorm to Q and K per head before RoPE (Qwen3-style)
    n_experts: int = 0              # 0 = dense; N > 0 = MoE with N total experts
    n_experts_per_token: int = 2    # top-k experts activated per token
    moe_intermediate_size: int = 0               # per-expert FFN hidden dim; 0 = same as intermediate_size
    moe_aux_loss_coef: float = 0.01 # Switch Transformer load-balancing loss coefficient
    moe_expert_capacity_factor: Optional[float] = None  # None = dynamic (no dropping); float = fixed capacity, enables torch.compile


    def __post_init__(self):
        if self.intermediate_size == 0:
            self.intermediate_size = 4 * self.d_model
        if self.n_kv_heads == 0:
            self.n_kv_heads = self.n_heads
        if self.moe_intermediate_size == 0:
            self.moe_intermediate_size = self.intermediate_size


@dataclass
class DataConfig:
    dataset: str = "openwebtext"
    tokenizer_path: str = "tokenizers/custom_bpe"
    data_dir: str = "data/"
    val_split: float = 0.01
    num_workers: int = 4
    packing: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_steps: int = 100000
    mixed_precision: str = "bf16"
    activation_checkpointing: bool = False
    backend: str = "torch"  # "torch" (torch.compile) or "triton" (custom kernels)
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints/"
    checkpoint_every: int = 5000
    eval_every: int = 1000
    eval_steps: int = 200
    intra_doc_masking: bool = True


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 6e-4
    weight_decay: float = 0.1
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 2000
    min_lr: float = 6e-5


@dataclass
class LoggingConfig:
    wandb_project: str = "pretrain"
    wandb_run_name: str = ""
    wandb_group: str = ""  # group name for comparing runs in W&B (e.g. "dtype-sweep")
    log_every: int = 10
    log_layer_grad_norms: bool = True  # log per-layer gradient norms to W&B


@dataclass
class SpikeConfig:
    enabled: bool = False
    grad_norm_threshold: float = 0.0    # save a full checkpoint when grad_norm exceeds this
    max_checkpoints: int = 10           # keep only top-N spikes by grad norm


@dataclass
class DebugConfig:
    spike: SpikeConfig = field(default_factory=SpikeConfig)
    max_steps: int = 0  # if > 0, stop training at this step (overrides training.max_steps without affecting the LR schedule)


@dataclass
class TrainConfig:
    max_seq_len: int = 1024
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def to_dict(self):
        return asdict(self)


def _apply_overrides(config: TrainConfig, overrides: List[str]):
    for override in overrides:
        key, value = override.split("=", 1)
        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        field_name = parts[-1]
        current = getattr(obj, field_name)
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
        if expected == float and isinstance(v, str):
            v = float(v)
        elif expected == int and isinstance(v, str):
            v = int(v)
        elif expected == bool and isinstance(v, str):
            v = v.lower() in ("true", "1", "yes")
        coerced[k] = v
    return coerced


def load_config(path: str, overrides: Optional[List[str]] = None) -> TrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    config = TrainConfig(
        max_seq_len=raw.get("max_seq_len", 1024),
        model=ModelConfig(**_coerce_types(ModelConfig, raw.get("model", {}))),
        data=DataConfig(**_coerce_types(DataConfig, raw.get("data", {}))),
        training=TrainingConfig(**_coerce_types(TrainingConfig, raw.get("training", {}))),
        optimizer=OptimizerConfig(**_coerce_types(OptimizerConfig, raw.get("optimizer", {}))),
        scheduler=SchedulerConfig(**_coerce_types(SchedulerConfig, raw.get("scheduler", {}))),
        logging=LoggingConfig(**_coerce_types(LoggingConfig, raw.get("logging", {}))),
        debug=DebugConfig(
            spike=SpikeConfig(**_coerce_types(SpikeConfig, raw.get("debug", {}).get("spike", {}))),
        ),
    )

    if overrides:
        _apply_overrides(config, overrides)

    return config
