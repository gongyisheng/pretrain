from dataclasses import dataclass, field, asdict
from typing import List, Optional
import yaml


@dataclass
class ModelConfig:
    arch: str = "gpt2"
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 0  # 0 means 4 * d_model, set in post_init
    vocab_size: int = 50257
    dropout: float = 0.1

    def __post_init__(self):
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model


@dataclass
class DataConfig:
    dataset: str = "openwebtext"
    tokenizer_path: str = "tokenizers/custom_bpe"
    data_dir: str = "data/"
    val_split: float = 0.01
    num_workers: int = 4


@dataclass
class TrainingConfig:
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_steps: int = 100000
    mixed_precision: str = "bf16"
    activation_checkpointing: bool = False
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints/"
    checkpoint_every: int = 5000
    eval_every: int = 1000
    eval_steps: int = 200


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 6e-4
    weight_decay: float = 0.1
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 2000
    min_lr: float = 6e-5


@dataclass
class LoggingConfig:
    wandb_project: str = "pretrain"
    wandb_run_name: str = ""
    log_every: int = 10


@dataclass
class TrainConfig:
    max_seq_len: int = 1024
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
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
            obj = getattr(obj, part)
        field_name = parts[-1]
        current = getattr(obj, field_name)
        if isinstance(current, bool):
            value = value.lower() in ("true", "1", "yes")
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        setattr(obj, field_name, value)


def load_config(path: str, overrides: Optional[List[str]] = None) -> TrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    config = TrainConfig(
        max_seq_len=raw.get("max_seq_len", 1024),
        model=ModelConfig(**{k: v for k, v in raw.get("model", {}).items()}),
        data=DataConfig(**{k: v for k, v in raw.get("data", {}).items()}),
        training=TrainingConfig(**{k: v for k, v in raw.get("training", {}).items()}),
        optimizer=OptimizerConfig(**{k: v for k, v in raw.get("optimizer", {}).items()}),
        scheduler=SchedulerConfig(**{k: v for k, v in raw.get("scheduler", {}).items()}),
        logging=LoggingConfig(**{k: v for k, v in raw.get("logging", {}).items()}),
    )

    if overrides:
        _apply_overrides(config, overrides)

    return config
