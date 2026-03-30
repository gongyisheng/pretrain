from src.model.gpt2 import GPT2Model
from src.model.qwen3 import Qwen3Model
from src.model.qwen3_moe import Qwen3MoEModel

MODEL_REGISTRY = {
    "gpt2": GPT2Model,
    "qwen3": Qwen3Model,
    "qwen3_moe": Qwen3MoEModel,
}


def build_model(config):
    """Build model from config. config must have .model.arch and .max_seq_len."""
    arch = config.model.arch
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[arch](config.model, max_seq_len=config.max_seq_len)
