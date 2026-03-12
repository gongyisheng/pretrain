from src.model.gpt2 import GPT2Model

MODEL_REGISTRY = {
    "gpt2": GPT2Model,
}


def build_model(config):
    """Build model from config. config must have .model.arch and .max_seq_len."""
    arch = config.model.arch
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[arch](config.model, max_seq_len=config.max_seq_len)
