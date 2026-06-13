from src.model.transformer import TransformerLM


def build_model(config):
    """Build the unified model from config (.model + .max_seq_len)."""
    return TransformerLM(config.model, max_seq_len=config.max_seq_len)
