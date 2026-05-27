"""tokenizer sub-package."""

from supreme_modeltx.model_core.tokenizer.modeltx_tokenizer import ModelTXTokenizer

# Backward-compatible alias
SMTXTokenizer = ModelTXTokenizer

__all__ = ["ModelTXTokenizer", "SMTXTokenizer"]
