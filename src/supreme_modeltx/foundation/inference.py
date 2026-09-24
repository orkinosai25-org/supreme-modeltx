from __future__ import annotations

from pathlib import Path

import torch

from supreme_modeltx.foundation.config import InferenceRunConfig
from supreme_modeltx.foundation.tokenization import CharacterTokenizer
from supreme_modeltx.foundation.training import TinyPilotLM, resolve_device


class FoundationResponder:
    def __init__(self, config: InferenceRunConfig) -> None:
        self.config = config
        self.device = resolve_device(config.inference.device)
        self.tokenizer = CharacterTokenizer(config.model.vocab_size)
        self.model: TinyPilotLM | None = None
        if config.inference.backend == "checkpoint":
            checkpoint_path = config.inference.checkpoint_path
            if not checkpoint_path:
                raise ValueError("checkpoint backend requires inference.checkpoint_path")
            self.model = TinyPilotLM(
                vocab_size=config.model.vocab_size,
                hidden_size=config.model.hidden_size,
                num_layers=config.model.num_layers,
                num_attention_heads=config.model.num_attention_heads,
                max_seq_len=config.model.max_seq_len,
                dropout=config.model.dropout,
            ).to(self.device)
            state = torch.load(Path(checkpoint_path), map_location=self.device, weights_only=True)
            self.model.load_state_dict(state["model_state"])
            self.model.eval()

    def generate(self, prompt: str) -> str:
        if self.config.inference.backend == "stub":
            return f"{self.config.inference.response_prefix} {prompt}".strip()

        assert self.model is not None
        input_ids = self.tokenizer.encode(prompt, max_length=self.config.model.max_seq_len)
        generated_tokens: list[int] = []
        tokens = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            for _ in range(self.config.inference.max_new_tokens):
                logits = self.model(tokens)["logits"][:, -1, :]
                next_token = int(torch.argmax(logits, dim=-1).item())
                generated_tokens.append(next_token)
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                tokens = torch.cat([tokens, next_token_tensor], dim=1)
                if tokens.size(1) > self.config.model.max_seq_len:
                    tokens = tokens[:, -self.config.model.max_seq_len :]
        decoded = self.tokenizer.decode(generated_tokens).strip()
        if not decoded:
            decoded = self.config.inference.response_prefix
        return f"{self.config.inference.response_prefix} {decoded}".strip()
