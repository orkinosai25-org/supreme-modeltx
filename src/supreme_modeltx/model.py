from __future__ import annotations

import torch
from torch import nn

from .config import TrainingConfig


class T100LM(nn.Module):
    def __init__(self, cfg: TrainingConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.position_embedding = nn.Embedding(cfg.sequence_length, cfg.embedding_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embedding_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffw_hidden_dim,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.ln_f = nn.LayerNorm(cfg.embedding_dim)
        self.lm_head = nn.Linear(cfg.embedding_dim, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions).unsqueeze(0)

        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
            diagonal=1,
        )
        x = self.blocks(x, mask=mask)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(batch_size * seq_len, -1),
                targets.view(batch_size * seq_len),
            )

        return logits, loss
