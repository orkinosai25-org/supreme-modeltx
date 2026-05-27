"""Training loop for T-Series models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from supreme_modeltx.model_core.config import TrainingConfig

logger = logging.getLogger(__name__)


class Trainer:
    """Minimal training loop for T-Series models.

    Handles training step, evaluation step, checkpointing, and basic
    logging. For large-scale distributed training, wrap with Accelerate
    or DeepSpeed before calling :meth:`train`.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        self.global_step = 0

    def _training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.train()
        outputs = self.model(**batch)
        loss: torch.Tensor = outputs["loss"]
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()
        return loss

    def _eval_step(self, batch: dict[str, torch.Tensor]) -> float:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**batch)
        return outputs["loss"].item()

    def save_checkpoint(self) -> None:
        ckpt_path = self.output_dir / f"checkpoint-{self.global_step}"
        ckpt_path.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), ckpt_path / "model.pt")
        logger.info("Saved checkpoint to %s", ckpt_path)

    def train(self) -> dict[str, Any]:
        """Run the training loop up to :attr:`config.max_steps`."""
        grad_accum = self.config.gradient_accumulation_steps
        log_loss = 0.0
        self.optimizer.zero_grad()

        data_iter = iter(self.train_dataloader)
        while self.global_step < self.config.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            loss = self._training_step(batch)
            log_loss += loss.item()

            if (self.global_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.global_step += 1

            if self.global_step % self.config.logging_steps == 0:
                logger.info(
                    "step=%d loss=%.4f", self.global_step, log_loss / self.config.logging_steps
                )
                log_loss = 0.0

            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint()

        return {"global_step": self.global_step}
