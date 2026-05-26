# T‑101 Base Model

## Overview

T‑101 is the 7B dense transformer base model for SMTX.

## Architecture

- 32 transformer layers
- Hidden size: 4096
- Intermediate size: 11008 (SwiGLU)
- 32 attention heads
- Rotary (RoPE) position embeddings
- BPE tokenizer — vocabulary size 32,000
- BF16 training precision

## Configuration

See `config.json` for the full model configuration.

## Training

See `../../training/train_t101.py` and `../../training/config_t101.json`.
