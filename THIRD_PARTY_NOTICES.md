# Third-Party Notices

Supreme ModelTX is original software. The implementation draws architectural inspiration from the following open-source projects and research papers. No code from these projects has been copied verbatim; where patterns are closely followed, this is noted explicitly.

---

## PyTorch

- **Project**: PyTorch
- **URL**: https://github.com/pytorch/pytorch
- **Licence**: BSD-3-Clause
- **Usage**: Core deep learning framework. All model, training, and inference code is built on PyTorch.

---

## Rotary Position Embeddings (RoPE)

- **Paper**: "RoFormer: Enhanced Transformer with Rotary Position Embedding", Su et al., 2021
- **arXiv**: https://arxiv.org/abs/2104.09864
- **Usage**: `supreme_modeltx.model_core.models.common.attention` — `precompute_freqs_cis`, `apply_rope`
- **Notes**: Implementation is original; the mathematical formulation follows the published paper.

---

## Grouped Query Attention (GQA)

- **Paper**: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", Ainslie et al., 2023
- **arXiv**: https://arxiv.org/abs/2305.13245
- **Usage**: `GroupedQueryAttention` in `attention.py`
- **Notes**: Original implementation following the paper's description.

---

## SwiGLU Feed-Forward Network

- **Paper**: "GLU Variants Improve Transformer", Noam Shazeer, 2020
- **arXiv**: https://arxiv.org/abs/2002.05202
- **Usage**: `SwiGLUFFN` in `baseline.py`
- **Notes**: Standard formulation; implementation is original.

---

## RMSNorm

- **Paper**: "Root Mean Square Layer Normalization", Zhang & Sennrich, 2019
- **arXiv**: https://arxiv.org/abs/1910.07467
- **Usage**: `RMSNorm` in `baseline.py`

---

## LLaMA / LLaMA 2

- **Project**: Meta AI LLaMA 2
- **Paper**: "Llama 2: Open Foundation and Fine-Tuned Chat Models", Touvron et al., 2023
- **arXiv**: https://arxiv.org/abs/2307.09288
- **Licence**: LLaMA 2 Community Licence
- **Usage**: Architectural inspiration for the T-series decoder-only design (RoPE + RMSNorm + SwiGLU combination). No code copied.

---

## Mistral 7B

- **Project**: Mistral AI
- **Paper**: "Mistral 7B", Jiang et al., 2023
- **arXiv**: https://arxiv.org/abs/2310.06825
- **Licence**: Apache-2.0
- **Usage**: Inspiration for GQA integration and attention implementation patterns. No code copied.

---

## LitGPT

- **Project**: Lightning AI LitGPT
- **URL**: https://github.com/Lightning-AI/litgpt
- **Licence**: Apache-2.0
- **Usage**: Inspiration for clean config / model separation patterns and training loop structure. No code copied.

---

## Sequence Packing

- **Paper**: "PaLM: Scaling Language Modeling with Pathways", Chowdhery et al., 2022
- **arXiv**: https://arxiv.org/abs/2204.02311
- **Usage**: Greedy sequence-packing approach in `preprocessing.py`. Standard technique; implementation is original.
- Also inspired by: `packed_dataset.py` in lit-gpt (Lightning AI, Apache-2.0).

---

## SentencePiece

- **Project**: Google SentencePiece
- **URL**: https://github.com/google/sentencepiece
- **Paper**: "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing", Kudo & Richardson, 2018
- **Licence**: Apache-2.0
- **Usage**: Tokenizer backend in `tokenizer/workflow.py`. Used as a library; no code copied.

---

## HuggingFace Datasets

- **Project**: HuggingFace Datasets
- **URL**: https://github.com/huggingface/datasets
- **Licence**: Apache-2.0
- **Usage**: Optional data backend in `data/sources.py`. Used as a library.

---

## HuggingFace Tokenizers

- **Project**: HuggingFace Tokenizers
- **URL**: https://github.com/huggingface/tokenizers
- **Licence**: Apache-2.0
- **Usage**: Optional tokenizer backend in `tokenizer/workflow.py`. Used as a library.

---

## FastAPI

- **Project**: FastAPI
- **URL**: https://github.com/tiangolo/fastapi
- **Licence**: MIT
- **Usage**: Platform API web framework (`platform_api/api/`). Used as a library.

---

## Pydantic

- **Project**: Pydantic
- **URL**: https://github.com/pydantic/pydantic
- **Licence**: MIT
- **Usage**: Configuration schema and API request/response validation throughout.

---

## Sampling algorithms

- **Paper**: "The Curious Case of Neural Text Degeneration", Holtzman et al., 2020
- **arXiv**: https://arxiv.org/abs/1904.09751
- **Usage**: Top-p (nucleus) sampling in `inference/sampling.py`. Standard formulation; implementation is original.

---

## DeepSeek (design inspiration only)

- **Project**: DeepSeek AI
- **URL**: https://github.com/deepseek-ai
- **Notes**: DeepSeek's open-source LLM training approach and scaling strategy were part of the motivation for this sovereign-stack direction. **No DeepSeek code is included in this repository.** The T-series architecture predates our awareness of DeepSeek-V3 and is not derived from it.

---

## Existing SMTX control-plane (retained components)

The `control-plane/` and `infra/` directories contain code originally developed for the SMTX project. These components are retained as-is and are documented in their own subdirectory READMEs.

---

*This file is updated as new dependencies and inspirations are incorporated. If you believe a notice is missing or inaccurate, please open an issue.*
