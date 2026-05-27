# Third-Party Notices

`supreme-modeltx` is an original work. Where components are inspired by, reference, or adapt patterns from open-source projects, those inspirations are documented below in compliance with their respective licences.

---

## PyTorch

- **Project:** PyTorch
- **URL:** https://github.com/pytorch/pytorch
- **Licence:** BSD 3-Clause
- **Usage:** Core deep learning framework. All model, training, and inference code is built on PyTorch primitives.

---

## HuggingFace Transformers

- **Project:** transformers
- **URL:** https://github.com/huggingface/transformers
- **Licence:** Apache 2.0
- **Usage:** Architectural patterns (RoPE, RMSNorm, SwiGLU FFN) are inspired by models published in this library. No source code is copied; implementation is original.

---

## HuggingFace Tokenizers

- **Project:** tokenizers
- **URL:** https://github.com/huggingface/tokenizers
- **Licence:** Apache 2.0
- **Usage:** `SMTXTokenizer` wraps the `tokenizers` library for BPE vocabulary training and encoding.

---

## HuggingFace Datasets

- **Project:** datasets
- **URL:** https://github.com/huggingface/datasets
- **Licence:** Apache 2.0
- **Usage:** `DataPipeline` optionally loads datasets via the `datasets` library.

---

## FastAPI

- **Project:** FastAPI
- **URL:** https://github.com/tiangolo/fastapi
- **Licence:** MIT
- **Usage:** The `platform_api` application is built on FastAPI.

---

## Pydantic

- **Project:** Pydantic
- **URL:** https://github.com/pydantic/pydantic
- **Licence:** MIT
- **Usage:** All configuration and API schemas use Pydantic v2 for validation.

---

## Accelerate

- **Project:** Accelerate
- **URL:** https://github.com/huggingface/accelerate
- **Licence:** Apache 2.0
- **Usage:** Optional training acceleration wrapper; referenced in training documentation.

---

## DeepSpeed

- **Project:** DeepSpeed
- **URL:** https://github.com/microsoft/DeepSpeed
- **Licence:** Apache 2.0
- **Usage:** Optional large-scale training backend; listed as an optional dependency.

---

## LLaMA / Meta AI (Architectural Inspiration)

- **Project:** LLaMA
- **URL:** https://github.com/meta-llama/llama
- **Licence:** Custom (Meta Community Licence)
- **Usage:** RoPE, RMSNorm, SwiGLU, and pre-normalisation architectural patterns were first published in LLaMA and are widely used in the open-source community. No source code from Meta is included in this repository. Our implementation is independent and original.

---

## Rotary Embeddings (RoPE)

- **Paper:** "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
- **URL:** https://arxiv.org/abs/2104.09864
- **Licence:** Not applicable (academic paper)
- **Usage:** The rotary positional embedding algorithm implemented in `model_core/models/t_series.py` is based on the mathematical formulation in this paper.

---

## SwiGLU

- **Paper:** "GLU Variants Improve Transformer" (Noam Shazeer, 2020)
- **URL:** https://arxiv.org/abs/2002.05202
- **Licence:** Not applicable (academic paper)
- **Usage:** The SwiGLU feed-forward network formulation used in `FeedForward` is from this paper.

---

*This document is maintained on a best-effort basis. If you identify a missing attribution, please open an issue.*
