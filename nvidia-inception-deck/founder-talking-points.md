# Founder talking points

## Core story

Orkinos AI should be presented as a staged company, not as a single vague model bet.

- **Stage 1:** Ebru is the commercial Turkish-first AI product
- **Foundation:** Supreme ModelTX supports Ebru through governed pretrained-model
  evaluation, adaptation planning, and controlled deployment

## What to emphasize

- We are not claiming a frontier model today
- Approved local pretrained artifacts are the near-term model path; original
  pretraining is not a product prerequisite
- Our differentiation is Turkish-first UX plus private deployment, governance, and auditability
- We want to support institutions that need more control than generic hosted AI products provide

## Why NVIDIA is a fit

- Current repository evidence shows training, checkpointing, evaluation, and API/governance foundations already exist
- CPU/custom models remain research and smoke tests, not proof of Turkish product readiness
- CUDA code paths exist but are not GPU-validated; no multi-GPU validation is claimed
- NeMo, Megatron Core, TensorRT-LLM, and ONNX are planned, not integrated;
  cuDNN is a runtime library, not a model backend
- Licensing, data, safety, reproducibility, and deployment controls matter alongside compute
- Inception guidance could help scope approved one-GPU comparisons and supported
  adaptation/serving experiments; neither admission nor compute access is promised

## Good concise phrasing

- “Ebru is the Turkish-first product direction; Supreme ModelTX is its governed model foundation.”
- “We are product-first in the market and infrastructure-serious underneath.”
- “Our ask is practical: GPU enablement, technical guidance, and Inception support to reach the next measurable milestone.”

## Avoid saying

- “We already have a frontier model”
- “We are production-ready across every layer”
- “We can replace all global models immediately”
- “GPU access is the only blocker”
- “The NVIDIA stack is already integrated”
- “Our application is upload-ready or Inception-approved”

## If asked about current maturity

Say:

- Ebru is the near-term product focus
- Supreme ModelTX is early-stage but repo-backed
- do not treat earlier TRL self-assessments as pretrained-product readiness
- `Qwen/Qwen2.5-0.5B-Instruct` (candidate Apache-2.0, not independently verified here)
  is a candidate pending explicit upstream
  license/revision and Turkish evaluation approval, with no results claimed
- the next gates are a frozen Turkish test set (at least 100 prompts), two local
  baseline passes, then an approved GPU comparison with latency/VRAM and quality evidence
- canonical T-Dev-6L is ~58M / 512 context; T-101 is an unvalidated legacy ~7B
  config, and larger targets are research concepts, not commitments

Use [the roadmap](../docs/roadmap.md) for measurable gates and
[sources](sources.md) for official documentation.
