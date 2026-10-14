# Orkinos AI / NVIDIA Inception Deck

## Slide 1 — Company overview

**Orkinosai Limited**  
UK AI company developing Turkish-first product workflows and governed model foundations.

- Near term: **Ebru**, a Turkish-first AI assistant/product layer for private and institutional use
- Foundation: **Supreme ModelTX**, an early-stage governed pretrained-model
  evaluation, adaptation, and deployment foundation supporting Ebru
- Core positioning: private deployment, auditability, local-language quality, and controlled operations

## Slide 2 — Mission and problem

**Mission:** make AI more usable, governable, and locally relevant for Turkish-language and regulated environments.

- Generic hosted AI tools do not always prioritize Turkish language depth, deployment control, or auditability
- Institutions increasingly need stronger boundaries around data flow, policy, and operational evidence
- Language quality, cultural context, and governance matter in education, public-sector, and enterprise workflows

## Slide 3 — Stage 1: Ebru

**Ebru is the near-term product direction and intended commercial wedge.**

- Turkish-first assistant and product experience direction; no customer traction is claimed here
- Intended use cases include knowledge access, document workflows, guided assistance, and customer-facing AI interactions
- Modular positioning already documented for general assistant, education/child-friendly, institutional, and document-oriented usage modes
- Can use OpenAI-compatible infrastructure in the short term to validate product, safety, and deployment patterns quickly

## Slide 4 — Target customers and immediate market

**Initial target market:** Turkish-speaking users and institutions that need more control than generic chat products provide.

- Schools, universities, and education-oriented deployments
- Public-sector and regulated organizations
- SMEs needing Turkish-language content, workflow, and support automation
- Teams that prefer private deployment or clearer governance boundaries

## Slide 5 — Why now

- AI adoption is rising, but many institutions still need safer deployment models and stronger local-language fit
- Turkish-first product evaluation can guide approved pretrained-model selection
  without requiring original large-model pretraining first
- The company already presents AI services and sector experience across government, heritage, and commercial work on its public website
- Repository evidence shows Supreme ModelTX has moved beyond concept into a reproducible early technical foundation

## Slide 6 — Stage 2: Supreme ModelTX

**Supreme ModelTX is the governed model foundation supporting Ebru.**

- `platform_api` provides API/governance scaffolding; `model_core` contains custom
  PyTorch research models, tokenizer/data/training, and local checkpoint inference
- Current repo evidence includes versioned tokenizer workflow, manifest-based data pipeline, checkpoint save/resume, validation loss and perplexity reporting, and checkpoint-backed inference
- API default remains `SMTX_INFERENCE_BACKEND=pytorch`; optional local pretrained
  inference uses `transformers`, `SMTX_PRETRAINED_MODEL_PATH`, and strict
  `SMTX_DEVICE=cpu|cuda` (unavailable requested CUDA fails)
- `foundation` TinyPilotLM checkpoints, `model_core` checkpoints, and pretrained
  snapshots are separate formats; CPU/custom workflows remain research/smoke paths
- `Qwen/Qwen2.5-0.5B-Instruct` (candidate Apache-2.0, not independently verified here)
  is a candidate pending explicit upstream
  license/revision and Turkish evaluation approval, not an approved deployment

## Slide 7 — Current status and credibility

**Current status: early, honest, and real.**

- Supreme ModelTX is an **early-stage governed pretrained-model foundation**,
  not an application-ready model or production-certified system
- CPU proxy experiments support narrow research plumbing, not pretrained Turkish
  quality; no candidate measurements or GPU runs are reported
- Canonical T-Dev-6L is ~58M / 512 context; T-101 is an unvalidated legacy ~7B
  config, with larger T-series ideas uncommitted
- Licensing, data quality, held-out Turkish evaluation, safety, reproducibility,
  and deployment controls remain gates alongside compute
- The project explicitly avoids frontier-model or production-readiness overclaims

## Slide 8 — Why NVIDIA

**NVIDIA support would accelerate the next credible milestone, not just add prestige.**

- CUDA paths exist but are not GPU-validated; no BF16 or multi-GPU validation is claimed
- **NeMo:** planned, not integrated — investigate approved Turkish adaptation
- **Megatron Core:** planned, not integrated — distributed research only if justified
- **TensorRT-LLM:** planned, not integrated — supported-model optimized serving investigation
- **ONNX:** planned, not integrated — export/runtime parity investigation
- **cuDNN:** runtime library in a compatible CUDA stack, not a model backend
- Seek Inception technical guidance on these gated choices; programme benefits
  and GPU allocation are not assumed

## Slide 9 — Differentiation

- Artifact custody, approved local inference, and accountable model selection
  matter independently of who originally pretrained the model
- Not just a model lab: Ebru provides the product wedge, user learning loop, and route to commercial validation
- Turkish-first positioning with private deployment and governance emphasis
- Conservative, evidence-based story with implemented components, documented limitations, and staged milestones

## Slide 10 — The ask

**Proposed NVIDIA Inception ask: guidance toward measurable, approved milestones.**

- Approve candidate license, immutable revision/hashes, and a frozen Turkish
  evaluation set of at least 100 prompts with predeclared thresholds
- Complete two reproducible local baseline passes before an approved one-GPU
  comparison of quality/safety, latency, tokens/s, peak VRAM, and cost
- Investigate adaptation and serving only after baseline approval and exact
  model/hardware support checks; require held-out comparison and output parity
- Gate an Ebru pilot on model card, access/retention policy, audit evidence,
  rollback test, safety review, and operator sign-off

These are planned gates, not completed experiments, customer traction, or an
application-readiness claim. See [the roadmap](../docs/roadmap.md) and
[official ecosystem sources](sources.md).
