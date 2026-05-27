# Sovereign AI Principles

`supreme-modeltx` is designed around the belief that powerful AI should be **governable, auditable, and owned by those it serves** — not by foreign hyperscalers or opaque third parties.

These principles guide every architectural and product decision.

---

## 1. British Sovereign Ownership

All model weights, training data, checkpoints, and inference infrastructure must be deployable within UK-controlled data centres and legal boundaries.

**Commitments:**
- Default infrastructure targets UK Azure regions
- No mandatory reliance on US or EU-controlled AI APIs for production inference
- Model weights are owned by the operator, not licensed from a third party

---

## 2. Trainable and Governable

The platform must support the full training lifecycle — from data ingestion through checkpointing to evaluation — without requiring external permission or external infrastructure.

**Commitments:**
- Open training pipeline based on PyTorch and Accelerate
- Reproducible data pipeline with documented provenance
- Checkpoint formats compatible with open tooling

---

## 3. API-First for Enterprises

Businesses must be able to integrate once and adopt better models over time without changing their integration.

**Commitments:**
- Stable, versioned API surface (`/v1/...`)
- OpenAI-compatible request/response schema where applicable
- Clear model versioning and routing in the registry

---

## 4. Auditable

Every significant action on the platform — key issuance, model deployment, API call, training job — must be logged and attributable.

**Commitments:**
- Per-request audit trail
- Token-level usage metering
- Model registry records deployment history
- No silent model substitution without operator notification

---

## 5. Provider-Independent

The platform must not have a hard dependency on any single cloud provider's AI services for core model functionality.

**Commitments:**
- Inference is served from sovereign weights, not third-party model APIs
- Infrastructure is deployable to any UK-region cloud or on-premises hardware
- Dependencies are open-source and vendorable

---

## 6. Portable

Model weights, tokenizers, configurations, and checkpoints use open formats so that operators can migrate, archive, or inspect them independently.

**Commitments:**
- PyTorch `.pt` checkpoint format
- HuggingFace-compatible config and tokenizer formats
- Pydantic-serialisable configuration objects

---

## 7. Designed for UK Business and Government Use Cases

The model family and API surface will be developed with UK enterprise and public-sector requirements in mind.

**Planned domain focus areas:**
- UK legal and regulatory text
- UK financial services
- UK public sector and government
- UK engineering and scientific domains

---

## Third-Party Attribution

Where components are inspired by or derived from open-source projects, we document provenance transparently. See [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
