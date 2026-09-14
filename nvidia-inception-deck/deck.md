# Orkinos AI / NVIDIA Inception Deck

## Slide 1 — Company overview

**Orkinosai Limited**  
UK AI company building practical AI products now and sovereign AI capability over time.

- Near term: **Ebru**, a Turkish-first AI assistant/product layer for private and institutional use
- Longer term: **Supreme ModelTX**, a sovereign model and governed deployment stack intended to power Ebru more directly over time
- Core positioning: private deployment, auditability, local-language quality, and controlled operations

## Slide 2 — Mission and problem

**Mission:** make AI more usable, governable, and locally relevant for Turkish-language and regulated environments.

- Generic hosted AI tools do not always prioritize Turkish language depth, deployment control, or auditability
- Institutions increasingly need stronger boundaries around data flow, policy, and operational evidence
- Language quality, cultural context, and governance matter in education, public-sector, and enterprise workflows

## Slide 3 — Stage 1: Ebru

**Ebru is the commercial wedge.**

- Turkish-first assistant and product experience
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
- Turkish-first product execution can create a practical market wedge before deeper model ownership is fully funded
- The company already presents AI services and sector experience across government, heritage, and commercial work on its public website
- Repository evidence shows Supreme ModelTX has moved beyond concept into a reproducible early technical foundation

## Slide 6 — Stage 2: Supreme ModelTX

**Supreme ModelTX is the longer-term sovereign model path behind Ebru.**

- Two-layer architecture: `model_core` for tokenizer/data/training/inference and `platform_api` for auth, tenancy, usage, audit, registry, and deployment metadata
- Current repo evidence includes versioned tokenizer workflow, manifest-based data pipeline, checkpoint save/resume, validation loss and perplexity reporting, and checkpoint-backed inference
- OpenAI-compatible API boundaries support staged integration while longer-term model independence is developed

## Slide 7 — Current status and credibility

**Current status: early, honest, and real.**

- Supreme ModelTX is documented in-repo as an **early-stage sovereign AI platform foundation**
- Current TRL self-assessment is **TRL 4**, with working POC evidence across model core, API, infrastructure, and tests
- Expanded CPU training runs show measurable learning and benchmark improvement, but GPU-backed scale-up remains the main constraint
- The project explicitly avoids frontier-model or production-readiness overclaims

## Slide 8 — Why NVIDIA

**NVIDIA support would accelerate the next credible milestone, not just add prestige.**

- Validate the canonical GPU-backed T-Dev-6L training path with CUDA/BF16 and longer schedules
- Increase corpus size, batch regime, and experiment velocity beyond CPU-only constraints
- Harden GPU inference and governed deployment pathways, including the repository's vLLM-ready serving path
- Benefit from Inception ecosystem support, technical guidance, and access to proven GPU software practices

## Slide 9 — Differentiation

- Not just a wrapper around third-party APIs: long-term plan includes owned model workflow and governed infrastructure
- Not just a model lab: Ebru provides the product wedge, user learning loop, and route to commercial validation
- Turkish-first positioning with private deployment and governance emphasis
- Conservative, evidence-based story with implemented components, documented limitations, and staged milestones

## Slide 10 — The ask

**We are applying for NVIDIA Inception to accelerate both stages of the roadmap.**

- Near-term: technical guidance and ecosystem support for Ebru deployments and institutional pilots
- Compute: GPU enablement for canonical T-Dev-6L validation, larger Turkish-language experiments, and faster iteration
- Platform: advice on efficient training/inference stack choices for secure, governed deployments
- Outcome: turn Ebru traction into a stronger sovereign-model pathway with measurable technical milestones
