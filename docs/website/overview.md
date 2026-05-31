# Supreme ModelTX (TAI) — Website Overview

## What Supreme ModelTX is

Supreme ModelTX (TAI) is an early-stage sovereign AI platform foundation focused on controlled model development and controlled deployment.

It combines two working layers:

- **`model_core`**: model architecture, tokenization, dataset manifest contracts, training/evaluation, and local checkpoint-backed inference
- **`platform_api`**: OpenAI-compatible API surface with authentication, project tenancy, usage metering, audit events, model registry metadata, and deployment metadata

## How this differs from generic hosted AI products

Generic hosted AI products optimize for convenience in a provider-owned stack. Supreme ModelTX is being built for environments that require control over model artifacts, runtime boundaries, and operational evidence.

Current design priorities are:

- local artifact ownership (tokenizer, checkpoint paths)
- explicit model lifecycle metadata
- persisted platform state (projects, usage, audit, registry)
- API compatibility with room for sovereign deployment controls

## Sovereign AI angle

In this project, “sovereign AI” means practical control rather than slogans:

- where models run
- where data flows
- who controls deployment and promotion decisions
- what evidence exists for usage and changes over time

The repository is structured to support customer-controlled or policy-constrained environments, not only managed SaaS usage.

## Capabilities currently implemented in this repository

- real model-core path with `TDev6LMini` and transformer primitives
- tokenizer workflow with versioned SentencePiece artifacts
- manifest-backed train/validation split loading
- first end-to-end local T-Dev-6L training run with checkpoint save/resume
- validation loss and perplexity reporting in trainer flow
- checkpoint-backed inference engine path configurable via local artifacts
- SQLite-backed persistence in platform API stores for projects, usage, audit events, and model registry metadata
- OpenAI-compatible `/v1/*` API scaffold with auth, model listing, and usage endpoints

## Current limitations (honest scope)

- this is an **early but credible foundation**, not a frontier-trained model release
- no claim of benchmark leadership or AGI-level capability
- no production-scale distributed training throughput claims
- some API surfaces remain scaffold/stub-level for full parity behavior
- production multi-region hardening and enterprise operations are roadmap work

## Why controlled deployment, audit, persistence, and model ownership matter

For regulated, mission-critical, or national-interest use cases, model quality is only part of the requirement. Teams also need:

- **controlled deployment**: enforce where and how model workloads execute
- **auditability**: retain evidence of requests, model selection, and operational events
- **persistence**: preserve usage, tenant, and model metadata beyond process lifetime
- **model ownership**: maintain direct control of checkpoints, tokenizers, and promotion state

These controls reduce platform risk, improve accountability, and make long-term governance possible.
