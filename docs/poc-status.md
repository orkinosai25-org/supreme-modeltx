# POC Evidence Matrix — Supreme ModelTX

> **Purpose:** Prove current POC status from repository artifacts, mapping every major capability claim to concrete evidence (file paths, commits, tests, and demo scripts).  
> **Last updated:** 2026-07-21  
> **Overall verdict:** [see bottom of document](#overall-maturity-verdict)

---

## Table of Contents

1. [Model Core](#1-model-core)
2. [Platform API](#2-platform-api)
3. [Control Plane (C#)](#3-control-plane-c)
4. [Multi-Model Orchestration](#4-multi-model-orchestration)
5. [Infrastructure (Azure / Bicep)](#5-infrastructure-azure--bicep)
6. [Testing](#6-testing)
7. [Documentation](#7-documentation)
8. [Component Maturity Summary](#component-maturity-summary)
9. [Gap Summary](#gap-summary)
10. [Overall Maturity Verdict](#overall-maturity-verdict)

---

## 1. Model Core

> Python-native LLM development layer (`src/supreme_modeltx/model_core/`). Entirely independent of the platform API.

| Feature | Evidence Link | Status | Notes |
|---|---|---|---|
| T-Dev-6L transformer architecture | [`src/supreme_modeltx/model_core/models/t_series/baseline.py`](../src/supreme_modeltx/model_core/models/t_series/baseline.py) | ✅ Done | 6-layer, 512-hidden, ~25M params; CPU-runnable |
| Grouped Query Attention (GQA) + RoPE | [`src/supreme_modeltx/model_core/models/common/attention.py`](../src/supreme_modeltx/model_core/models/common/attention.py) | ✅ Done | RoPE (Su et al., 2023); GQA (Ainslie et al., 2023); uses `scaled_dot_product_attention` |
| Pydantic v2 configuration schema | [`src/supreme_modeltx/model_core/config/schema.py`](../src/supreme_modeltx/model_core/config/schema.py) | ✅ Done | `ModelConfig`, `TrainingConfig`, `DataConfig`, `TokenizerConfig`, `SMTXConfig` |
| Canonical training configs | [`configs/real_training/`](../configs/real_training/) | ✅ Done | JSON + YAML configs for T-Dev-6L first run, expanded run, and GPU corpus runs |
| SentencePiece tokenizer training workflow | [`src/supreme_modeltx/model_core/tokenizer/train.py`](../src/supreme_modeltx/model_core/tokenizer/train.py), [`workflow.py`](../src/supreme_modeltx/model_core/tokenizer/workflow.py) | ✅ Done | Versioned BPE tokenizer; trained artifact in `artifacts/tokenizers/` |
| Manifest-based dataset contract | [`src/supreme_modeltx/model_core/data/manifest.py`](../src/supreme_modeltx/model_core/data/manifest.py), [`data/manifests/`](../data/manifests/) | ✅ Done | Split-aware loaders; processed slices in `data/processed/` |
| Data sources + preprocessing | [`src/supreme_modeltx/model_core/data/sources.py`](../src/supreme_modeltx/model_core/data/sources.py), [`preprocessing.py`](../src/supreme_modeltx/model_core/data/preprocessing.py) | ✅ Done | Tokenize-and-pack pipeline; raw corpus in `data/raw/` |
| Training loop (single-process + distributed) | [`src/supreme_modeltx/model_core/training/trainer.py`](../src/supreme_modeltx/model_core/training/trainer.py) | ✅ Done | `torchrun` multi-GPU support; gradient accumulation; mixed precision (BF16/FP16) |
| Checkpoint save/resume | [`src/supreme_modeltx/model_core/training/checkpoint.py`](../src/supreme_modeltx/model_core/training/checkpoint.py) | ✅ Done | `save_checkpoint`, `load_checkpoint`, `find_latest_checkpoint` |
| Optimizer + LR scheduler | [`src/supreme_modeltx/model_core/training/optimizer.py`](../src/supreme_modeltx/model_core/training/optimizer.py), [`scheduler.py`](../src/supreme_modeltx/model_core/training/scheduler.py) | ✅ Done | AdamW, gradient clipping; cosine/linear/constant schedulers |
| Mixed-precision context + grad scaler | [`src/supreme_modeltx/model_core/training/precision.py`](../src/supreme_modeltx/model_core/training/precision.py) | ✅ Done | `get_autocast_context`, `get_grad_scaler` |
| Distributed training setup | [`src/supreme_modeltx/model_core/training/distributed/setup.py`](../src/supreme_modeltx/model_core/training/distributed/setup.py) | ✅ Done | `init_distributed`, `cleanup_distributed`, `is_main_process` |
| Inference engine (checkpoint-backed generation) | [`src/supreme_modeltx/model_core/inference/engine.py`](../src/supreme_modeltx/model_core/inference/engine.py) | ✅ Done | Loads `.pt` checkpoint + tokenizer; runs autoregressive generation |
| Token sampling strategies | [`src/supreme_modeltx/model_core/inference/sampling.py`](../src/supreme_modeltx/model_core/inference/sampling.py) | ✅ Done | `sample_tokens`; temperature, top-k, top-p |
| Perplexity evaluator | [`src/supreme_modeltx/model_core/eval/perplexity.py`](../src/supreme_modeltx/model_core/eval/perplexity.py) | ✅ Done | Runs during training validation loop; reported per checkpoint |
| Benchmark scorer (code/reasoning) | [`src/supreme_modeltx/model_core/eval/benchmark.py`](../src/supreme_modeltx/model_core/eval/benchmark.py) | ✅ Done | `keyword_ratio` and `contains` scoring; eval set at `configs/benchmark_eval_set.json` |
| First benchmarked training run (CPU) | [`docs/first-experiment-findings.md`](first-experiment-findings.md) | ✅ Done | Run `t_dev_6l_first_run`; checkpoint artifacts in `artifacts/runs/` |
| Expanded GPU training run | [`docs/first-gpu-experiment-findings.md`](first-gpu-experiment-findings.md), [`docs/t_dev_6l_first_gpu_run.md`](t_dev_6l_first_gpu_run.md) | ✅ Done | GPU corpus plan + findings documented |
| Training dataset pipeline (legacy entrypoint) | [`training/train_t101.py`](../training/train_t101.py), [`training/dataset_pipeline.py`](../training/dataset_pipeline.py) | ✅ Done | Profile-aware training; raw-text → tokenized-JSONL pipeline |

---

## 2. Platform API

> FastAPI business API layer (`src/supreme_modeltx/platform_api/`). SQLite-backed persistence.

| Feature | Evidence Link | Status | Notes |
|---|---|---|---|
| FastAPI application factory + router mounting | [`src/supreme_modeltx/platform_api/api/app.py`](../src/supreme_modeltx/platform_api/api/app.py) | ✅ Done | `create_app()`; all routers mounted; CORS middleware; OpenAPI docs at `/docs` |
| Health probe endpoint | [`src/supreme_modeltx/platform_api/api/routers/health.py`](../src/supreme_modeltx/platform_api/api/routers/health.py) | ✅ Done | `GET /health` liveness probe |
| API key issuance (scrypt-hashed, display-once) | [`src/supreme_modeltx/platform_api/auth/keys.py`](../src/supreme_modeltx/platform_api/auth/keys.py), [`api/routers/keys.py`](../src/supreme_modeltx/platform_api/api/routers/keys.py) | ✅ Done | Keys hashed with scrypt (N=16384); plain-text returned once only; `issue_key`, `revoke_key` |
| API key validation (****** dependency) | [`src/supreme_modeltx/platform_api/api/routers/auth.py`](../src/supreme_modeltx/platform_api/api/routers/auth.py) | ✅ Done | `require_api_key` FastAPI dependency; `POST /v1/auth/validate` |
| API key metadata store | [`src/supreme_modeltx/platform_api/auth/key_store.py`](../src/supreme_modeltx/platform_api/auth/key_store.py) | ✅ Done | Stores label, prefix, project_id, created_at alongside hashed keys |
| Project / tenant model | [`src/supreme_modeltx/platform_api/tenants/models.py`](../src/supreme_modeltx/platform_api/tenants/models.py), [`store.py`](../src/supreme_modeltx/platform_api/tenants/store.py) | ✅ Done | `Project`, `ProjectCreate`; SQLite-backed `ProjectStore` |
| Project CRUD endpoints | [`src/supreme_modeltx/platform_api/api/routers/projects.py`](../src/supreme_modeltx/platform_api/api/routers/projects.py) | ✅ Done | `GET/POST /v1/projects` |
| Model registry (metadata + stages) | [`src/supreme_modeltx/platform_api/model_registry/registry.py`](../src/supreme_modeltx/platform_api/model_registry/registry.py), [`api/routers/models.py`](../src/supreme_modeltx/platform_api/api/routers/models.py) | ✅ Done | `ModelEntry` with stage tags (`development/staging/production/deprecated`); default T-Dev-6L + T-101 entries |
| Usage metering (SQLite-backed token ledger) | [`src/supreme_modeltx/platform_api/usage/metering.py`](../src/supreme_modeltx/platform_api/usage/metering.py), [`api/routers/usage.py`](../src/supreme_modeltx/platform_api/api/routers/usage.py) | ✅ Done | Per-project prompt/completion token accumulation; `GET /v1/usage/{project_id}` |
| Audit log (append-only, SQLite) | [`src/supreme_modeltx/platform_api/audit/log.py`](../src/supreme_modeltx/platform_api/audit/log.py), [`api/routers/audit.py`](../src/supreme_modeltx/platform_api/api/routers/audit.py) | ✅ Done | `AuditEvent` schema; `AuditLog.record()` is append-only; events emitted per chat request |
| SQLite persistence layer | [`src/supreme_modeltx/platform_api/persistence/sqlite.py`](../src/supreme_modeltx/platform_api/persistence/sqlite.py) | ✅ Done | Shared `connect()` / `resolve_db_path()` helpers; path configurable via env var |
| OpenAI-compatible schemas | [`src/supreme_modeltx/platform_api/api/schemas.py`](../src/supreme_modeltx/platform_api/api/schemas.py) | ✅ Done | `ChatRequest/Response`, `EmbeddingsRequest/Response`, `ResponsesRequest/Response`, `KeyIssueRequest/Response` |
| Chat/completions endpoint | [`src/supreme_modeltx/platform_api/api/routers/chat.py`](../src/supreme_modeltx/platform_api/api/routers/chat.py) | ⚠️ Partial | Wired to `InferenceEngine` when checkpoint env vars set; returns HTTP 503 otherwise |
| Embeddings endpoint | [`src/supreme_modeltx/platform_api/api/routers/embeddings.py`](../src/supreme_modeltx/platform_api/api/routers/embeddings.py) | ⚠️ Partial | Schema complete; embedding computation stubbed; returns zeros |
| Responses endpoint (structured output) | [`src/supreme_modeltx/platform_api/api/routers/responses.py`](../src/supreme_modeltx/platform_api/api/routers/responses.py) | ⚠️ Partial | Schema complete; model dispatch wired in future phase |
| Inference engine integration (API gateway) | [`src/supreme_modeltx/platform_api/api/engine.py`](../src/supreme_modeltx/platform_api/api/engine.py) | ⚠️ Partial | `InferenceEngine` singleton loaded at startup if checkpoint path configured |
| Deployment service (lifecycle scaffold) | [`src/supreme_modeltx/platform_api/deployment/service.py`](../src/supreme_modeltx/platform_api/deployment/service.py) | ⚠️ Partial | `DeploymentService` with in-memory state; production backend (K8s/vLLM) not integrated |
| OpenAPI specification | [`docs/openapi.json`](openapi.json), [`docs/openapi.yaml`](openapi.yaml) | ✅ Done | Full spec exported; interactive docs at `/docs` and `/redoc` |

---

## 3. Control Plane (C#)

> ASP.NET Core 9 governance and administration layer (`control-plane/`).

| Feature | Evidence Link | Status | Notes |
|---|---|---|---|
| ASP.NET Core API project | [`control-plane/src/SMTX.ControlPlane.Api/`](../control-plane/src/SMTX.ControlPlane.Api/) | ✅ Done | REST API for control-plane operations |
| Blazor Server dashboard | [`control-plane/src/SMTX.ControlPlane.Blazor/`](../control-plane/src/SMTX.ControlPlane.Blazor/) | ✅ Done | Admin UI; seeded from `SumotxDbContext` on startup |
| Core domain models | [`control-plane/src/SMTX.ControlPlane.Core/`](../control-plane/src/SMTX.ControlPlane.Core/) | ✅ Done | Business domain entities |
| Infrastructure / EF Core persistence | [`control-plane/src/SMTX.ControlPlane.Infrastructure/`](../control-plane/src/SMTX.ControlPlane.Infrastructure/) | ✅ Done | `SumotxDbContext`; `SumotxDbContextInitialization.InitializeAsync` |
| Governed-models API | [`api/routers/governed_models.py`](../api/routers/governed_models.py) | ✅ Done | Additional governance surface over model registry |
| API governed-model tests | [`api/tests/test_governed_models.py`](../api/tests/test_governed_models.py) | ✅ Done | FastAPI test client tests for governed-model endpoints |

---

## 4. Multi-Model Orchestration

> T-Series pipeline routing (`tmodels/`, `inference/`).

| Feature | Evidence Link | Status | Notes |
|---|---|---|---|
| T-X Orchestrator (CPU/GPU routing) | [`tmodels/tx/orchestrator.py`](../tmodels/tx/orchestrator.py) | ✅ Done | Routes Prompt → T-101 → T-201 (reasoning) → T-301 (retrieval) → T-501 (verification); CPU-default, vLLM optional |
| CPU inference server (HF Transformers) | [`inference/cpu_inference_server.py`](../inference/cpu_inference_server.py) | ✅ Done | `/health` + `/generate`; `AutoModelForCausalLM`; compatible REST API with vLLM |
| vLLM GPU inference server | [`inference/vllm_server.py`](../inference/vllm_server.py) | ⚠️ Partial | Config and server scaffold present; requires GPU quota to activate |
| Retrieval service (T-301) | [`inference/retrieval_service.py`](../inference/retrieval_service.py) | ⚠️ Partial | Service scaffold; full semantic retrieval integration pending |
| Verification service (T-501) | [`inference/verification_service.py`](../inference/verification_service.py) | ⚠️ Partial | Service scaffold; dual-tower classifier integration pending |
| Model stubs (T-101, T-201, T-301, T-501) | [`tmodels/`](../tmodels/) | ⚠️ Partial | Directory structure and stub placeholders; model weights not yet promoted |

---

## 5. Infrastructure (Azure / Bicep)

> Azure cloud infrastructure definitions (`infra/`).

| Feature | Evidence Link | Status | Notes |
|---|---|---|---|
| Main Bicep deployment template | [`infra/main.bicep`](../infra/main.bicep) | ✅ Done | Top-level Azure deployment |
| RBAC Bicep module | [`infra/rbac.bicep`](../infra/rbac.bicep) | ✅ Done | Azure RBAC role assignments |
| Bicep modules | [`infra/modules/`](../infra/modules/) | ✅ Done | Modular resource definitions |
| Infra outputs | [`infra/outputs.md`](../infra/outputs.md) | ✅ Done | Documented output values |
| Azure parameters | [`infra/parameters.json`](../infra/parameters.json) | ✅ Done | Deployment parameters |
| Azure runbook | [`docs/azure-uk-gpu-runner-runbook.md`](azure-uk-gpu-runner-runbook.md) | ✅ Done | UK GPU runner setup and operation |
| VM scaling guide | [`docs/vm-scaling.md`](vm-scaling.md) | ✅ Done | GPU VM scaling strategy |
| GPU readiness + scaling plan | [`docs/gpu-readiness-scaling-plan.md`](gpu-readiness-scaling-plan.md) | ✅ Done | Full GPU training readiness assessment |

---

## 6. Testing

| Feature | Evidence Link | Status | Notes |
|---|---|---|---|
| Config schema unit tests | [`tests/unit/test_config.py`](../tests/unit/test_config.py) | ✅ Done | `ModelConfig`, `TrainingConfig`, `SMTXConfig` validation |
| Platform API unit tests | [`tests/unit/test_platform_api.py`](../tests/unit/test_platform_api.py) | ✅ Done | Auth keys, model registry, project store, usage ledger, deployment service, audit log, all schemas |
| Data manifest unit tests | [`tests/unit/test_data_manifest.py`](../tests/unit/test_data_manifest.py) | ✅ Done | Manifest loading and split contract |
| Tokenizer workflow unit tests | [`tests/unit/test_tokenizer_workflow.py`](../tests/unit/test_tokenizer_workflow.py) | ✅ Done | Tokenizer training + encoding roundtrip |
| Trainer preflight unit tests | [`tests/unit/test_trainer_preflight.py`](../tests/unit/test_trainer_preflight.py) | ✅ Done | Config validation and setup checks |
| Trainer real-run unit tests | [`tests/unit/test_trainer_real_run.py`](../tests/unit/test_trainer_real_run.py) | ✅ Done | Actual training loop execution |
| Benchmark unit tests | [`tests/unit/test_benchmark.py`](../tests/unit/test_benchmark.py) | ✅ Done | Benchmark scorer logic |
| Chat endpoint unit tests | [`tests/unit/test_chat_endpoint.py`](../tests/unit/test_chat_endpoint.py) | ✅ Done | FastAPI test client for chat completions |
| Run artifacts unit tests | [`tests/unit/test_run_artifacts.py`](../tests/unit/test_run_artifacts.py) | ✅ Done | Checkpoint + artifact path validation |
| Model smoke tests | [`tests/smoke/test_model_smoke.py`](../tests/smoke/test_model_smoke.py) | ✅ Done | `TSeriesBaseline` instantiation and forward pass |
| Governed-model API tests | [`api/tests/test_governed_models.py`](../api/tests/test_governed_models.py) | ✅ Done | FastAPI test client for governed-model routes |
| End-to-end test scaffold | [`scripts/e2e_test.py`](../scripts/e2e_test.py) | ⚠️ Partial | Script exists; requires live orchestrator URL to populate results |

---

## 7. Documentation

| Feature | Evidence Link | Status | Notes |
|---|---|---|---|
| Model core architecture doc | [`docs/architecture/model-core.md`](architecture/model-core.md) | ✅ Done | Complete module breakdown with usage examples |
| Platform API architecture doc | [`docs/architecture/platform-api.md`](architecture/platform-api.md) | ✅ Done | Full router, schema, and persistence design |
| Architecture overview | [`docs/architecture/overview.md`](architecture/overview.md) | ✅ Done | System-level diagram and component relationships |
| Sovereign AI application brief | [`docs/sovereign-ai/application-brief.md`](sovereign-ai/application-brief.md) | ✅ Done | TAI programme positioning and repository evidence summary |
| Repository readiness review | [`docs/repository-readiness-review.md`](repository-readiness-review.md) | ✅ Done | Honest assessment of ready vs. partial vs. missing areas |
| Development roadmap | [`docs/roadmap.md`](roadmap.md) | ✅ Done | Phase 1–5 milestone plan |
| 90-day delivery plan | [`docs/delivery-plan-90d.md`](delivery-plan-90d.md) | ✅ Done | Detailed 90-day execution plan |
| Risk register | [`docs/risk-register.md`](risk-register.md) | ✅ Done | Risk identification and mitigations |
| Evaluation methodology | [`docs/evaluation.md`](evaluation.md) | ✅ Done | Evaluation framework description |
| Benchmarking guide | [`docs/benchmarking.md`](benchmarking.md) | ✅ Done | Benchmark tooling and methodology |
| Dataset overview | [`docs/dataset-overview.md`](dataset-overview.md) | ✅ Done | Training corpus composition |
| First experiment findings | [`docs/first-experiment-findings.md`](first-experiment-findings.md) | ✅ Done | CPU training run metrics and validation loss |
| Expanded experiment findings | [`docs/expanded-experiment-findings.md`](expanded-experiment-findings.md) | ✅ Done | Extended run analysis |
| First GPU experiment findings | [`docs/first-gpu-experiment-findings.md`](first-gpu-experiment-findings.md) | ✅ Done | GPU run metrics |
| Run artifacts documentation | [`docs/run-artifacts.md`](run-artifacts.md) | ✅ Done | Checkpoint artifact catalogue |
| GPU corpus training plan | [`docs/t_dev_6l_gpu_corpus_plan.md`](t_dev_6l_gpu_corpus_plan.md) | ✅ Done | Corpus scale-up plan |
| OpenAPI specification | [`docs/openapi.json`](openapi.json), [`docs/openapi.yaml`](openapi.yaml) | ✅ Done | Machine-readable API contract |
| SUMOTX API documentation | [`docs/sumotx/api.md`](sumotx/api.md) | ✅ Done | Full API surface documentation |

---

## Component Maturity Summary

| Component | Done | Partial | Missing | Maturity |
|---|---|---|---|---|
| Model Core (architecture, training, tokenizer, inference, eval) | 16 | 0 | 0 | **POC — Complete** |
| Platform API (auth, registry, metering, audit, projects) | 12 | 5 | 0 | **POC — Core complete; inference stub** |
| Control Plane (C#, Blazor, EF Core, governed-models) | 6 | 0 | 0 | **POC — Complete** |
| Multi-Model Orchestration (T-X, CPU server, vLLM, T-301, T-501) | 2 | 4 | 0 | **POC — CPU path complete; GPU/retrieval partial** |
| Infrastructure (Bicep, RBAC, runbooks) | 8 | 0 | 0 | **POC — Complete** |
| Testing (unit, smoke, e2e) | 11 | 1 | 0 | **POC — Strong** |
| Documentation (architecture, roadmap, findings, API spec) | 17 | 0 | 0 | **POC — Strong** |
| RBAC / permission enforcement (API middleware) | 0 | 0 | 1 | **Missing — planned (Issue #5)** |
| Policy engine (model/data/region controls) | 0 | 0 | 1 | **Missing — planned (Issue #8)** |
| Multi-tenant isolation enforcement | 0 | 1 | 0 | **Partial — project model exists; cross-tenant blocking not enforced** |
| Scoped token lifecycle (revocation, expiry) | 0 | 1 | 0 | **Partial — revocation exists; expiry not enforced** |
| Tamper-evident audit chaining | 0 | 0 | 1 | **Missing — planned (Issue #7)** |
| Experiment tracking / run lineage | 0 | 0 | 1 | **Missing — planned (Issue #11)** |
| Production serving hardening (SLO, autoscaling) | 0 | 0 | 1 | **Missing — planned (Issue #12)** |

---

## Gap Summary

The following capabilities are absent or incomplete relative to pilot/deployment readiness. Each maps to an open issue in the backlog.

| Gap | Severity | Effort | Linked Issue |
|---|---|---|---|
| RBAC middleware enforcement (role-based API guards) | High | Medium | Issue #5 |
| Policy engine (model/data/region/retention controls) | High | High | Issue #8 |
| Tamper-evident audit log chaining (hash-chain events) | High | Medium | Issue #7 |
| Multi-tenant cross-tenant isolation enforcement | High | Medium | Issue #4 |
| Scoped API key expiry and full token lifecycle | Medium | Low | Issue #6 |
| Model promotion/rollback workflow | Medium | Medium | Issue #9 |
| Inference provider abstraction (CPU → GPU swappable) | Medium | Medium | Issue #10 |
| Evaluation harness (reproducible, CI-friendly) | Medium | Medium | Issue #11 |
| Usage metering + cost rollups + operational runbooks | Medium | Low | Issue #12 |
| Experiment tracking / artifact governance integration | High | High | Not yet filed |
| End-to-end production serving hardening | High | High | Issue #12 |

---

## Overall Maturity Verdict

**Supreme ModelTX is a credible, functional POC.**

The repository demonstrates a real end-to-end foundation:

- A working local transformer training pipeline with documented run results (loss, perplexity, checkpoints).
- A functional FastAPI platform API with auth, project management, model registry, usage metering, and audit logging — all backed by SQLite persistence.
- An operational C# control-plane with Blazor UI and EF Core persistence.
- A CPU-capable multi-model orchestration path (T-X routing, HF Transformers inference server).
- Azure Bicep infrastructure definitions sufficient for a first cloud deployment.
- A healthy test suite (11 unit test modules + smoke tests) providing reasonable coverage of core behaviour.

**However, it is not yet pilot-ready or deployment-ready.** The primary blockers are:

1. No RBAC middleware enforcement — any authenticated key can call any route.
2. No policy engine — no configurable model/data/region/retention controls.
3. No tamper-evident audit chaining — audit events are stored but not hash-linked.
4. No enforced multi-tenant isolation — projects exist as metadata but cross-project access is not blocked at the service layer.
5. Inference integration is conditional — live model responses require a configured checkpoint path; without it the API returns HTTP 503.
6. GPU-scale training has been planned and runbook-documented but not yet validated at full scale in production.

**Recommended classification:** **TRL 3–4** (proof-of-concept demonstrated in laboratory/controlled environment; key sub-systems validated individually).  
**Target for pilot-prep:** TRL 5 (technology validated in operationally relevant environment with security, governance, and reliability controls in place).
