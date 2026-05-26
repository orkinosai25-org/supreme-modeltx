# SUMOTX Overview

SUMOTX is a governed AI control plane that runs in customer-owned Azure environments.

It helps organizations control the AI lifecycle inside enterprise boundaries:

- register models and execution profiles
- enforce policy and approval gates
- deploy governed AI workloads
- ground responses with enterprise knowledge sources
- audit decisions and operations end-to-end

---

## What SUMOTX Enables

- **Customer-owned deployment** — resources run in the customer subscription
- **Control-plane governance** — policy, approvals, lifecycle state, and orchestration
- **Grounded enterprise AI** — retrieval/verification patterns for enterprise data (including SharePoint ingestion flows)
- **Auditability** — decision traceability for regulated environments
- **Provider flexibility** — open or hosted model providers behind the same governance boundary

---

## Product Positioning

> SUMOTX is the governed AI control plane for customer-boundary deployments.
>
> It is not a chatbot product and not a GPU-first hosting platform. It is the operating layer that governs how AI is approved, deployed, grounded, and audited.

---

## Execution Strategy

- **Default now:** CPU-first governed deployments
- **Optional:** GPU-accelerated execution where customer policy and quota permit
- **Future state:** deeper model programs (including T-Series expansion) under the same governance lifecycle

---

## What SUMOTX Deliberately Does NOT Do

- ❌ Force SaaS data custody outside customer boundaries
- ❌ Assume GPU-heavy execution as the default architecture
- ❌ Depend on a single model provider
- ❌ Treat governance as an afterthought

---

## Further Reading

- [Architecture](architecture.md)
- [API Reference](api.md)
- [Deployment Model](deployment.md)
- [Technology Stack](stack.md)
