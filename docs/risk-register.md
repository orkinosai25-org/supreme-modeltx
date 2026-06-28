# Risk Register

Version: 0.1.0  
Last updated: 2026-06

---

## Format

| ID | Category | Risk description | Likelihood | Impact | Overall | Mitigation | Owner | Status |
|---|---|---|---|---|---|---|---|---|
| R01–R15 — see rows below | | | H/M/L | H/M/L | H/M/L | | | Open/Mitigated/Closed |

---

## Technical Risks

| ID | Risk | Likelihood | Impact | Overall | Mitigation |
|---|---|---|---|---|---|
| R01 | GPU quota not available in UK Azure region within 30 days | M | H | H | Pre-request Azure UK South/UK West quota; CPU-only fallback for Phase 1 deliverables |
| R02 | Training instability at T-101 scale (7B) — gradient explosion, NaN loss | M | M | M | Gradient clipping already in trainer; validate on T-Dev-6L first; incremental scale-up |
| R03 | SentencePiece tokeniser vocab mismatch when switching corpora | L | M | L | Tokeniser versioning enforced in manifest contract; version pinned in all configs |
| R04 | PyTorch or CUDA version incompatibility on Azure GPU VMs | M | M | M | Pin `torch==2.3.x` in `requirements.txt`; use validated Docker images (`scripts/Dockerfile`) |
| R05 | Checkpoint storage costs exceed budget at scale | L | L | L | Use Azure Blob lifecycle policies; keep only N best checkpoints per run |
| R06 | Distributed training (FSDP) adds deadlock or OOM risk | M | M | M | Default to single-GPU; FSDP only after single-GPU baseline is stable |

---

## Data and Provenance Risks

| ID | Risk | Likelihood | Impact | Overall | Mitigation |
|---|---|---|---|---|---|
| R07 | Training data contains PII or restricted content | M | H | H | Manifest-level data provenance; pre-processing audit step; use public/open datasets only for initial runs |
| R08 | Third-party dataset licence incompatible with sovereign deployment | M | H | H | Review licence per dataset source in manifest; document in `THIRD_PARTY_NOTICES.md` |
| R09 | Data pipeline produces inconsistent splits between runs | L | M | L | Manifest-driven deterministic splits; checksums in manifest |

---

## Governance and Compliance Risks

| ID | Risk | Likelihood | Impact | Overall | Mitigation |
|---|---|---|---|---|---|
| R10 | UK AI Act / DSIT policy requirements change before deployment | M | H | H | Track DSIT AI Safety Institute publications; keep governance layer modular and policy-configurable |
| R11 | Missing model card leads to fund rejection | M | H | H | Draft model card (T-Dev-6L) by Day 60; align with CDEI/DSIT template |
| R12 | Audit log gaps in control-plane governance workflow | M | H | H | Audit event pipeline wired in `platform_api/`; end-to-end integration test by Day 60 |
| R13 | API key / secret leakage in repository | L | H | M | `.env.example` template committed (no real values); CI secret scan on all PRs; `.gitignore` excludes `.env` |

---

## Operational and Strategic Risks

| ID | Risk | Likelihood | Impact | Overall | Mitigation |
|---|---|---|---|---|---|
| R14 | Key engineer unavailability disrupts delivery | M | H | H | Document all run procedures in scripts and runbooks (`docs/azure-uk-gpu-runner-runbook.md`) |
| R15 | Competitor sovereign AI product announced, reducing differentiation | M | M | M | Emphasise governance-first, UK data residency, control-plane architecture; accelerate public-sector use case |

---

## Risk Summary

| Severity | Count |
|---|---|
| High overall | R01, R07, R08, R10, R11, R12, R14 (7) |
| Medium overall | R02, R03, R04, R06, R13, R15 (6) |
| Low overall | R05, R09 (2) |

All High risks have documented mitigations. Residual risk after mitigation is assessed as Medium or below.

---

## Review Schedule

| Review | Target date | Owner |
|---|---|---|
| Initial register (v0.1) | Day 1 | PM/Eng |
| Mid-point review (v0.2) | Day 45 | PM/Eng |
| Final review before demo (v0.3) | Day 85 | PM/Eng |

---

## References

- [`docs/delivery-plan-90d.md`](delivery-plan-90d.md) — 90-day delivery plan and go/no-go criteria
- [`docs/evaluation.md`](evaluation.md) — evaluation framework including safety assessment
- [`docs/azure-uk-gpu-runner-runbook.md`](azure-uk-gpu-runner-runbook.md) — operational runbook
- [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) — open-source provenance and licences
