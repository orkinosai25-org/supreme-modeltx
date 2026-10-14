# Supreme ModelTX / TAI Messaging Guide

## 1) Approved positioning language

Use:

- “Supreme ModelTX (TAI) is an early-stage governed pretrained-model evaluation, adaptation, and deployment foundation supporting Ebru.”
- “The platform emphasizes approved local artifacts, controlled deployment, and auditable operations.”
- “Current capabilities are implemented in repository artifacts and tests; production hardening is on the roadmap.”

## 2) How to describe TAI / Supreme ModelTX

Preferred description:

> Supreme ModelTX supports Ebru's Turkish-first direction through governed
> evaluation of existing pretrained models, with adaptation and deployment as
> gated next steps. Custom PyTorch models and CPU workflows remain research and
> smoke-test paths alongside API and governance scaffolding.

Short form:

> Governed model evaluation and deployment foundations for Turkish-first AI.

## 3) What not to claim

Do **not** claim:

- AGI or “synthetic intelligence” breakthroughs
- frontier benchmark leadership without published reproducible evidence
- full production readiness across all components
- complete parity with every hosted AI provider API feature
- NVIDIA NeMo, Megatron Core, TensorRT-LLM, or ONNX integration: all are planned, not integrated
- completed CUDA/BF16 or multi-GPU validation merely because code/configs exist
- GPU access as the only blocker, application readiness, or an approved Inception application
- approved Qwen deployment or measured Turkish pretrained results: candidate review remains pending

## 4) How to compare with ChatGPT / Gemini / Claude / Copilot

Use this structure:

- “Hosted assistants prioritize convenience on provider-managed infrastructure.”
- “Supreme ModelTX focuses on sovereign controls: artifact custody, deployment boundaries, and auditable records.”
- “This project can interoperate through compatible API patterns while serving different governance requirements.”

Avoid this structure:

- dismissive language (“they are all the same”)
- superiority claims without evidence
- zero-sum framing that implies hosted tools are useless

## 5) How to talk about sovereign AI credibly

Use concrete control language:

- where execution happens
- how model/tokenizer/checkpoint artifacts are managed
- what persistence and audit evidence is retained
- how deployment decisions are governed

Model sovereignty means approved artifact custody and accountable operation; it
does not require original pretraining. `Qwen/Qwen2.5-0.5B-Instruct` is a candidate
with a candidate Apache-2.0 license, not independently verified here, pending
explicit upstream license/revision and Turkish
evaluation approval. Do not confuse this with an endorsement or a deployed model.
Canonical T-Dev-6L is ~58M / 512 context; T-101 is an unvalidated legacy ~7B config.
Larger targets are research ideas, not commitments. TinyPilotLM foundation
checkpoints are separate from model_core checkpoints and pretrained snapshots.

CUDA paths exist but are not GPU-validated. cuDNN is a runtime library, not a
model backend. Use [the roadmap](../roadmap.md) for measurable gates and
[GPU planning](../gpu-readiness-scaling-plan.md) for official ecosystem links.

Keep claims evidence-based:

- point to implemented components, configs, and tests
- clearly separate “implemented now” from “planned next”
- prefer operational proof over visionary wording

## 6) Consistency checklist for future copy

Before publishing, confirm copy:

- distinguishes implemented capability vs roadmap
- includes at least one concrete repo-backed evidence reference
- avoids hype and unverifiable superiority language
- reflects the project as “early but credible foundation”
