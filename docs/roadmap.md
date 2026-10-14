# SMTX Evidence-Gated Development Roadmap

Supreme ModelTX is an early-stage governed pretrained-model evaluation,
adaptation, and deployment foundation supporting Ebru. Phases below are
conditional proposals, not calendar promises, completed runs, or funded commitments.

## Current foundation

- Lightweight API, local checkpoint inference, governance metadata, and CPU
  training/evaluation scaffolding exist.
- Optional local Transformers inference is separate from the default `pytorch`
  backend. A local snapshot and strict `SMTX_DEVICE=cpu|cuda` are required.
- CUDA paths exist but are not GPU-validated. No multi-GPU validation is claimed.
- `foundation` TinyPilotLM and `model_core` TSeriesBaseline have distinct
  checkpoint formats. Canonical T-Dev-6L is ~58M / 512 context; smaller CPU proxy
  results are not validation of that configuration.
- T-101 is an unvalidated legacy ~7B config. Larger T-series targets are research
  concepts, not near-term product dependencies.

## Gate 1 — Candidate and data approval

- [ ] Review `Qwen/Qwen2.5-0.5B-Instruct` as a candidate only (candidate
  Apache-2.0, not independently verified here); obtain explicit upstream approval.
- [ ] Record license/notice review, immutable revision, artifact hashes, intended
  use, and owner approval before inference; no automatic Hub downloads.
- [ ] Freeze a held-out Turkish evaluation set of at least 100 prompts spanning
  instruction following, grounding, language quality, and safety.
- [ ] Approve dataset provenance, consent/redaction rules, and numerical quality,
  safety, latency, and memory thresholds before evaluating.

**Exit evidence:** signed-off candidate manifest, dataset manifest, frozen
evaluation version, and predeclared acceptance thresholds. Approval is pending.

## Gate 2 — Reproducible local pretrained baseline

- [ ] Complete a CPU baseline on the approved local snapshot.
- [ ] Archive all prompt outputs, scores, failures, model/tokenizer hashes, config,
  package versions, device, and timing/memory measurements.
- [ ] Repeat the evaluation twice with the same inputs and seeds, documenting
  any nondeterminism.
- [ ] Verify missing artifacts and unavailable requested CUDA produce an explicit
  failure, not an unapproved fallback.

**Exit evidence:** complete results for the frozen prompt set, threshold review,
and reproducibility report. No pretrained results are claimed today.
The optional four-case Turkish lexical suite in
`configs/foundation/eval_cases/turkish-baseline-v1.yaml` and CPU/float32
`configs/pretrained_baseline.json` provide smoke scaffolding only; they do not
satisfy the at-least-100-prompt held-out gate.

## Gate 3 — Approved single-GPU evaluation

- [ ] Obtain budget and hardware approval; record GPU, driver, PyTorch, CUDA,
  cuDNN, and precision compatibility.
- [ ] Repeat the same baseline workload on one GPU and compare output quality.
- [ ] Measure warm/cold latency, tokens/s, peak VRAM, failure rate, and cost/run;
  record workload and batch/context settings for meaningful comparisons.
- [ ] Require zero runtime failures in the approved test workload and satisfaction
  of predeclared quality/safety and resource thresholds before promotion.

**Exit evidence:** archived CPU/GPU comparison and explicit approval. Compute
access alone does not resolve licensing, data, safety, or deployment gaps.

## Gate 4 — Adaptation and serving feasibility

- [ ] Investigate **NeMo** for adaptation on approved Turkish data; compare with
  the unchanged baseline on the frozen held-out set and report safety regressions.
- [ ] Investigate **TensorRT-LLM** only after checking exact model/hardware support;
  require output-quality parity within approved tolerance and measured latency/VRAM.
- [ ] Investigate **ONNX** export/runtime compatibility with output-parity checks.
- [ ] Consider **Megatron Core** distributed research only after single-GPU
  evidence establishes a need, a supported recipe, and an approved budget.

All four integrations are **planned, not integrated**. cuDNN is a compatible
runtime library, not a serving backend. No distributed validation is implied.

## Gate 5 — Governed Ebru pilot

- [ ] Approve a model card, deployment boundary, artifact registry entry, rollback
  procedure, retention/access policy, and audit evidence.
- [ ] Evaluate Turkish user workflows and safety with an approved pilot cohort.
- [ ] Review against the frozen acceptance thresholds; block promotion on failed
  safety or governance gates, even if throughput improves.

**Exit evidence:** pilot review and operator sign-off, not automatic public release.

Official ecosystem references are in the
[GPU readiness plan](gpu-readiness-scaling-plan.md).
