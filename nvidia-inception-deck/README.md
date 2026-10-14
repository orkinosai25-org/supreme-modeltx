# NVIDIA Inception deck package

This folder contains draft materials for review before an NVIDIA Inception
application. They are not an application-readiness assessment, evidence of
programme admission, or a promise of GPU allocation.

## Files

- `deck.md` — concise 10-slide pitch deck in Markdown
- `application-description.txt` — 200–1,000 character company/application description for the form
- `founder-talking-points.md` — one-page narrative and talking points for calls or form review
- `sources.md` — grounding references from this repository and the public company website

## Positioning

The package uses a two-stage story:

1. **Ebru** as the near-term commercial Turkish-first AI product
2. **Supreme ModelTX** as an early-stage governed pretrained-model evaluation,
   adaptation, and deployment foundation supporting Ebru

All claims are intentionally conservative and separated into:

- what is implemented now
- what is planned next

CPU/custom models remain research and smoke-test paths. CUDA code exists but is
not GPU-validated; NeMo, Megatron Core, TensorRT-LLM, and ONNX are planned, not
integrated. Candidate model approval and measured Turkish/GPU results are pending.
Use [the gated roadmap](../docs/roadmap.md) and
[official resources](sources.md) when reviewing claims.
