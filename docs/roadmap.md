# SMTX Development Roadmap

## Phase 1 — Foundation (Weeks 1–3)

- [ ] Initialize repo and documentation
- [ ] Implement repo directory structure
- [ ] Add T‑101 config + tokenizer specification
- [ ] Implement training loop using DeepSpeed FSDP
- [ ] Dataset ingestion pipeline
- [ ] Azure distributed training skeleton
- [ ] First passing unit tests

## Phase 2 — Initial LLM (Weeks 4–8)

- [ ] Train T‑101 prototype on Azure GPU
- [ ] Basic inference with vLLM
- [ ] Connect Supreme Model T‑X UI
- [ ] Add benchmark runner (GSM8K, ARC)
- [ ] Test 4‑bit quantized inference
- [ ] Release v0.1.0 checkpoint

## Phase 3 — Reasoning Expansion (Weeks 9–12)

- [ ] Build T‑201 reasoning module
- [ ] Distillation from synthetic CoT dataset
- [ ] ORPO / DPO fine‑tuning pipeline
- [ ] Integrate T‑201 into T‑X orchestrator
- [ ] Evaluation: GSM8K, MATH, ARC

## Phase 4 — Retrieval + Verification (Weeks 12–20)

- [ ] Add T‑301 embedding store + RAG pipeline
- [ ] Add T‑501 evidence scoring module
- [ ] Dual‑tower verification classifier
- [ ] Expand T‑Chain distributed compute
- [ ] Full orchestration flow end‑to‑end

## Phase 5 — Public Release (6 months)

- [ ] v1.0.0 open release on Hugging Face
- [ ] Technical report / whitepaper
- [ ] Azure deployment guide
- [ ] Model cards for all T‑Series models
- [ ] Community contribution guidelines
