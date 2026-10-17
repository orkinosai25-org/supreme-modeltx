# GPU Readiness and Evidence-Gated Scaling Plan

Supreme ModelTX is an early-stage governed pretrained-model evaluation,
adaptation, and deployment foundation supporting Ebru. GPU access can enable
experiments, but is not the only blocker and does not establish application
readiness. Licensing, data quality, Turkish evaluation, safety, reproducibility,
and operational controls are independent gates.

This document proposes work; it reports no new measurements, candidate results,
actual GPU executions, or validated NVIDIA integrations.

## 1) Current evidence boundary

- CPU custom-model workflows provide research and smoke-test infrastructure.
  Existing small proxy artifacts are directional engineering evidence, not
  broad generalization, Turkish product quality, or deployment readiness.
- Canonical `model_core` T-Dev-6L is approximately **58M parameters / 512-token
  context** (`configs/t_dev_6l.json`). Smaller 256-hidden / 128-context CPU
  proxies are not canonical-model GPU validation.
- `foundation` TinyPilotLM uses its own tokenizer/checkpoint format; it is not
  `model_core` TSeriesBaseline or a Transformers pretrained snapshot.
- T-101 is an **unvalidated legacy ~7B config**. Larger T-series targets and
  distributed training are research options, not commitments.
- CUDA and mixed-precision code paths exist but are **not GPU-validated**.
  Distributed initialization does not establish multi-GPU correctness.

## 2) Integration status matrix

| Component | Status today | Proposed role | Evidence before adoption |
|---|---|---|---|
| PyTorch CUDA | Existing code paths; not GPU-validated | Single-GPU inference/research | Compatible host, version inventory, complete approved run |
| cuDNN | Runtime library where provided by compatible CUDA/PyTorch stack | Runtime support, not an independent backend | Record versions and compatibility; do not infer performance |
| NVIDIA NeMo | Planned; not integrated | Pretrained adaptation | Supported model/data recipe and held-out comparison |
| NVIDIA Megatron Core | Planned; not integrated | Distributed research if justified | Single-GPU baseline, budget, synchronization/checkpoint validation |
| NVIDIA TensorRT-LLM | Planned; not integrated | Optimized GPU serving | Supported model export/build and output-quality parity |
| ONNX | Planned; not integrated | Portable export/runtime investigation | Supported export and output-parity checks |

No runtime dependency or toolkit installation is implied by this matrix.

## 3) Candidate and local runtime contract

`Qwen/Qwen2.5-0.5B-Instruct` is a small **candidate only**, with a candidate
Apache-2.0 license not independently verified here (upstream access was blocked
by DNS in this task). Explicit upstream approval requires review of the exact snapshot's
license/notices, immutable revision and hashes, intended use and safety, and
Turkish evaluation design. Approval is pending; no measured results are claimed.

The existing API backend remains `SMTX_INFERENCE_BACKEND=pytorch` by default.
Optional local pretrained inference uses:

```bash
export SMTX_INFERENCE_BACKEND=transformers
export SMTX_PRETRAINED_MODEL_PATH=/absolute/path/to/approved/local-snapshot
export SMTX_DEVICE=cpu  # or cuda on an approved compatible host
```

The snapshot must include local model/tokenizer artifacts. This is not a Hub-ID
download interface. Pretrained device selection is strict: unavailable requested
CUDA must fail, not silently fall back. The optional runtime is not bundled in
the default lightweight API image. This backend supplies inference, not a NeMo
adaptation pipeline or TensorRT-LLM/ONNX execution engine.

The provider requires the existing `foundation` extra and a verified
`smtx-artifact.json`: source model, immutable 40-character lowercase hexadecimal
revision, reviewed license, matching Hugging Face `model_type`, conversion
lineage, and SHA-256 inventory of every other snapshot file. Model/tokenizer
configs, safetensors weights, and a tokenizer chat template are required.
Loading is local-only with remote code disabled; defaults are CPU float32 and
CUDA BF16 on supported hardware. Runtime registry descriptions retain artifact
metadata. See [README snapshot preparation](../README.md#required-snapshot-manifest)
for manifest generation using existing tooling, not a new helper script.

After approval/preparation, opt into the existing foundation evaluation harness:

```bash
python -m supreme_modeltx.foundation.evaluation --config configs/pretrained_baseline.json
```

The example config selects CPU/float32 and the illustrative local path
`/approved/models/qwen2.5-0.5b-instruct`; set its path to your approved snapshot.
Its four-case `turkish-baseline-v1.yaml` suite is **lexical smoke coverage**, not
the larger held-out quality/safety gate. Reports include provenance, token
counts, serial generation latency/throughput, peak allocated CUDA VRAM where
applicable, and optional generation-only cost using
`evaluation.compute_cost_per_hour_usd`. CPU VRAM and an unspecified cost rate
are `null`; model loading, warmup, and idle costs are excluded. No downloaded
model or executed pretrained results are reported.

### Dense Mistral local demo

The opt-in `MistralProvider` reuses local Transformers inference, not an
NVIDIA-specific runtime. Scope is dense `model_type: "mistral"` snapshots only;
Mixtral, multimodal models, GGUF and unmerged adapters are excluded. This is
backend enablement, not an Ebru transport switch or production Azure OpenAI
replacement. No real pretrained Mistral snapshot or GPU run is validated here.
Validation against a separately approved real snapshot remains an outstanding
gate; passing synthetic tests does not establish migration readiness.

1. Separately approve the exact model's license/notices, immutable revision,
   intended use and tokenizer/chat template. Follow
   [snapshot preparation](../README.md#required-snapshot-manifest); hashes
   establish integrity, not approval. Set both manifest `architecture` and
   config `model_type` to `mistral`. Include `tokenizer.json` or
   `tokenizer.model`, `tokenizer_config.json` with a chat template, and
   `model.safetensors` or its complete index and every referenced shard.
2. Install the existing `api,foundation` extras on an isolated demo host,
   keep the approved snapshot read-only, and select the backend:

   ```bash
   export SMTX_INFERENCE_BACKEND=mistral
   export SMTX_PRETRAINED_MODEL_PATH=/approved/models/dense-mistral-demo
   export SMTX_DEVICE=cpu
   export SMTX_INFERENCE_DTYPE=float32
   export SMTX_MODEL_ID=dense-mistral-demo
   smtx-serve
   ```

   `SUPREME_MODELTX_DEVICE` overrides `SMTX_DEVICE`. CPU float32 may require
   substantial RAM (7B weights alone are roughly 28 GB). CUDA and supported
   precision require separate host validation; unavailable CUDA fails closed.
3. Exercise the non-streaming endpoint with a securely configured API key:

   ```bash
   curl http://localhost:9000/v1/chat/completions \
     --oauth2-bearer "$SMTX_API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"model":"dense-mistral-demo","messages":[{"role":"user","content":"Merhaba"}],"max_tokens":64,"temperature":0}'
   ```

4. Set the approved snapshot path in `configs/mistral_baseline.json`, then run
   `python -m supreme_modeltx.foundation.evaluation --config configs/mistral_baseline.json`.
   This is an illustrative CPU/float32 lexical-smoke config, not an approved
   pretrained baseline or a held-out quality/safety evaluation.

Missing or incompatible artifacts leave API inference disabled (503); startup
logs retain the error. Health is not a readiness check. Evaluation raises
rather than substituting a stub or another backend. Loading stays local-only,
with remote code disabled and safetensors weights required.

Tests use mocked loaders and a tiny random model with a synthetic tokenizer
and template: they check wiring, provenance and fail-closed behavior only.
They do not prove real pretrained/tokenizer compatibility or product quality.
Templates are not rewritten; system roles and conversation ordering may be
rejected. Any future Ebru integration needs independent validation of message
formats, authentication, URLs, streaming, tools and structured outputs, plus
the approval gates below. No general Mistral-family or NVIDIA integration is
implied.

## 4) Minimum plan — licensed, reproducible baseline

Before allocating compute:

1. Approve the candidate manifest and immutable artifact revision/hashes.
2. Freeze at least 100 held-out Turkish prompts covering instruction following,
   grounding, language quality, and safety; review dataset provenance/redaction.
3. Predeclare numerical quality/safety thresholds, latency/VRAM budgets, context,
   batch size, generation settings, and maximum spend.
4. Complete two local CPU evaluation passes with recorded seeds, outputs,
   scores, errors, package versions, timings, and memory.

**Exit gate:** complete frozen-set results and an approved reproducibility and
threshold review. A successful load or fluent sample alone is insufficient.

## 5) Target plan — approved one-GPU evaluation

After baseline approval:

1. Inventory GPU, driver, PyTorch, CUDA, cuDNN, and supported precision.
2. Execute the same frozen workload on one approved GPU.
3. Record warm/cold latency, tokens/s, peak VRAM, failures, and cost/run, alongside
   exact batch/context settings and quality/safety scores.
4. Require zero runtime failures in the approved workload and quality/safety
   within predeclared tolerance; retain every failure and artifact.
5. Review CPU/GPU differences before promotion; do not compare losses across
   changed architectures, datasets, or tokenizers as if they were equivalent.

**Exit gate:** archived comparable results and operator sign-off. BF16 support,
speedup, and efficiency must be measured, not assumed.

## 6) Conditional adaptation, optimized serving, and scale

- **NeMo:** investigate an approved Turkish adaptation recipe only if a baseline
  gap justifies it. Compare unchanged/adapted models on the frozen held-out set;
  record quality, safety regressions, cost, and artifact provenance.
- **TensorRT-LLM:** check exact model/precision/GPU support before export/build.
  Require parity within approved tolerance and measured latency/VRAM benefit.
- **ONNX:** check export/operator/runtime support and output parity; portability
  is not guaranteed for every architecture or generation workload.
- **Megatron Core:** consider only if single-GPU evidence establishes a research
  need and budget. Require explicit distributed-gradient, checkpoint/resume,
  reproducibility, and scaling-efficiency tests. No multi-GPU result is claimed.
- **Custom T-series:** retain CPU smoke and bounded research experiments. A
  canonical T-Dev-6L CUDA experiment needs its own dataset, precision, stability,
  and checkpoint gates; it does not automatically justify T-101 pretraining.

**Ebru pilot gate:** approved model card, Turkish quality/safety review, access
and retention policies, deployment boundary, audit evidence, rollback test, and
operator sign-off. Hardware access alone cannot satisfy these requirements.

See [the roadmap](roadmap.md) for phase-level acceptance evidence.

## Official ecosystem resources

- [NVIDIA Inception](https://www.nvidia.com/en-us/startups/) — programme information;
  no admission, benefits entitlement, or GPU allocation is claimed.
- [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html)
- [NVIDIA Megatron Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/)
- [NVIDIA TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/)
- [NVIDIA CUDA documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA cuDNN documentation](https://docs.nvidia.com/deeplearning/cudnn/)
- [ONNX documentation](https://onnx.ai/onnx/)
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/)
- [Candidate model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) —
  upstream reference, not model approval.

## Existing research references

- `configs/t_dev_6l.json` — canonical research geometry
- `configs/real_training/t_dev_6l_first_run.json` — smaller CPU proxy configuration
- `configs/real_training/t_dev_6l_expanded_run.json` — smaller CPU proxy configuration
- `docs/architecture/model-core.md` — custom-model contract and limitations
- `docs/dataset-overview.md` — research data context, not pretrained evaluation approval
