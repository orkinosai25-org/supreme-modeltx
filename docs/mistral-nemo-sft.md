# Turkish-domain Mistral LoRA SFT pilot v1

**Preparation only. No weights, approved corpus, GPU run, quality result or
deployment is supplied.** This is supervised fine-tuning (SFT) of
`mistralai/Mistral-7B-v0.3` (base, not Instruct) on reviewed input/answer pairs,
not random-init pretraining, continued pretraining, or a new foundation model.
Domains: Turkish culture, tourism and Islamic knowledge. The output is a **NeMo
LoRA adapter checkpoint** plus run metadata; it still needs the exact frozen
base and tokenizer. It is not standalone full weights, a Hugging Face PEFT
adapter, or an artifact accepted by the existing API. Merging/export/deployment
are separate reviewed work.

This path does not modify T-101, T-Dev-6L or the Lambda bring-up work in PR #73.
Azure may be used for separately authorized preparation/governance. Lambda
credits are intended for separately authorized training. There is **no assumed
Azure–Lambda integration**, automatic data transfer, provisioning or credentials.
The inaccessible OneNote discussion is not a source for this plan.

## Verified upstream contract (2026-11-04)

Use the **versioned NeMo 2.4.0 LLM stack**, not unpinned `pip install nemo_toolkit`.
NeMo's current 3.0 release is speech-focused after its repository split; this
is a deliberately pinned older LLM API, not a claim of latest-version support.
Before real use, the operator must review the older environment's security,
driver and dependency support and record an approved container digest.

Official NVIDIA references inspected for this implementation:

- [Mistral recipe, v2.4.0](https://github.com/NVIDIA/NeMo/blob/v2.4.0/nemo/collections/llm/recipes/mistral_7b.py):
  explicitly selects `mistralai/Mistral-7B-v0.3` and supports `peft_scheme="lora"`.
- [Default fine-tune recipe](https://github.com/NVIDIA/NeMo/blob/v2.4.0/nemo/collections/llm/recipes/finetune_default.py):
  BF16, TP/PP/CP=1; replace its SQuAD data and cached-model restore path.
- [Mistral importer](https://github.com/NVIDIA/NeMo/blob/v2.4.0/nemo/collections/llm/gpt/model/mistral.py),
  [import API](https://github.com/NVIDIA/NeMo/blob/v2.4.0/nemo/collections/llm/api.py),
  [FineTuningDataModule](https://github.com/NVIDIA/NeMo/blob/v2.4.0/nemo/collections/llm/gpt/data/fine_tuning.py)
  and [SFT dataset contract](https://github.com/NVIDIA/NeMo/blob/v2.4.0/nemo/collections/llm/gpt/data/core.py):
  local converted checkpoint, `training.jsonl` / `validation.jsonl`, input/output
  fields, answer-only loss.
- [LoRA parameters](https://github.com/NVIDIA/NeMo/blob/v2.4.0/nemo/collections/llm/peft/lora.py):
  NeMo fused linear targets, `dim`, `alpha`, `dropout`.
- [NeMo-Run direct execution](https://github.com/NVIDIA/NeMo-Run/blob/v0.4.0/nemo_run/run/api.py)
  and [NeMo 3.0 migration](https://github.com/NVIDIA/NeMo/releases/tag/v3.0.0).

NeMo v2.4.0 tag resolved to commit
`2381f42f6979449b5b99538f8f80135831009b51`. These source checks establish
API/model compatibility, **not** successful execution on Lambda or A10 fit.

## Offline validation (no NeMo, PyTorch, tokenizer or network needed)

From the repository root, using Python 3.10+:

```bash
python training/nemo_sft.py --config configs/nemo/mistral-7b-v03-lora-v1.json
python -m unittest tests.unit.test_nemo_sft -v
```

The config is one GPU, BF16, LoRA rank 8/alpha 16/dropout .05, 512 tokens,
microbatch 1/global batch 4, activation recomputation, 100 steps and validation
every 25 steps. These are conservative **pilot starting values**, not tuning
recommendations or demonstrated memory/quality results. No FP8 or QLoRA.

## Data contract and governance

Keep all private inputs and outputs under ignored
`artifacts/private_foundation/nemo-sft/`, or another approved private volume.
Do not commit data, receipts identifying people, logs containing content,
weights, tokens or SSH keys. Do not reuse the existing random-init corpora
without a fresh source, rights and suitability audit.

Copy `data/manifests/nemo-sft-v1.template.json` beside your two JSONL files as
`manifest.json`. The template deliberately fails validation. Fill in a versioned
dataset ID, approvals, exact 40-character HF base revision and verified model
license. Approvals are human attestations, not automated PII/license detection.
Supply `sources` entries with these required fields:

| Field | Requirement |
|---|---|
| `id` | Unique source key used by records |
| `group_id` | Canonical document/work/family ID shared by all excerpts, translations and paraphrases |
| `uri` | Original stable source location; do not put credentials in URLs |
| `license` | SPDX identifier or reviewed custom permission identifier |
| `license_evidence` | Durable location of permission/license evidence, including derivative/training rights |
| `training_allowed` | Boolean `true` only after rights review |
| `retrieved_at` | ISO date/time of acquisition |
| `content_sha256` | SHA-256 of original source bytes, retained privately |
| `transform` | Versioned extraction/normalization/translation steps and responsible tool/person |
| `reviewed_by` | Accountable rights/content reviewer identifier |

Each UTF-8 JSONL line is an object containing non-empty strings:
`id`, `input`, `output`, `source_id`, `source_locator` (page/section/verse
reference as applicable), `reviewed_by`; plus `language: "tr"` and `domain`
equal to `turkish_culture`, `tourism` or `islamic_knowledge`.
Metadata is not included in the model prompt. Prompt format is exactly
`Soru: {input}\nYanıt: {output}` with BOS/EOS and answer-only loss; it is a
plain SFT template, not the Instruct chat template. Keep the same format in
later evaluations.

For Islamic knowledge, each record additionally requires
`religious_reviewed_by`: a qualified reviewer selected by the user. Preserve
precise citations, source editions, translation rights and interpretive context.
Distinguish quotation, translation and interpretation; represent differences
respectfully. Do not fabricate scripture, hadith, rulings or attribution, and
do not treat the model or automated validator as scholarly authority. No
religious training content is supplied here.

Split **by source/document family before producing examples**. All excerpts,
translations and paraphrases of the same work belong to one split. Reserve a
separate locked test set outside this manifest for final evaluation. Do not
tune on validation/test examples. Human near-duplicate review must cover both
splits and the reserved test set. Freeze split hashes after all reviews:

```bash
sha256sum artifacts/private_foundation/nemo-sft/data/training.jsonl \
          artifacts/private_foundation/nemo-sft/data/validation.jsonl
```

Enter those hashes under `splits`. Validation checks hashes, required source
and review fields, IDs, normalized pair duplication, cross-split prompt
duplication and source-group leakage. It cannot verify truthful citations,
rights, language, reviewer qualifications, semantic duplicates or PII removal.
Check domain counts and deliberately balance the domains; the validator reports
counts but does not choose a mixture or require every domain in a small pilot.
Audit tourism freshness and Turkish regional/cultural diversity.

```bash
python training/nemo_sft.py \
  --config configs/nemo/mistral-7b-v03-lora-v1.json \
  --manifest artifacts/private_foundation/nemo-sft/data/manifest.json
```

## Lambda operator runbook — only after separate authorization

Nothing below is executed by offline validation. Do not provision resources or
spend credits just to test this PR. On an **already authorized** Lambda GPU
host, verify persistent storage, NVIDIA driver/container toolkit, disk/RAM
headroom and `nvidia-smi`. A10 has 24 GB: frozen BF16 7B weights alone are
roughly 14–15 GB, before activations, workspaces, adapters and checkpoint
conversion. **Single-A10 fit is not established.** Stop on OOM rather than
automatically renting another GPU; obtain budget approval for a larger GPU.

Use a private, approved NVIDIA NeMo container with `nemo_toolkit==2.4.0`,
NeMo-Run, Megatron Core, Transformer Engine and CUDA dependencies. NVIDIA's
[2.4 installation guidance](https://github.com/NVIDIA/NeMo/blob/v2.4.0/README.md#install-nemo-framework)
describes the complete stack. Do not install this repo's `train`/`foundation`
extras into it: their Torch constraints are for the existing trainers.
The container is a prerequisite, not supplied or validated here; do not assume
a calendar-tagged image contains the right API. Verify the version and imports,
record `pip freeze`, image digest, driver/GPU details and repository commit.

Before entering the offline container, an authorized operator must separately
accept/check model access and license terms, acquire the exact approved HF
revision, and verify its file hashes. This PR never downloads models. Keep
credentials only in an approved secret store, not commands/history/manifests.
Mount the approved complete HF snapshot at `/work/artifacts/private_foundation/nemo-sft/hf-base`.

The following is a runnable container shell command after setting the approved
digest; it does not provision Lambda or connect Azure:

```bash
export NEMO_IMAGE='nvcr.io/nvidia/nemo@sha256:REPLACE_WITH_APPROVED_DIGEST'
docker run --rm -it --gpus '"device=0"' --network none \
  --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e WANDB_MODE=disabled \
  -e NEMO_HOME=/work/artifacts/private_foundation/nemo-sft/cache \
  -v "$PWD:/work" -w /work "$NEMO_IMAGE" bash
```

Inside that shell (offline import conversion may itself require substantial
host/GPU memory; do not run it without authorization):

```bash
python -c 'from importlib.metadata import version; assert version("nemo_toolkit") == "2.4.0"; import nemo_run; from nemo.collections.llm.recipes import mistral_7b'
mkdir -p artifacts/private_foundation/nemo-sft/cache
python - <<'PY'
from pathlib import Path
from nemo.collections import llm
root = Path("artifacts/private_foundation/nemo-sft").resolve()
llm.import_ckpt(
    model=llm.MistralModel(), source=f"hf://{root / 'hf-base'}",
    output_path=root / "base-nemo", overwrite=False,
)
PY
```

After verifying conversion identity, write `base-nemo/smtx-base.json` with
`{"nemo_version": "2.4.0", "base_model": ...}` where `base_model` is the exact
approved object from the dataset manifest. This is a conversion attestation,
not a cryptographic proof of model identity: privately retain snapshot and
converted-file hashes, conversion logs and reviewer sign-off. Never attach a
receipt to untrusted or unrelated checkpoint contents.

Explicit training command, **only after go/no-go and credit approval**:

```bash
python training/nemo_sft.py \
  --config configs/nemo/mistral-7b-v03-lora-v1.json \
  --manifest artifacts/private_foundation/nemo-sft/data/manifest.json \
  --base-checkpoint artifacts/private_foundation/nemo-sft/base-nemo \
  --output artifacts/private_foundation/nemo-sft/runs/pilot-v1 \
  --train
```

It rechecks data, version, base receipt, exactly one CUDA GPU and BF16 support.
It loads the local tokenizer and rejects over-budget pairs before training:
shorten/re-review/re-hash data instead of silently truncating citations or
answers. A fresh output directory prevents overwriting or accidental resume.
Training has a bounded step count; interruption/restart requires a separate
reviewed resumption procedure. No remote executor, tracking service, download
fallback, or cloud integration is configured. Keep `smtx-run.json`, adapters,
tokenizer/base identity, environment inventory and logs private.

Stop for non-finite loss, memory failures, missing validation or checkpoint
errors. Verify actual saved adapter checkpoints and validation logs after any
authorized run; merely creating run metadata is not success. Compare with the
unchanged base on a locked Turkish/domain evaluation, citation faithfulness,
respectful religious handling and memorization checks, with human reviewers.
Loss or these offline tests alone cannot establish domain quality, safety,
scholarly reliability or production readiness.
