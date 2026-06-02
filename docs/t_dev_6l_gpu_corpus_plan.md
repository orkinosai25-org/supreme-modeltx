# T-Dev-6L next corpus plan for GPU experiments

## Goal

Prepare a larger, more realistic, and more reproducible corpus than the small and partly synthetic manifests used to validate the current CPU and first-GPU training pipeline.

The versioned target for the next experiment is:

- manifest: `data/manifests/t_dev_6l_gpu_corpus_v1.yaml`
- corpus version: `t_dev_6l_gpu_corpus_v1`

This keeps the existing trainer, benchmark, and run-artifact contract unchanged. The future GPU run only needs to point `data.manifest_path` at the new manifest once the planned processed files are materialized.

## Corpus composition target

| Category | Intended share | Purpose |
| --- | ---: | --- |
| Code | ~55% | Improve code completion, editing, refactoring, and bug-fixing behaviour on the repository-style benchmark path. |
| Reasoning / instruction text | ~25% | Improve canonical prompt following, structured task framing, and multi-step answer quality. |
| Documentation / technical prose | ~15% | Improve README-style explanations, architecture narration, and operations-oriented answers. |
| Justified synthetic bridge data | ~5% | Cover narrow formats that are hard to source naturally, such as patch-style edits or tool-call shells. |

The synthetic category remains intentionally small so the mixture is still dominated by natural code and technical text.

## Provenance and licensing discipline

Each source in `data/manifests/t_dev_6l_gpu_corpus_v1.yaml` records:

- provenance
- license or usage basis
- inclusion rationale
- source-specific preprocessing expectations
- benchmark alignment

The preparation bar for inclusion is:

1. the source must be traceable,
2. the usage basis must be documented,
3. preprocessing must be describable and repeatable,
4. benchmark contamination review must happen before the split freeze.

Untraceable, ambiguously licensed, or benchmark-contaminating data should not be materialized into this corpus version.

## Reproducible materialization plan

The manifest uses stable output locations under:

- `data/processed/t_dev_6l_gpu_corpus_v1/train/`
- `data/processed/t_dev_6l_gpu_corpus_v1/validation/`

Expected prepared files:

- `code.jsonl`
- `reasoning_instructions.jsonl`
- `technical_docs.jsonl`
- `synthetic_bridge.jsonl` (train only)

The split contract is fixed in the manifest:

- `train`: main optimization split
- `validation`: stable holdout split

Split assignment should be frozen by content hash after deduplication so reruns stay comparable.

## Preprocessing expectations

The shared preprocessing path for this corpus version is:

1. collect only approved inputs,
2. normalize encoding and whitespace,
3. remove empty / generated / boilerplate-heavy records,
4. deduplicate exact and near-duplicate records,
5. run benchmark decontamination review,
6. freeze train/validation assignment,
7. emit text-only JSONL records,
8. keep trainer packing assumptions unchanged (`max_seq_len`, EOS-delimited packing, existing tokenizer contract).

Category-specific expectations are captured per source in the manifest.

## Alignment with evaluation goals

This corpus is intended to improve training signal for the benchmark path already in the repository:

- **code tasks:** more realistic repository code and fewer synthetic-only patterns
- **reasoning tasks:** more instruction-style and stepwise task framing examples
- **canonical prompts:** better match between training examples and the prompt/task styles already emitted in run artifacts and benchmark reports

This document does not claim guaranteed benchmark gains. It documents why the next experiment should be a stronger test than the current small baseline corpora.

## Comparability with previous runs

Compared with:

- `data/manifests/t_dev_6l_first_run.yaml`
- `data/manifests/t_dev_6l_expanded_run.yaml`

the new corpus plan changes the data layer, not the benchmark or artifact layer.

What stays the same:

- trainer entrypoint
- benchmark workflow
- run-artifact schema
- canonical prompt evaluation path

What changes:

- corpus scale target
- category coverage
- provenance detail
- preprocessing discipline
- explicit versioning of the next corpus target

## Lightweight validation

Repository tests now validate that:

- the new corpus-plan manifest has the required metadata fields,
- train/validation split declarations stay consistent,
- planned manifests can resolve expected paths without pretending files already exist,
- materialized manifests still require existing files.
