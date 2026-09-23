# Data governance

## Principles

- Keep all training and evaluation data private unless it is explicitly cleared for internal use
- Track source and license metadata for every JSONL record set
- Validate before training, not after
- Never upload private data to external services without approval

## Required dataset shape

The default JSONL contract expects:

- `prompt`
- `response`
- `source`
- `license`

Additional metadata may be added, but the required fields must remain present and non-empty.

## Validation workflow

Run:

```bash
python -m supreme_modeltx.foundation.dataset_tools validate /path/to/dataset.jsonl --scan-pii --scan-secrets
```

The validator checks:

- JSONL parsing
- required fields
- duplicate records
- source metadata summary
- license metadata summary
- optional PII pattern matches
- optional secret pattern matches

## Splitting workflow

Run:

```bash
python -m supreme_modeltx.foundation.dataset_tools split /path/to/dataset.jsonl --output-dir /path/to/splits
```

Outputs:

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `split_manifest.json`

The split manifest records the source dataset path, ratios, seed, counts, and output file paths.

## Source and license tracking

Every approved dataset slice should preserve:

- where the text came from
- the license or internal-use basis
- the approval owner
- the date the slice was created

Do not mix unlicensed text into the same training file as approved text.

## Optional PII and secret scanning

The built-in scanner is a lightweight preflight check. It helps catch obvious:

- email addresses
- phone-like strings
- AWS access key patterns
- GitHub token patterns
- OpenAI-style API key patterns

This does not replace a full legal or compliance review.

## Versioning rules

- version dataset manifests and split manifests
- record the seed used for splitting
- treat transformed datasets as new versions
- tie every training run to one dataset version and one config snapshot

## Private-data handling

- store raw private data outside the repository when possible
- keep checkpoints and training logs in approved private storage
- avoid copying private data into issue trackers, PR comments, or public notebooks
- document retention and deletion decisions alongside each experiment
