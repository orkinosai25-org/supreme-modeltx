# Run Artifacts

Every real training run (i.e. any run that is **not** a `--dry-run`) automatically produces a structured set of artifacts alongside the checkpoint directory.

---

## Artifact layout

Given a checkpoint directory configured as:

```text
artifacts/runs/<run-name>/checkpoints/
```

The trainer writes a `run_artifacts/` directory at the same level:

```text
artifacts/runs/<run-name>/
├── checkpoints/
│   ├── checkpoint_step_00000010.pt
│   └── checkpoint_step_00000020.pt
└── run_artifacts/
    ├── config_used.json          ← effective config snapshot
    ├── training_summary.json     ← full structured metadata + metrics
    ├── training_summary.md       ← human-readable version of the above
    ├── samples.json              ← consolidated sample generations (all checkpoints)
    ├── samples.md                ← human-readable sample outputs
    └── samples/
        ├── checkpoint_step_00000010_samples.json
        └── checkpoint_step_00000020_samples.json
```

---

## File descriptions

### `config_used.json`

A snapshot of the full `SMTXConfig` that was active when the run started, serialised as JSON.  Use this to reproduce any run exactly.

### `training_summary.json`

Structured metadata covering the complete provenance of the run.  Every field is described below.

```json
{
  "run_name": "t_dev_6l_first_run",
  "training_end_status": "completed",
  "timestamps": {
    "started_at_utc": "2026-01-01T00:00:00+00:00",
    "ended_at_utc":   "2026-01-01T01:00:00+00:00"
  },
  "git_commit": "abc1234...",
  "config_path": ".../run_artifacts/config_used.json",
  "data": {
    "manifest_path":     "data/manifests/t_dev_6l_first_run.yaml",
    "train_split":       "train",
    "validation_split":  "validation"
  },
  "device":    "cpu",
  "precision": { "dtype": "float32", "enabled": false },
  "eval_cadence": {
    "eval_every_n_steps": 10,
    "eval_max_batches":    4
  },
  "tokenizer": {
    "path":    "artifacts/tokenizers/t-dev-6l/v1/tokenizer.model",
    "version": "v1",
    "backend": "sentencepiece"
  },
  "checkpoint_paths":    ["...checkpoint_step_00000010.pt", "..."],
  "best_checkpoint_path": "...checkpoint_step_00000020.pt",
  "validation_history": [
    { "step": 10, "val_loss": 7.81, "perplexity": 2471.83, "timestamp_utc": "..." },
    { "step": 20, "val_loss": 7.64, "perplexity": 2077.31, "timestamp_utc": "..." }
  ],
  "latest_validation_loss": 7.64,
  "latest_perplexity":      2077.31,
  "sample_artifact_paths":  ["...samples/checkpoint_step_00000010_samples.json", "..."]
}
```

| Field | Description |
|---|---|
| `run_name` | Name derived from the run directory |
| `training_end_status` | Always `"completed"` for a clean run |
| `timestamps` | UTC ISO-8601 start and end times |
| `git_commit` | SHA of `HEAD` at training time, or `null` |
| `config_path` | Path to the `config_used.json` snapshot |
| `data.*` | Manifest path, train and validation split names |
| `device` | Torch device used (e.g. `"cpu"`, `"cuda:0"`) |
| `precision.*` | Mixed-precision dtype and enabled flag |
| `eval_cadence.*` | How often validation ran and how many batches |
| `tokenizer.*` | Path, version, and backend |
| `checkpoint_paths` | All checkpoint files produced, sorted by step |
| `best_checkpoint_path` | Checkpoint closest to the step with the lowest validation loss |
| `validation_history` | Step-by-step validation loss and perplexity |
| `latest_validation_loss` | Final (most recent) validation loss |
| `latest_perplexity` | Final (most recent) perplexity |
| `sample_artifact_paths` | Paths to per-checkpoint sample JSON files |

### `training_summary.md`

A human-readable markdown rendering of the key provenance fields and checkpoint/sample lists.  Suitable for pasting into internal reviews or funding applications.

### `samples.json`

An aggregated JSON array of all per-checkpoint sample payloads.  Each entry corresponds to one checkpoint and contains:

```json
[
  {
    "checkpoint_path": "...checkpoint_step_00000010.pt",
    "generated_at_utc": "2026-01-01T00:30:00+00:00",
    "generation": {
      "max_new_tokens": 24,
      "temperature": 0.0,
      "top_p": 1.0,
      "top_k": 0
    },
    "samples": [
      {
        "prompt": "Sovereign AI enables",
        "prompt_token_count": 3,
        "completion_token_count": 24,
        "completion_text": "...",
        "full_output_text": "..."
      }
    ]
  }
]
```

Generation always uses `temperature=0.0` (greedy decoding) for full reproducibility.

### `samples.md`

A human-readable rendering of all sample outputs, grouped by checkpoint.

### `samples/checkpoint_step_*_samples.json`

Per-checkpoint sample files.  Each contains the same structure as one element of `samples.json`.  Produced immediately after the checkpoint is saved.

---

## Canonical prompts

The fixed prompt set used for sample generation is defined in:

```text
configs/canonical_prompts.json
```

These prompts are used for **every** run so outputs can be compared across experiments:

```json
{
  "prompts": [
    "Write a Python function max_of_two(a, b) that returns the larger integer.",
    "What does this Python expression output: [x*x for x in range(4)]",
    "All birds have wings. A sparrow is a bird. Does a sparrow have wings? Answer yes or no.",
    "If five machines make five widgets in five minutes, how many widgets do 100 machines make in five minutes?"
  ]
}
```

To override the prompts for a specific run, place a `canonical_prompts.json` file inside the run directory (the parent of `checkpoints/`).  The trainer will load it automatically.  If the file is missing or empty, the built-in defaults are used.

---

## How to run a real training experiment

```bash
python -m supreme_modeltx.model_core.training.trainer \
  --config configs/real_training/t_dev_6l_first_run.json
```

See the [Quick start](../README.md#first-real-t-dev-6l-training-experiment-manifest--checkpoint--perplexity) section of the README for the full workflow including tokenizer training and manifest setup.

---

## Inspecting outputs

### View the structured summary

```bash
cat artifacts/runs/t_dev_6l_first_run/run_artifacts/training_summary.json | python -m json.tool
```

### View the markdown summary

```bash
cat artifacts/runs/t_dev_6l_first_run/run_artifacts/training_summary.md
```

### View sample outputs

```bash
cat artifacts/runs/t_dev_6l_first_run/run_artifacts/samples.md
```

### Compare two runs

Both `training_summary.json` files from different runs share the same schema.  To compare:

```bash
# Diff two run summaries (requires jq)
diff \
  <(jq '{loss: .latest_validation_loss, ppl: .latest_perplexity, ckpt: .best_checkpoint_path}' \
      artifacts/runs/run_a/run_artifacts/training_summary.json) \
  <(jq '{loss: .latest_validation_loss, ppl: .latest_perplexity, ckpt: .best_checkpoint_path}' \
      artifacts/runs/run_b/run_artifacts/training_summary.json)
```

---

## Current limitations

- Single-process local execution; distributed multi-node provenance is not yet tracked separately.
- Sample generation is best-effort: if the tokenizer or checkpoint is unavailable the trainer logs a warning and continues without samples.
- The benchmark prompt set is intentionally small and directional; use full benchmark suites for release gating.
- No external experiment tracking integration (MLflow, W&B, etc.) — intentionally local-first.
