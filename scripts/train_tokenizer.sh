#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

INPUT_PATH="${INPUT_PATH:-data/raw}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/tokenizers}"
MODEL_VARIANT="${MODEL_VARIANT:-t-dev-6l}"
TOKENIZER_VERSION="${TOKENIZER_VERSION:-}"
VOCAB_SIZE="${VOCAB_SIZE:-32000}"
MANIFEST_PATH="${MANIFEST_PATH:-}"

cmd=(
  python -m supreme_modeltx.model_core.tokenizer.train
  --input-path "${INPUT_PATH}"
  --artifact-root "${ARTIFACT_ROOT}"
  --model-variant "${MODEL_VARIANT}"
  --vocab-size "${VOCAB_SIZE}"
)

if [ -n "${TOKENIZER_VERSION}" ]; then
  cmd+=(--version "${TOKENIZER_VERSION}")
fi

if [ -n "${MANIFEST_PATH}" ]; then
  cmd+=(--manifest-path "${MANIFEST_PATH}")
fi

"${cmd[@]}"
