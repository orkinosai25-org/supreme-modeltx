#!/usr/bin/env bash
# scripts/evaluate.sh
# Run the Supreme ModelTX evaluation benchmark.
#
# What this does:
#   1. Compiles all source files (syntax check)
#   2. Runs the full unit + smoke test suite
#   3. Runs perplexity evaluation against the validation set
#      (skipped if no checkpoint exists — set SMTX_CHECKPOINT_PATH)
#   4. Runs the benchmark scoring against configs/benchmark_eval_set.json
#      (skipped if no checkpoint exists)
#   5. Writes a summary to results/latest.json
#
# Usage: bash scripts/evaluate.sh [--checkpoint PATH] [--config PATH]
#
# Environment variables (or use --flags):
#   SMTX_CHECKPOINT_PATH   Path to a trained model checkpoint (.pt)
#   SMTX_TOKENIZER_PATH    Path to trained tokenizer (.model)
#   SMTX_MODEL_CONFIG_PATH Path to model JSON config
#
# Prerequisites:
#   bash scripts/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

CHECKPOINT_PATH="${SMTX_CHECKPOINT_PATH:-}"
CONFIG_PATH="${SMTX_MODEL_CONFIG_PATH:-configs/real_training/t_dev_6l_first_run.json}"

for arg in "$@"; do
  case "$arg" in
    --checkpoint) shift; CHECKPOINT_PATH="$1" ;;
    --config) shift; CONFIG_PATH="$1" ;;
  esac
done

TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
RESULTS_DIR="${REPO_ROOT}/results"
mkdir -p "${RESULTS_DIR}"

echo "============================================================"
echo "  Supreme ModelTX — Evaluation Run"
echo "  ${TIMESTAMP}"
echo "============================================================"

# --- Step 1: Syntax check ---
echo ""
echo "[1/4] Syntax check (compileall)"
echo "--------------------------------"
python3 -m compileall src/supreme_modeltx -q
echo "✓ No syntax errors"

# --- Step 2: Test suite ---
echo ""
echo "[2/4] Unit and smoke test suite"
echo "--------------------------------"
# Build ignore list for tests that need optional extras not installed
PYTEST_IGNORE_FLAGS=""
if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "  fastapi not installed — ignoring API-dependent tests (install with: pip install -e '.[api]')"
  PYTEST_IGNORE_FLAGS="--ignore=tests/unit/test_chat_endpoint.py"
fi
# shellcheck disable=SC2086
python3 -m pytest tests/ -v --tb=short ${PYTEST_IGNORE_FLAGS}
echo "✓ All tests passed"

# --- Step 3: Perplexity evaluation ---
echo ""
echo "[3/4] Perplexity evaluation"
echo "---------------------------"

if [ -z "${CHECKPOINT_PATH}" ]; then
  echo "  SMTX_CHECKPOINT_PATH not set — skipping perplexity evaluation"
  echo "  To enable: export SMTX_CHECKPOINT_PATH=<path_to_checkpoint.pt>"
  PERPLEXITY_SKIPPED=true
else
  python3 -m supreme_modeltx.model_core.eval.perplexity \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    2>&1 | tee /tmp/smtx_eval_perplexity.txt
  echo "✓ Perplexity evaluation complete"
  PERPLEXITY_SKIPPED=false
fi

# --- Step 4: Benchmark scoring ---
echo ""
echo "[4/4] Benchmark scoring"
echo "-----------------------"

if [ -z "${CHECKPOINT_PATH}" ]; then
  echo "  SMTX_CHECKPOINT_PATH not set — skipping benchmark scoring"
  BENCHMARK_SKIPPED=true
else
  python3 -m supreme_modeltx.model_core.eval.benchmark \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --eval-set configs/benchmark_eval_set.json \
    --baselines configs/benchmark_baselines.json \
    --output "${RESULTS_DIR}/latest.json" \
    2>&1 | tee /tmp/smtx_eval_benchmark.txt
  echo "✓ Benchmark scoring complete → ${RESULTS_DIR}/latest.json"
  BENCHMARK_SKIPPED=false
fi

# --- Summary ---
echo ""
echo "============================================================"
echo "  Evaluation Summary"
echo "  Timestamp: ${TIMESTAMP}"
echo "  Config:    ${CONFIG_PATH}"
echo "  Checkpoint: ${CHECKPOINT_PATH:-'(not set — skipped checkpoint steps)'}"
echo "  Tests:     PASSED"
echo "  Perplexity eval: $([ "${PERPLEXITY_SKIPPED:-true}" = true ] && echo 'SKIPPED' || echo 'DONE')"
echo "  Benchmark:       $([ "${BENCHMARK_SKIPPED:-true}" = true ] && echo 'SKIPPED' || echo "DONE → results/latest.json")"
echo "============================================================"
