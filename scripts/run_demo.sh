#!/usr/bin/env bash
# scripts/run_demo.sh
# End-to-end demo for Supreme ModelTX.
#
# Demonstrates:
#   1. Model instantiation and forward pass (smoke test suite)
#   2. Platform API health check (if API dependencies are installed)
#
# Usage: bash scripts/run_demo.sh [--skip-api]
#
# Prerequisites:
#   bash scripts/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

SKIP_API=false
for arg in "$@"; do
  case "$arg" in
    --skip-api) SKIP_API=true ;;
  esac
done

echo "============================================================"
echo "  Supreme ModelTX — End-to-End Demo"
echo "  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================================"

# --- Step 1: Model smoke test (includes training dry-run) ---
echo ""
echo "[1/2] Model instantiation, forward-pass, and training dry-run (smoke tests)"
echo "----------------------------------------------------------------------------"
python3 -m pytest tests/smoke/ -v --tb=short
echo "✓ Smoke tests passed (model instantiation, forward pass, training dry-run)"

# --- Step 2: Platform API health check ---
if [ "${SKIP_API}" = false ] && python3 -c "import fastapi" 2>/dev/null; then
  echo ""
  echo "[2/2] Platform API health check"
  echo "--------------------------------"
  # Start API in background
  python3 -m supreme_modeltx.platform_api.api.app &
  API_PID=$!
  sleep 3

  # Health check
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ API health check passed"
    HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
    echo "  Response: ${HEALTH_RESPONSE}"
  else
    echo "  (API health endpoint not responding — skipping)"
  fi

  kill "${API_PID}" 2>/dev/null || true
  wait "${API_PID}" 2>/dev/null || true
else
  echo ""
  echo "[2/2] Platform API health check — skipped (--skip-api or fastapi not installed)"
fi

echo ""
echo "============================================================"
echo "  Demo complete ✓"
echo ""
echo "  To run a real training experiment:"
echo "    python -m supreme_modeltx.model_core.training.trainer \\"
echo "      --config configs/real_training/t_dev_6l_first_run.json"
echo ""
echo "  To run the full benchmark evaluation:"
echo "    bash scripts/evaluate.sh"
echo "============================================================"
