#!/usr/bin/env bash
# scripts/setup.sh
# One-step environment setup for Supreme ModelTX.
# Usage: bash scripts/setup.sh [--api] [--gpu]
#
# Flags:
#   --api   Also install API platform dependencies (fastapi, uvicorn, httpx)
#   --gpu   Use GPU-accelerated torch install (requires CUDA 12.x)
#
# After running this script:
#   - Python package is installed in editable mode
#   - Run `python -m pytest tests/ -v` to verify
#   - Copy .env.example to .env and fill in any required values

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

INSTALL_API=false
INSTALL_GPU=false

for arg in "$@"; do
  case "$arg" in
    --api) INSTALL_API=true ;;
    --gpu) INSTALL_GPU=true ;;
  esac
done

echo "==> Supreme ModelTX setup"
echo "    Repo root: ${REPO_ROOT}"

# --- Python version check ---
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "${PYTHON_VERSION}" | cut -d. -f1)
PYTHON_MINOR=$(echo "${PYTHON_VERSION}" | cut -d. -f2)

if [ "${PYTHON_MAJOR}" -lt 3 ] || { [ "${PYTHON_MAJOR}" -eq 3 ] && [ "${PYTHON_MINOR}" -lt 10 ]; }; then
  echo "ERROR: Python 3.10+ required (found ${PYTHON_VERSION})" >&2
  exit 1
fi
echo "    Python: ${PYTHON_VERSION} ✓"

# --- pip upgrade ---
python3 -m pip install --upgrade pip --quiet

# --- Build extras list ---
EXTRAS="train,eval,dev"
if [ "${INSTALL_API}" = true ]; then
  EXTRAS="${EXTRAS},api"
fi

# --- GPU torch ---
if [ "${INSTALL_GPU}" = true ]; then
  echo "==> Installing GPU-accelerated torch (cu121)"
  python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
fi

# --- Install package ---
echo "==> Installing supreme-modeltx[${EXTRAS}] in editable mode"
python3 -m pip install -e ".[${EXTRAS}]" --quiet

# --- Copy .env.example if .env does not exist ---
if [ ! -f "${REPO_ROOT}/.env" ] && [ -f "${REPO_ROOT}/.env.example" ]; then
  cp "${REPO_ROOT}/.env.example" "${REPO_ROOT}/.env"
  echo "==> Copied .env.example → .env (update values as needed)"
fi

# --- Verify install ---
echo "==> Verifying syntax"
python3 -m compileall src/supreme_modeltx -q

echo ""
echo "✓ Setup complete."
echo ""
echo "Next steps:"
echo "  python -m pytest tests/ -v          # run all tests"
echo "  bash scripts/run_demo.sh            # run end-to-end demo"
echo "  bash scripts/evaluate.sh            # run evaluation benchmark"
