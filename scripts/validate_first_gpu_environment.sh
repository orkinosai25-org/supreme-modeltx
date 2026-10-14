#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Validate Azure GPU readiness for the first T-Dev-6L run.

Usage:
  bash scripts/validate_first_gpu_environment.sh

Optional environment variables:
  SMTX_REPO_DIR            Repo root (default: script parent)
  SMTX_VENV_DIR            Python virtualenv path (default: $SMTX_REPO_DIR/.venv)
  SMTX_CONFIG_PATH         Training config path
  SMTX_MANIFEST_PATH       Corpus manifest path
  SMTX_TOKENIZER_MODEL     Tokenizer model path
  SMTX_RUN_ROOT            Run output root
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMTX_REPO_DIR="${SMTX_REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SMTX_VENV_DIR="${SMTX_VENV_DIR:-${SMTX_REPO_DIR}/.venv}"
SMTX_CONFIG_PATH="${SMTX_CONFIG_PATH:-${SMTX_REPO_DIR}/configs/real_training/t_dev_6l_first_gpu_run.json}"
cd "${SMTX_REPO_DIR}"
SMTX_RUN_ROOT="${SMTX_RUN_ROOT:-${SMTX_REPO_DIR}/artifacts/runs/t_dev_6l_first_gpu_run}"

PYTHON_BIN="python3"
if [[ -x "${SMTX_VENV_DIR}/bin/python" ]]; then
  PYTHON_BIN="${SMTX_VENV_DIR}/bin/python"
fi

[[ -f "${SMTX_CONFIG_PATH}" ]] || {
  echo "[ERROR] Missing config: ${SMTX_CONFIG_PATH}" >&2
  exit 1
}

mapfile -t config_paths < <("${PYTHON_BIN}" - "${SMTX_CONFIG_PATH}" <<'PYEOF'
import sys
from supreme_modeltx.model_core.config.schema import SMTXConfig
cfg = SMTXConfig.from_file(sys.argv[1])
print(cfg.data.manifest_path or "")
print(cfg.data.tokenizer_path or cfg.tokenizer.model_path or "")
PYEOF
)
SMTX_MANIFEST_PATH="${SMTX_MANIFEST_PATH:-${config_paths[0]:-}}"
SMTX_TOKENIZER_MODEL="${SMTX_TOKENIZER_MODEL:-${config_paths[1]:-}}"
[[ "${SMTX_MANIFEST_PATH}" == "${config_paths[0]:-}" && "${SMTX_TOKENIZER_MODEL}" == "${config_paths[1]:-}" ]] || {
  echo "[ERROR] Manifest/tokenizer overrides must match the selected training config." >&2
  exit 1
}
[[ -f "${SMTX_MANIFEST_PATH}" ]] || { echo "[ERROR] Missing manifest: ${SMTX_MANIFEST_PATH}" >&2; exit 1; }

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi is unavailable; CUDA driver stack is not ready." >&2
  exit 1
fi

nvidia-smi
"${PYTHON_BIN}" -m supreme_modeltx.foundation.gpu_diagnostics --require-cuda

"${PYTHON_BIN}" - <<'PYEOF'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Expected failure condition: bootstrap cannot continue until the GPU driver/extension is healthy.")
print(f"CUDA verification passed with {torch.cuda.device_count()} device(s).")
PYEOF

if [[ ! -f "${SMTX_TOKENIZER_MODEL}" ]]; then
  echo "[ERROR] Missing immutable tokenizer artifact: ${SMTX_TOKENIZER_MODEL}" >&2
  echo "[ERROR] Restore the approved artifact; retraining at the same path does not preserve identity." >&2
  exit 1
fi

mkdir -p \
  "${SMTX_RUN_ROOT}/checkpoints" \
  "${SMTX_RUN_ROOT}/run_artifacts" \
  "${SMTX_RUN_ROOT}/benchmark_outputs" \
  "${SMTX_RUN_ROOT}/comparison_vs_cpu"

(
  cd "${SMTX_REPO_DIR}"
  SMTX_REQUIRE_GPU=1 "${PYTHON_BIN}" -m supreme_modeltx.model_core.training.trainer \
    --config "${SMTX_CONFIG_PATH}" \
    --preflight
)

cat <<EOF
[OK] Candidate first-run GPU readiness checks complete; training remains GPU-unvalidated.

Verified:
  - manifest path: ${SMTX_MANIFEST_PATH}
  - tokenizer path: ${SMTX_TOKENIZER_MODEL}
  - output root   : ${SMTX_RUN_ROOT}
  - CUDA readiness: passed

Manual launch:
  cd ${SMTX_REPO_DIR}
  SMTX_REQUIRE_GPU=1 ${PYTHON_BIN} -m supreme_modeltx.model_core.training.trainer \
    --config ${SMTX_CONFIG_PATH}
EOF
