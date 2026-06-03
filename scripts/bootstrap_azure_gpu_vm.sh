#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Bootstrap an Azure GPU VM for SMTX GPU runs.

Run this on the VM after the NVIDIA driver is installed.

Usage:
  sudo bash scripts/bootstrap_azure_gpu_vm.sh

Optional environment variables:
  SMTX_REPO_URL           Repo clone URL
  SMTX_REPO_DIR           Repo checkout directory
  SMTX_GIT_REF            Git ref to check out after clone/fetch
  SMTX_VENV_DIR           Python virtualenv path
  GITHUB_RUNNER_URL       Repo/org runner URL for optional registration
  GITHUB_RUNNER_TOKEN     Ephemeral registration token
  GITHUB_RUNNER_LABELS    Runner labels (default: self-hosted,linux,x64,gpu)
  GITHUB_RUNNER_ROOT      Runner install root (default: /opt/actions-runner)
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

SUDO=""
if [[ ${EUID} -ne 0 ]]; then
  SUDO="sudo"
fi

RUN_AS_USER="${SUDO_USER:-$(id -un)}"
RUN_AS_GROUP="$(id -gn "${RUN_AS_USER}")"
RUN_AS_HOME="$(getent passwd "${RUN_AS_USER}" | cut -d: -f6)"

run_as_user() {
  if [[ "$(id -un)" == "${RUN_AS_USER}" ]]; then
    "$@"
  else
    sudo -u "${RUN_AS_USER}" "$@"
  fi
}

SMTX_REPO_URL="${SMTX_REPO_URL:-https://github.com/orkinosai25-org/supreme-modeltx.git}"
SMTX_REPO_DIR="${SMTX_REPO_DIR:-${RUN_AS_HOME}/supreme-modeltx}"
SMTX_GIT_REF="${SMTX_GIT_REF:-}"
SMTX_VENV_DIR="${SMTX_VENV_DIR:-${SMTX_REPO_DIR}/.venv}"
GITHUB_RUNNER_URL="${GITHUB_RUNNER_URL:-}"
GITHUB_RUNNER_TOKEN="${GITHUB_RUNNER_TOKEN:-}"
GITHUB_RUNNER_LABELS="${GITHUB_RUNNER_LABELS:-self-hosted,linux,x64,gpu}"
GITHUB_RUNNER_ROOT="${GITHUB_RUNNER_ROOT:-/opt/actions-runner}"

${SUDO} apt-get update -y
${SUDO} apt-get install -y \
  build-essential \
  ca-certificates \
  curl \
  git \
  pkg-config \
  python3 \
  python3-pip \
  python3-venv

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi is not available yet." >&2
  echo "[ERROR] From your workstation, install the Azure driver extension first:" >&2
  echo "  az vm extension set --publisher Microsoft.HpcCompute --name NvidiaGpuDriverLinux ..." >&2
  exit 1
fi

nvidia-smi

if [[ ! -d "${SMTX_REPO_DIR}/.git" ]]; then
  run_as_user git clone "${SMTX_REPO_URL}" "${SMTX_REPO_DIR}"
else
  run_as_user git -C "${SMTX_REPO_DIR}" fetch --all --tags --prune
  run_as_user git -C "${SMTX_REPO_DIR}" pull --ff-only
fi

if [[ -n "${SMTX_GIT_REF}" ]]; then
  run_as_user git -C "${SMTX_REPO_DIR}" checkout "${SMTX_GIT_REF}"
fi

run_as_user python3 -m venv "${SMTX_VENV_DIR}"
run_as_user "${SMTX_VENV_DIR}/bin/pip" install --upgrade pip
run_as_user "${SMTX_VENV_DIR}/bin/pip" install -e "${SMTX_REPO_DIR}[dev]"

run_as_user "${SMTX_VENV_DIR}/bin/python" - <<'PYEOF'
import torch

print(f"Torch version: {torch.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available after bootstrap.")
print(f"Detected CUDA devices: {torch.cuda.device_count()}")
print(f"Primary GPU: {torch.cuda.get_device_name(0)}")
PYEOF

if [[ -n "${GITHUB_RUNNER_URL}" && -n "${GITHUB_RUNNER_TOKEN}" ]]; then
  runner_version="$(
    curl -fsSL https://api.github.com/repos/actions/runner/releases/latest \
      | python3 -c 'import json, sys; print(json.load(sys.stdin)["tag_name"].lstrip("v"))'
  )"
  runner_tarball="actions-runner-linux-x64-${runner_version}.tar.gz"

  ${SUDO} mkdir -p "${GITHUB_RUNNER_ROOT}"
  ${SUDO} chown -R "${RUN_AS_USER}":"${RUN_AS_GROUP}" "${GITHUB_RUNNER_ROOT}"

  run_as_user bash -lc "cd '${GITHUB_RUNNER_ROOT}' && curl -fsSL -o '${runner_tarball}' 'https://github.com/actions/runner/releases/download/v${runner_version}/${runner_tarball}' && tar xzf '${runner_tarball}'"
  run_as_user bash -lc "cd '${GITHUB_RUNNER_ROOT}' && ./config.sh \
    --unattended \
    --replace \
    --url '${GITHUB_RUNNER_URL}' \
    --token '${GITHUB_RUNNER_TOKEN}' \
    --name '$(hostname)' \
    --labels '${GITHUB_RUNNER_LABELS}'"
  (cd "${GITHUB_RUNNER_ROOT}" && ${SUDO} ./svc.sh install "${RUN_AS_USER}")
  (cd "${GITHUB_RUNNER_ROOT}" && ${SUDO} ./svc.sh start)
fi

cat <<EOF
[OK] Azure GPU VM bootstrap complete.

Repo      : ${SMTX_REPO_DIR}
Virtualenv: ${SMTX_VENV_DIR}

Next step:
  bash ${SMTX_REPO_DIR}/scripts/validate_first_gpu_environment.sh
EOF
