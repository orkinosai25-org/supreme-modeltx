#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Provision a plain Ubuntu 22.04 Azure GPU VM for SMTX GPU runs.

Usage:
  bash scripts/provision_azure_gpu_vm.sh \
    --resource-group smtx-gpu-rg \
    --vm-name smtx-gpu-runner \
    --ssh-public-key ~/.ssh/id_ed25519.pub

Options:
  --resource-group NAME     Azure resource group to create/use (required)
  --vm-name NAME            VM name (default: smtx-gpu-runner)
  --admin-username NAME     VM admin username (default: smtxadmin)
  --ssh-public-key PATH     SSH public key file path (required)
  --region REGION           Preferred region; repeat to override defaults
  --vm-size SIZE            Candidate GPU VM size; repeat to override defaults
  --image URN               Azure image URN
  --os-disk-size-gb N       OS disk size in GB (default: 256)
  --skip-driver-extension   Do not install the Azure NVIDIA driver extension
  --help                    Show this help

Defaults:
  regions : uksouth, then ukwest
  image   : Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest
  sizes   : Standard_NC24ads_A100_v4, Standard_NC8as_T4_v3, Standard_NC4as_T4_v3

Notes:
  - The script tries each region/size pair until one succeeds.
  - The default size order starts with a canonical-run-capable option, then
    falls back to cheaper single-GPU bootstrap-friendly sizes.
EOF
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[ERROR] Required command not found: $1" >&2
    exit 1
  }
}

RESOURCE_GROUP=""
VM_NAME="smtx-gpu-runner"
ADMIN_USERNAME="smtxadmin"
SSH_PUBLIC_KEY=""
IMAGE="Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest"
OS_DISK_SIZE_GB="256"
INSTALL_DRIVER_EXTENSION="1"
declare -a REGIONS=()
declare -a VM_SIZES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resource-group)
      RESOURCE_GROUP="${2:?missing value for --resource-group}"
      shift 2
      ;;
    --vm-name)
      VM_NAME="${2:?missing value for --vm-name}"
      shift 2
      ;;
    --admin-username)
      ADMIN_USERNAME="${2:?missing value for --admin-username}"
      shift 2
      ;;
    --ssh-public-key)
      SSH_PUBLIC_KEY="${2:?missing value for --ssh-public-key}"
      shift 2
      ;;
    --region)
      REGIONS+=("${2:?missing value for --region}")
      shift 2
      ;;
    --vm-size)
      VM_SIZES+=("${2:?missing value for --vm-size}")
      shift 2
      ;;
    --image)
      IMAGE="${2:?missing value for --image}"
      shift 2
      ;;
    --os-disk-size-gb)
      OS_DISK_SIZE_GB="${2:?missing value for --os-disk-size-gb}"
      shift 2
      ;;
    --skip-driver-extension)
      INSTALL_DRIVER_EXTENSION="0"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

[[ -n "${RESOURCE_GROUP}" ]] || {
  echo "[ERROR] --resource-group is required." >&2
  usage
  exit 1
}

[[ -f "${SSH_PUBLIC_KEY}" ]] || {
  echo "[ERROR] SSH public key not found: ${SSH_PUBLIC_KEY}" >&2
  exit 1
}

if [[ ${#REGIONS[@]} -eq 0 ]]; then
  REGIONS=("uksouth" "ukwest")
fi

if [[ ${#VM_SIZES[@]} -eq 0 ]]; then
  VM_SIZES=(
    "Standard_NC24ads_A100_v4"
    "Standard_NC8as_T4_v3"
    "Standard_NC4as_T4_v3"
  )
fi

require_cmd az
require_cmd python3

az account show >/dev/null
az group create \
  --name "${RESOURCE_GROUP}" \
  --location "${REGIONS[0]}" \
  --output none

for region in "${REGIONS[@]}"; do
  for vm_size in "${VM_SIZES[@]}"; do
    echo "[INFO] Attempting Azure GPU VM in region=${region} size=${vm_size}"
    set +e
    vm_output="$(
      az vm create \
        --resource-group "${RESOURCE_GROUP}" \
        --name "${VM_NAME}" \
        --location "${region}" \
        --image "${IMAGE}" \
        --size "${vm_size}" \
        --admin-username "${ADMIN_USERNAME}" \
        --ssh-key-values "${SSH_PUBLIC_KEY}" \
        --authentication-type ssh \
        --os-disk-size-gb "${OS_DISK_SIZE_GB}" \
        --storage-sku Premium_LRS \
        --public-ip-sku Standard \
        --tags purpose=t-dev-6l-gpu-run owner="${USER:-unknown}" \
        --output json 2>&1
    )"
    status=$?
    set -e

    if [[ ${status} -ne 0 ]]; then
      echo "[WARN] Provisioning failed for ${region}/${vm_size}." >&2
      echo "${vm_output}" >&2
      continue
    fi

    public_ip="$(
      printf '%s' "${vm_output}" | python3 -c 'import json, sys; print(json.load(sys.stdin)["publicIpAddress"])'
    )"

    az vm open-port \
      --resource-group "${RESOURCE_GROUP}" \
      --name "${VM_NAME}" \
      --port 22 \
      --priority 1001 \
      --output none

    if [[ "${INSTALL_DRIVER_EXTENSION}" == "1" ]]; then
      echo "[INFO] Installing Azure NVIDIA driver extension"
      az vm extension set \
        --resource-group "${RESOURCE_GROUP}" \
        --vm-name "${VM_NAME}" \
        --publisher Microsoft.HpcCompute \
        --name NvidiaGpuDriverLinux \
        --enable-auto-upgrade true \
        --output none
    fi

    cat <<EOF
[OK] Provisioned VM successfully.

Resource group : ${RESOURCE_GROUP}
VM name        : ${VM_NAME}
Region         : ${region}
VM size        : ${vm_size}
Public IP      : ${public_ip}
Image          : ${IMAGE}

Next steps:
  ssh ${ADMIN_USERNAME}@${public_ip}
  sudo bash /path/to/repo/scripts/bootstrap_azure_gpu_vm.sh
  bash /path/to/repo/scripts/validate_first_gpu_environment.sh

Cost controls:
  az vm deallocate --resource-group ${RESOURCE_GROUP} --name ${VM_NAME}
  az vm start --resource-group ${RESOURCE_GROUP} --name ${VM_NAME}
  az vm delete --yes --resource-group ${RESOURCE_GROUP} --name ${VM_NAME}
EOF
    exit 0
  done
done

echo "[ERROR] Failed to provision a GPU VM in preferred regions (${REGIONS[*]})." >&2
echo "[ERROR] Try a different GPU SKU list with repeated --vm-size flags." >&2
exit 1
