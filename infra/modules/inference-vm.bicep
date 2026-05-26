// inference-vm.bicep — SMTX CPU Inference VM (Layer 6 — CPU-only)
//
// Purpose  : CPU-based LLM inference using llama-cpp-python (GGUF/quantised
//            models). Serves the SUMOTX chat and API layers with no GPU quota
//            requirement. Suitable for small-to-medium quantised models
//            (Q4/Q5/Q8 GGUF) with 4–16 CPU threads.
//
// VM count : configurable (default 1, min 1, max 5)
// VM size  : Standard_D8s_v5 (8 vCPU / 32 GiB, default — good for Q4 7B)
//            Standard_D16s_v5 (16 vCPU / 64 GiB — 13B / 20B models)
//            Standard_E8s_v5  (8 vCPU / 64 GiB — memory-heavy models)
//            Standard_E16s_v5 (16 vCPU / 128 GiB — larger quantised models)
//
// Services : smtx-inference (llama-cpp-python OpenAI-compatible API, port 8000)

@description('Azure region for all resources')
param location string

@description('Environment tag (dev, staging, prod)')
param environment string

@description('Name prefix for all resources')
param prefix string

@description('Subnet resource ID for inference VMs')
param subnetId string

@description('NSG resource ID for inference NICs')
param nsgId string

@description('User-assigned managed identity resource ID')
param identityId string

@description('Name of the storage account holding model weights')
param storageAccountName string

@description('Number of inference VMs to provision')
@minValue(1)
@maxValue(5)
param vmCount int = 1

@description('CPU VM size for LLM inference. Larger sizes handle bigger quantised models.')
@allowed([
  'Standard_D8s_v5'
  'Standard_D16s_v5'
  'Standard_E8s_v5'
  'Standard_E16s_v5'
])
param vmSize string = 'Standard_D8s_v5'

@description('Admin username for the VMs')
param adminUsername string = 'smtxadmin'

@description('SSH public key data for the admin user')
@secure()
param sshPublicKey string

// ── Network Interfaces ────────────────────────────────────────────────────────

resource nics 'Microsoft.Network/networkInterfaces@2023-04-01' = [for i in range(0, vmCount): {
  name: '${prefix}-inference-nic-${i}'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'inference', instance: '${i}' }
  properties: {
    networkSecurityGroup: { id: nsgId }
    ipConfigurations: [
      {
        name: 'ipconfig1'
        properties: {
          subnet: { id: subnetId }
          privateIPAllocationMethod: 'Dynamic'
        }
      }
    ]
  }
}]

// ── Virtual Machines ──────────────────────────────────────────────────────────

resource inferenceVms 'Microsoft.Compute/virtualMachines@2023-07-01' = [for i in range(0, vmCount): {
  name: '${prefix}-inference-vm-${i}'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'inference', role: 'cpu-llm-inference', instance: '${i}' }
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identityId}': {}
    }
  }
  properties: {
    hardwareProfile: { vmSize: vmSize }
    storageProfile: {
      imageReference: {
        publisher: 'Canonical'
        offer: '0001-com-ubuntu-server-jammy'
        sku: '22_04-lts-gen2'
        version: 'latest'
      }
      osDisk: {
        name: '${prefix}-inference-osdisk-${i}'
        createOption: 'FromImage'
        managedDisk: { storageAccountType: 'Premium_LRS' }
        diskSizeGB: 128
      }
      dataDisks: [
        {
          // Model weights disk (GGUF files can be 4–10 GB each)
          name: '${prefix}-inference-datadisk-${i}'
          lun: 0
          createOption: 'Empty'
          diskSizeGB: 256
          managedDisk: { storageAccountType: 'Premium_LRS' }
        }
      ]
    }
    osProfile: {
      computerName: '${prefix}-inf-${i}'
      adminUsername: adminUsername
      linuxConfiguration: {
        disablePasswordAuthentication: true
        ssh: {
          publicKeys: [
            {
              path: '/home/${adminUsername}/.ssh/authorized_keys'
              keyData: sshPublicKey
            }
          ]
        }
      }
    }
    networkProfile: {
      networkInterfaces: [ { id: nics[i].id } ]
    }
    diagnosticsProfile: { bootDiagnostics: { enabled: true } }
  }
}]

// ── Custom Script Extension ───────────────────────────────────────────────────

resource setupExtensions 'Microsoft.Compute/virtualMachines/extensions@2023-07-01' = [for i in range(0, vmCount): {
  parent: inferenceVms[i]
  name: 'inference-setup'
  location: location
  properties: {
    publisher: 'Microsoft.Azure.Extensions'
    type: 'CustomScript'
    typeHandlerVersion: '2.1'
    autoUpgradeMinorVersion: true
    settings: {
      script: base64('''
#!/usr/bin/env bash
# Usage: script.sh <storage_account_name>
set -euo pipefail

STORAGE_ACCOUNT="$1"
SMTX_REPO="https://github.com/orkinosai25-org/SMTX.git"
SMTX_DIR="/opt/smtx"
VENV_DIR="/opt/smtx-venv"
MODEL_DIR="/mnt/smtx-models"

apt-get update -y
apt-get install -y python3-pip python3-venv git curl build-essential cmake

if [ ! -d "${SMTX_DIR}/.git" ]; then
  git clone "${SMTX_REPO}" "${SMTX_DIR}"
else
  git -C "${SMTX_DIR}" pull --ff-only
fi

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip

# CPU-only inference stack using llama-cpp-python (GGUF quantised models)
# CMAKE_ARGS selects CPU-only build (no CUDA)
CMAKE_ARGS="-DLLAMA_CUBLAS=off" pip install llama-cpp-python
pip install fastapi uvicorn[standard] pydantic \
            azure-identity azure-storage-blob httpx

chown -R smtxadmin:smtxadmin "${SMTX_DIR}" "${VENV_DIR}"

# ── Format and mount model disk ───────────────────────────────────────────────
DATA_DISK="/dev/disk/azure/scsi1/lun0"
if [ -b "$DATA_DISK" ] && ! blkid "$DATA_DISK" &>/dev/null; then
  mkfs.ext4 -F "$DATA_DISK"
fi
mkdir -p "${MODEL_DIR}"
if ! mountpoint -q "${MODEL_DIR}"; then
  mount "$DATA_DISK" "${MODEL_DIR}" 2>/dev/null || true
fi
if ! grep -q "${MODEL_DIR}" /etc/fstab; then
  echo "${DATA_DISK}  ${MODEL_DIR}  ext4  defaults,nofail  0  2" >> /etc/fstab
fi
chown smtxadmin:smtxadmin "${MODEL_DIR}"

# ── Download GGUF model weights from Azure Blob (managed identity) ────────────
python3 - <<PYEOF
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient
import os
import pathlib

dest_dir = pathlib.Path("${MODEL_DIR}")
dest_dir.mkdir(parents=True, exist_ok=True)

cred = ManagedIdentityCredential()
svc = BlobServiceClient(
    account_url=f"https://${STORAGE_ACCOUNT}.blob.core.windows.net",
    credential=cred,
)
try:
    container = svc.get_container_client("models")
    for blob in container.list_blobs(name_starts_with="gguf/"):
        dest = dest_dir / pathlib.Path(blob.name).name
        if not dest.exists():
            print(f"Downloading {blob.name}...")
            dest.write_bytes(container.download_blob(blob.name).readall())
    print("Model download complete.")
except Exception as exc:
    print(f"Warning: model download skipped ({exc}). Upload GGUF weights to models/gguf/ in storage.")
PYEOF

# ── Systemd service for llama-cpp-python OpenAI-compatible server ─────────────
cat > /etc/systemd/system/smtx-inference.service <<SVCEOF
[Unit]
Description=SMTX CPU Inference Service (llama-cpp-python)
After=network.target

[Service]
Type=simple
User=smtxadmin
WorkingDirectory=/opt/smtx
ExecStart=/bin/bash -c '\
  MODEL_FILE=$(find "${MODEL_DIR}" -name "*.gguf" | sort | head -1); \
  if [ -z "$MODEL_FILE" ]; then \
    echo "No GGUF model found in ${MODEL_DIR}. Upload a GGUF file to the models/gguf/ blob container." >&2; \
    exit 1; \
  fi; \
  exec /opt/smtx-venv/bin/python -m llama_cpp.server \
    --host 0.0.0.0 \
    --port 8000 \
    --n_threads $(nproc) \
    --model "$MODEL_FILE" \
    --model_alias smtx-model'
Restart=on-failure
RestartSec=10
Environment=SMTX_MODEL_DIR=${MODEL_DIR}

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable smtx-inference
systemctl start smtx-inference
''')
    }
    protectedSettings: {
      commandToExecute: 'bash /var/lib/waagent/custom-script/download/0/script.sh ${storageAccountName}'
    }
  }
}]

// ── Outputs ───────────────────────────────────────────────────────────────────

output inferenceVmNames array = [for i in range(0, vmCount): inferenceVms[i].name]
output inferenceVmPrivateIps array = [for i in range(0, vmCount): nics[i].properties.ipConfigurations[0].properties.privateIPAddress]
