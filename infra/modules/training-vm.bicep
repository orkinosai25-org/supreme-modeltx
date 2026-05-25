// training-vm.bicep — SMTX LLM Training VM(s) (Layer 7 — CPU-only)
//
// Purpose  : CPU-based LoRA/QLoRA fine-tuning of open base models on Turkish
//            culture, history, and Islamic AI datasets. SUMOTX is a CPU-only
//            platform — no GPU quota required.
//
//            Exposes a training job API so GitHub Actions can submit jobs via
//            az vm run-command (no public SSH required).
//
// VM count : configurable (default 1, min 1, max 10)
//            — Start with 1. Add more via the scale-vm-layer workflow as
//              budget allows.
//
// VM size  : Standard_D16s_v5 (16 vCPU / 64 GiB, default — good for LoRA)
//            Standard_D32s_v5 (32 vCPU / 128 GiB, faster multi-thread training)
//            Standard_E16s_v5 (16 vCPU / 128 GiB, memory-optimised for large models)
//            Standard_E32s_v5 (32 vCPU / 256 GiB, heavy fine-tuning workloads)
//
// Services : smtx-train-api (job submission API, port 8200)

@description('Azure region for all resources')
param location string

@description('Environment tag (dev, staging, prod)')
param environment string

@description('Name prefix for all resources')
param prefix string

@description('Subnet resource ID for training VMs')
param subnetId string

@description('NSG resource ID for training NICs')
param nsgId string

@description('User-assigned managed identity resource ID')
param identityId string

@description('Name of the storage account holding datasets and checkpoints')
param storageAccountName string

@description('Number of training VMs to provision')
@minValue(1)
@maxValue(10)
param vmCount int = 1

@description('CPU VM size for LLM fine-tuning. Larger sizes give faster multi-threaded training throughput.')
@allowed([
  'Standard_D16s_v5'
  'Standard_D32s_v5'
  'Standard_E16s_v5'
  'Standard_E32s_v5'
])
param vmSize string = 'Standard_D16s_v5'

@description('Admin username for the VMs')
param adminUsername string = 'smtxadmin'

@description('SSH public key data for the admin user')
@secure()
param sshPublicKey string

// ── Network Interfaces ────────────────────────────────────────────────────────

resource nics 'Microsoft.Network/networkInterfaces@2023-04-01' = [for i in range(0, vmCount): {
  name: '${prefix}-training-nic-${i}'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'training', instance: '${i}' }
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

resource trainingVms 'Microsoft.Compute/virtualMachines@2023-07-01' = [for i in range(0, vmCount): {
  name: '${prefix}-training-vm-${i}'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'training', role: 'llm-training', instance: '${i}' }
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
        name: '${prefix}-training-osdisk-${i}'
        createOption: 'FromImage'
        managedDisk: { storageAccountType: 'Premium_LRS' }
        diskSizeGB: 256
      }
      dataDisks: [
        {
          // Large data disk for datasets, checkpoints, and model weights
          name: '${prefix}-training-datadisk-${i}'
          lun: 0
          createOption: 'Empty'
          diskSizeGB: 1024
          managedDisk: { storageAccountType: 'Premium_LRS' }
        }
      ]
    }
    osProfile: {
      computerName: '${prefix}-train-${i}'
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
  parent: trainingVms[i]
  name: 'training-setup'
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

apt-get update -y
apt-get install -y python3-pip python3-venv git curl

if [ ! -d "${SMTX_DIR}/.git" ]; then
  git clone "${SMTX_REPO}" "${SMTX_DIR}"
else
  git -C "${SMTX_DIR}" pull --ff-only
fi

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip

# CPU-only training stack — no GPU or CUDA required
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate peft datasets \
            sentencepiece tokenizers tqdm \
            azure-identity azure-storage-blob \
            fastapi uvicorn[standard] pydantic

chown -R smtxadmin:smtxadmin "${SMTX_DIR}" "${VENV_DIR}"

# ── Format and mount data disk ────────────────────────────────────────────────
# Use Azure-stable device path (LUN 0 → lun0 symlink)
DATA_DISK="/dev/disk/azure/scsi1/lun0"
if [ -b "$DATA_DISK" ] && ! blkid "$DATA_DISK" &>/dev/null; then
  mkfs.ext4 -F "$DATA_DISK"
fi
mkdir -p /mnt/smtx-data
if ! mountpoint -q /mnt/smtx-data; then
  mount "$DATA_DISK" /mnt/smtx-data 2>/dev/null || true
fi
# Only add fstab entry if it doesn't already exist
if ! grep -q "/mnt/smtx-data" /etc/fstab; then
  echo "${DATA_DISK}  /mnt/smtx-data  ext4  defaults,nofail  0  2" >> /etc/fstab
fi

# ── Job submission API (accepts training jobs from GitHub Actions) ─────────────
cat > /etc/systemd/system/smtx-train-api.service <<SVCEOF
[Unit]
Description=SMTX Training Job Submission API
After=network.target

[Service]
Type=simple
User=smtxadmin
WorkingDirectory=/opt/smtx
ExecStart=/opt/smtx-venv/bin/uvicorn training.train_api:app --host 0.0.0.0 --port 8200
Restart=on-failure
RestartSec=5
Environment=STORAGE_ACCOUNT=${STORAGE_ACCOUNT}
Environment=SMTX_DIR=/opt/smtx
Environment=DATA_DIR=/mnt/smtx-data

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable smtx-train-api
systemctl start smtx-train-api
''')
    }
    protectedSettings: {
      commandToExecute: 'bash /var/lib/waagent/custom-script/download/0/script.sh ${storageAccountName}'
    }
  }
}]

// ── Outputs ───────────────────────────────────────────────────────────────────

output trainingVmNames array = [for i in range(0, vmCount): trainingVms[i].name]
output trainingVmPrivateIps array = [for i in range(0, vmCount): nics[i].properties.ipConfigurations[0].properties.privateIPAddress]
