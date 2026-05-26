// gpu-vm.bicep — SMTX GPU Virtual Machine for vLLM Inference
// Supports Standard_NC48ads_A100_v4 (A100) or Standard_NC40ads_H100_v5 (H100/L40S)

@description('Azure region for all resources')
param location string

@description('Environment tag (dev, staging, prod)')
param environment string

@description('Name prefix for all resources')
param prefix string

@description('Subnet resource ID for the GPU VM')
param subnetId string

@description('NSG resource ID for the GPU NIC')
param nsgId string

@description('User-assigned managed identity resource ID')
param identityId string

@description('VM size — use A100 (Standard_NC48ads_A100_v4) or L40S (Standard_NC40ads_L40S)')
@allowed([
  'Standard_NC48ads_A100_v4'
  'Standard_NC40ads_L40S'
  'Standard_NC24ads_A100_v4'
])
param vmSize string = 'Standard_NC48ads_A100_v4'

@description('Admin username for the VM')
param adminUsername string = 'smtxadmin'

@description('SSH public key data for the admin user')
@secure()
param sshPublicKey string

@description('Name of the storage account holding model weights')
param storageAccountName string

@description('Name of the blob container holding model weights')
param modelContainer string = 'models'

// ── Network Interface ─────────────────────────────────────────────────────────

resource nic 'Microsoft.Network/networkInterfaces@2023-04-01' = {
  name: '${prefix}-gpu-nic'
  location: location
  tags: { environment: environment, project: 'smtx', component: 'gpu-vm' }
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
}

// ── Virtual Machine ───────────────────────────────────────────────────────────

resource gpuVm 'Microsoft.Compute/virtualMachines@2023-07-01' = {
  name: '${prefix}-gpu-vm'
  location: location
  tags: { environment: environment, project: 'smtx', component: 'gpu-vm', role: 'vllm-inference' }
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identityId}': {}
    }
  }
  properties: {
    hardwareProfile: {
      vmSize: vmSize
    }
    storageProfile: {
      imageReference: {
        publisher: 'microsoft-dsvm'
        offer: 'ubuntu-hpc'
        sku: '2204'
        version: 'latest'
      }
      osDisk: {
        name: '${prefix}-gpu-osdisk'
        createOption: 'FromImage'
        managedDisk: {
          storageAccountType: 'Premium_LRS'
        }
        diskSizeGB: 128
      }
      dataDisks: [
        {
          name: '${prefix}-gpu-datadisk'
          lun: 0
          createOption: 'Empty'
          diskSizeGB: 1024
          managedDisk: {
            storageAccountType: 'Premium_LRS'
          }
        }
      ]
    }
    osProfile: {
      computerName: '${prefix}-gpu-vm'
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
      networkInterfaces: [
        { id: nic.id }
      ]
    }
    diagnosticsProfile: {
      bootDiagnostics: {
        enabled: true
      }
    }
  }
}

// ── Custom Script Extension: Install vLLM and start server ────────────────────

resource vllmSetupExtension 'Microsoft.Compute/virtualMachines/extensions@2023-07-01' = {
  parent: gpuVm
  name: 'vllm-setup'
  location: location
  properties: {
    publisher: 'Microsoft.Azure.Extensions'
    type: 'CustomScript'
    typeHandlerVersion: '2.1'
    autoUpgradeMinorVersion: true
    settings: {
      // Script is stored at /var/lib/waagent/custom-script/download/0/script.sh
      // and executed via commandToExecute in protectedSettings (which injects variables).
      script: base64('''
#!/usr/bin/env bash
# Usage: script.sh <storage_account> <model_container>
set -euo pipefail

STORAGE_ACCOUNT="$1"
MODEL_CONTAINER="$2"
STORAGE_DOMAIN="$3"
SMTX_REPO="https://github.com/orkinosai25-org/SMTX.git"
SMTX_DIR="/opt/smtx"
VENV_DIR="/opt/smtx-venv"

# ── System dependencies ───────────────────────────────────────────────────────
apt-get update -y
apt-get install -y python3-pip python3-venv git

# ── Clone application code ────────────────────────────────────────────────────
if [ ! -d "${SMTX_DIR}/.git" ]; then
  git clone "${SMTX_REPO}" "${SMTX_DIR}"
else
  git -C "${SMTX_DIR}" pull --ff-only
fi

# ── Python virtual environment ────────────────────────────────────────────────
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install vllm fastapi uvicorn[standard] pydantic \
            azure-storage-blob azure-identity

# ── Download model weights from Azure Blob (managed identity) ─────────────────
python3 - <<PYEOF
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient
import pathlib

dest_dir = pathlib.Path("${SMTX_DIR}/tmodels/t101")
dest_dir.mkdir(parents=True, exist_ok=True)

cred = ManagedIdentityCredential()
svc = BlobServiceClient(
    account_url=f"https://${STORAGE_ACCOUNT}.blob.{STORAGE_DOMAIN}",
    credential=cred,
)
client = svc.get_container_client("${MODEL_CONTAINER}")
for blob in client.list_blobs(name_starts_with="t101/"):
    dest = dest_dir / pathlib.Path(blob.name).name
    with open(dest, "wb") as fh:
        fh.write(client.download_blob(blob.name).readall())
print("Model weights downloaded successfully.")
PYEOF

# ── Fix ownership ─────────────────────────────────────────────────────────────
chown -R smtxadmin:smtxadmin "${SMTX_DIR}" "${VENV_DIR}"

# ── Systemd service for vLLM ──────────────────────────────────────────────────
cat > /etc/systemd/system/vllm.service <<'SVCEOF'
[Unit]
Description=SMTX vLLM Inference Server
After=network.target

[Service]
Type=simple
User=smtxadmin
WorkingDirectory=/opt/smtx
ExecStart=/opt/smtx-venv/bin/python inference/vllm_server.py --model tmodels/t101 --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable vllm
systemctl start vllm
''')
    }
    protectedSettings: {
      // commandToExecute injects Bicep parameter values as positional arguments.
      commandToExecute: 'bash /var/lib/waagent/custom-script/download/0/script.sh ${storageAccountName} ${modelContainer} ${az.environment().suffixes.storage}'
    }
  }
}

// ── Outputs ───────────────────────────────────────────────────────────────────

output gpuVmId string = gpuVm.id
output gpuVmName string = gpuVm.name
output gpuVmPrivateIp string = nic.properties.ipConfigurations[0].properties.privateIPAddress
