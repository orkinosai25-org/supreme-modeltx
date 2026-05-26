// lifecycle-vm.bicep — SMTX Model Lifecycle / Registry VM (Layer 4)
//
// Purpose  : Model management, artifact storage, versioning, and job tracking.
//            Acts as the internal model registry and admin services hub.
// VM count : 1 (add a second for HA via the scale-vm-layer workflow)
// VM size  : Standard_D4s_v5 (4 vCPU / 16 GiB) — balanced general-purpose
// Services : smtx-lifecycle (FastAPI model registry, port 8090)

@description('Azure region for all resources')
param location string

@description('Environment tag (dev, staging, prod)')
param environment string

@description('Name prefix for all resources')
param prefix string

@description('Subnet resource ID for the lifecycle VM')
param subnetId string

@description('NSG resource ID for the lifecycle NIC')
param nsgId string

@description('User-assigned managed identity resource ID')
param identityId string

@description('Name of the storage account holding model artifacts')
param storageAccountName string

@description('VM size for the lifecycle VM')
@allowed([
  'Standard_D2s_v5'
  'Standard_D4s_v5'
  'Standard_D8s_v5'
])
param vmSize string = 'Standard_D4s_v5'

@description('Admin username for the VM')
param adminUsername string = 'smtxadmin'

@description('SSH public key data for the admin user')
@secure()
param sshPublicKey string

// ── Network Interface ─────────────────────────────────────────────────────────

resource nic 'Microsoft.Network/networkInterfaces@2023-04-01' = {
  name: '${prefix}-lifecycle-nic'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'lifecycle' }
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

resource lifecycleVm 'Microsoft.Compute/virtualMachines@2023-07-01' = {
  name: '${prefix}-lifecycle-vm'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'lifecycle', role: 'model-registry' }
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
        name: '${prefix}-lifecycle-osdisk'
        createOption: 'FromImage'
        managedDisk: { storageAccountType: 'Premium_LRS' }
        diskSizeGB: 128
      }
      dataDisks: [
        {
          name: '${prefix}-lifecycle-datadisk'
          lun: 0
          createOption: 'Empty'
          diskSizeGB: 256
          managedDisk: { storageAccountType: 'Premium_LRS' }
        }
      ]
    }
    osProfile: {
      computerName: '${prefix}-lifecycle-vm'
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
      networkInterfaces: [ { id: nic.id } ]
    }
    diagnosticsProfile: { bootDiagnostics: { enabled: true } }
  }
}

// ── Custom Script Extension ───────────────────────────────────────────────────

resource setupExtension 'Microsoft.Compute/virtualMachines/extensions@2023-07-01' = {
  parent: lifecycleVm
  name: 'lifecycle-setup'
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
pip install fastapi uvicorn[standard] pydantic \
            azure-identity azure-storage-blob mlflow

chown -R smtxadmin:smtxadmin "${SMTX_DIR}" "${VENV_DIR}"

cat > /etc/systemd/system/smtx-lifecycle.service <<SVCEOF
[Unit]
Description=SMTX Model Lifecycle / Registry Service
After=network.target

[Service]
Type=simple
User=smtxadmin
WorkingDirectory=/opt/smtx
ExecStart=/opt/smtx-venv/bin/uvicorn lifecycle.registry_server:app --host 0.0.0.0 --port 8090
Restart=on-failure
RestartSec=5
Environment=STORAGE_ACCOUNT=${STORAGE_ACCOUNT}

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable smtx-lifecycle
systemctl start smtx-lifecycle
''')
    }
    protectedSettings: {
      commandToExecute: 'bash /var/lib/waagent/custom-script/download/0/script.sh ${storageAccountName}'
    }
  }
}

// ── Outputs ───────────────────────────────────────────────────────────────────

output lifecycleVmId string = lifecycleVm.id
output lifecycleVmName string = lifecycleVm.name
output lifecycleVmPrivateIp string = nic.properties.ipConfigurations[0].properties.privateIPAddress
