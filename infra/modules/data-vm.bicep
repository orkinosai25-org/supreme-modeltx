// data-vm.bicep — SMTX Data / Governance VMs (Layer 5)
//
// Purpose  : Document storage, SharePoint-like ingestion, content governance,
//            vector search, and structured data serving for all SMTX layers.
// VM count : configurable (default 2, min 1, max 8 for SharePoint-scale)
// VM size  : Standard_D8s_v5 (8 vCPU / 32 GiB) — memory-optimised for search
// Services : smtx-data (FastAPI data gateway, port 8100)

@description('Azure region for all resources')
param location string

@description('Environment tag (dev, staging, prod)')
param environment string

@description('Name prefix for all resources')
param prefix string

@description('Subnet resource ID for data VMs')
param subnetId string

@description('NSG resource ID for data NICs')
param nsgId string

@description('User-assigned managed identity resource ID')
param identityId string

@description('Name of the storage account for data blobs')
param storageAccountName string

@description('Number of data/governance VMs to provision')
@minValue(1)
@maxValue(8)
param vmCount int = 2

@description('VM size for data VMs')
@allowed([
  'Standard_D4s_v5'
  'Standard_D8s_v5'
  'Standard_D16s_v5'
  'Standard_E8s_v5'
])
param vmSize string = 'Standard_D8s_v5'

@description('Admin username for the VMs')
param adminUsername string = 'smtxadmin'

@description('SSH public key data for the admin user')
@secure()
param sshPublicKey string

// ── Network Interfaces ────────────────────────────────────────────────────────

resource nics 'Microsoft.Network/networkInterfaces@2023-04-01' = [for i in range(0, vmCount): {
  name: '${prefix}-data-nic-${i}'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'data', instance: '${i}' }
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

resource dataVms 'Microsoft.Compute/virtualMachines@2023-07-01' = [for i in range(0, vmCount): {
  name: '${prefix}-data-vm-${i}'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'data', role: 'data-governance', instance: '${i}' }
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
        name: '${prefix}-data-osdisk-${i}'
        createOption: 'FromImage'
        managedDisk: { storageAccountType: 'Premium_LRS' }
        diskSizeGB: 128
      }
      dataDisks: [
        {
          name: '${prefix}-data-datadisk-${i}'
          lun: 0
          createOption: 'Empty'
          diskSizeGB: 512
          managedDisk: { storageAccountType: 'Premium_LRS' }
        }
      ]
    }
    osProfile: {
      computerName: '${prefix}-data-vm-${i}'
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
  parent: dataVms[i]
  name: 'data-setup'
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
            azure-identity azure-storage-blob \
            sentence-transformers faiss-cpu \
            langchain

chown -R smtxadmin:smtxadmin "${SMTX_DIR}" "${VENV_DIR}"

cat > /etc/systemd/system/smtx-data.service <<SVCEOF
[Unit]
Description=SMTX Data / Governance Service
After=network.target

[Service]
Type=simple
User=smtxadmin
WorkingDirectory=/opt/smtx
ExecStart=/opt/smtx-venv/bin/uvicorn data.gateway:app --host 0.0.0.0 --port 8100 --workers 2
Restart=on-failure
RestartSec=5
Environment=STORAGE_ACCOUNT=${STORAGE_ACCOUNT}

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable smtx-data
systemctl start smtx-data
''')
    }
    protectedSettings: {
      commandToExecute: 'bash /var/lib/waagent/custom-script/download/0/script.sh ${storageAccountName}'
    }
  }
}]

// ── Outputs ───────────────────────────────────────────────────────────────────

output dataVmNames array = [for i in range(0, vmCount): dataVms[i].name]
output dataVmPrivateIps array = [for i in range(0, vmCount): nics[i].properties.ipConfigurations[0].properties.privateIPAddress]
