// api-vm.bicep — SMTX API / Backend VMs (Layer 2)
//
// Purpose  : REST API gateway and backend application logic. Multiple VMs
//            provide redundancy for public/investor-demo load.
// VM count : configurable (default 2, min 1, max 10)
// VM size  : Standard_D4s_v5 (4 vCPU / 16 GiB) — balanced general-purpose
// Services : smtx-api (FastAPI, port 8080)

@description('Azure region for all resources')
param location string

@description('Environment tag (dev, staging, prod)')
param environment string

@description('Name prefix for all resources')
param prefix string

@description('Subnet resource ID for API VMs')
param subnetId string

@description('NSG resource ID for API NICs')
param nsgId string

@description('User-assigned managed identity resource ID')
param identityId string

@description('Number of API/backend VMs to provision')
@minValue(1)
@maxValue(10)
param vmCount int = 2

@description('VM size for API VMs')
@allowed([
  'Standard_D2s_v5'
  'Standard_D4s_v5'
  'Standard_D8s_v5'
])
param vmSize string = 'Standard_D4s_v5'

@description('Admin username for the VMs')
param adminUsername string = 'smtxadmin'

@description('SSH public key data for the admin user')
@secure()
param sshPublicKey string

// ── Network Interfaces (one per VM) ──────────────────────────────────────────

resource nics 'Microsoft.Network/networkInterfaces@2023-04-01' = [for i in range(0, vmCount): {
  name: '${prefix}-api-nic-${i}'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'api', instance: '${i}' }
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

resource apiVms 'Microsoft.Compute/virtualMachines@2023-07-01' = [for i in range(0, vmCount): {
  name: '${prefix}-api-vm-${i}'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'api', role: 'api-backend', instance: '${i}' }
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
        name: '${prefix}-api-osdisk-${i}'
        createOption: 'FromImage'
        managedDisk: { storageAccountType: 'Premium_LRS' }
        diskSizeGB: 64
      }
    }
    osProfile: {
      computerName: '${prefix}-api-vm-${i}'
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

// ── Custom Script Extension (runs on each VM) ─────────────────────────────────

resource setupExtensions 'Microsoft.Compute/virtualMachines/extensions@2023-07-01' = [for i in range(0, vmCount): {
  parent: apiVms[i]
  name: 'api-setup'
  location: location
  properties: {
    publisher: 'Microsoft.Azure.Extensions'
    type: 'CustomScript'
    typeHandlerVersion: '2.1'
    autoUpgradeMinorVersion: true
    settings: {
      script: base64('''
#!/usr/bin/env bash
set -euo pipefail

SMTX_REPO="https://github.com/orkinosai25-org/SMTX.git"
SMTX_DIR="/opt/smtx"
VENV_DIR="/opt/smtx-venv"

apt-get update -y
apt-get install -y python3-pip python3-venv git curl nginx

if [ ! -d "${SMTX_DIR}/.git" ]; then
  git clone "${SMTX_REPO}" "${SMTX_DIR}"
else
  git -C "${SMTX_DIR}" pull --ff-only
fi

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install fastapi uvicorn[standard] pydantic httpx \
            azure-identity azure-storage-blob

chown -R smtxadmin:smtxadmin "${SMTX_DIR}" "${VENV_DIR}"

cat > /etc/systemd/system/smtx-api.service <<'SVCEOF'
[Unit]
Description=SMTX API Backend Service
After=network.target

[Service]
Type=simple
User=smtxadmin
WorkingDirectory=/opt/smtx
ExecStart=/opt/smtx-venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8080 --workers 4
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable smtx-api
systemctl start smtx-api
''')
    }
  }
}]

// ── Outputs ───────────────────────────────────────────────────────────────────

output apiVmNames array = [for i in range(0, vmCount): apiVms[i].name]
output apiVmPrivateIps array = [for i in range(0, vmCount): nics[i].properties.ipConfigurations[0].properties.privateIPAddress]
