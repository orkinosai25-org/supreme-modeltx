// control-vm.bicep — SMTX Control-Plane / Orchestration VM (Layer 1)
//
// Purpose  : Job scheduling, deployment control, admin tasks, and service
//            health monitoring for all other SMTX layers.
// VM count : 1 (baseline; add a second for HA via the scale-vm-layer workflow)
// VM size  : Standard_B2ms (2 vCPU / 8 GiB) — small CPU VM, cost-efficient
// Services : smtx-control (Python orchestrator, port 9000)

@description('Azure region for all resources')
param location string

@description('Environment tag (dev, staging, prod)')
param environment string

@description('Name prefix for all resources')
param prefix string

@description('Subnet resource ID for the control VM')
param subnetId string

@description('NSG resource ID for the control NIC')
param nsgId string

@description('User-assigned managed identity resource ID')
param identityId string

@description('VM size for the control-plane VM')
@allowed([
  'Standard_B2ms'
  'Standard_B4ms'
  'Standard_D2s_v5'
  'Standard_D4s_v5'
])
param vmSize string = 'Standard_B2ms'

@description('Admin username for the VM')
param adminUsername string = 'smtxadmin'

@description('SSH public key data for the admin user')
@secure()
param sshPublicKey string

// ── Network Interface ─────────────────────────────────────────────────────────

resource nic 'Microsoft.Network/networkInterfaces@2023-04-01' = {
  name: '${prefix}-control-nic'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'control' }
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

resource controlVm 'Microsoft.Compute/virtualMachines@2023-07-01' = {
  name: '${prefix}-control-vm'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'control', role: 'orchestration' }
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
        name: '${prefix}-control-osdisk'
        createOption: 'FromImage'
        managedDisk: { storageAccountType: 'Premium_LRS' }
        diskSizeGB: 64
      }
    }
    osProfile: {
      computerName: '${prefix}-control-vm'
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
  parent: controlVm
  name: 'control-setup'
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
apt-get install -y python3-pip python3-venv git curl jq

if [ ! -d "${SMTX_DIR}/.git" ]; then
  git clone "${SMTX_REPO}" "${SMTX_DIR}"
else
  git -C "${SMTX_DIR}" pull --ff-only
fi

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install fastapi uvicorn[standard] pydantic httpx azure-identity azure-mgmt-compute

chown -R smtxadmin:smtxadmin "${SMTX_DIR}" "${VENV_DIR}"

cat > /etc/systemd/system/smtx-control.service <<'SVCEOF'
[Unit]
Description=SMTX Control-Plane Orchestrator
After=network.target

[Service]
Type=simple
User=smtxadmin
WorkingDirectory=/opt/smtx
ExecStart=/opt/smtx-venv/bin/uvicorn control_plane.orchestrator:app --host 0.0.0.0 --port 9000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable smtx-control
systemctl start smtx-control
''')
    }
  }
}

// ── Outputs ───────────────────────────────────────────────────────────────────

output controlVmId string = controlVm.id
output controlVmName string = controlVm.name
output controlVmPrivateIp string = nic.properties.ipConfigurations[0].properties.privateIPAddress
