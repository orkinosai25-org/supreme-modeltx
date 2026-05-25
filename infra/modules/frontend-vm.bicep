// frontend-vm.bicep — SMTX Web / Chat Frontend VMs (Layer 3)
//
// Purpose  : Public-facing website and chat interface. Multiple VMs handle
//            concurrent traffic and provide one spare for zero-downtime deploys.
// VM count : configurable (default 2, min 1, max 10)
// VM size  : Standard_D4s_v5 (4 vCPU / 16 GiB) — balanced general-purpose
// Services : smtx-frontend (Node.js/Nginx, ports 80 + 443)

@description('Azure region for all resources')
param location string

@description('Environment tag (dev, staging, prod)')
param environment string

@description('Name prefix for all resources')
param prefix string

@description('Subnet resource ID for frontend VMs')
param subnetId string

@description('NSG resource ID for frontend NICs')
param nsgId string

@description('User-assigned managed identity resource ID')
param identityId string

@description('Number of frontend VMs to provision')
@minValue(1)
@maxValue(10)
param vmCount int = 2

@description('VM size for frontend VMs')
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

// ── Public IP Addresses (one per VM — frontend is internet-facing) ────────────

resource publicIps 'Microsoft.Network/publicIPAddresses@2023-04-01' = [for i in range(0, vmCount): {
  name: '${prefix}-frontend-pip-${i}'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'frontend', instance: '${i}' }
  sku: { name: 'Standard' }
  properties: {
    publicIPAllocationMethod: 'Static'
    dnsSettings: {
      domainNameLabel: '${prefix}-frontend-${i}'
    }
  }
}]

// ── Network Interfaces ────────────────────────────────────────────────────────

resource nics 'Microsoft.Network/networkInterfaces@2023-04-01' = [for i in range(0, vmCount): {
  name: '${prefix}-frontend-nic-${i}'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'frontend', instance: '${i}' }
  properties: {
    networkSecurityGroup: { id: nsgId }
    ipConfigurations: [
      {
        name: 'ipconfig1'
        properties: {
          subnet: { id: subnetId }
          privateIPAllocationMethod: 'Dynamic'
          publicIPAddress: { id: publicIps[i].id }
        }
      }
    ]
  }
}]

// ── Virtual Machines ──────────────────────────────────────────────────────────

resource frontendVms 'Microsoft.Compute/virtualMachines@2023-07-01' = [for i in range(0, vmCount): {
  name: '${prefix}-frontend-vm-${i}'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'frontend', role: 'web-chat', instance: '${i}' }
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
        name: '${prefix}-frontend-osdisk-${i}'
        createOption: 'FromImage'
        managedDisk: { storageAccountType: 'Premium_LRS' }
        diskSizeGB: 64
      }
    }
    osProfile: {
      computerName: '${prefix}-frontend-${i}'
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
  parent: frontendVms[i]
  name: 'frontend-setup'
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

# ── System packages ───────────────────────────────────────────────────────────
apt-get update -y
apt-get install -y nginx curl git

# Install Node.js 20 LTS via NodeSource
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs

SMTX_REPO="https://github.com/orkinosai25-org/SMTX.git"
SMTX_DIR="/opt/smtx"
FRONTEND_DIR="${SMTX_DIR}/frontend"

# ── Clone / update repo ────────────────────────────────────────────────────────
if [ ! -d "${SMTX_DIR}/.git" ]; then
  git clone "${SMTX_REPO}" "${SMTX_DIR}"
else
  git -C "${SMTX_DIR}" pull --ff-only
fi

# ── Build and install the frontend app ────────────────────────────────────────
if [ -f "${FRONTEND_DIR}/package.json" ]; then
  cd "${FRONTEND_DIR}"
  npm ci --prefer-offline
  npm run build
fi

# Set ownership after npm build so all generated files are owned by smtxadmin
chown -R smtxadmin:smtxadmin "${SMTX_DIR}"

# ── Systemd service: smtx-frontend ────────────────────────────────────────────
cat > /etc/systemd/system/smtx-frontend.service <<'SVCEOF'
[Unit]
Description=SMTX Web / Chat Frontend (Node.js)
After=network.target

[Service]
Type=simple
User=smtxadmin
WorkingDirectory=/opt/smtx/frontend
# Use "node .next/standalone/server.js" for Next.js standalone builds, or
# "node dist/server.js" for other frameworks, to improve signal handling.
# Falls back to "npm run start" if a direct entry point is not yet configured.
ExecStart=/usr/bin/npm run start
Restart=on-failure
RestartSec=5
Environment=PORT=3000
Environment=NODE_ENV=production

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload

# Only enable/start the service if the frontend app is present
if [ -f "${FRONTEND_DIR}/package.json" ]; then
  systemctl enable smtx-frontend
  systemctl start smtx-frontend
else
  echo "WARNING: ${FRONTEND_DIR}/package.json not found -- smtx-frontend service will not be started automatically."
  echo "Deploy the frontend app and run: systemctl enable --now smtx-frontend"
fi

# ── Configure nginx as reverse proxy ─────────────────────────────────────────
cat > /etc/nginx/sites-available/smtx <<'NGINXEOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
NGINXEOF

ln -sf /etc/nginx/sites-available/smtx /etc/nginx/sites-enabled/smtx
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx
systemctl enable nginx
''')
    }
  }
}]

// ── Outputs ───────────────────────────────────────────────────────────────────

output frontendVmNames array = [for i in range(0, vmCount): frontendVms[i].name]
output frontendPublicIps array = [for i in range(0, vmCount): publicIps[i].properties.ipAddress]
output frontendPrivateIps array = [for i in range(0, vmCount): nics[i].properties.ipConfigurations[0].properties.privateIPAddress]
