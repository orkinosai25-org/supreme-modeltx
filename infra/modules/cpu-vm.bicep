// cpu-vm.bicep — SMTX CPU Virtual Machine
// Hosts: CPU Inference (T-101, port 8003), Retrieval (T-301, port 8001),
//        and Verification (T-501, port 8002) services.

@description('Azure region for all resources')
param location string

@description('Environment tag (dev, staging, prod)')
param environment string

@description('Name prefix for all resources')
param prefix string

@description('Subnet resource ID for the CPU VM')
param subnetId string

@description('NSG resource ID for the CPU NIC')
param nsgId string

@description('User-assigned managed identity resource ID')
param identityId string

@description('VM size for the CPU workload (inference, retrieval + verification)')
@allowed([
  'Standard_D8s_v5'
  'Standard_D16s_v5'
  'Standard_D32s_v5'
])
param vmSize string = 'Standard_D16s_v5'

@description('Admin username for the VM')
param adminUsername string = 'smtxadmin'

@description('SSH public key data for the admin user')
@secure()
param sshPublicKey string

@description('Private IP address of the GPU VM (used to reach the vLLM endpoint)')
param gpuVmPrivateIp string

@description('Path to the T-101 model directory used by the CPU inference service')
param modelPath string = 'tmodels/t101'

// ── Network Interface ─────────────────────────────────────────────────────────

resource nic 'Microsoft.Network/networkInterfaces@2023-04-01' = {
  name: '${prefix}-cpu-nic'
  location: location
  tags: { environment: environment, project: 'smtx', component: 'cpu-vm' }
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

resource cpuVm 'Microsoft.Compute/virtualMachines@2023-07-01' = {
  name: '${prefix}-cpu-vm'
  location: location
  tags: { environment: environment, project: 'smtx', component: 'cpu-vm', role: 'inference-retrieval-verification' }
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
        publisher: 'Canonical'
        offer: '0001-com-ubuntu-server-jammy'
        sku: '22_04-lts-gen2'
        version: 'latest'
      }
      osDisk: {
        name: '${prefix}-cpu-osdisk'
        createOption: 'FromImage'
        managedDisk: {
          storageAccountType: 'Premium_LRS'
        }
        diskSizeGB: 128
      }
      dataDisks: [
        {
          name: '${prefix}-cpu-datadisk'
          lun: 0
          createOption: 'Empty'
          diskSizeGB: 512
          managedDisk: {
            storageAccountType: 'Premium_LRS'
          }
        }
      ]
    }
    osProfile: {
      computerName: '${prefix}-cpu-vm'
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

// ── Custom Script Extension: Install inference, retrieval + verification services ──

resource cpuSetupExtension 'Microsoft.Compute/virtualMachines/extensions@2023-07-01' = {
  parent: cpuVm
  name: 'cpu-setup'
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
# Usage: script.sh <gpu_vm_private_ip> [model_path]
set -euo pipefail

GPU_VM_IP="$1"
MODEL_PATH="${2:-tmodels/t101}"
SMTX_REPO="https://github.com/orkinosai25-org/SMTX.git"
SMTX_DIR="/opt/smtx"
VENV_DIR="/opt/smtx-venv"

# ── System dependencies ───────────────────────────────────────────────────────
apt-get update -y
apt-get install -y python3-pip python3-venv git curl

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
pip install fastapi uvicorn[standard] pydantic \
            sentence-transformers faiss-cpu \
            azure-storage-blob azure-identity httpx \
            torch transformers accelerate sentencepiece tokenizers

# ── Fix ownership ─────────────────────────────────────────────────────────────
chown -R smtxadmin:smtxadmin "${SMTX_DIR}" "${VENV_DIR}"

# ── CPU Inference service (default inference backend, port 8003) ──────────────
cat > /etc/systemd/system/smtx-inference.service <<SVCEOF
[Unit]
Description=SMTX CPU Inference Service (T-101)
After=network.target

[Service]
Type=simple
User=smtxadmin
WorkingDirectory=/opt/smtx
ExecStart=/opt/smtx-venv/bin/python inference/cpu_inference_server.py --host 0.0.0.0 --port 8003
Restart=on-failure
RestartSec=5
Environment=MODEL_PATH=${MODEL_PATH}

[Install]
WantedBy=multi-user.target
SVCEOF

# ── T-301 Retrieval service ────────────────────────────────────────────────────
cat > /etc/systemd/system/smtx-retrieval.service <<SVCEOF
[Unit]
Description=SMTX T-301 Retrieval Service
After=network.target

[Service]
Type=simple
User=smtxadmin
WorkingDirectory=/opt/smtx
ExecStart=/opt/smtx-venv/bin/uvicorn tmodels.t301.retrieval_server:app --host 0.0.0.0 --port 8001
Restart=on-failure
RestartSec=5
Environment=VLLM_ENDPOINT=http://${GPU_VM_IP}:8000

[Install]
WantedBy=multi-user.target
SVCEOF

# ── T-501 Verification service ─────────────────────────────────────────────────
cat > /etc/systemd/system/smtx-verification.service <<SVCEOF
[Unit]
Description=SMTX T-501 Verification Service
After=network.target

[Service]
Type=simple
User=smtxadmin
WorkingDirectory=/opt/smtx
ExecStart=/opt/smtx-venv/bin/uvicorn tmodels.t501.verification_server:app --host 0.0.0.0 --port 8002
Restart=on-failure
RestartSec=5
Environment=VLLM_ENDPOINT=http://${GPU_VM_IP}:8000

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable smtx-inference smtx-retrieval smtx-verification
systemctl start smtx-inference smtx-retrieval smtx-verification
''')
    }
    protectedSettings: {
      // commandToExecute injects the GPU VM private IP and model path as positional arguments.
      commandToExecute: 'bash /var/lib/waagent/custom-script/download/0/script.sh ${gpuVmPrivateIp} ${modelPath}'
    }
  }
}

// ── Outputs ───────────────────────────────────────────────────────────────────

output cpuVmId string = cpuVm.id
output cpuVmName string = cpuVm.name
output cpuVmPrivateIp string = nic.properties.ipConfigurations[0].properties.privateIPAddress
