// main.bicep — SMTX Azure Infrastructure (top-level orchestration)
//
// SUMOTX is a CPU-only LLM platform. All VM layers use CPU VMs.
// No GPU quota is required.
//
// Provisions:
//   • VNet with subnets for all 7 SMTX layers
//   • NSGs per layer
//   • NEW: Control-plane VM       (Layer 1, 1×)
//   • NEW: API/backend VMs        (Layer 2, default 2×)
//   • NEW: Frontend/web VMs       (Layer 3, default 2×)
//   • NEW: Model lifecycle VM     (Layer 4, 1×)
//   • NEW: Data/governance VMs    (Layer 5, default 2×)
//   • NEW: CPU Inference VM(s)    (Layer 6, default 1×, llama-cpp-python)
//   • NEW: CPU Training VM(s)     (Layer 7, default 1×, LoRA/QLoRA CPU)
//   • Legacy App Service / GPU VM path kept but disabled by default
//   • User-assigned managed identities + RBAC
//   • Storage account for model weights/checkpoints
//   • Key Vault for secrets
//   • Private endpoints for Storage and Key Vault
//
// VM-layer deployment is controlled by the deployVms parameter (default: true).
// Legacy GPU VM deployment remains controlled by deployGpuVm (default: false).
//
// Deploy with:
//   az deployment group create \
//     --resource-group smtx-rg \
//     --template-file infra/main.bicep \
//     --parameters @infra/parameters.json

targetScope = 'resourceGroup'

// ── Parameters ────────────────────────────────────────────────────────────────

@description('Azure region for all resources')
param location string = resourceGroup().location

@description('Short name prefix used for all resource names')
@minLength(3)
@maxLength(10)
param prefix string = 'smtx'

@description('Environment name (dev, staging, prod)')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'dev'

@description('VM size for the GPU node')
@allowed([
  'Standard_NC48ads_A100_v4'
  'Standard_NC40ads_L40S'
  'Standard_NC24ads_A100_v4'
])
param gpuVmSize string = 'Standard_NC48ads_A100_v4'

@description('VM size for the CPU node')
@allowed([
  'Standard_D8s_v5'
  'Standard_D16s_v5'
  'Standard_D32s_v5'
])
param cpuVmSize string = 'Standard_D16s_v5'

@description('SSH public key data for VM admin user (required when deployGpuVm is true)')
@secure()
param sshPublicKey string = ''

@description('Deploy GPU VM, CPU VM, and App Service (set to false when GPU quota is unavailable)')
param deployGpuVm bool = false

@description('Deploy all VM-layer infrastructure (control, API, frontend, lifecycle, data, training). Set to false for storage/network-only deploys.')
param deployVms bool = true

@description('Admin username for VMs')
param adminUsername string = 'smtxadmin'

// ── VM layer counts ───────────────────────────────────────────────────────────

@description('Number of API/backend VMs (min 1, max 10)')
@minValue(1)
@maxValue(10)
param apiVmCount int = 2

@description('Number of frontend/web VMs (min 1, max 10)')
@minValue(1)
@maxValue(10)
param frontendVmCount int = 2

@description('Number of data/governance VMs (min 1, max 8)')
@minValue(1)
@maxValue(8)
param dataVmCount int = 2

@description('Number of LLM training VMs (min 1, max 10). Start with 1 CPU training VM.')
@minValue(1)
@maxValue(10)
param trainingVmCount int = 1

@description('Number of CPU inference VMs (min 1, max 5)')
@minValue(1)
@maxValue(5)
param inferenceVmCount int = 1

// ── VM layer sizes ────────────────────────────────────────────────────────────

@description('VM size for control-plane VM')
@allowed(['Standard_B2ms', 'Standard_B4ms', 'Standard_D2s_v5', 'Standard_D4s_v5'])
param controlVmSize string = 'Standard_B2ms'

@description('VM size for API/backend VMs')
@allowed(['Standard_D2s_v5', 'Standard_D4s_v5', 'Standard_D8s_v5'])
param apiVmSize string = 'Standard_D4s_v5'

@description('VM size for frontend/web VMs')
@allowed(['Standard_D2s_v5', 'Standard_D4s_v5', 'Standard_D8s_v5'])
param frontendVmSize string = 'Standard_D4s_v5'

@description('VM size for model lifecycle VM')
@allowed(['Standard_D2s_v5', 'Standard_D4s_v5', 'Standard_D8s_v5'])
param lifecycleVmSize string = 'Standard_D4s_v5'

@description('CPU VM size for data/governance VMs')
@allowed(['Standard_D4s_v5', 'Standard_D8s_v5', 'Standard_D16s_v5', 'Standard_E8s_v5'])
param dataVmSize string = 'Standard_D8s_v5'

@description('CPU VM size for LLM training VMs. Larger sizes give faster multi-threaded fine-tuning.')
@allowed([
  'Standard_D16s_v5'
  'Standard_D32s_v5'
  'Standard_E16s_v5'
  'Standard_E32s_v5'
])
param trainingVmSize string = 'Standard_D16s_v5'

@description('CPU VM size for inference VMs. Larger sizes support bigger quantised models.')
@allowed([
  'Standard_D8s_v5'
  'Standard_D16s_v5'
  'Standard_E8s_v5'
  'Standard_E16s_v5'
])
param inferenceVmSize string = 'Standard_D8s_v5'

// ── Storage Account ───────────────────────────────────────────────────────────

resource storage 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: take('${prefix}${uniqueString(resourceGroup().id)}sa', 24)
  location: location
  tags: { environment: environment, project: 'smtx' }
  sku: { name: 'Standard_ZRS' }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    allowBlobPublicAccess: false
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Deny'
    }
  }
}

resource modelContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  name: '${storage.name}/default/models'
  properties: {
    publicAccess: 'None'
  }
}

resource checkpointContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  name: '${storage.name}/default/checkpoints'
  properties: {
    publicAccess: 'None'
  }
}

// ── Azure Container Registry ──────────────────────────────────────────────────

resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: '${prefix}${uniqueString(resourceGroup().id)}acr'
  location: location
  tags: { environment: environment, project: 'smtx' }
  sku: { name: 'Premium' }
  properties: {
    adminUserEnabled: false
    publicNetworkAccess: 'Enabled'
    zoneRedundancy: 'Disabled'
  }
}

// ── Azure Batch Account ───────────────────────────────────────────────────────

resource batchAccount 'Microsoft.Batch/batchAccounts@2023-05-01' = {
  name: take('${prefix}${uniqueString(resourceGroup().id)}batch', 24)
  location: location
  tags: { environment: environment, project: 'smtx' }
  properties: {
    poolAllocationMode: 'BatchService'
    autoStorage: {
      storageAccountId: storage.id
      authenticationMode: 'StorageKeys'
    }
  }
}

// ── Key Vault ─────────────────────────────────────────────────────────────────

resource keyVault 'Microsoft.KeyVault/vaults@2023-02-01' = {
  name: '${prefix}-kv-${uniqueString(resourceGroup().id)}'
  location: location
  tags: { environment: environment, project: 'smtx' }
  properties: {
    sku: { family: 'A', name: 'standard' }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 90
    enablePurgeProtection: true
    publicNetworkAccess: 'Disabled'
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Deny'
    }
  }
}

// ── VNet module ───────────────────────────────────────────────────────────────

module vnet 'modules/vnet.bicep' = {
  name: 'vnet-deploy'
  params: {
    location: location
    environment: environment
    prefix: prefix
  }
}

// ── Identity module ───────────────────────────────────────────────────────────

module identity 'modules/identity.bicep' = {
  name: 'identity-deploy'
  params: {
    location: location
    environment: environment
    prefix: prefix
  }
}

// ── GPU VM module ─────────────────────────────────────────────────────────────

module gpuVm 'modules/gpu-vm.bicep' = if (deployGpuVm) {
  name: 'gpu-vm-deploy'
  params: {
    location: location
    environment: environment
    prefix: prefix
    subnetId: vnet.outputs.gpuSubnetId
    nsgId: vnet.outputs.gpuNsgId
    identityId: identity.outputs.gpuVmIdentityId
    vmSize: gpuVmSize
    adminUsername: adminUsername
    sshPublicKey: sshPublicKey
    storageAccountName: storage.name
    modelContainer: 'models'
  }
}

// ── CPU VM module ─────────────────────────────────────────────────────────────

module cpuVm 'modules/cpu-vm.bicep' = if (deployGpuVm) {
  name: 'cpu-vm-deploy'
  params: {
    location: location
    environment: environment
    prefix: prefix
    subnetId: vnet.outputs.cpuSubnetId
    nsgId: vnet.outputs.cpuNsgId
    identityId: identity.outputs.cpuVmIdentityId
    vmSize: cpuVmSize
    adminUsername: adminUsername
    sshPublicKey: sshPublicKey
    gpuVmPrivateIp: deployGpuVm ? gpuVm.outputs.gpuVmPrivateIp : ''
  }
}

// ── App Service module ────────────────────────────────────────────────────────

module appService 'modules/app-service.bicep' = if (deployGpuVm) {
  name: 'app-service-deploy'
  params: {
    location: location
    environment: environment
    prefix: prefix
    appSubnetId: vnet.outputs.appSubnetId
    identityId: identity.outputs.appServiceIdentityId
    identityClientId: identity.outputs.appServiceClientId
    gpuVmPrivateIp: deployGpuVm ? gpuVm.outputs.gpuVmPrivateIp : ''
    cpuVmPrivateIp: deployGpuVm ? cpuVm.outputs.cpuVmPrivateIp : ''
  }
}

// ── VM-layer modules (new) ────────────────────────────────────────────────────

// Layer 1: Control-plane / Orchestration (1 VM)
module controlVm 'modules/control-vm.bicep' = if (deployVms) {
  name: 'control-vm-deploy'
  params: {
    location: location
    environment: environment
    prefix: prefix
    subnetId: vnet.outputs.controlSubnetId
    nsgId: vnet.outputs.controlNsgId
    identityId: identity.outputs.controlVmIdentityId
    vmSize: controlVmSize
    adminUsername: adminUsername
    sshPublicKey: sshPublicKey
  }
}

// Layer 2: API / Backend (default 2 VMs)
module apiVm 'modules/api-vm.bicep' = if (deployVms) {
  name: 'api-vm-deploy'
  params: {
    location: location
    environment: environment
    prefix: prefix
    subnetId: vnet.outputs.apiSubnetId
    nsgId: vnet.outputs.apiNsgId
    identityId: identity.outputs.apiVmIdentityId
    vmCount: apiVmCount
    vmSize: apiVmSize
    adminUsername: adminUsername
    sshPublicKey: sshPublicKey
  }
}

// Layer 3: Frontend / Web / Chat (default 2 VMs)
module frontendVm 'modules/frontend-vm.bicep' = if (deployVms) {
  name: 'frontend-vm-deploy'
  params: {
    location: location
    environment: environment
    prefix: prefix
    subnetId: vnet.outputs.frontendSubnetId
    nsgId: vnet.outputs.frontendNsgId
    identityId: identity.outputs.frontendVmIdentityId
    vmCount: frontendVmCount
    vmSize: frontendVmSize
    adminUsername: adminUsername
    sshPublicKey: sshPublicKey
  }
}

// Layer 4: Model Lifecycle / Registry (1 VM)
module lifecycleVm 'modules/lifecycle-vm.bicep' = if (deployVms) {
  name: 'lifecycle-vm-deploy'
  params: {
    location: location
    environment: environment
    prefix: prefix
    subnetId: vnet.outputs.lifecycleSubnetId
    nsgId: vnet.outputs.lifecycleNsgId
    identityId: identity.outputs.lifecycleVmIdentityId
    storageAccountName: storage.name
    vmSize: lifecycleVmSize
    adminUsername: adminUsername
    sshPublicKey: sshPublicKey
  }
}

// Layer 5: Data / Governance / SharePoint-like (default 2 VMs, expandable to 8)
module dataVm 'modules/data-vm.bicep' = if (deployVms) {
  name: 'data-vm-deploy'
  params: {
    location: location
    environment: environment
    prefix: prefix
    subnetId: vnet.outputs.dataSubnetId
    nsgId: vnet.outputs.dataNsgId
    identityId: identity.outputs.dataVmIdentityId
    storageAccountName: storage.name
    vmCount: dataVmCount
    vmSize: dataVmSize
    adminUsername: adminUsername
    sshPublicKey: sshPublicKey
  }
}

// Layer 6: CPU Inference (default 1 VM; llama-cpp-python OpenAI-compatible)
module inferenceVm 'modules/inference-vm.bicep' = if (deployVms) {
  name: 'inference-vm-deploy'
  params: {
    location: location
    environment: environment
    prefix: prefix
    subnetId: vnet.outputs.inferenceSubnetId
    nsgId: vnet.outputs.inferenceNsgId
    identityId: identity.outputs.inferenceVmIdentityId
    storageAccountName: storage.name
    vmCount: inferenceVmCount
    vmSize: inferenceVmSize
    adminUsername: adminUsername
    sshPublicKey: sshPublicKey
  }
}

// Layer 7: LLM Training (default 1 VM; scale up as budget allows)
module trainingVm 'modules/training-vm.bicep' = if (deployVms) {
  name: 'training-vm-deploy'
  params: {
    location: location
    environment: environment
    prefix: prefix
    subnetId: vnet.outputs.trainingSubnetId
    nsgId: vnet.outputs.trainingNsgId
    identityId: identity.outputs.trainingVmIdentityId
    storageAccountName: storage.name
    vmCount: trainingVmCount
    vmSize: trainingVmSize
    adminUsername: adminUsername
    sshPublicKey: sshPublicKey
  }
}

// ── Private Endpoints ─────────────────────────────────────────────────────────

resource storagePrivateEndpoint 'Microsoft.Network/privateEndpoints@2023-04-01' = {
  name: '${prefix}-storage-pe'
  location: location
  tags: { environment: environment, project: 'smtx' }
  properties: {
    subnet: { id: vnet.outputs.privateEndpointSubnetId }
    privateLinkServiceConnections: [
      {
        name: '${prefix}-storage-plsc'
        properties: {
          privateLinkServiceId: storage.id
          groupIds: ['blob']
        }
      }
    ]
  }
}

resource storagePrivateDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2023-04-01' = {
  parent: storagePrivateEndpoint
  name: 'storage-dns-zone-group'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'blob-dns-config'
        properties: {
          privateDnsZoneId: vnet.outputs.blobPrivateDnsZoneId
        }
      }
    ]
  }
}

resource kvPrivateEndpoint 'Microsoft.Network/privateEndpoints@2023-04-01' = {
  name: '${prefix}-kv-pe'
  location: location
  tags: { environment: environment, project: 'smtx' }
  properties: {
    subnet: { id: vnet.outputs.privateEndpointSubnetId }
    privateLinkServiceConnections: [
      {
        name: '${prefix}-kv-plsc'
        properties: {
          privateLinkServiceId: keyVault.id
          groupIds: ['vault']
        }
      }
    ]
  }
}

resource kvPrivateDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2023-04-01' = {
  parent: kvPrivateEndpoint
  name: 'kv-dns-zone-group'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'kv-dns-config'
        properties: {
          privateDnsZoneId: vnet.outputs.kvPrivateDnsZoneId
        }
      }
    ]
  }
}

// ── Outputs ───────────────────────────────────────────────────────────────────

// Core resources
output storageAccountId string = storage.id
output storageAccountName string = storage.name
output acrId string = acr.id
output acrLoginServer string = acr.properties.loginServer
output batchAccountId string = batchAccount.id
output batchAccountName string = batchAccount.name
output keyVaultId string = keyVault.id
output keyVaultName string = keyVault.name
output vnetId string = vnet.outputs.vnetId

// Existing inference layer
output gpuVmName string = deployGpuVm ? gpuVm.outputs.gpuVmName : ''
output gpuVmPrivateIp string = deployGpuVm ? gpuVm.outputs.gpuVmPrivateIp : ''
output cpuVmName string = deployGpuVm ? cpuVm.outputs.cpuVmName : ''
output cpuVmPrivateIp string = deployGpuVm ? cpuVm.outputs.cpuVmPrivateIp : ''
output txOrchestratorUrl string = deployGpuVm ? 'https://${appService.outputs.appServiceDefaultHostname}' : ''
output appInsightsConnectionString string = deployGpuVm ? appService.outputs.appInsightsConnectionString : ''

// Identity principal IDs (needed by RBAC template)
output gpuVmPrincipalId string = identity.outputs.gpuVmPrincipalId
output cpuVmPrincipalId string = identity.outputs.cpuVmPrincipalId
output appServicePrincipalId string = identity.outputs.appServicePrincipalId
output controlVmPrincipalId string = identity.outputs.controlVmPrincipalId
output apiVmPrincipalId string = identity.outputs.apiVmPrincipalId
output frontendVmPrincipalId string = identity.outputs.frontendVmPrincipalId
output lifecycleVmPrincipalId string = identity.outputs.lifecycleVmPrincipalId
output dataVmPrincipalId string = identity.outputs.dataVmPrincipalId
output trainingVmPrincipalId string = identity.outputs.trainingVmPrincipalId
output inferenceVmPrincipalId string = identity.outputs.inferenceVmPrincipalId

// New VM-layer outputs
output controlVmName string = deployVms ? controlVm.outputs.controlVmName : ''
output controlVmPrivateIp string = deployVms ? controlVm.outputs.controlVmPrivateIp : ''
output apiVmNames array = deployVms ? apiVm.outputs.apiVmNames : []
output apiVmPrivateIps array = deployVms ? apiVm.outputs.apiVmPrivateIps : []
output frontendVmNames array = deployVms ? frontendVm.outputs.frontendVmNames : []
output frontendPublicIps array = deployVms ? frontendVm.outputs.frontendPublicIps : []
output lifecycleVmName string = deployVms ? lifecycleVm.outputs.lifecycleVmName : ''
output lifecycleVmPrivateIp string = deployVms ? lifecycleVm.outputs.lifecycleVmPrivateIp : ''
output dataVmNames array = deployVms ? dataVm.outputs.dataVmNames : []
output dataVmPrivateIps array = deployVms ? dataVm.outputs.dataVmPrivateIps : []
output inferenceVmNames array = deployVms ? inferenceVm.outputs.inferenceVmNames : []
output inferenceVmPrivateIps array = deployVms ? inferenceVm.outputs.inferenceVmPrivateIps : []
output trainingVmNames array = deployVms ? trainingVm.outputs.trainingVmNames : []
output trainingVmPrivateIps array = deployVms ? trainingVm.outputs.trainingVmPrivateIps : []
