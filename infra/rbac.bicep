// rbac.bicep — SMTX RBAC Role Assignments
//
// Deploys role assignments for managed identities.
// Requires Microsoft.Authorization/roleAssignments/write permission at the
// resource group scope (Owner or User Access Administrator role).
//
// Deploy with:
//   az deployment group create \
//     --resource-group smtx-rg \
//     --template-file infra/rbac.bicep \
//     --parameters gpuVmPrincipalId=<id> cpuVmPrincipalId=<id> \
//                  appServicePrincipalId=<id> storageAccountName=<name> keyVaultName=<name>

targetScope = 'resourceGroup'

@description('Principal ID of the GPU VM managed identity')
param gpuVmPrincipalId string

@description('Principal ID of the CPU VM managed identity')
param cpuVmPrincipalId string

@description('Principal ID of the App Service managed identity')
param appServicePrincipalId string

// ── New VM-layer principal IDs ────────────────────────────────────────────────

@description('Principal ID of the control-plane VM managed identity')
param controlVmPrincipalId string = ''

@description('Principal ID of the API/backend VM managed identity')
param apiVmPrincipalId string = ''

@description('Principal ID of the frontend VM managed identity')
param frontendVmPrincipalId string = ''

@description('Principal ID of the model lifecycle VM managed identity')
param lifecycleVmPrincipalId string = ''

@description('Principal ID of the data/governance VM managed identity')
param dataVmPrincipalId string = ''

@description('Principal ID of the LLM training VM managed identity')
param trainingVmPrincipalId string = ''

@description('Principal ID of the CPU inference VM managed identity')
param inferenceVmPrincipalId string = ''

@description('Name of the storage account used for model weights and checkpoints')
param storageAccountName string

@description('Name of the Key Vault used for secrets')
param keyVaultName string

// ── Existing resource references ──────────────────────────────────────────────

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' existing = {
  name: storageAccountName
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-02-01' existing = {
  name: keyVaultName
}

// ── Built-in role definition IDs ──────────────────────────────────────────────
// Storage Blob Data Contributor
var storageBlobContributorRoleId = 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
// Key Vault Secrets User
var kvSecretsUserRoleId = '4633458b-17de-408a-b874-0445c86b69e6'

// ── GPU VM: Storage Blob Data Contributor (read model weights) ────────────────

resource gpuStorageRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storageAccount.id, gpuVmPrincipalId, storageBlobContributorRoleId)
  scope: storageAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageBlobContributorRoleId)
    principalId: gpuVmPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource gpuKvRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, gpuVmPrincipalId, kvSecretsUserRoleId)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', kvSecretsUserRoleId)
    principalId: gpuVmPrincipalId
    principalType: 'ServicePrincipal'
  }
}

// ── CPU VM: Storage Blob Data Contributor + Key Vault Secrets User ────────────

resource cpuStorageRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storageAccount.id, cpuVmPrincipalId, storageBlobContributorRoleId)
  scope: storageAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageBlobContributorRoleId)
    principalId: cpuVmPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource cpuKvRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, cpuVmPrincipalId, kvSecretsUserRoleId)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', kvSecretsUserRoleId)
    principalId: cpuVmPrincipalId
    principalType: 'ServicePrincipal'
  }
}

// ── App Service: Key Vault Secrets User ───────────────────────────────────────

resource appKvRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, appServicePrincipalId, kvSecretsUserRoleId)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', kvSecretsUserRoleId)
    principalId: appServicePrincipalId
    principalType: 'ServicePrincipal'
  }
}

// ── New VM layers: Storage Blob Data Contributor + Key Vault Secrets User ──────
// Each new layer gets read/write access to storage and secrets.
// Assignments are skipped (condition) when principal IDs are not provided.

resource controlStorageRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(controlVmPrincipalId)) {
  name: guid(storageAccount.id, controlVmPrincipalId, storageBlobContributorRoleId)
  scope: storageAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageBlobContributorRoleId)
    principalId: controlVmPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource controlKvRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(controlVmPrincipalId)) {
  name: guid(keyVault.id, controlVmPrincipalId, kvSecretsUserRoleId)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', kvSecretsUserRoleId)
    principalId: controlVmPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource apiStorageRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(apiVmPrincipalId)) {
  name: guid(storageAccount.id, apiVmPrincipalId, storageBlobContributorRoleId)
  scope: storageAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageBlobContributorRoleId)
    principalId: apiVmPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource apiKvRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(apiVmPrincipalId)) {
  name: guid(keyVault.id, apiVmPrincipalId, kvSecretsUserRoleId)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', kvSecretsUserRoleId)
    principalId: apiVmPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource frontendKvRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(frontendVmPrincipalId)) {
  name: guid(keyVault.id, frontendVmPrincipalId, kvSecretsUserRoleId)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', kvSecretsUserRoleId)
    principalId: frontendVmPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource lifecycleStorageRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(lifecycleVmPrincipalId)) {
  name: guid(storageAccount.id, lifecycleVmPrincipalId, storageBlobContributorRoleId)
  scope: storageAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageBlobContributorRoleId)
    principalId: lifecycleVmPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource lifecycleKvRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(lifecycleVmPrincipalId)) {
  name: guid(keyVault.id, lifecycleVmPrincipalId, kvSecretsUserRoleId)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', kvSecretsUserRoleId)
    principalId: lifecycleVmPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource dataStorageRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(dataVmPrincipalId)) {
  name: guid(storageAccount.id, dataVmPrincipalId, storageBlobContributorRoleId)
  scope: storageAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageBlobContributorRoleId)
    principalId: dataVmPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource dataKvRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(dataVmPrincipalId)) {
  name: guid(keyVault.id, dataVmPrincipalId, kvSecretsUserRoleId)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', kvSecretsUserRoleId)
    principalId: dataVmPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource trainingStorageRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(trainingVmPrincipalId)) {
  name: guid(storageAccount.id, trainingVmPrincipalId, storageBlobContributorRoleId)
  scope: storageAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageBlobContributorRoleId)
    principalId: trainingVmPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource trainingKvRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(trainingVmPrincipalId)) {
  name: guid(keyVault.id, trainingVmPrincipalId, kvSecretsUserRoleId)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', kvSecretsUserRoleId)
    principalId: trainingVmPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource inferenceStorageRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(inferenceVmPrincipalId)) {
  name: guid(storageAccount.id, inferenceVmPrincipalId, storageBlobContributorRoleId)
  scope: storageAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageBlobContributorRoleId)
    principalId: inferenceVmPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource inferenceKvRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(inferenceVmPrincipalId)) {
  name: guid(keyVault.id, inferenceVmPrincipalId, kvSecretsUserRoleId)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', kvSecretsUserRoleId)
    principalId: inferenceVmPrincipalId
    principalType: 'ServicePrincipal'
  }
}
