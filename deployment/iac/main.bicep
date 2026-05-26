targetScope = 'subscription'

@allowed([
  'dev'
  'prod'
])
@description('Deployment environment label.')
param env string = 'dev'

@description('Azure region for all resources.')
param location string = 'uksouth'

@description('Resource group for SUMOTX resources.')
param resourceGroupName string = 'sumotx-rg'

@description('Admin username for both VMs.')
param adminUsername string = 'sumotxadmin'

@description('SSH public key injected into both VMs.')
param sshPublicKey string

@secure()
@description('SQL admin password.')
param sqlAdminPassword string

@description('SQL admin login.')
param sqlAdminLogin string = 'sumotxsqladmin'

@description('Azure OpenAI model deployment name for chat/completion workloads.')
param openAiChatDeploymentName string = 'gpt-4o-mini'

@description('Azure OpenAI model deployment name for embedding workloads.')
param openAiEmbeddingDeploymentName string = 'text-embedding-3-small'

@description('Frontend VM size.')
param frontendVmSize string = 'Standard_D4s_v5'

@description('Backend VM size.')
param backendVmSize string = 'Standard_D4s_v5'

@description('Checkpoint data disk size for backend VM.')
param backendCheckpointDiskSizeGB int = 256

@description('CIDR allowed to SSH to the frontend VM.')
param frontendSshSourceCidr string

var suffix = toLower(uniqueString(subscription().id, resourceGroupName, env))
var tags = {
  project: 'sumotx'
  environment: env
  managedBy: 'bicep'
}

resource rg 'Microsoft.Resources/resourceGroups@2024-03-01' = {
  name: resourceGroupName
  location: location
  tags: tags
}

module network './network.bicep' = {
  name: 'sumotx-network-${env}'
  scope: rg
  params: {
    location: location
    tags: tags
    vnetName: 'sumotx-vnet'
    frontendSubnetName: 'frontend-subnet'
    backendSubnetName: 'backend-subnet'
    frontendSshSourceCidr: frontendSshSourceCidr
  }
}

module identity './identity.bicep' = {
  name: 'sumotx-identity-${env}'
  scope: rg
  params: {
    location: location
    tags: tags
    identityName: 'sumotx-mi'
  }
}

module storage './storage.bicep' = {
  name: 'sumotx-storage-${env}'
  scope: rg
  params: {
    location: location
    tags: tags
    storageAccountName: take('sumotx${suffix}', 24)
  }
}

module search './search.bicep' = {
  name: 'sumotx-search-${env}'
  scope: rg
  params: {
    location: location
    tags: tags
    searchServiceName: 'sumotx-search-${take(suffix, 8)}'
  }
}

module openAi './openai.bicep' = {
  name: 'sumotx-openai-${env}'
  scope: rg
  params: {
    location: location
    tags: tags
    openAiAccountName: 'sumotx-openai-${take(suffix, 8)}'
  }
}

module database './database.bicep' = {
  name: 'sumotx-database-${env}'
  scope: rg
  params: {
    location: location
    tags: tags
    sqlServerName: 'sumotx-sql-${take(suffix, 8)}'
    databaseName: 'sumotxdb'
    sqlAdminLogin: sqlAdminLogin
    sqlAdminPassword: sqlAdminPassword
  }
}

module frontendVm './frontend-vm.bicep' = {
  name: 'sumotx-frontend-vm-${env}'
  scope: rg
  params: {
    location: location
    tags: tags
    vmName: 'sumotx-frontend-vm'
    vmSize: frontendVmSize
    adminUsername: adminUsername
    sshPublicKey: sshPublicKey
    subnetId: network.outputs.frontendSubnetId
    userAssignedIdentityResourceId: identity.outputs.identityId
  }
}

module backendVm './backend-vm.bicep' = {
  name: 'sumotx-backend-vm-${env}'
  scope: rg
  params: {
    location: location
    tags: tags
    vmName: 'sumotx-backend-vm'
    vmSize: backendVmSize
    adminUsername: adminUsername
    sshPublicKey: sshPublicKey
    subnetId: network.outputs.backendSubnetId
    userAssignedIdentityResourceId: identity.outputs.identityId
    checkpointDiskSizeGB: backendCheckpointDiskSizeGB
  }
}

output resourceGroup string = rg.name
output frontendPublicIp string = frontendVm.outputs.publicIpAddress
output frontendPrivateIp string = frontendVm.outputs.privateIpAddress
output backendPrivateIp string = backendVm.outputs.privateIpAddress
output storageAccountName string = storage.outputs.storageAccountName
output storageBlobEndpoint string = storage.outputs.blobEndpoint
output searchServiceName string = search.outputs.searchServiceName
output searchEndpoint string = search.outputs.searchEndpoint
output sqlServerName string = database.outputs.sqlServerName
output sqlServerFqdn string = database.outputs.sqlServerFqdn
output sqlDatabaseName string = database.outputs.databaseName
output sqlAdminLogin string = database.outputs.sqlAdminLogin
output openAiAccountName string = openAi.outputs.openAiAccountName
output openAiEndpoint string = openAi.outputs.openAiEndpoint
output openAiChatDeploymentName string = openAiChatDeploymentName
output openAiEmbeddingDeploymentName string = openAiEmbeddingDeploymentName
