targetScope = 'resourceGroup'

param location string
param tags object = {}
param storageAccountName string

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  name: 'default'
  parent: storage
}

resource trainingContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  name: 'training'
  parent: blobService
  properties: {
    publicAccess: 'None'
  }
}

resource checkpointsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  name: 'checkpoints'
  parent: blobService
  properties: {
    publicAccess: 'None'
  }
}

resource documentsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  name: 'documents'
  parent: blobService
  properties: {
    publicAccess: 'None'
  }
}

resource artifactsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  name: 'artifacts'
  parent: blobService
  properties: {
    publicAccess: 'None'
  }
}

resource policiesContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  name: 'policies'
  parent: blobService
  properties: {
    publicAccess: 'None'
  }
}

output storageAccountId string = storage.id
output storageAccountName string = storage.name
output blobEndpoint string = storage.properties.primaryEndpoints.blob
