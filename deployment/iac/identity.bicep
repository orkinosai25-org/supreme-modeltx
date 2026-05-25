targetScope = 'resourceGroup'

param location string
param tags object = {}
param identityName string = 'sumotx-mi'

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
  tags: tags
}

output identityId string = identity.id
output identityPrincipalId string = identity.properties.principalId
output identityClientId string = identity.properties.clientId
