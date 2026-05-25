targetScope = 'resourceGroup'

param location string
param tags object = {}
param openAiAccountName string

resource openAiAccount 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: openAiAccountName
  location: location
  tags: tags
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: openAiAccountName
    publicNetworkAccess: 'Enabled'
  }
}

output openAiAccountId string = openAiAccount.id
output openAiAccountName string = openAiAccount.name
output openAiEndpoint string = openAiAccount.properties.endpoint
