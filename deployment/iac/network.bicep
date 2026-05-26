targetScope = 'resourceGroup'

param location string
param tags object = {}
param vnetName string
param frontendSubnetName string
param backendSubnetName string
param frontendSshSourceCidr string = '0.0.0.0/0'

resource frontendNsg 'Microsoft.Network/networkSecurityGroups@2024-05-01' = {
  name: '${vnetName}-frontend-nsg'
  location: location
  tags: tags
  properties: {
    securityRules: [
      {
        name: 'allow-https-inbound'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '443'
          sourceAddressPrefix: '*'
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 100
          direction: 'Inbound'
        }
      }
      {
        name: 'allow-ssh-inbound'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '22'
          sourceAddressPrefix: frontendSshSourceCidr
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 110
          direction: 'Inbound'
        }
      }
    ]
  }
}

resource backendNsg 'Microsoft.Network/networkSecurityGroups@2024-05-01' = {
  name: '${vnetName}-backend-nsg'
  location: location
  tags: tags
  properties: {
    securityRules: [
      {
        name: 'allow-ssh-from-vnet'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '22'
          sourceAddressPrefix: 'VirtualNetwork'
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 100
          direction: 'Inbound'
        }
      }
      {
        name: 'allow-backend-internal'
        properties: {
          protocol: '*'
          sourcePortRange: '*'
          destinationPortRange: '*'
          sourceAddressPrefix: 'VirtualNetwork'
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 110
          direction: 'Inbound'
        }
      }
    ]
  }
}

resource vnet 'Microsoft.Network/virtualNetworks@2024-05-01' = {
  name: vnetName
  location: location
  tags: tags
  properties: {
    addressSpace: {
      addressPrefixes: [
        '10.40.0.0/16'
      ]
    }
    subnets: [
      {
        name: frontendSubnetName
        properties: {
          addressPrefix: '10.40.1.0/24'
          networkSecurityGroup: {
            id: frontendNsg.id
          }
        }
      }
      {
        name: backendSubnetName
        properties: {
          addressPrefix: '10.40.2.0/24'
          networkSecurityGroup: {
            id: backendNsg.id
          }
        }
      }
    ]
  }
}

output frontendSubnetId string = '${vnet.id}/subnets/${frontendSubnetName}'
output backendSubnetId string = '${vnet.id}/subnets/${backendSubnetName}'
