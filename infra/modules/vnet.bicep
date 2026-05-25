// vnet.bicep — SMTX Virtual Network, Subnets, and Private Endpoints
//
// Subnet layout (10.0.0.0/16):
//   10.0.1.0/24  inference-subnet    — Layer 6: CPU inference VM (llama-cpp-python, :8000)
//   10.0.2.0/24  cpu-subnet          — Legacy: CPU retrieval/verification VMs
//   10.0.3.0/24  app-subnet          — App Service (VNet integrated)
//   10.0.4.0/24  private-endpoint    — Storage + Key Vault private endpoints
//   10.0.5.0/24  control-subnet      — Layer 1: control-plane / orchestration VM
//   10.0.6.0/24  api-subnet          — Layer 2: API / backend VMs
//   10.0.7.0/24  frontend-subnet     — Layer 3: web / chat frontend VMs
//   10.0.8.0/24  lifecycle-subnet    — Layer 4: model lifecycle / registry VM
//   10.0.9.0/24  data-subnet         — Layer 5: data / governance VMs
//   10.0.10.0/24 training-subnet     — Layer 7: LLM training VM(s)

@description('Azure region for all resources')
param location string

@description('Environment tag (dev, staging, prod)')
param environment string

@description('Name prefix for all resources')
param prefix string

// ── VNet ──────────────────────────────────────────────────────────────────────

resource vnet 'Microsoft.Network/virtualNetworks@2023-04-01' = {
  name: '${prefix}-vnet'
  location: location
  tags: { environment: environment, project: 'smtx' }
  properties: {
    addressSpace: {
      addressPrefixes: ['10.0.0.0/16']
    }
    subnets: [
      {
        name: 'gpu-subnet'
        properties: {
          addressPrefix: '10.0.1.0/24'
          privateEndpointNetworkPolicies: 'Disabled'
        }
      }
      {
        name: 'cpu-subnet'
        properties: {
          addressPrefix: '10.0.2.0/24'
          privateEndpointNetworkPolicies: 'Disabled'
        }
      }
      {
        name: 'app-subnet'
        properties: {
          addressPrefix: '10.0.3.0/24'
          delegations: [
            {
              name: 'appServiceDelegation'
              properties: {
                serviceName: 'Microsoft.Web/serverFarms'
              }
            }
          ]
        }
      }
      {
        name: 'private-endpoint-subnet'
        properties: {
          addressPrefix: '10.0.4.0/24'
          privateEndpointNetworkPolicies: 'Disabled'
        }
      }
      // ── VM-layer subnets (new) ──────────────────────────────────────────────
      {
        name: 'control-subnet'
        properties: {
          addressPrefix: '10.0.5.0/24'
          privateEndpointNetworkPolicies: 'Disabled'
        }
      }
      {
        name: 'api-subnet'
        properties: {
          addressPrefix: '10.0.6.0/24'
          privateEndpointNetworkPolicies: 'Disabled'
        }
      }
      {
        name: 'frontend-subnet'
        properties: {
          addressPrefix: '10.0.7.0/24'
          privateEndpointNetworkPolicies: 'Disabled'
        }
      }
      {
        name: 'lifecycle-subnet'
        properties: {
          addressPrefix: '10.0.8.0/24'
          privateEndpointNetworkPolicies: 'Disabled'
        }
      }
      {
        name: 'data-subnet'
        properties: {
          addressPrefix: '10.0.9.0/24'
          privateEndpointNetworkPolicies: 'Disabled'
        }
      }
      {
        name: 'training-subnet'
        properties: {
          addressPrefix: '10.0.10.0/24'
          privateEndpointNetworkPolicies: 'Disabled'
        }
      }
    ]
  }
}

// ── Network Security Groups ───────────────────────────────────────────────────
// Existing NSGs (gpu, cpu) are unchanged.
// New NSGs follow the same deny-all-inbound + explicit-allow pattern.

resource gpuNsg 'Microsoft.Network/networkSecurityGroups@2023-04-01' = {
  name: '${prefix}-gpu-nsg'
  location: location
  tags: { environment: environment, project: 'smtx' }
  properties: {
    securityRules: [
      {
        name: 'allow-vllm-inbound'
        properties: {
          priority: 100
          protocol: 'Tcp'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: '10.0.0.0/16'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '8000'
        }
      }
      {
        name: 'allow-ssh-inbound'
        properties: {
          priority: 200
          protocol: 'Tcp'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: 'VirtualNetwork'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '22'
        }
      }
      {
        name: 'deny-all-inbound'
        properties: {
          priority: 4096
          protocol: '*'
          access: 'Deny'
          direction: 'Inbound'
          sourceAddressPrefix: '*'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '*'
        }
      }
    ]
  }
}

resource cpuNsg 'Microsoft.Network/networkSecurityGroups@2023-04-01' = {
  name: '${prefix}-cpu-nsg'
  location: location
  tags: { environment: environment, project: 'smtx' }
  properties: {
    securityRules: [
      {
        name: 'allow-retrieval-inbound'
        properties: {
          priority: 100
          protocol: 'Tcp'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: '10.0.0.0/16'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRanges: ['8001', '8002']
        }
      }
      {
        name: 'allow-ssh-inbound'
        properties: {
          priority: 200
          protocol: 'Tcp'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: 'VirtualNetwork'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '22'
        }
      }
      {
        name: 'deny-all-inbound'
        properties: {
          priority: 4096
          protocol: '*'
          access: 'Deny'
          direction: 'Inbound'
          sourceAddressPrefix: '*'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '*'
        }
      }
    ]
  }
}

// ── Private DNS Zone (for storage + key vault) ────────────────────────────────

resource blobPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: 'privatelink.blob.${az.environment().suffixes.storage}'
  location: 'global'
  tags: { environment: environment, project: 'smtx' }
}

resource kvPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: 'privatelink.vaultcore.azure.net'
  location: 'global'
  tags: { environment: environment, project: 'smtx' }
}

resource blobDnsVnetLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: blobPrivateDnsZone
  name: '${prefix}-blob-dns-link'
  location: 'global'
  properties: {
    virtualNetwork: { id: vnet.id }
    registrationEnabled: false
  }
}

resource kvDnsVnetLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: kvPrivateDnsZone
  name: '${prefix}-kv-dns-link'
  location: 'global'
  properties: {
    virtualNetwork: { id: vnet.id }
    registrationEnabled: false
  }
}

// ── New layer NSGs ────────────────────────────────────────────────────────────

resource controlNsg 'Microsoft.Network/networkSecurityGroups@2023-04-01' = {
  name: '${prefix}-control-nsg'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'control' }
  properties: {
    securityRules: [
      {
        name: 'allow-ssh-vnet'
        properties: {
          priority: 100
          protocol: 'Tcp'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: 'VirtualNetwork'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '22'
        }
      }
      {
        name: 'deny-all-inbound'
        properties: {
          priority: 4096
          protocol: '*'
          access: 'Deny'
          direction: 'Inbound'
          sourceAddressPrefix: '*'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '*'
        }
      }
    ]
  }
}

resource apiNsg 'Microsoft.Network/networkSecurityGroups@2023-04-01' = {
  name: '${prefix}-api-nsg'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'api' }
  properties: {
    securityRules: [
      {
        name: 'allow-api-inbound'
        properties: {
          priority: 100
          protocol: 'Tcp'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: '10.0.0.0/16'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRanges: ['80', '443', '8080']
        }
      }
      {
        name: 'allow-ssh-vnet'
        properties: {
          priority: 200
          protocol: 'Tcp'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: 'VirtualNetwork'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '22'
        }
      }
      {
        name: 'deny-all-inbound'
        properties: {
          priority: 4096
          protocol: '*'
          access: 'Deny'
          direction: 'Inbound'
          sourceAddressPrefix: '*'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '*'
        }
      }
    ]
  }
}

resource frontendNsg 'Microsoft.Network/networkSecurityGroups@2023-04-01' = {
  name: '${prefix}-frontend-nsg'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'frontend' }
  properties: {
    securityRules: [
      {
        name: 'allow-http-inbound'
        properties: {
          priority: 100
          protocol: 'Tcp'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: '*'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRanges: ['80', '443']
        }
      }
      {
        name: 'allow-ssh-vnet'
        properties: {
          priority: 200
          protocol: 'Tcp'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: 'VirtualNetwork'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '22'
        }
      }
      {
        name: 'deny-all-inbound'
        properties: {
          priority: 4096
          protocol: '*'
          access: 'Deny'
          direction: 'Inbound'
          sourceAddressPrefix: '*'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '*'
        }
      }
    ]
  }
}

resource lifecycleNsg 'Microsoft.Network/networkSecurityGroups@2023-04-01' = {
  name: '${prefix}-lifecycle-nsg'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'lifecycle' }
  properties: {
    securityRules: [
      {
        name: 'allow-lifecycle-inbound'
        properties: {
          priority: 100
          protocol: 'Tcp'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: '10.0.0.0/16'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '8090'
        }
      }
      {
        name: 'allow-ssh-vnet'
        properties: {
          priority: 200
          protocol: 'Tcp'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: 'VirtualNetwork'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '22'
        }
      }
      {
        name: 'deny-all-inbound'
        properties: {
          priority: 4096
          protocol: '*'
          access: 'Deny'
          direction: 'Inbound'
          sourceAddressPrefix: '*'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '*'
        }
      }
    ]
  }
}

resource dataNsg 'Microsoft.Network/networkSecurityGroups@2023-04-01' = {
  name: '${prefix}-data-nsg'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'data' }
  properties: {
    securityRules: [
      {
        name: 'allow-data-inbound'
        properties: {
          priority: 100
          protocol: 'Tcp'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: '10.0.0.0/16'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRanges: ['80', '443', '8100']
        }
      }
      {
        name: 'allow-ssh-vnet'
        properties: {
          priority: 200
          protocol: 'Tcp'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: 'VirtualNetwork'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '22'
        }
      }
      {
        name: 'deny-all-inbound'
        properties: {
          priority: 4096
          protocol: '*'
          access: 'Deny'
          direction: 'Inbound'
          sourceAddressPrefix: '*'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '*'
        }
      }
    ]
  }
}

resource trainingNsg 'Microsoft.Network/networkSecurityGroups@2023-04-01' = {
  name: '${prefix}-training-nsg'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'training' }
  properties: {
    securityRules: [
      {
        name: 'allow-ssh-vnet'
        properties: {
          priority: 100
          protocol: 'Tcp'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: 'VirtualNetwork'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '22'
        }
      }
      {
        name: 'deny-all-inbound'
        properties: {
          priority: 4096
          protocol: '*'
          access: 'Deny'
          direction: 'Inbound'
          sourceAddressPrefix: '*'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '*'
        }
      }
    ]
  }
}

// ── Outputs ───────────────────────────────────────────────────────────────────

output vnetId string = vnet.id
// Legacy subnets kept for backward compatibility (indices 0–3)
// subnet[0] is named 'gpu-subnet' in Azure but used as the CPU inference subnet (Layer 6)
output gpuSubnetId string = vnet.properties.subnets[0].id
output inferenceSubnetId string = vnet.properties.subnets[0].id   // Layer 6: CPU inference reuses the legacy gpu-subnet address space
output cpuSubnetId string = vnet.properties.subnets[1].id
output appSubnetId string = vnet.properties.subnets[2].id
output privateEndpointSubnetId string = vnet.properties.subnets[3].id
// New VM-layer subnets (indices 4–9)
output controlSubnetId string = vnet.properties.subnets[4].id
output apiSubnetId string = vnet.properties.subnets[5].id
output frontendSubnetId string = vnet.properties.subnets[6].id
output lifecycleSubnetId string = vnet.properties.subnets[7].id
output dataSubnetId string = vnet.properties.subnets[8].id
output trainingSubnetId string = vnet.properties.subnets[9].id
// NSG IDs
output gpuNsgId string = gpuNsg.id
output inferenceNsgId string = gpuNsg.id   // Layer 6: CPU inference reuses the legacy gpu-nsg (port 8000 is already open)
output cpuNsgId string = cpuNsg.id
output controlNsgId string = controlNsg.id
output apiNsgId string = apiNsg.id
output frontendNsgId string = frontendNsg.id
output lifecycleNsgId string = lifecycleNsg.id
output dataNsgId string = dataNsg.id
output trainingNsgId string = trainingNsg.id
// DNS zones
output blobPrivateDnsZoneId string = blobPrivateDnsZone.id
output kvPrivateDnsZoneId string = kvPrivateDnsZone.id
