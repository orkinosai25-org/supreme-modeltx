// app-service.bicep — SMTX App Service for the T-X Orchestrator

@description('Azure region for all resources')
param location string

@description('Environment tag (dev, staging, prod)')
param environment string

@description('Name prefix for all resources')
param prefix string

@description('Subnet resource ID for App Service VNet integration')
param appSubnetId string

@description('User-assigned managed identity resource ID')
param identityId string

@description('Client ID of the user-assigned managed identity (for AZURE_CLIENT_ID app setting)')
param identityClientId string

@description('Private IP of the GPU VM (vLLM endpoint)')
param gpuVmPrivateIp string

@description('Private IP of the CPU VM (retrieval + verification endpoints)')
param cpuVmPrivateIp string

// ── App Service Plan (Linux, P2v3) ────────────────────────────────────────────

resource appServicePlan 'Microsoft.Web/serverfarms@2023-01-01' = {
  name: '${prefix}-asp'
  location: location
  tags: { environment: environment, project: 'smtx', component: 'app-service' }
  sku: {
    name: 'P2v3'
    tier: 'PremiumV3'
    size: 'P2v3'
    family: 'Pv3'
    capacity: 1
  }
  kind: 'linux'
  properties: {
    reserved: true
  }
}

// ── App Service (T-X Orchestrator) ────────────────────────────────────────────

resource txOrchestrator 'Microsoft.Web/sites@2023-01-01' = {
  name: '${prefix}-tx-orchestrator'
  location: location
  tags: { environment: environment, project: 'smtx', component: 'tx-orchestrator', role: 'governance-engine' }
  kind: 'app,linux'
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identityId}': {}
    }
  }
  properties: {
    serverFarmId: appServicePlan.id
    reserved: true
    httpsOnly: true
    virtualNetworkSubnetId: appSubnetId
    vnetRouteAllEnabled: true
    siteConfig: {
      linuxFxVersion: 'PYTHON|3.11'
      alwaysOn: true
      ftpsState: 'Disabled'
      minTlsVersion: '1.2'
      http20Enabled: true
      appSettings: [
        {
          name: 'VLLM_ENDPOINT'
          value: 'http://${gpuVmPrivateIp}:8000'
        }
        {
          name: 'RETRIEVAL_ENDPOINT'
          value: 'http://${cpuVmPrivateIp}:8001'
        }
        {
          name: 'VERIFICATION_ENDPOINT'
          value: 'http://${cpuVmPrivateIp}:8002'
        }
        {
          name: 'AZURE_CLIENT_ID'
          value: identityClientId
        }
        {
          name: 'SCM_DO_BUILD_DURING_DEPLOYMENT'
          value: '1'
        }
        {
          name: 'WEBSITES_PORT'
          value: '8080'
        }
      ]
      cors: {
        allowedOrigins: ['https://portal.azure.com']
        supportCredentials: false
      }
    }
  }
}

// ── Deployment slot: staging ──────────────────────────────────────────────────

resource stagingSlot 'Microsoft.Web/sites/slots@2023-01-01' = {
  parent: txOrchestrator
  name: 'staging'
  location: location
  tags: { environment: 'staging', project: 'smtx', component: 'tx-orchestrator' }
  kind: 'app,linux'
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identityId}': {}
    }
  }
  properties: {
    serverFarmId: appServicePlan.id
    reserved: true
    httpsOnly: true
    virtualNetworkSubnetId: appSubnetId
    vnetRouteAllEnabled: true
    siteConfig: {
      linuxFxVersion: 'PYTHON|3.11'
      alwaysOn: true
      ftpsState: 'Disabled'
      minTlsVersion: '1.2'
      appSettings: [
        {
          name: 'VLLM_ENDPOINT'
          value: 'http://${gpuVmPrivateIp}:8000'
        }
        {
          name: 'RETRIEVAL_ENDPOINT'
          value: 'http://${cpuVmPrivateIp}:8001'
        }
        {
          name: 'VERIFICATION_ENDPOINT'
          value: 'http://${cpuVmPrivateIp}:8002'
        }
      ]
    }
  }
}

// ── Diagnostic settings (Application Insights) ───────────────────────────────

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${prefix}-appinsights'
  location: location
  tags: { environment: environment, project: 'smtx' }
  kind: 'web'
  properties: {
    Application_Type: 'web'
    Request_Source: 'rest'
    RetentionInDays: 90
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

// ── Outputs ───────────────────────────────────────────────────────────────────

output appServiceId string = txOrchestrator.id
output appServiceName string = txOrchestrator.name
output appServiceDefaultHostname string = txOrchestrator.properties.defaultHostName
output appInsightsConnectionString string = appInsights.properties.ConnectionString
output appInsightsInstrumentationKey string = appInsights.properties.InstrumentationKey
