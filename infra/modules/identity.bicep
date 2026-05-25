// identity.bicep — SMTX Managed Identity and RBAC Role Assignments

@description('Azure region for all resources')
param location string

@description('Environment tag (dev, staging, prod)')
param environment string

@description('Name prefix for all resources')
param prefix string

// ── User-Assigned Managed Identities ─────────────────────────────────────────
// Existing identities (inference layer)

resource gpuVmIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-gpu-vm-identity'
  location: location
  tags: { environment: environment, project: 'smtx', component: 'gpu-vm' }
}

resource cpuVmIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-cpu-vm-identity'
  location: location
  tags: { environment: environment, project: 'smtx', component: 'cpu-vm' }
}

resource appServiceIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-app-identity'
  location: location
  tags: { environment: environment, project: 'smtx', component: 'app-service' }
}

// ── New VM-layer managed identities ──────────────────────────────────────────

resource controlVmIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-control-identity'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'control' }
}

resource apiVmIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-api-identity'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'api' }
}

resource frontendVmIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-frontend-identity'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'frontend' }
}

resource lifecycleVmIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-lifecycle-identity'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'lifecycle' }
}

resource dataVmIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-data-identity'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'data' }
}

resource trainingVmIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-training-identity'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'training' }
}

resource inferenceVmIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-inference-identity'
  location: location
  tags: { environment: environment, project: 'smtx', layer: 'inference' }
}

// ── Outputs ───────────────────────────────────────────────────────────────────

output gpuVmIdentityId string = gpuVmIdentity.id
output gpuVmPrincipalId string = gpuVmIdentity.properties.principalId
output gpuVmClientId string = gpuVmIdentity.properties.clientId

output cpuVmIdentityId string = cpuVmIdentity.id
output cpuVmPrincipalId string = cpuVmIdentity.properties.principalId
output cpuVmClientId string = cpuVmIdentity.properties.clientId

output appServiceIdentityId string = appServiceIdentity.id
output appServicePrincipalId string = appServiceIdentity.properties.principalId
output appServiceClientId string = appServiceIdentity.properties.clientId

output controlVmIdentityId string = controlVmIdentity.id
output controlVmPrincipalId string = controlVmIdentity.properties.principalId

output apiVmIdentityId string = apiVmIdentity.id
output apiVmPrincipalId string = apiVmIdentity.properties.principalId

output frontendVmIdentityId string = frontendVmIdentity.id
output frontendVmPrincipalId string = frontendVmIdentity.properties.principalId

output lifecycleVmIdentityId string = lifecycleVmIdentity.id
output lifecycleVmPrincipalId string = lifecycleVmIdentity.properties.principalId

output dataVmIdentityId string = dataVmIdentity.id
output dataVmPrincipalId string = dataVmIdentity.properties.principalId

output trainingVmIdentityId string = trainingVmIdentity.id
output trainingVmPrincipalId string = trainingVmIdentity.properties.principalId

output inferenceVmIdentityId string = inferenceVmIdentity.id
output inferenceVmPrincipalId string = inferenceVmIdentity.properties.principalId
