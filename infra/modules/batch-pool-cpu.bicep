// batch-pool-cpu.bicep — CPU-first Azure Batch pool for SMTX training
//
// Provisions a Batch pool backed by Standard_Dv5 / Dsv5 VMs (no GPU required).
// Uses dedicated nodes by default for predictable scheduling.
// Designed to work within Azure Startup Sponsorship quotas.

@description('Azure region for the Batch pool')
param location string

@description('Batch account name')
param batchAccountName string

@description('Storage account name (for file-share mounts)')
param storageAccountName string

@description('Storage account key (for file-share mounts)')
@secure()
param storageAccountKey string

@description('Pool ID — must be unique within the Batch account')
param poolId string = 'smtx-cpu-training-pool'

@description('VM size for CPU nodes. No GPU dependency.')
@allowed([
  'Standard_D8s_v5'
  'Standard_D16s_v5'
  'Standard_D32s_v5'
  'Standard_E8s_v5'
  'Standard_E16s_v5'
])
param vmSize string = 'Standard_D16s_v5'

@description('Number of low-priority (Spot) nodes. Default is 0 for dedicated-only runs.')
@minValue(0)
@maxValue(20)
param targetLowPriorityNodes int = 0

@description('Number of dedicated nodes. Default is 1 for reliable CPU validation runs.')
@minValue(0)
@maxValue(10)
param targetDedicatedNodes int = 1

// ── Reference existing Batch account ─────────────────────────────────────────

resource batchAccount 'Microsoft.Batch/batchAccounts@2023-05-01' existing = {
  name: batchAccountName
}

// ── CPU Batch pool ────────────────────────────────────────────────────────────

resource cpuPool 'Microsoft.Batch/batchAccounts/pools@2023-05-01' = {
  parent: batchAccount
  name: poolId
  properties: {
    vmSize: vmSize
    // Ubuntu 22.04 LTS — widely supported, no HPC image required for CPU runs
    deploymentConfiguration: {
      virtualMachineConfiguration: {
        imageReference: {
          publisher: 'canonical'
          offer: '0001-com-ubuntu-server-jammy'
          sku: '22_04-lts'
          version: 'latest'
        }
        nodeAgentSKUId: 'batch.node.ubuntu 22.04'
      }
    }
    scaleSettings: {
      fixedScale: {
        targetDedicatedNodes: targetDedicatedNodes
        targetLowPriorityNodes: targetLowPriorityNodes
        // Allow resizing without blocking deploys
        resizeTimeout: 'PT15M'
      }
    }
    taskSlotsPerNode: 1
    taskSchedulingPolicy: {
      nodeFillType: 'Pack'
    }
    // Mount data and checkpoint file shares on every node
    mountConfiguration: [
      {
        azureFileShareConfiguration: {
          accountName: storageAccountName
          azureFileUrl: 'https://${storageAccountName}.file.core.windows.net/smtxdata'
          relativeMountPath: 'smtxdata'
          accountKey: storageAccountKey
          mountOptions: '-o vers=3.0,dir_mode=0777,file_mode=0777,serverino'
        }
      }
      {
        azureFileShareConfiguration: {
          accountName: storageAccountName
          azureFileUrl: 'https://${storageAccountName}.file.core.windows.net/smtxcheckpoints'
          relativeMountPath: 'smtxcheckpoints'
          accountKey: storageAccountKey
          mountOptions: '-o vers=3.0,dir_mode=0777,file_mode=0777,serverino'
        }
      }
    ]
  }
}

// ── Outputs ───────────────────────────────────────────────────────────────────

output poolId string = cpuPool.name
output poolResourceId string = cpuPool.id
