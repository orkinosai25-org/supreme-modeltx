using '../main.bicep'

param env = 'prod'
param location = 'westeurope'
param resourceGroupName = 'sumotx-rg'
param frontendVmSize = 'Standard_D4s_v5'
param backendVmSize = 'Standard_D8s_v5'
param backendCheckpointDiskSizeGB = 512
param frontendSshSourceCidr = readEnvironmentVariable('SUMOTX_FRONTEND_SSH_CIDR')
param sshPublicKey = readEnvironmentVariable('SUMOTX_SSH_PUBLIC_KEY')
param sqlAdminPassword = readEnvironmentVariable('SUMOTX_SQL_ADMIN_PASSWORD')
