using '../main.bicep'

param env = 'dev'
param location = 'westeurope'
param resourceGroupName = 'sumotx-rg-dev'
param frontendVmSize = 'Standard_D2s_v5'
param backendVmSize = 'Standard_D2s_v5'
param backendCheckpointDiskSizeGB = 128
param frontendSshSourceCidr = readEnvironmentVariable('SUMOTX_FRONTEND_SSH_CIDR')
param sshPublicKey = readEnvironmentVariable('SUMOTX_SSH_PUBLIC_KEY')
param sqlAdminPassword = readEnvironmentVariable('SUMOTX_SQL_ADMIN_PASSWORD')
