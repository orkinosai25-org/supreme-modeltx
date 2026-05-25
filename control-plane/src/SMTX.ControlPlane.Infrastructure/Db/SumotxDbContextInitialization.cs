using Microsoft.EntityFrameworkCore;
using SMTX.ControlPlane.Core.Entities;
using SMTX.ControlPlane.Core.Enums;

namespace SMTX.ControlPlane.Infrastructure.Db;

public static class SumotxDbContextInitialization
{
    public static async Task InitializeAsync(
        SumotxDbContext dbContext,
        CancellationToken cancellationToken = default)
    {
        if (dbContext.Database.IsRelational() && dbContext.Database.GetMigrations().Any())
        {
            await dbContext.Database.MigrateAsync(cancellationToken);
        }
        else
        {
            await dbContext.Database.EnsureCreatedAsync(cancellationToken);
        }

        var hasChanges = false;

        if (!await dbContext.ModelVersions.AnyAsync(cancellationToken))
        {
            await dbContext.ModelVersions.AddRangeAsync(
                new ModelVersion
                {
                    Name = "t101-baseline",
                    ArtifactUri = "blob://checkpoints/t101-baseline",
                    IsApproved = true,
                    BenchmarkScore = 72.5,
                    MaxDataSensitivity = DataSensitivity.High,
                    SupportedUseCases = $"{ModelUseCase.GeneralPurpose}"
                },
                new ModelVersion
                {
                    Name = "t301-retrieval",
                    ArtifactUri = "blob://checkpoints/t301-retrieval",
                    IsApproved = true,
                    BenchmarkScore = 81.0,
                    MaxDataSensitivity = DataSensitivity.Medium,
                    SupportedUseCases = $"{ModelUseCase.Retrieval}"
                },
                new ModelVersion
                {
                    Name = "t501-verification",
                    ArtifactUri = "blob://checkpoints/t501-verification",
                    IsApproved = true,
                    BenchmarkScore = 78.3,
                    MaxDataSensitivity = DataSensitivity.High,
                    SupportedUseCases = $"{ModelUseCase.Verification}"
                });
            hasChanges = true;
        }

        if (!await dbContext.GovernedDataSourcePolicies.AnyAsync(cancellationToken))
        {
            await dbContext.GovernedDataSourcePolicies.AddRangeAsync(
                new GovernedDataSourcePolicy
                {
                    DataSourceId = "ds_finance_docs",
                    Name = "Finance Documents",
                    MaxDataSensitivity = DataSensitivity.High,
                    SupportedUseCases = $"{ModelUseCase.GeneralPurpose},{ModelUseCase.Retrieval}",
                    IndexingProfiles = $"{IndexingProfile.KeywordBasic},{IndexingProfile.HybridEnterprise}",
                    RagEnabled = true
                },
                new GovernedDataSourcePolicy
                {
                    DataSourceId = "ds_hr_kb",
                    Name = "HR Knowledge Base",
                    MaxDataSensitivity = DataSensitivity.Restricted,
                    SupportedUseCases = $"{ModelUseCase.GeneralPurpose},{ModelUseCase.Retrieval}",
                    IndexingProfiles = $"{IndexingProfile.VectorBalanced},{IndexingProfile.HybridEnterprise}",
                    RagEnabled = true
                },
                new GovernedDataSourcePolicy
                {
                    DataSourceId = "ds_compliance_archive",
                    Name = "Compliance Archive",
                    MaxDataSensitivity = DataSensitivity.Restricted,
                    SupportedUseCases = $"{ModelUseCase.Verification},{ModelUseCase.Orchestration}",
                    IndexingProfiles = $"{IndexingProfile.KeywordBasic}",
                    RagEnabled = false
                });
            hasChanges = true;
        }

        if (hasChanges)
        {
            await dbContext.SaveChangesAsync(cancellationToken);
        }
    }
}
