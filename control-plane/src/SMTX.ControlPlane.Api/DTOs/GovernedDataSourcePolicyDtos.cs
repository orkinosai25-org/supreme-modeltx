using SMTX.ControlPlane.Core.Entities;
using SMTX.ControlPlane.Core.Enums;
using SMTX.ControlPlane.Infrastructure.Services;

namespace SMTX.ControlPlane.Api.DTOs;

public record UpsertGovernedDataSourcePolicyRequest(
    string DataSourceId,
    string Name,
    DataSensitivity MaxDataSensitivity,
    IReadOnlyList<ModelUseCase> SupportedUseCases,
    IReadOnlyList<IndexingProfile> IndexingProfiles,
    bool RagEnabled)
{
    public GovernedDataSourcePolicyUpsert ToCommand() => new(
        DataSourceId,
        Name,
        MaxDataSensitivity,
        SupportedUseCases,
        IndexingProfiles,
        RagEnabled);
}

public record GovernedDataSourcePolicyResponse(
    Guid Id,
    string DataSourceId,
    string Name,
    DataSensitivity MaxDataSensitivity,
    IReadOnlyList<ModelUseCase> SupportedUseCases,
    IReadOnlyList<IndexingProfile> IndexingProfiles,
    bool RagEnabled)
{
    public static GovernedDataSourcePolicyResponse FromEntity(GovernedDataSourcePolicy policy) =>
        new(
            policy.Id,
            policy.DataSourceId,
            policy.Name,
            policy.MaxDataSensitivity,
            policy.GetSupportedUseCasesSet().OrderBy(value => value).ToArray(),
            policy.GetIndexingProfilesSet().OrderBy(value => value).ToArray(),
            policy.RagEnabled);
}
