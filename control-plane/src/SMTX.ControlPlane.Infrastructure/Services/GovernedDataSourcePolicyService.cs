using Microsoft.EntityFrameworkCore;
using SMTX.ControlPlane.Core.Entities;
using SMTX.ControlPlane.Core.Enums;
using SMTX.ControlPlane.Infrastructure.Db;

namespace SMTX.ControlPlane.Infrastructure.Services;

public sealed record GovernedDataSourcePolicyUpsert(
    string DataSourceId,
    string Name,
    DataSensitivity MaxDataSensitivity,
    IReadOnlyCollection<ModelUseCase> SupportedUseCases,
    IReadOnlyCollection<IndexingProfile> IndexingProfiles,
    bool RagEnabled);

public class GovernedDataSourcePolicyService(SumotxDbContext dbContext)
{
    public Task<List<GovernedDataSourcePolicy>> ListAsync(CancellationToken cancellationToken)
    {
        return dbContext.GovernedDataSourcePolicies
            .AsNoTracking()
            .OrderBy(policy => policy.Name)
            .ThenBy(policy => policy.DataSourceId)
            .ToListAsync(cancellationToken);
    }

    public async Task<GovernedDataSourcePolicy> CreateAsync(
        GovernedDataSourcePolicyUpsert request,
        CancellationToken cancellationToken)
    {
        var policy = new GovernedDataSourcePolicy();
        await ApplyAsync(policy, request, isCreate: true, cancellationToken);

        dbContext.GovernedDataSourcePolicies.Add(policy);
        await dbContext.SaveChangesAsync(cancellationToken);
        return policy;
    }

    public async Task<GovernedDataSourcePolicy?> UpdateAsync(
        Guid id,
        GovernedDataSourcePolicyUpsert request,
        CancellationToken cancellationToken)
    {
        var policy = await dbContext.GovernedDataSourcePolicies
            .SingleOrDefaultAsync(existing => existing.Id == id, cancellationToken);

        if (policy is null)
        {
            return null;
        }

        await ApplyAsync(policy, request, isCreate: false, cancellationToken);
        await dbContext.SaveChangesAsync(cancellationToken);
        return policy;
    }

    public async Task<bool> DeleteAsync(Guid id, CancellationToken cancellationToken)
    {
        var policy = await dbContext.GovernedDataSourcePolicies
            .SingleOrDefaultAsync(existing => existing.Id == id, cancellationToken);

        if (policy is null)
        {
            return false;
        }

        if (await HasActiveAssignmentsAsync(policy.DataSourceId, cancellationToken))
        {
            throw new InvalidOperationException(
                $"Data source '{policy.Name}' is still referenced by active model assignments and cannot be deleted.");
        }

        dbContext.GovernedDataSourcePolicies.Remove(policy);
        await dbContext.SaveChangesAsync(cancellationToken);
        return true;
    }

    private async Task ApplyAsync(
        GovernedDataSourcePolicy policy,
        GovernedDataSourcePolicyUpsert request,
        bool isCreate,
        CancellationToken cancellationToken)
    {
        var dataSourceId = request.DataSourceId.Trim().ToLowerInvariant();
        var name = request.Name.Trim();

        if (string.IsNullOrWhiteSpace(dataSourceId))
        {
            throw new InvalidOperationException("A dataSourceId is required.");
        }

        if (string.IsNullOrWhiteSpace(name))
        {
            throw new InvalidOperationException("A policy name is required.");
        }

        if (!isCreate
            && !string.Equals(policy.DataSourceId, dataSourceId, StringComparison.OrdinalIgnoreCase)
            && await HasActiveAssignmentsAsync(policy.DataSourceId, cancellationToken))
        {
            throw new InvalidOperationException(
                $"Data source '{policy.Name}' is still referenced by active model assignments and its dataSourceId cannot be changed.");
        }

        await EnsureUniqueDataSourceIdAsync(policy.Id, dataSourceId, cancellationToken);

        policy.DataSourceId = dataSourceId;
        policy.Name = name;
        policy.MaxDataSensitivity = request.MaxDataSensitivity;
        policy.SupportedUseCases = SerializeEnumValues(
            request.SupportedUseCases,
            "At least one supported use case is required.");
        policy.IndexingProfiles = SerializeEnumValues(
            request.IndexingProfiles,
            "At least one indexing profile is required.");
        policy.RagEnabled = request.RagEnabled;
    }

    private async Task EnsureUniqueDataSourceIdAsync(
        Guid currentPolicyId,
        string dataSourceId,
        CancellationToken cancellationToken)
    {
        if (await dbContext.GovernedDataSourcePolicies
                .AsNoTracking()
                .AnyAsync(
                    policy => policy.Id != currentPolicyId
                              && policy.DataSourceId == dataSourceId,
                    cancellationToken))
        {
            throw new InvalidOperationException($"A governed policy already exists for data source '{dataSourceId}'.");
        }
    }

    private Task<bool> HasActiveAssignmentsAsync(string dataSourceId, CancellationToken cancellationToken)
    {
        return dbContext.ModelAssignments
            .AsNoTracking()
            .AnyAsync(assignment => assignment.DataSourceId == dataSourceId, cancellationToken);
    }

    private static string SerializeEnumValues<TEnum>(
        IEnumerable<TEnum> values,
        string emptyMessage)
        where TEnum : struct, Enum
    {
        var serialized = string.Join(',',
            values
                .Distinct()
                .Order()
                .Select(value => value.ToString()));

        if (string.IsNullOrWhiteSpace(serialized))
        {
            throw new InvalidOperationException(emptyMessage);
        }

        return serialized;
    }
}
