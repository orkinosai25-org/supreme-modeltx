using Microsoft.EntityFrameworkCore;
using SMTX.ControlPlane.Core.Entities;
using SMTX.ControlPlane.Core.Enums;
using SMTX.ControlPlane.Infrastructure.Db;

namespace SMTX.ControlPlane.Infrastructure.Services;

/// <summary>
/// Implements enterprise-governed model assignment.
/// Ensures that:
/// <list type="bullet">
/// <item>Only admin-approved models are returned or assigned.</item>
/// <item>Policy rules (data sensitivity, use case) are enforced before assignment.</item>
/// <item>Every assignment change produces an immutable audit record.</item>
/// <item>Benchmark scores inform recommendations but do not override governance approval.</item>
/// </list>
/// </summary>
public class GovernedModelService(SumotxDbContext dbContext)
{
    // ── Approved model listing ──────────────────────────────────────────────

    /// <summary>
    /// Returns models that have been approved and satisfy the given policy filters.
    /// Benchmark scores are included as informational metadata only.
    /// </summary>
    public async Task<List<ModelVersion>> GetApprovedModelsAsync(
        ModelUseCase? useCase,
        DataSensitivity? maxDataSensitivity,
        CancellationToken cancellationToken)
    {
        // Materialise the approved set first, then apply use-case filtering in memory
        // using GetSupportedUseCasesSet() to avoid fragile comma-string Contains checks.
        var candidates = await dbContext.ModelVersions
            .Where(m => m.IsApproved)
            .ToListAsync(cancellationToken);

        return candidates
            .Where(m => !useCase.HasValue || m.GetSupportedUseCasesSet().Contains(useCase.Value))
            .Where(m => !maxDataSensitivity.HasValue || m.MaxDataSensitivity >= maxDataSensitivity.Value)
            // Surface highest-benchmarked models first (informational ordering only)
            .OrderByDescending(m => m.BenchmarkScore)
            .ThenBy(m => m.Name)
            .ToList();
    }

    // ── Model assignment ────────────────────────────────────────────────────

    /// <summary>
    /// Returns the active model assignment for a given tenant / workspace / use-case scope.
    /// </summary>
    public Task<ModelAssignment?> GetAssignmentAsync(
        string tenantId,
        string workspaceId,
        ModelUseCase useCase,
        CancellationToken cancellationToken)
    {
        return dbContext.ModelAssignments
            .Include(a => a.ModelVersion)
            .FirstOrDefaultAsync(
                a => a.TenantId == tenantId
                     && a.WorkspaceId == workspaceId
                     && a.UseCase == useCase,
                cancellationToken);
    }

    /// <summary>
    /// Assigns an approved model to a tenant/workspace/use-case scope.
    /// Enforces governance policy and writes an audit entry.
    /// Returns the updated assignment, or throws <see cref="InvalidOperationException"/>
    /// when policy is violated.
    /// </summary>
    public async Task<ModelAssignment> AssignModelAsync(
        string tenantId,
        string workspaceId,
        ModelUseCase useCase,
        DataSensitivity dataSensitivity,
        string dataSourceId,
        IndexingProfile indexingProfile,
        bool ragEnabled,
        int ragTopK,
        Guid modelVersionId,
        string assignedByUserId,
        string reason,
        CancellationToken cancellationToken)
    {
        var model = await dbContext.ModelVersions
            .SingleOrDefaultAsync(m => m.Id == modelVersionId, cancellationToken)
            ?? throw new InvalidOperationException($"Model {modelVersionId} not found.");

        // ── Governance gate 1: model must be admin-approved ─────────────────
        if (!model.IsApproved)
        {
            throw new InvalidOperationException(
                $"Model '{model.Name}' is not approved for enterprise use and cannot be assigned.");
        }

        // ── Governance gate 2: data-sensitivity ceiling ─────────────────────
        // Enum ordering: Low(1) < Medium(2) < High(3) < Restricted(4).
        // The requested dataSensitivity must not exceed the model's clearance ceiling.
        if (dataSensitivity > model.MaxDataSensitivity)
        {
            throw new InvalidOperationException(
                $"Model '{model.Name}' is cleared for {model.MaxDataSensitivity} data but the " +
                $"requested sensitivity is {dataSensitivity}. Select a model cleared for higher sensitivity.");
        }

        // ── Governance gate 3: use-case support ─────────────────────────────
        if (!model.GetSupportedUseCasesSet().Contains(useCase))
        {
            throw new InvalidOperationException(
                $"Model '{model.Name}' does not support the '{useCase}' use case.");
        }

        // ── Governance gate 4: data-source, indexing, and RAG linkage ───────
        var sourcePolicy = await dbContext.GovernedDataSourcePolicies
            .AsNoTracking()
            .SingleOrDefaultAsync(p => p.DataSourceId == dataSourceId, cancellationToken);

        if (sourcePolicy is null)
        {
            throw new InvalidOperationException($"Data source '{dataSourceId}' is not registered for enterprise assignment.");
        }

        var sourceUseCases = sourcePolicy.GetSupportedUseCasesSet();
        if (!sourceUseCases.Contains(useCase))
        {
            throw new InvalidOperationException(
                $"Data source '{sourcePolicy.Name}' is not approved for the '{useCase}' use case.");
        }

        if (dataSensitivity > sourcePolicy.MaxDataSensitivity)
        {
            throw new InvalidOperationException(
                $"Data source '{sourcePolicy.Name}' is cleared for {sourcePolicy.MaxDataSensitivity} data but the " +
                $"requested sensitivity is {dataSensitivity}.");
        }

        var sourceProfiles = sourcePolicy.GetIndexingProfilesSet();
        if (!sourceProfiles.Contains(indexingProfile))
        {
            throw new InvalidOperationException(
                $"Data source '{sourcePolicy.Name}' does not support indexing profile '{indexingProfile}'.");
        }

        if (ragEnabled && !sourcePolicy.RagEnabled)
        {
            throw new InvalidOperationException(
                $"Data source '{sourcePolicy.Name}' is not approved for RAG-enabled assignments.");
        }

        if (useCase == ModelUseCase.Retrieval && !ragEnabled)
        {
            throw new InvalidOperationException("Retrieval use-case assignments require ragEnabled=true.");
        }

        if (ragEnabled && !model.GetSupportedUseCasesSet().Contains(ModelUseCase.Retrieval))
        {
            throw new InvalidOperationException(
                $"Model '{model.Name}' does not support retrieval but ragEnabled=true requires retrieval capability.");
        }

        if (ragTopK is < 1 or > 50)
        {
            throw new InvalidOperationException("ragTopK must be between 1 and 50 (inclusive).");
        }

        // ── Apply assignment ────────────────────────────────────────────────
        var existing = await GetAssignmentAsync(tenantId, workspaceId, useCase, cancellationToken);

        var auditEntry = new ModelAssignmentAuditEntry
        {
            TenantId = tenantId,
            WorkspaceId = workspaceId,
            UseCase = useCase,
            PreviousModelVersionId = existing?.ModelVersionId,
            NewModelVersionId = modelVersionId,
            DataSourceId = dataSourceId,
            IndexingProfile = indexingProfile,
            RagEnabled = ragEnabled,
            RagTopK = ragTopK,
            ChangedByUserId = assignedByUserId,
            ChangedAtUtc = DateTimeOffset.UtcNow,
            Reason = reason
        };

        if (existing is null)
        {
            existing = new ModelAssignment
            {
                TenantId = tenantId,
                WorkspaceId = workspaceId,
                UseCase = useCase,
                DataSensitivity = dataSensitivity,
                DataSourceId = dataSourceId,
                IndexingProfile = indexingProfile,
                RagEnabled = ragEnabled,
                RagTopK = ragTopK,
                ModelVersionId = modelVersionId,
                AssignedByUserId = assignedByUserId
            };
            dbContext.ModelAssignments.Add(existing);
        }
        else
        {
            existing.ModelVersionId = modelVersionId;
            existing.DataSensitivity = dataSensitivity;
            existing.DataSourceId = dataSourceId;
            existing.IndexingProfile = indexingProfile;
            existing.RagEnabled = ragEnabled;
            existing.RagTopK = ragTopK;
            existing.AssignedByUserId = assignedByUserId;
            existing.UpdatedAtUtc = DateTimeOffset.UtcNow;
        }

        dbContext.ModelAssignmentAuditEntries.Add(auditEntry);
        await dbContext.SaveChangesAsync(cancellationToken);

        // Re-load with navigation property populated
        existing.ModelVersion = model;
        return existing;
    }

    // ── Audit trail ─────────────────────────────────────────────────────────

    /// <summary>
    /// Returns the audit trail for a tenant/workspace scope, most recent first.
    /// When <paramref name="useCase"/> is provided, only entries for that use case
    /// are returned; otherwise all use cases are included.
    /// </summary>
    public Task<List<ModelAssignmentAuditEntry>> GetAuditTrailAsync(
        string tenantId,
        string workspaceId,
        ModelUseCase? useCase,
        CancellationToken cancellationToken)
    {
        var query = dbContext.ModelAssignmentAuditEntries
            .Where(e => e.TenantId == tenantId && e.WorkspaceId == workspaceId);

        if (useCase.HasValue)
        {
            query = query.Where(e => e.UseCase == useCase.Value);
        }

        return query
            .OrderByDescending(e => e.ChangedAtUtc)
            .ToListAsync(cancellationToken);
    }
}
