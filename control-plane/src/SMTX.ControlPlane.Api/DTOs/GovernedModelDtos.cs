using SMTX.ControlPlane.Core.Entities;
using SMTX.ControlPlane.Core.Enums;

namespace SMTX.ControlPlane.Api.DTOs;

// ── Approved model query ────────────────────────────────────────────────────

public record ApprovedModelResponse(
    Guid Id,
    string Name,
    string ArtifactUri,
    bool IsApproved,
    double? BenchmarkScore,
    DataSensitivity MaxDataSensitivity,
    string SupportedUseCases,
    DateTimeOffset RegisteredAtUtc)
{
    public static ApprovedModelResponse FromEntity(ModelVersion m) =>
        new(m.Id, m.Name, m.ArtifactUri, m.IsApproved, m.BenchmarkScore,
            m.MaxDataSensitivity, m.SupportedUseCases, m.RegisteredAtUtc);
}

// ── Model assignment ────────────────────────────────────────────────────────

public record AssignModelRequest(
    string TenantId,
    string WorkspaceId,
    ModelUseCase UseCase,
    DataSensitivity DataSensitivity,
    string DataSourceId,
    IndexingProfile IndexingProfile,
    bool RagEnabled,
    int RagTopK,
    Guid ModelVersionId,
    string AssignedByUserId,
    string Reason);

public record ModelAssignmentResponse(
    Guid Id,
    string TenantId,
    string WorkspaceId,
    ModelUseCase UseCase,
    DataSensitivity DataSensitivity,
    string DataSourceId,
    IndexingProfile IndexingProfile,
    bool RagEnabled,
    int RagTopK,
    Guid ModelVersionId,
    string ModelName,
    string AssignedByUserId,
    DateTimeOffset AssignedAtUtc,
    DateTimeOffset UpdatedAtUtc)
{
    public static ModelAssignmentResponse FromEntity(ModelAssignment a) =>
        new(a.Id, a.TenantId, a.WorkspaceId, a.UseCase, a.DataSensitivity,
            a.DataSourceId, a.IndexingProfile,
            a.RagEnabled, a.RagTopK,
            a.ModelVersionId, a.ModelVersion?.Name ?? string.Empty,
            a.AssignedByUserId, a.AssignedAtUtc, a.UpdatedAtUtc);
}

// ── Audit trail ─────────────────────────────────────────────────────────────

public record ModelAssignmentAuditResponse(
    Guid Id,
    string TenantId,
    string WorkspaceId,
    ModelUseCase UseCase,
    Guid? PreviousModelVersionId,
    Guid NewModelVersionId,
    string DataSourceId,
    IndexingProfile IndexingProfile,
    bool RagEnabled,
    int RagTopK,
    string ChangedByUserId,
    DateTimeOffset ChangedAtUtc,
    string Reason)
{
    public static ModelAssignmentAuditResponse FromEntity(ModelAssignmentAuditEntry e) =>
        new(e.Id, e.TenantId, e.WorkspaceId, e.UseCase,
            e.PreviousModelVersionId, e.NewModelVersionId,
            e.DataSourceId, e.IndexingProfile,
            e.RagEnabled, e.RagTopK,
            e.ChangedByUserId, e.ChangedAtUtc, e.Reason);
}
