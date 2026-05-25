using SMTX.ControlPlane.Core.Enums;

namespace SMTX.ControlPlane.Core.Entities;

/// <summary>
/// Records the active model assigned to a specific tenant / workspace / use-case combination.
/// All assignment changes are captured in <see cref="ModelAssignmentAuditEntry"/>.
/// </summary>
public class ModelAssignment
{
    public Guid Id { get; set; } = Guid.NewGuid();

    /// <summary>Tenant (organisation) scope for this assignment.</summary>
    public string TenantId { get; set; } = string.Empty;

    /// <summary>Workspace or project scope within the tenant.</summary>
    public string WorkspaceId { get; set; } = string.Empty;

    /// <summary>Enterprise use case this assignment covers.</summary>
    public ModelUseCase UseCase { get; set; }

    /// <summary>Maximum data-sensitivity level requested for this assignment.</summary>
    public DataSensitivity DataSensitivity { get; set; }

    /// <summary>Enterprise data source attached to this model assignment.</summary>
    public string DataSourceId { get; set; } = string.Empty;

    /// <summary>Indexing profile selected for data retrieval in this assignment.</summary>
    public IndexingProfile IndexingProfile { get; set; }

    /// <summary>Whether retrieval-augmented generation is enabled for this assignment.</summary>
    public bool RagEnabled { get; set; }

    /// <summary>Maximum number of chunks retrieved when RAG is enabled.</summary>
    public int RagTopK { get; set; } = 5;

    /// <summary>The approved model assigned to this scope.</summary>
    public Guid ModelVersionId { get; set; }
    public ModelVersion? ModelVersion { get; set; }

    /// <summary>Identity (user/service principal) that created or last updated this assignment.</summary>
    public string AssignedByUserId { get; set; } = string.Empty;

    public DateTimeOffset AssignedAtUtc { get; set; } = DateTimeOffset.UtcNow;
    public DateTimeOffset UpdatedAtUtc { get; set; } = DateTimeOffset.UtcNow;
}
