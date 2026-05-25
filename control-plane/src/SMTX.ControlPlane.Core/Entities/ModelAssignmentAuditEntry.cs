using SMTX.ControlPlane.Core.Enums;

namespace SMTX.ControlPlane.Core.Entities;

/// <summary>
/// Immutable audit record produced every time an admin assigns or changes a model.
/// Records who changed the model, from which version, to which version, and why.
/// </summary>
public class ModelAssignmentAuditEntry
{
    public Guid Id { get; set; } = Guid.NewGuid();

    public string TenantId { get; set; } = string.Empty;
    public string WorkspaceId { get; set; } = string.Empty;
    public ModelUseCase UseCase { get; set; }

    /// <summary>Model that was previously assigned (null on first assignment).</summary>
    public Guid? PreviousModelVersionId { get; set; }

    /// <summary>Model that is now assigned.</summary>
    public Guid NewModelVersionId { get; set; }

    /// <summary>Data source attached to this assignment revision.</summary>
    public string DataSourceId { get; set; } = string.Empty;

    /// <summary>Indexing profile selected for this assignment revision.</summary>
    public IndexingProfile IndexingProfile { get; set; }

    /// <summary>Whether RAG was enabled for this assignment revision.</summary>
    public bool RagEnabled { get; set; }

    /// <summary>Configured top-k retrieval depth for this assignment revision.</summary>
    public int RagTopK { get; set; } = 5;

    /// <summary>Identity of the admin who made the change.</summary>
    public string ChangedByUserId { get; set; } = string.Empty;

    public DateTimeOffset ChangedAtUtc { get; set; } = DateTimeOffset.UtcNow;

    /// <summary>Mandatory justification for the model change (governance requirement).</summary>
    public string Reason { get; set; } = string.Empty;
}
