using SMTX.ControlPlane.Core.Enums;

namespace SMTX.ControlPlane.Core.Entities;

public class ModelVersion
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public string Name { get; set; } = string.Empty;
    public string ArtifactUri { get; set; } = string.Empty;
    public bool IsActive { get; set; }
    public DateTimeOffset RegisteredAtUtc { get; set; } = DateTimeOffset.UtcNow;

    // ── Governance fields ──────────────────────────────────────────────────────

    /// <summary>
    /// Indicates that an admin has explicitly approved this model for enterprise use.
    /// Only approved models may be assigned or activated.
    /// </summary>
    public bool IsApproved { get; set; }

    /// <summary>
    /// Optional benchmark/capability score (0–100). Used as an informational
    /// recommendation to admins; it does not override governance approval.
    /// </summary>
    public double? BenchmarkScore { get; set; }

    /// <summary>
    /// The highest data-sensitivity category this model is cleared to handle.
    /// Assignments are rejected when the requested sensitivity exceeds this value.
    /// </summary>
    public DataSensitivity MaxDataSensitivity { get; set; } = DataSensitivity.Low;

    /// <summary>
    /// Comma-separated list of <see cref="ModelUseCase"/> values this model supports.
    /// Stored as a string for EF Core compatibility.
    /// </summary>
    public string SupportedUseCases { get; set; } = string.Empty;

    /// <summary>
    /// Returns the supported use-case values as a parsed set.
    /// </summary>
    public IReadOnlySet<ModelUseCase> GetSupportedUseCasesSet()
    {
        if (string.IsNullOrEmpty(SupportedUseCases))
        {
            return new HashSet<ModelUseCase>();
        }

        var result = new HashSet<ModelUseCase>();
        foreach (var part in SupportedUseCases.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            if (Enum.TryParse<ModelUseCase>(part, out var useCase))
            {
                result.Add(useCase);
            }
        }
        return result;
    }
}
