using SMTX.ControlPlane.Core.Enums;

namespace SMTX.ControlPlane.Core.Entities;

/// <summary>
/// Persisted policy record for a governed enterprise data source.
/// This is the source of truth for assignment linkage controls (use case, indexing, RAG).
/// </summary>
public class GovernedDataSourcePolicy
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public string DataSourceId { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public DataSensitivity MaxDataSensitivity { get; set; } = DataSensitivity.Low;
    public string SupportedUseCases { get; set; } = string.Empty;
    public string IndexingProfiles { get; set; } = string.Empty;
    public bool RagEnabled { get; set; }

    public IReadOnlySet<ModelUseCase> GetSupportedUseCasesSet()
    {
        if (string.IsNullOrEmpty(SupportedUseCases))
        {
            return new HashSet<ModelUseCase>();
        }

        var result = new HashSet<ModelUseCase>();
        foreach (var part in SupportedUseCases.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            if (Enum.TryParse<ModelUseCase>(part, ignoreCase: true, out var useCase))
            {
                result.Add(useCase);
            }
        }

        return result;
    }

    public IReadOnlySet<IndexingProfile> GetIndexingProfilesSet()
    {
        if (string.IsNullOrEmpty(IndexingProfiles))
        {
            return new HashSet<IndexingProfile>();
        }

        var result = new HashSet<IndexingProfile>();
        foreach (var part in IndexingProfiles.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            if (Enum.TryParse<IndexingProfile>(part, ignoreCase: true, out var profile))
            {
                result.Add(profile);
            }
        }

        return result;
    }
}
