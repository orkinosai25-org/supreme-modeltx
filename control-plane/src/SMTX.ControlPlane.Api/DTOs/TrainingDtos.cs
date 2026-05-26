using SMTX.ControlPlane.Core.Entities;
using SMTX.ControlPlane.Core.Enums;

namespace SMTX.ControlPlane.Api.DTOs;

public record CreateSharePointTrainingRequest(string SiteUrl, string LibraryName, string Scope);

public record TrainingJobResponse(
    Guid Id,
    string Status,
    DateTimeOffset CreatedAtUtc,
    string SiteUrl,
    string LibraryName,
    string Scope)
{
    public static TrainingJobResponse FromEntity(TrainingJob job)
    {
        return new TrainingJobResponse(
            job.Id,
            job.Status.ToString(),
            job.CreatedAtUtc,
            job.TrainingSource?.SiteUrl ?? string.Empty,
            job.TrainingSource?.LibraryName ?? string.Empty,
            job.TrainingSource?.Scope ?? string.Empty);
    }
}
