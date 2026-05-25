using SMTX.ControlPlane.Core.Entities;
using SMTX.ControlPlane.Core.Enums;

namespace SMTX.ControlPlane.Api.DTOs;

public record ModelVersionResponse(
    Guid Id,
    string Name,
    string ArtifactUri,
    bool IsActive,
    bool IsApproved,
    double? BenchmarkScore,
    DataSensitivity MaxDataSensitivity,
    string SupportedUseCases,
    DateTimeOffset RegisteredAtUtc)
{
    public static ModelVersionResponse FromEntity(ModelVersion model)
    {
        return new ModelVersionResponse(
            model.Id,
            model.Name,
            model.ArtifactUri,
            model.IsActive,
            model.IsApproved,
            model.BenchmarkScore,
            model.MaxDataSensitivity,
            model.SupportedUseCases,
            model.RegisteredAtUtc);
    }
}

public record InferenceReloadRequest(Guid? ModelVersionId);

