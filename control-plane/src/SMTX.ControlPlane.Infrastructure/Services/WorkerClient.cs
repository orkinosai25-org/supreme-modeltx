using SMTX.ControlPlane.Core.Entities;

namespace SMTX.ControlPlane.Infrastructure.Services;

/// <summary>
/// Abstract intent-dispatch contract used by the control plane to coordinate external services.
/// </summary>
public abstract class WorkerClient
{
    public abstract Task DispatchTrainingJobAsync(TrainingJob job, CancellationToken cancellationToken);

    public abstract Task TriggerInferenceReloadAsync(Guid? modelVersionId, CancellationToken cancellationToken);
}
