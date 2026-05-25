using SMTX.ControlPlane.Core.Entities;

namespace SMTX.ControlPlane.Infrastructure.Services;

/// <summary>
/// No-op adapter that preserves intent-only dispatch semantics for the control-plane skeleton.
/// </summary>
public sealed class NoOpWorkerClient : WorkerClient
{
    public override Task DispatchTrainingJobAsync(TrainingJob job, CancellationToken cancellationToken)
    {
        return Task.CompletedTask;
    }

    public override Task TriggerInferenceReloadAsync(Guid? modelVersionId, CancellationToken cancellationToken)
    {
        return Task.CompletedTask;
    }
}
