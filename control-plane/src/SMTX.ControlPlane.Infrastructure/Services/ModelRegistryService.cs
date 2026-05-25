using Microsoft.EntityFrameworkCore;
using SMTX.ControlPlane.Core.Entities;
using SMTX.ControlPlane.Infrastructure.Db;

namespace SMTX.ControlPlane.Infrastructure.Services;

public class ModelRegistryService(SumotxDbContext dbContext, WorkerClient workerClient)
{
    public Task<List<ModelVersion>> ListAsync(CancellationToken cancellationToken)
    {
        return dbContext.ModelVersions
            .OrderByDescending(model => model.RegisteredAtUtc)
            .ToListAsync(cancellationToken);
    }

    /// <summary>
    /// Activates a model version for inference, enforcing that the model is
    /// admin-approved before activation is permitted.
    /// </summary>
    public async Task<ModelVersion?> ActivateAsync(Guid modelId, CancellationToken cancellationToken)
    {
        var model = await dbContext.ModelVersions.SingleOrDefaultAsync(x => x.Id == modelId, cancellationToken);
        if (model is null)
        {
            return null;
        }

        if (!model.IsApproved)
        {
            throw new InvalidOperationException(
                $"Model '{model.Name}' is not approved and cannot be activated. " +
                "An admin must approve the model before it can serve inference traffic.");
        }

        await dbContext.ModelVersions
            .Where(x => x.IsActive)
            .ExecuteUpdateAsync(setters => setters.SetProperty(x => x.IsActive, false), cancellationToken);

        model.IsActive = true;
        await dbContext.SaveChangesAsync(cancellationToken);
        await workerClient.TriggerInferenceReloadAsync(model.Id, cancellationToken);

        return model;
    }
}
