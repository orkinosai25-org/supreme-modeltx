using Microsoft.EntityFrameworkCore;
using SMTX.ControlPlane.Core.Entities;
using SMTX.ControlPlane.Infrastructure.Db;

namespace SMTX.ControlPlane.Infrastructure.Services;

public class TrainingOrchestrator(SumotxDbContext dbContext, WorkerClient workerClient)
{
    public async Task<TrainingJob> RegisterSharePointTrainingJobAsync(
        string siteUrl,
        string libraryName,
        string scope,
        CancellationToken cancellationToken)
    {
        var source = new TrainingSource
        {
            SiteUrl = siteUrl,
            LibraryName = libraryName,
            Scope = scope
        };

        var job = new TrainingJob
        {
            TrainingSource = source
        };

        dbContext.TrainingSources.Add(source);
        dbContext.TrainingJobs.Add(job);
        await dbContext.SaveChangesAsync(cancellationToken);
        await workerClient.DispatchTrainingJobAsync(job, cancellationToken);

        return job;
    }

    public Task<TrainingJob?> GetJobAsync(Guid jobId, CancellationToken cancellationToken)
    {
        return dbContext.TrainingJobs
            .Include(job => job.TrainingSource)
            .SingleOrDefaultAsync(job => job.Id == jobId, cancellationToken);
    }

    public Task<List<TrainingJob>> ListJobsAsync(CancellationToken cancellationToken)
    {
        return dbContext.TrainingJobs
            .Include(job => job.TrainingSource)
            .OrderByDescending(job => job.CreatedAtUtc)
            .ToListAsync(cancellationToken);
    }
}
