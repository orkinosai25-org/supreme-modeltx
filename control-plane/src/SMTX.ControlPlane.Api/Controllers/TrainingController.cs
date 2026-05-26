using Microsoft.AspNetCore.Mvc;
using SMTX.ControlPlane.Api.DTOs;
using SMTX.ControlPlane.Infrastructure.Services;

namespace SMTX.ControlPlane.Api.Controllers;

[ApiController]
[Route("api/train")]
public class TrainingController(TrainingOrchestrator orchestrator) : ControllerBase
{
    [HttpPost("sharepoint")]
    public async Task<ActionResult<TrainingJobResponse>> CreateSharePointTrainingJob(
        [FromBody] CreateSharePointTrainingRequest request,
        CancellationToken cancellationToken)
    {
        var job = await orchestrator.RegisterSharePointTrainingJobAsync(
            request.SiteUrl,
            request.LibraryName,
            request.Scope,
            cancellationToken);

        return Accepted($"/api/train/{job.Id}", TrainingJobResponse.FromEntity(job));
    }

    [HttpGet("{jobId:guid}")]
    public async Task<ActionResult<TrainingJobResponse>> GetTrainingJob(Guid jobId, CancellationToken cancellationToken)
    {
        var job = await orchestrator.GetJobAsync(jobId, cancellationToken);
        if (job is null)
        {
            return NotFound();
        }

        return Ok(TrainingJobResponse.FromEntity(job));
    }

    [HttpGet]
    public async Task<ActionResult<IReadOnlyList<TrainingJobResponse>>> ListTrainingJobs(CancellationToken cancellationToken)
    {
        var jobs = await orchestrator.ListJobsAsync(cancellationToken);
        return Ok(jobs.Select(TrainingJobResponse.FromEntity).ToArray());
    }
}
