using Microsoft.AspNetCore.Mvc;
using SMTX.ControlPlane.Api.DTOs;
using SMTX.ControlPlane.Infrastructure.Services;

namespace SMTX.ControlPlane.Api.Controllers;

[ApiController]
[Route("api/inference")]
public class InferenceController(WorkerClient workerClient) : ControllerBase
{
    [HttpPost("reload")]
    public async Task<IActionResult> Reload([FromBody] InferenceReloadRequest? request, CancellationToken cancellationToken)
    {
        await workerClient.TriggerInferenceReloadAsync(request?.ModelVersionId, cancellationToken);

        return Accepted(new { status = "reload_queued", modelVersionId = request?.ModelVersionId });
    }
}
