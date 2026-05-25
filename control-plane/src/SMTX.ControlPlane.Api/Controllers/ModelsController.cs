using Microsoft.AspNetCore.Mvc;
using SMTX.ControlPlane.Api.DTOs;
using SMTX.ControlPlane.Infrastructure.Services;

namespace SMTX.ControlPlane.Api.Controllers;

[ApiController]
[Route("api/models")]
public class ModelsController(ModelRegistryService modelRegistryService) : ControllerBase
{
    [HttpGet]
    public async Task<ActionResult<IReadOnlyList<ModelVersionResponse>>> ListModels(CancellationToken cancellationToken)
    {
        var models = await modelRegistryService.ListAsync(cancellationToken);
        return Ok(models.Select(ModelVersionResponse.FromEntity).ToArray());
    }

    [HttpPost("{id:guid}/activate")]
    public async Task<ActionResult<ModelVersionResponse>> Activate(Guid id, CancellationToken cancellationToken)
    {
        try
        {
            var model = await modelRegistryService.ActivateAsync(id, cancellationToken);
            if (model is null)
            {
                return NotFound();
            }

            return Ok(ModelVersionResponse.FromEntity(model));
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(ex.Message);
        }
    }
}
