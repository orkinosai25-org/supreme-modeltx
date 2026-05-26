using Microsoft.AspNetCore.Mvc;
using SMTX.ControlPlane.Api.DTOs;
using SMTX.ControlPlane.Infrastructure.Services;

namespace SMTX.ControlPlane.Api.Controllers;

[ApiController]
[Route("api/governed-data-source-policies")]
public class GovernedDataSourcePoliciesController(GovernedDataSourcePolicyService policyService) : ControllerBase
{
    [HttpGet]
    public async Task<ActionResult<IReadOnlyList<GovernedDataSourcePolicyResponse>>> ListPolicies(
        CancellationToken cancellationToken)
    {
        var policies = await policyService.ListAsync(cancellationToken);
        return Ok(policies.Select(GovernedDataSourcePolicyResponse.FromEntity).ToArray());
    }

    [HttpPost]
    public async Task<ActionResult<GovernedDataSourcePolicyResponse>> CreatePolicy(
        [FromBody] UpsertGovernedDataSourcePolicyRequest request,
        CancellationToken cancellationToken)
    {
        try
        {
            var policy = await policyService.CreateAsync(request.ToCommand(), cancellationToken);
            return Ok(GovernedDataSourcePolicyResponse.FromEntity(policy));
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(ex.Message);
        }
    }

    [HttpPut("{id:guid}")]
    public async Task<ActionResult<GovernedDataSourcePolicyResponse>> UpdatePolicy(
        Guid id,
        [FromBody] UpsertGovernedDataSourcePolicyRequest request,
        CancellationToken cancellationToken)
    {
        try
        {
            var policy = await policyService.UpdateAsync(id, request.ToCommand(), cancellationToken);
            if (policy is null)
            {
                return NotFound();
            }

            return Ok(GovernedDataSourcePolicyResponse.FromEntity(policy));
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(ex.Message);
        }
    }

    [HttpDelete("{id:guid}")]
    public async Task<IActionResult> DeletePolicy(Guid id, CancellationToken cancellationToken)
    {
        try
        {
            var deleted = await policyService.DeleteAsync(id, cancellationToken);
            if (!deleted)
            {
                return NotFound();
            }

            return NoContent();
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(ex.Message);
        }
    }
}
