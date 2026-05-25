using Microsoft.AspNetCore.Mvc;
using SMTX.ControlPlane.Api.DTOs;
using SMTX.ControlPlane.Core.Enums;
using SMTX.ControlPlane.Infrastructure.Services;

namespace SMTX.ControlPlane.Api.Controllers;

/// <summary>
/// Governed model listing endpoint.
///
/// <para>
/// Returns only admin-approved models, optionally filtered by enterprise policy.
/// Benchmark scores surface as informational metadata only — they do not determine
/// approval status or override governance decisions.
/// </para>
/// </summary>
[ApiController]
[Route("api/governed-models")]
public class GovernedModelsController(GovernedModelService governedModelService) : ControllerBase
{
    /// <summary>
    /// Returns the list of admin-approved models, optionally filtered by use case and
    /// data-sensitivity ceiling.
    /// </summary>
    [HttpGet]
    public async Task<ActionResult<IReadOnlyList<ApprovedModelResponse>>> GetApprovedModels(
        [FromQuery] ModelUseCase? useCase,
        [FromQuery] DataSensitivity? maxDataSensitivity,
        CancellationToken cancellationToken)
    {
        var models = await governedModelService.GetApprovedModelsAsync(
            useCase, maxDataSensitivity, cancellationToken);

        return Ok(models.Select(ApprovedModelResponse.FromEntity).ToArray());
    }
}

/// <summary>
/// Model assignment and audit endpoints.
///
/// <para>
/// These endpoints implement the assignment half of the enterprise governance lifecycle:
/// <list type="bullet">
/// <item>Policy rules (approval status, data sensitivity, use case) are enforced on every assignment.</item>
/// <item>Every assignment change is recorded in an immutable audit trail.</item>
/// <item>Audit records can be filtered by use case to reduce noise in multi-use-case workspaces.</item>
/// </list>
/// </para>
/// </summary>
[ApiController]
[Route("api/model-assignments")]
public class ModelAssignmentsController(GovernedModelService governedModelService) : ControllerBase
{
    /// <summary>
    /// Returns the active model assignment for a given tenant / workspace / use-case scope.
    /// </summary>
    [HttpGet("{tenantId}/{workspaceId}")]
    public async Task<ActionResult<ModelAssignmentResponse>> GetAssignment(
        string tenantId,
        string workspaceId,
        [FromQuery] ModelUseCase useCase,
        CancellationToken cancellationToken)
    {
        var assignment = await governedModelService.GetAssignmentAsync(
            tenantId, workspaceId, useCase, cancellationToken);

        if (assignment is null)
        {
            return NotFound();
        }

        return Ok(ModelAssignmentResponse.FromEntity(assignment));
    }

    /// <summary>
    /// Assigns an approved model to a tenant/workspace/use-case scope.
    /// Policy is enforced (approval status, data-sensitivity ceiling, use-case support)
    /// and an immutable audit record is written.
    /// </summary>
    [HttpPost]
    public async Task<ActionResult<ModelAssignmentResponse>> AssignModel(
        [FromBody] AssignModelRequest request,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(request.Reason))
        {
            return BadRequest("A justification reason is required for model assignment.");
        }

        try
        {
            var assignment = await governedModelService.AssignModelAsync(
                request.TenantId,
                request.WorkspaceId,
                request.UseCase,
                request.DataSensitivity,
                request.DataSourceId,
                request.IndexingProfile,
                request.RagEnabled,
                request.RagTopK,
                request.ModelVersionId,
                request.AssignedByUserId,
                request.Reason,
                cancellationToken);

            return Ok(ModelAssignmentResponse.FromEntity(assignment));
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(ex.Message);
        }
    }

    /// <summary>
    /// Returns the audit trail for a tenant/workspace scope, most recent first.
    /// Optionally filter by <paramref name="useCase"/> to narrow results to a single
    /// use-case assignment history.
    /// </summary>
    [HttpGet("{tenantId}/{workspaceId}/audit")]
    public async Task<ActionResult<IReadOnlyList<ModelAssignmentAuditResponse>>> GetAuditTrail(
        string tenantId,
        string workspaceId,
        [FromQuery] ModelUseCase? useCase,
        CancellationToken cancellationToken)
    {
        var entries = await governedModelService.GetAuditTrailAsync(
            tenantId, workspaceId, useCase, cancellationToken);

        return Ok(entries.Select(ModelAssignmentAuditResponse.FromEntity).ToArray());
    }
}
