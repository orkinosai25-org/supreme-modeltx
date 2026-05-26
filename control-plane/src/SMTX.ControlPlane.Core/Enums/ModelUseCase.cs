namespace SMTX.ControlPlane.Core.Enums;

/// <summary>
/// Describes the intended enterprise use case for a model assignment.
/// Governance policy is evaluated per use case to enforce appropriate model selection.
/// </summary>
public enum ModelUseCase
{
    GeneralPurpose,
    Retrieval,
    Verification,
    Orchestration
}
