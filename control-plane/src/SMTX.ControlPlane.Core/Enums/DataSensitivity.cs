namespace SMTX.ControlPlane.Core.Enums;

/// <summary>
/// Classifies the maximum data-sensitivity level a model may handle.
/// Higher values indicate more restricted data categories.
/// </summary>
public enum DataSensitivity
{
    Low = 1,
    Medium = 2,
    High = 3,
    Restricted = 4
}
