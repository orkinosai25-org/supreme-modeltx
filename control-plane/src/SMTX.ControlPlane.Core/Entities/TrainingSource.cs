namespace SMTX.ControlPlane.Core.Entities;

public class TrainingSource
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public string SiteUrl { get; set; } = string.Empty;
    public string LibraryName { get; set; } = string.Empty;
    public string Scope { get; set; } = string.Empty;
}
