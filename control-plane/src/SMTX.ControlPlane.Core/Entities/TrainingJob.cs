using SMTX.ControlPlane.Core.Enums;

namespace SMTX.ControlPlane.Core.Entities;

public class TrainingJob
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public Guid TrainingSourceId { get; set; }
    public TrainingSource? TrainingSource { get; set; }
    public TrainingJobStatus Status { get; set; } = TrainingJobStatus.Queued;
    public DateTimeOffset CreatedAtUtc { get; set; } = DateTimeOffset.UtcNow;
    public DateTimeOffset? StartedAtUtc { get; set; }
    public DateTimeOffset? CompletedAtUtc { get; set; }
}
