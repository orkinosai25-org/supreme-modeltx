using Microsoft.EntityFrameworkCore;
using SMTX.ControlPlane.Core.Entities;

namespace SMTX.ControlPlane.Infrastructure.Db;

/// <summary>
/// Control-plane metadata store. Contains orchestration registry state only (jobs, sources, model metadata).
/// It must not store training payloads, inference execution data, or embeddings.
/// </summary>
public class SumotxDbContext(DbContextOptions<SumotxDbContext> options) : DbContext(options)
{
    public DbSet<TrainingJob> TrainingJobs => Set<TrainingJob>();
    public DbSet<ModelVersion> ModelVersions => Set<ModelVersion>();
    public DbSet<TrainingSource> TrainingSources => Set<TrainingSource>();
    public DbSet<GovernedDataSourcePolicy> GovernedDataSourcePolicies => Set<GovernedDataSourcePolicy>();
    public DbSet<ModelAssignment> ModelAssignments => Set<ModelAssignment>();
    public DbSet<ModelAssignmentAuditEntry> ModelAssignmentAuditEntries => Set<ModelAssignmentAuditEntry>();
}
