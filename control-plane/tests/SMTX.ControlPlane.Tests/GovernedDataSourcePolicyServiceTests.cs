using Microsoft.EntityFrameworkCore;
using SMTX.ControlPlane.Core.Entities;
using SMTX.ControlPlane.Core.Enums;
using SMTX.ControlPlane.Infrastructure.Db;
using SMTX.ControlPlane.Infrastructure.Services;

namespace SMTX.ControlPlane.Tests;

public class GovernedDataSourcePolicyServiceTests : IDisposable
{
    private readonly SumotxDbContext _db;
    private readonly GovernedDataSourcePolicyService _service;
    private readonly GovernedDataSourcePolicy _existingPolicy;

    public GovernedDataSourcePolicyServiceTests()
    {
        var options = new DbContextOptionsBuilder<SumotxDbContext>()
            .UseInMemoryDatabase(Guid.NewGuid().ToString())
            .Options;

        _db = new SumotxDbContext(options);
        _service = new GovernedDataSourcePolicyService(_db);
        _existingPolicy = new GovernedDataSourcePolicy
        {
            DataSourceId = "ds_finance_docs",
            Name = "Finance Documents",
            MaxDataSensitivity = DataSensitivity.High,
            SupportedUseCases = $"{ModelUseCase.GeneralPurpose},{ModelUseCase.Retrieval}",
            IndexingProfiles = $"{IndexingProfile.KeywordBasic},{IndexingProfile.HybridEnterprise}",
            RagEnabled = true
        };

        _db.GovernedDataSourcePolicies.Add(_existingPolicy);
        _db.SaveChanges();
    }

    public void Dispose() => _db.Dispose();

    [Fact]
    public async Task CreateAsync_PersistsNormalizedPolicyRecord()
    {
        var created = await _service.CreateAsync(
            new GovernedDataSourcePolicyUpsert(
                "  ds_new_policy  ",
                "  New Policy  ",
                DataSensitivity.Restricted,
                [ModelUseCase.Orchestration, ModelUseCase.GeneralPurpose, ModelUseCase.Orchestration],
                [IndexingProfile.VectorBalanced, IndexingProfile.KeywordBasic],
                true),
            default);

        Assert.Equal("ds_new_policy", created.DataSourceId);
        Assert.Equal("New Policy", created.Name);
        Assert.Equal(
            $"{ModelUseCase.GeneralPurpose},{ModelUseCase.Orchestration}",
            created.SupportedUseCases);
        Assert.Equal(
            $"{IndexingProfile.KeywordBasic},{IndexingProfile.VectorBalanced}",
            created.IndexingProfiles);
        Assert.True(created.RagEnabled);
    }

    [Fact]
    public async Task UpdateAsync_RejectsDataSourceIdChangesWhileAssignmentsExist()
    {
        _db.ModelAssignments.Add(new ModelAssignment
        {
            TenantId = "tenant-a",
            WorkspaceId = "workspace-1",
            UseCase = ModelUseCase.GeneralPurpose,
            DataSensitivity = DataSensitivity.Low,
            DataSourceId = _existingPolicy.DataSourceId,
            IndexingProfile = IndexingProfile.KeywordBasic,
            ModelVersionId = Guid.NewGuid(),
            AssignedByUserId = "admin@corp"
        });
        await _db.SaveChangesAsync();

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            _service.UpdateAsync(
                _existingPolicy.Id,
                new GovernedDataSourcePolicyUpsert(
                    "ds_finance_docs_v2",
                    _existingPolicy.Name,
                    _existingPolicy.MaxDataSensitivity,
                    [ModelUseCase.GeneralPurpose],
                    [IndexingProfile.KeywordBasic],
                    false),
                default));

        Assert.Contains("cannot be changed", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task DeleteAsync_RejectsPoliciesReferencedByActiveAssignments()
    {
        _db.ModelAssignments.Add(new ModelAssignment
        {
            TenantId = "tenant-a",
            WorkspaceId = "workspace-2",
            UseCase = ModelUseCase.Retrieval,
            DataSensitivity = DataSensitivity.Low,
            DataSourceId = _existingPolicy.DataSourceId,
            IndexingProfile = IndexingProfile.HybridEnterprise,
            RagEnabled = true,
            ModelVersionId = Guid.NewGuid(),
            AssignedByUserId = "admin@corp"
        });
        await _db.SaveChangesAsync();

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            _service.DeleteAsync(_existingPolicy.Id, default));

        Assert.Contains("cannot be deleted", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task DeleteAsync_RemovesUnassignedPolicies()
    {
        var deleted = await _service.DeleteAsync(_existingPolicy.Id, default);

        Assert.True(deleted);
        Assert.Empty(await _db.GovernedDataSourcePolicies.ToListAsync());
    }
}
