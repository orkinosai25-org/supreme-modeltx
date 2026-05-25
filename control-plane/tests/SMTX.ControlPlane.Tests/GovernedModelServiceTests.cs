using Microsoft.EntityFrameworkCore;
using SMTX.ControlPlane.Core.Entities;
using SMTX.ControlPlane.Core.Enums;
using SMTX.ControlPlane.Infrastructure.Db;
using SMTX.ControlPlane.Infrastructure.Services;

namespace SMTX.ControlPlane.Tests;

/// <summary>
/// Unit tests for <see cref="GovernedModelService"/> covering all three governance gates
/// and the audit-trail write behaviour.
/// </summary>
public class GovernedModelServiceTests : IDisposable
{
    private const string DefaultDataSourceId = "ds_finance_docs";
    private const IndexingProfile DefaultIndexingProfile = IndexingProfile.HybridEnterprise;
    private const int DefaultRagTopK = 5;

    private readonly SumotxDbContext _db;
    private readonly GovernedModelService _service;

    // Pre-registered models used across tests
    private readonly ModelVersion _approvedGeneralHighModel;
    private readonly ModelVersion _approvedRetrievalMediumModel;
    private readonly ModelVersion _approvedVerificationHighModel;
    private readonly ModelVersion _unapprovedModel;

    public GovernedModelServiceTests()
    {
        var options = new DbContextOptionsBuilder<SumotxDbContext>()
            .UseInMemoryDatabase(Guid.NewGuid().ToString()) // isolated per test class instance
            .Options;
        _db = new SumotxDbContext(options);
        _service = new GovernedModelService(_db);

        _approvedGeneralHighModel = new ModelVersion
        {
            Name = "t101-baseline",
            ArtifactUri = "blob://checkpoints/t101-baseline",
            IsApproved = true,
            BenchmarkScore = 72.5,
            MaxDataSensitivity = DataSensitivity.High,
            SupportedUseCases = $"{ModelUseCase.GeneralPurpose}"
        };
        _approvedRetrievalMediumModel = new ModelVersion
        {
            Name = "t301-retrieval",
            ArtifactUri = "blob://checkpoints/t301-retrieval",
            IsApproved = true,
            BenchmarkScore = 81.0,
            MaxDataSensitivity = DataSensitivity.Medium,
            SupportedUseCases = $"{ModelUseCase.Retrieval}"
        };
        _approvedVerificationHighModel = new ModelVersion
        {
            Name = "t501-verification",
            ArtifactUri = "blob://checkpoints/t501-verification",
            IsApproved = true,
            BenchmarkScore = 78.3,
            MaxDataSensitivity = DataSensitivity.High,
            SupportedUseCases = $"{ModelUseCase.Verification}"
        };
        _unapprovedModel = new ModelVersion
        {
            Name = "t201-draft",
            ArtifactUri = "blob://checkpoints/t201-draft",
            IsApproved = false,
            MaxDataSensitivity = DataSensitivity.High,
            SupportedUseCases = $"{ModelUseCase.GeneralPurpose}"
        };

        _db.ModelVersions.AddRange(
            _approvedGeneralHighModel,
            _approvedRetrievalMediumModel,
            _approvedVerificationHighModel,
            _unapprovedModel);
        _db.GovernedDataSourcePolicies.AddRange(
            new GovernedDataSourcePolicy
            {
                DataSourceId = "ds_finance_docs",
                Name = "Finance Documents",
                MaxDataSensitivity = DataSensitivity.High,
                SupportedUseCases = $"{ModelUseCase.GeneralPurpose},{ModelUseCase.Retrieval}",
                IndexingProfiles = $"{IndexingProfile.KeywordBasic},{IndexingProfile.HybridEnterprise}",
                RagEnabled = true
            },
            new GovernedDataSourcePolicy
            {
                DataSourceId = "ds_hr_kb",
                Name = "HR Knowledge Base",
                MaxDataSensitivity = DataSensitivity.Restricted,
                SupportedUseCases = $"{ModelUseCase.GeneralPurpose},{ModelUseCase.Retrieval}",
                IndexingProfiles = $"{IndexingProfile.VectorBalanced},{IndexingProfile.HybridEnterprise}",
                RagEnabled = true
            },
            new GovernedDataSourcePolicy
            {
                DataSourceId = "ds_compliance_archive",
                Name = "Compliance Archive",
                MaxDataSensitivity = DataSensitivity.Restricted,
                SupportedUseCases = $"{ModelUseCase.Verification},{ModelUseCase.Orchestration}",
                IndexingProfiles = $"{IndexingProfile.KeywordBasic}",
                RagEnabled = false
            });
        _db.SaveChanges();
    }

    public void Dispose() => _db.Dispose();

    // ── Approved model listing ──────────────────────────────────────────────

    [Fact]
    public async Task GetApprovedModels_ReturnsOnlyApprovedModels()
    {
        var result = await _service.GetApprovedModelsAsync(null, null, default);

        Assert.DoesNotContain(result, m => m.Name == _unapprovedModel.Name);
        Assert.Contains(result, m => m.Name == _approvedGeneralHighModel.Name);
        Assert.Contains(result, m => m.Name == _approvedRetrievalMediumModel.Name);
    }

    [Fact]
    public async Task GetApprovedModels_FilteredByUseCase_ReturnsMatchingOnly()
    {
        var result = await _service.GetApprovedModelsAsync(ModelUseCase.Retrieval, null, default);

        Assert.Single(result);
        Assert.Equal(_approvedRetrievalMediumModel.Name, result[0].Name);
    }

    [Fact]
    public async Task GetApprovedModels_FilteredBySensitivity_ExcludesModelsBelowCeiling()
    {
        // Requesting High → t301 (Medium ceiling) must be excluded; t101 (High ceiling) must be included
        var result = await _service.GetApprovedModelsAsync(null, DataSensitivity.High, default);

        Assert.DoesNotContain(result, m => m.Name == _approvedRetrievalMediumModel.Name);
        Assert.Contains(result, m => m.Name == _approvedGeneralHighModel.Name);
    }

    [Fact]
    public async Task GetApprovedModels_BenchmarkOrderingDoesNotBypassApproval()
    {
        // The unapproved model must never appear even if it had a higher benchmark score
        var result = await _service.GetApprovedModelsAsync(null, null, default);

        Assert.True(result.Count > 0);
        Assert.All(result, m => Assert.True(m.IsApproved));
    }

    // ── Gate 1: model must be admin-approved ───────────────────────────────

    [Fact]
    public async Task AssignModel_Gate1_UnapprovedModel_Throws()
    {
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            _service.AssignModelAsync(
                "tenant-a", "ws-1", ModelUseCase.GeneralPurpose,
                DataSensitivity.Low, DefaultDataSourceId, DefaultIndexingProfile, false, DefaultRagTopK, _unapprovedModel.Id,
                "admin@corp", "Governance test", default));

        Assert.Contains("not approved", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    // ── Gate 2: data-sensitivity ceiling ──────────────────────────────────

    [Fact]
    public async Task AssignModel_Gate2_RequestedSensitivityExceedsCeiling_Throws()
    {
        // t301 is cleared for Medium; requesting High must be rejected
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            _service.AssignModelAsync(
                "tenant-a", "ws-1", ModelUseCase.Retrieval,
                DataSensitivity.High, DefaultDataSourceId, DefaultIndexingProfile, true, DefaultRagTopK, _approvedRetrievalMediumModel.Id,
                "admin@corp", "Governance test", default));

        Assert.Contains("Medium", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("High", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task AssignModel_Gate2_EqualSensitivity_Succeeds()
    {
        // t301 is cleared for Medium; requesting Medium must succeed
        var result = await _service.AssignModelAsync(
            "tenant-a", "ws-1", ModelUseCase.Retrieval,
            DataSensitivity.Medium, DefaultDataSourceId, DefaultIndexingProfile, true, DefaultRagTopK, _approvedRetrievalMediumModel.Id,
            "admin@corp", "Governance test", default);

        Assert.Equal(_approvedRetrievalMediumModel.Id, result.ModelVersionId);
    }

    // ── Gate 3: use-case support ────────────────────────────────────────────

    [Fact]
    public async Task AssignModel_Gate3_UnsupportedUseCase_Throws()
    {
        // t101 supports GeneralPurpose only; assigning it as Retrieval must be rejected
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            _service.AssignModelAsync(
                "tenant-a", "ws-1", ModelUseCase.Retrieval,
                DataSensitivity.Low, DefaultDataSourceId, DefaultIndexingProfile, true, DefaultRagTopK, _approvedGeneralHighModel.Id,
                "admin@corp", "Governance test", default));

        Assert.Contains("Retrieval", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task AssignModel_Gate3_SupportedUseCase_Succeeds()
    {
        var result = await _service.AssignModelAsync(
            "tenant-a", "ws-1", ModelUseCase.GeneralPurpose,
            DataSensitivity.Low, DefaultDataSourceId, DefaultIndexingProfile, false, DefaultRagTopK, _approvedGeneralHighModel.Id,
            "admin@corp", "Governance test", default);

        Assert.Equal(_approvedGeneralHighModel.Id, result.ModelVersionId);
    }

    // ── Gate 4: data-source, indexing, and RAG linkage ─────────────────────

    [Fact]
    public async Task AssignModel_Gate4_UnknownDataSource_Throws()
    {
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            _service.AssignModelAsync(
                "tenant-a", "ws-1", ModelUseCase.GeneralPurpose,
                DataSensitivity.Low, "ds-unknown", DefaultIndexingProfile, false, DefaultRagTopK, _approvedGeneralHighModel.Id,
                "admin@corp", "Unknown data source check", default));

        Assert.Contains("not registered", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task AssignModel_Gate4_RetrievalRequiresRagEnabled_Throws()
    {
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            _service.AssignModelAsync(
                "tenant-a", "ws-1", ModelUseCase.Retrieval,
                DataSensitivity.Low, DefaultDataSourceId, DefaultIndexingProfile, false, DefaultRagTopK, _approvedRetrievalMediumModel.Id,
                "admin@corp", "RAG requirement check", default));

        Assert.Contains("ragEnabled=true", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task AssignModel_Gate4_UnsupportedIndexingProfile_Throws()
    {
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            _service.AssignModelAsync(
                "tenant-a", "ws-verify", ModelUseCase.Verification,
                DataSensitivity.High, "ds_compliance_archive", IndexingProfile.HybridEnterprise, false, DefaultRagTopK, _approvedVerificationHighModel.Id,
                "admin@corp", "Indexing profile check", default));

        Assert.Contains("indexing profile", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task AssignModel_Gate4_ComplianceArchiveKeywordBasic_Succeeds()
    {
        var result = await _service.AssignModelAsync(
            "tenant-a", "ws-verify", ModelUseCase.Verification,
            DataSensitivity.High, "ds_compliance_archive", IndexingProfile.KeywordBasic, false, DefaultRagTopK, _approvedVerificationHighModel.Id,
            "admin@corp", "Compliance archive policy check", default);

        Assert.Equal(_approvedVerificationHighModel.Id, result.ModelVersionId);
        Assert.Equal("ds_compliance_archive", result.DataSourceId);
        Assert.Equal(IndexingProfile.KeywordBasic, result.IndexingProfile);
    }

    [Fact]
    public async Task AssignModel_RejectsInvalidIndexingProfile_WhenPolicyUpdatedAtRuntime()
    {
        var financePolicy = await _db.GovernedDataSourcePolicies
            .SingleAsync(p => p.DataSourceId == "ds_finance_docs");
        financePolicy.IndexingProfiles = $"{IndexingProfile.KeywordBasic}";
        await _db.SaveChangesAsync();

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            _service.AssignModelAsync(
                "tenant-a", "ws-persisted", ModelUseCase.GeneralPurpose,
                DataSensitivity.Low, "ds_finance_docs", IndexingProfile.HybridEnterprise, false, DefaultRagTopK, _approvedGeneralHighModel.Id,
                "admin@corp", "Persisted policy enforcement check", default));

        Assert.Contains("indexing profile", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    // ── Audit trail ─────────────────────────────────────────────────────────

    [Fact]
    public async Task AssignModel_WritesAuditEntryWithCorrectFields()
    {
        await _service.AssignModelAsync(
            "tenant-a", "ws-audit", ModelUseCase.GeneralPurpose,
            DataSensitivity.Low, DefaultDataSourceId, DefaultIndexingProfile, false, DefaultRagTopK, _approvedGeneralHighModel.Id,
            "admin@corp", "Initial assignment for audit test", default);

        var audit = await _service.GetAuditTrailAsync("tenant-a", "ws-audit", null, default);

        Assert.Single(audit);
        var entry = audit[0];
        Assert.Equal(_approvedGeneralHighModel.Id, entry.NewModelVersionId);
        Assert.Null(entry.PreviousModelVersionId); // first assignment has no previous
        Assert.Equal(DefaultDataSourceId, entry.DataSourceId);
        Assert.Equal(DefaultIndexingProfile, entry.IndexingProfile);
        Assert.False(entry.RagEnabled);
        Assert.Equal(DefaultRagTopK, entry.RagTopK);
        Assert.Equal("admin@corp", entry.ChangedByUserId);
        Assert.Equal("Initial assignment for audit test", entry.Reason);
    }

    [Fact]
    public async Task AssignModel_SecondAssignment_AuditEntryRecordsPreviousModel()
    {
        await _service.AssignModelAsync(
            "tenant-a", "ws-chain", ModelUseCase.Retrieval,
            DataSensitivity.Low, DefaultDataSourceId, DefaultIndexingProfile, true, DefaultRagTopK, _approvedRetrievalMediumModel.Id,
            "admin@corp", "First assignment", default);

        // Re-assign the same scope to a different model (add a second approved retrieval model)
        var secondModel = new ModelVersion
        {
            Name = "t301-v2",
            IsApproved = true,
            MaxDataSensitivity = DataSensitivity.Medium,
            SupportedUseCases = $"{ModelUseCase.Retrieval}"
        };
        _db.ModelVersions.Add(secondModel);
        await _db.SaveChangesAsync();

        await _service.AssignModelAsync(
            "tenant-a", "ws-chain", ModelUseCase.Retrieval,
            DataSensitivity.Low, DefaultDataSourceId, DefaultIndexingProfile, true, DefaultRagTopK, secondModel.Id,
            "admin2@corp", "Upgrade to v2", default);

        var audit = await _service.GetAuditTrailAsync("tenant-a", "ws-chain", null, default);

        Assert.Equal(2, audit.Count);
        // Most-recent first: second assignment is at index 0
        Assert.Equal(secondModel.Id, audit[0].NewModelVersionId);
        Assert.Equal(_approvedRetrievalMediumModel.Id, audit[0].PreviousModelVersionId);
    }

    [Fact]
    public async Task GetAuditTrail_FilteredByUseCase_ReturnsOnlyMatchingEntries()
    {
        // Two assignments in the same workspace but different use cases
        await _service.AssignModelAsync(
            "tenant-b", "ws-filter", ModelUseCase.GeneralPurpose,
            DataSensitivity.Low, DefaultDataSourceId, DefaultIndexingProfile, false, DefaultRagTopK, _approvedGeneralHighModel.Id,
            "admin@corp", "General purpose assignment", default);

        await _service.AssignModelAsync(
            "tenant-b", "ws-filter", ModelUseCase.Retrieval,
            DataSensitivity.Low, DefaultDataSourceId, DefaultIndexingProfile, true, DefaultRagTopK, _approvedRetrievalMediumModel.Id,
            "admin@corp", "Retrieval assignment", default);

        var retrievalOnly = await _service.GetAuditTrailAsync(
            "tenant-b", "ws-filter", ModelUseCase.Retrieval, default);

        Assert.Single(retrievalOnly);
        Assert.Equal(ModelUseCase.Retrieval, retrievalOnly[0].UseCase);
    }

    [Fact]
    public async Task GetAuditTrail_NoUseCaseFilter_ReturnsAllUseCases()
    {
        await _service.AssignModelAsync(
            "tenant-c", "ws-all", ModelUseCase.GeneralPurpose,
            DataSensitivity.Low, DefaultDataSourceId, DefaultIndexingProfile, false, DefaultRagTopK, _approvedGeneralHighModel.Id,
            "admin@corp", "GP assignment", default);

        await _service.AssignModelAsync(
            "tenant-c", "ws-all", ModelUseCase.Retrieval,
            DataSensitivity.Low, DefaultDataSourceId, DefaultIndexingProfile, true, DefaultRagTopK, _approvedRetrievalMediumModel.Id,
            "admin@corp", "Retrieval assignment", default);

        var all = await _service.GetAuditTrailAsync("tenant-c", "ws-all", null, default);

        Assert.Equal(2, all.Count);
    }
}
