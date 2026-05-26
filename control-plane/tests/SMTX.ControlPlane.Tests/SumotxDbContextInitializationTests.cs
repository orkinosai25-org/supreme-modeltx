using Microsoft.EntityFrameworkCore;
using SMTX.ControlPlane.Infrastructure.Db;

namespace SMTX.ControlPlane.Tests;

public class SumotxDbContextInitializationTests : IDisposable
{
    private readonly SumotxDbContext _dbContext;

    public SumotxDbContextInitializationTests()
    {
        var options = new DbContextOptionsBuilder<SumotxDbContext>()
            .UseInMemoryDatabase(Guid.NewGuid().ToString())
            .Options;
        _dbContext = new SumotxDbContext(options);
    }

    public void Dispose() => _dbContext.Dispose();

    [Fact]
    public async Task InitializeAsync_IsIdempotentAndSeedsSharedDefaults()
    {
        await SumotxDbContextInitialization.InitializeAsync(_dbContext);
        await SumotxDbContextInitialization.InitializeAsync(_dbContext);

        Assert.Equal(3, await _dbContext.ModelVersions.CountAsync());
        Assert.Equal(3, await _dbContext.GovernedDataSourcePolicies.CountAsync());
        Assert.Contains(await _dbContext.GovernedDataSourcePolicies.Select(policy => policy.DataSourceId).ToListAsync(),
            id => id == "ds_finance_docs");
    }
}
