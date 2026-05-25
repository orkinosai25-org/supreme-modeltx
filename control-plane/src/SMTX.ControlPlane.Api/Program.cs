using Microsoft.EntityFrameworkCore;
using SMTX.ControlPlane.Infrastructure.Db;
using SMTX.ControlPlane.Infrastructure.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddOpenApi();

var controlPlaneConnectionString = builder.Configuration.GetConnectionString("SumotxControlPlane")
    ?? throw new InvalidOperationException("Connection string 'SumotxControlPlane' is required.");

builder.Services.AddDbContext<SumotxDbContext>(options => options.UseSqlServer(controlPlaneConnectionString));
builder.Services.AddScoped<WorkerClient, NoOpWorkerClient>();
builder.Services.AddScoped<TrainingOrchestrator>();
builder.Services.AddScoped<ModelRegistryService>();
builder.Services.AddScoped<GovernedModelService>();
builder.Services.AddScoped<GovernedDataSourcePolicyService>();

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.MapControllers();

using (var scope = app.Services.CreateScope())
{
    var dbContext = scope.ServiceProvider.GetRequiredService<SumotxDbContext>();
    await SumotxDbContextInitialization.InitializeAsync(dbContext);
}

app.Run();
