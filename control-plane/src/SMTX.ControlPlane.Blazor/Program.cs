using Microsoft.EntityFrameworkCore;
using SMTX.ControlPlane.Blazor.Components;
using SMTX.ControlPlane.Infrastructure.Db;
using SMTX.ControlPlane.Infrastructure.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();
var controlPlaneConnectionString = builder.Configuration.GetConnectionString("SumotxControlPlane")
    ?? throw new InvalidOperationException("Connection string 'SumotxControlPlane' is required.");

builder.Services.AddDbContext<SumotxDbContext>(options => options.UseSqlServer(controlPlaneConnectionString));
builder.Services.AddScoped<GovernedDataSourcePolicyService>();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}
app.UseStatusCodePagesWithReExecute("/not-found", createScopeForStatusCodePages: true);
app.UseHttpsRedirection();

app.UseAntiforgery();

app.MapStaticAssets();
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

using (var scope = app.Services.CreateScope())
{
    var dbContext = scope.ServiceProvider.GetRequiredService<SumotxDbContext>();
    await SumotxDbContextInitialization.InitializeAsync(dbContext);
}

app.Run();
