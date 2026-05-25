using Microsoft.AspNetCore.Mvc;

namespace SMTX.ControlPlane.Api.Controllers;

[ApiController]
[Route("api/health")]
public class HealthController : ControllerBase
{
    [HttpGet]
    public IActionResult Get()
    {
        return Ok(new { status = "ok", service = "control-plane", timestampUtc = DateTimeOffset.UtcNow });
    }
}
