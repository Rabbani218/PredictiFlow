# Docker smoke test: build, run, and poll backend health
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $projectRoot

Write-Host "Building and starting services with docker-compose..."
docker-compose up --build -d

# wait for backend to be healthy
$maxRetries = 30
$delay = 5
$attempt = 0
$healthy = $false

while ($attempt -lt $maxRetries -and -not $healthy) {
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri http://localhost:8000/health -TimeoutSec 5 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            Write-Host "Backend healthy."; $healthy = $true; break
        }
    } catch {
        Write-Host "Waiting for backend... (attempt $($attempt+1))"
    }
    Start-Sleep -Seconds $delay
    $attempt += 1
}

if (-not $healthy) {
    Write-Host "Backend did not become healthy within timeout. Check docker-compose logs.";
    docker-compose logs --tail=200
    exit 1
}

Write-Host "Smoke test passed: backend is responsive."; exit 0
