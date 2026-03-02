## Asuré Flow — Start client only (Windows)
## Usage: powershell -File scripts\start-client.ps1 -Server http://192.168.1.50:8000
##    or: $env:ASUREFLOW_SERVER="http://192.168.1.50:8000"; .\scripts\start-client.ps1

param(
    [string]$Server
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path "$PSScriptRoot\.."

Write-Host ""
Write-Host "=== Asuré Flow — Client ===" -ForegroundColor Cyan
Write-Host ""

# Accept server URL from: -Server param > env var > default
if ($Server) {
    $env:ASUREFLOW_SERVER = $Server
} elseif (-not $env:ASUREFLOW_SERVER) {
    $env:ASUREFLOW_SERVER = "http://localhost:8000"
}

$ServerUrl = $env:ASUREFLOW_SERVER

# Auto-install on first run
if (-not (Test-Path "$Root\client\node_modules")) {
    Write-Host "First run — installing client dependencies..." -ForegroundColor Yellow
    Set-Location "$Root\client"
    npm install
    Write-Host ""
}

Write-Host "  Connecting to server: $ServerUrl" -ForegroundColor White
Write-Host ""

# Verify server is reachable (non-blocking — just a warning)
try {
    $response = Invoke-WebRequest -Uri "$ServerUrl/api/health" -TimeoutSec 3 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "  Server status: online" -ForegroundColor Green
    }
} catch {
    Write-Host "  WARNING: Server at $ServerUrl is not reachable yet." -ForegroundColor Yellow
    Write-Host "  Make sure the server is running and the URL is correct." -ForegroundColor Yellow
}
Write-Host ""
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""

# Start Electron client
Set-Location "$Root\client"
npm run dev
