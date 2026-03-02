## Asuré Flow — Start server only (Windows)
## Usage: powershell -File scripts\start-server.ps1

$ErrorActionPreference = "Stop"
$Root = Resolve-Path "$PSScriptRoot\.."

Write-Host ""
Write-Host "=== Asure Flow — Server ===" -ForegroundColor Cyan
Write-Host ""

# Auto-setup on first run
if (-not (Test-Path "$Root\server\.venv")) {
    Write-Host "First run detected — running setup..." -ForegroundColor Yellow
    & powershell -ExecutionPolicy Bypass -File "$Root\scripts\setup.ps1"
}

# Load .env
$envFile = "$Root\.env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $key = $Matches[1].Trim()
            $val = $Matches[2].Trim()
            [Environment]::SetEnvironmentVariable($key, $val, "Process")
        }
    }
}

$Host_ = if ($env:HOST) { $env:HOST } else { "0.0.0.0" }
$Port  = if ($env:PORT) { $env:PORT } else { "8000" }

# Detect LAN IP
function Get-LanIP {
    try {
        $ip = (Get-NetIPAddress -AddressFamily IPv4 |
            Where-Object { $_.PrefixOrigin -eq "Dhcp" -or $_.PrefixOrigin -eq "Manual" } |
            Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' } |
            Select-Object -First 1).IPAddress
        if ($ip) { return $ip }
    } catch {}
    # Fallback
    try {
        $ip = (Test-Connection -ComputerName (hostname) -Count 1 -ErrorAction SilentlyContinue).IPV4Address.IPAddressToString
        if ($ip) { return $ip }
    } catch {}
    return "YOUR_WINDOWS_IP"
}

$LanIP = Get-LanIP

Write-Host "  Local:    http://localhost:$Port" -ForegroundColor White
Write-Host "  Network:  http://${LanIP}:$Port" -ForegroundColor Green
Write-Host "  API docs: http://localhost:$Port/docs" -ForegroundColor White
Write-Host ""
Write-Host "  To connect from another machine:" -ForegroundColor Yellow
Write-Host "    ASUREFLOW_SERVER=http://${LanIP}:$Port ./scripts/start-client.sh" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""

# Start server (foreground)
Set-Location "$Root\server"
$env:PYTHONPATH = "$Root\server\src"
& ".venv\Scripts\python.exe" -m uvicorn asure_flow.main:app `
    --host $Host_ --port $Port --reload --ws-max-size 1048576
