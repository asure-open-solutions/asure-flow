## Asuré Flow — Start server + client
## Right-click → "Run with PowerShell", or run: powershell -File scripts\start.ps1

$ErrorActionPreference = "Stop"
$Root = Resolve-Path "$PSScriptRoot\.."

Write-Host ""
Write-Host "=== Asure Flow ===" -ForegroundColor Cyan
Write-Host ""

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

# ── Start server ──

Write-Host "-> Starting server on ${Host_}:${Port}..." -ForegroundColor Green
$serverJob = Start-Job -ScriptBlock {
    param($root, $host_, $port)
    Set-Location "$root\server"
    & ".venv\Scripts\python.exe" -m uvicorn asure_flow.main:app `
        --host $host_ --port $port --reload --ws-max-size 1048576
} -ArgumentList $Root, $Host_, $Port

# ── Start client ──

Write-Host "-> Starting client..." -ForegroundColor Green
$clientJob = Start-Job -ScriptBlock {
    param($root)
    Set-Location "$root\client"
    npm run dev
} -ArgumentList $Root

Write-Host ""
Write-Host "Server:  http://${Host_}:${Port}" -ForegroundColor White
Write-Host "API docs: http://${Host_}:${Port}/docs" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""

# Stream output from both jobs
try {
    while ($true) {
        Receive-Job $serverJob 2>&1 | ForEach-Object { Write-Host "[server] $_" -ForegroundColor DarkCyan }
        Receive-Job $clientJob 2>&1 | ForEach-Object { Write-Host "[client] $_" -ForegroundColor DarkMagenta }
        Start-Sleep -Milliseconds 500
    }
} finally {
    Write-Host ""
    Write-Host "Shutting down..." -ForegroundColor Yellow
    Stop-Job $serverJob -ErrorAction SilentlyContinue
    Stop-Job $clientJob -ErrorAction SilentlyContinue
    Remove-Job $serverJob, $clientJob -Force -ErrorAction SilentlyContinue
}
