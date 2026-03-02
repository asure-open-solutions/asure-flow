## Asure Flow -- One-click setup for Windows
## Right-click this file -> "Run with PowerShell", or run: powershell -File scripts\setup.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Root) { $Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }
# Handle running from scripts/ dir
if (Test-Path "$PSScriptRoot\..\server") { $Root = Resolve-Path "$PSScriptRoot\.." }

Write-Host ""
Write-Host "=== Asure Flow Setup ===" -ForegroundColor Cyan
Write-Host ""

# -- Check prerequisites --

function Test-Command($cmd, $installUrl) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Host "ERROR: '$cmd' is not installed." -ForegroundColor Red
        Write-Host "  Install it from: $installUrl" -ForegroundColor Yellow
        Write-Host ""
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Test-Command "python" "https://www.python.org/downloads/"
Test-Command "node"   "https://nodejs.org/"

# -- Python server --

Write-Host "-> Setting up Python server..." -ForegroundColor Green
Set-Location "$Root\server"

if (Get-Command "uv" -ErrorAction SilentlyContinue) {
    Write-Host "   Using uv for Python dependencies"
    if (-not (Test-Path ".venv")) {
        uv venv .venv
    }
    & ".venv\Scripts\activate.ps1"
    uv pip install -e ".[dev]"
} else {
    Write-Host "   Using pip (install 'uv' for faster installs: pip install uv)"
    if (-not (Test-Path ".venv")) {
        python -m venv .venv
    }
    & ".venv\Scripts\activate.ps1"
    pip install -e ".[dev]"
}

Write-Host "   Server dependencies installed." -ForegroundColor Green

# -- Client --

Write-Host "-> Setting up Electron client..." -ForegroundColor Green
Set-Location "$Root\client"
npm install
Write-Host "   Client dependencies installed." -ForegroundColor Green

# -- .env --

Set-Location $Root
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host ""
    Write-Host "-> Created .env from .env.example" -ForegroundColor Yellow
    Write-Host "   Edit .env and add at least one LLM API key to enable AI features." -ForegroundColor Yellow
} else {
    Write-Host "-> .env already exists, skipping."
}

Write-Host ""
Write-Host "=== Setup complete! ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start:"
Write-Host "  Double-click:  start.bat  (Windows)" -ForegroundColor White
Write-Host "  Double-click:  start.command  (macOS)" -ForegroundColor White
Write-Host "  Terminal:      ./start.sh  (Linux)" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to exit"
