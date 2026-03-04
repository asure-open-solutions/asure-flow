@echo off
chcp 65001 >nul
setlocal
title AsuréFlow - Client
cd /d "%~dp0"

echo.
echo   AsuréFlow - Client
echo   ==================
echo.

:: ---- Check Node.js ----
where node >nul 2>&1
if errorlevel 1 (
    echo   ERROR: Node.js is not installed.
    echo   Download: https://nodejs.org/
    echo.
    pause
    exit /b 1
)

:: ---- Server URL ----
if not "%~1"=="" (
    set "ASUREFLOW_SERVER=%~1"
) else if not defined ASUREFLOW_SERVER (
    set "ASUREFLOW_SERVER=http://localhost:8000"
)

:: ---- Auto-setup ----
if not exist "client\node_modules" (
    echo   [setup] First run - installing client dependencies...
    cd /d "%~dp0client"
    call npm install
    cd /d "%~dp0"
    echo.
    echo   [setup] Client ready.
    echo.
)

echo   Server: %ASUREFLOW_SERVER%

:: ---- Health check ----
powershell -NoProfile -Command "try{$r=Invoke-WebRequest -Uri '%ASUREFLOW_SERVER%/api/health' -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop;Write-Host '  Status: online'}catch{Write-Host '  Status: server not reachable (start the server first)'}" 2>nul

echo.
echo   Tip: Change server URL in Settings inside the app.
echo   Press Ctrl+C to stop.
echo.

:: ---- Run ----
cd /d "%~dp0client"
call npm run dev
pause
