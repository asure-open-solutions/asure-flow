@echo off
chcp 65001 >nul
setlocal
title AsuréFlow - Server
cd /d "%~dp0"

echo.
echo   AsuréFlow - Server
echo   ==================
echo.

:: ---- Check Python ----
where python >nul 2>&1
if errorlevel 1 (
    echo   ERROR: Python is not installed.
    echo   Download: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

:: ---- Auto-setup ----
if not exist "server\.venv" call :setup_server

:: ---- Load .env ----
if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do (
        if not "%%a"=="" set "%%a=%%b"
    )
)

if not defined PORT set "PORT=8000"

:: ---- Detect LAN IP ----
set "LAN_IP=your-ip"
for /f "tokens=*" %%i in ('powershell -NoProfile -Command "try{(Get-NetIPAddress -AddressFamily IPv4|Where-Object{$_.PrefixOrigin -eq 'Dhcp' -or $_.PrefixOrigin -eq 'Manual'}|Where-Object{$_.IPAddress -notmatch '^(127\.|169\.254\.)'}|Select-Object -First 1).IPAddress}catch{'your-ip'}" 2^>nul') do set "LAN_IP=%%i"

echo   Local:   http://localhost:%PORT%
echo   Network: http://%LAN_IP%:%PORT%
echo   Docs:    http://localhost:%PORT%/docs
echo.
echo   Remote client:
echo     start-client.bat http://%LAN_IP%:%PORT%
echo.
echo   Press Ctrl+C to stop.
echo.

:: ---- Run ----
set "PYTHONPATH=%~dp0server\src"
cd /d "%~dp0server"
.venv\Scripts\python.exe -m uvicorn asure_flow.main:app --host 0.0.0.0 --port %PORT% --reload --reload-exclude .venv --ws-max-size 1048576
pause
exit /b 0

:: ---- Setup subroutine ----
:setup_server
echo   [setup] First run - installing server dependencies...
echo.
cd /d "%~dp0server"
where uv >nul 2>&1
if not errorlevel 1 (
    echo   Using uv for fast install...
    uv venv .venv
    call .venv\Scripts\activate.bat
    uv pip install -e ".[dev]"
) else (
    echo   Creating virtual environment...
    python -m venv .venv
    echo   Installing dependencies (this may take a few minutes^)...
    .venv\Scripts\pip.exe install -e ".[dev]"
)
cd /d "%~dp0"
echo.
echo   [setup] Server ready.
echo.
goto :eof
