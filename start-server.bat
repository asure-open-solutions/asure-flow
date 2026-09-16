@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo   AsureFlow - Server
echo   ===================
echo.

rem Prefer the project virtual environment. Fall back to system Python/py only for first setup.
set "PY_EXE=%~dp0server\.venv\Scripts\python.exe"
if exist "%PY_EXE%" goto have_python

where python >nul 2>nul
if not errorlevel 1 (
    set "PY_BOOT=python"
    goto setup_server
)
where py >nul 2>nul
if not errorlevel 1 (
    set "PY_BOOT=py -3"
    goto setup_server
)

echo   ERROR: Python is not installed or not on PATH.
echo   Install Python 3.11+ from https://www.python.org/downloads/
echo.
pause
exit /b 1

:setup_server
echo   [setup] Creating server virtual environment...
cd /d "%~dp0server"
%PY_BOOT% -m venv .venv
if errorlevel 1 (
    echo   ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m ensurepip --upgrade
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -e ".[dev]"
if errorlevel 1 (
    echo   ERROR: Failed to install server dependencies.
    pause
    exit /b 1
)
cd /d "%~dp0"
set "PY_EXE=%~dp0server\.venv\Scripts\python.exe"

:have_python
if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do (
        if not "%%a"=="" set "%%a=%%b"
    )
)
if not defined HOST set "HOST=127.0.0.1"
if not defined PORT set "PORT=8000"
if /i "%~1"=="lan" set "HOST=0.0.0.0"

echo   Python: %PY_EXE%
echo   Local:  http://localhost:%PORT%
echo   Docs:   http://localhost:%PORT%/docs
if "%HOST%"=="0.0.0.0" (
    for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$ip=(Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -notlike '127.*' -and $_.PrefixOrigin -ne 'WellKnown'} | Select-Object -First 1 -ExpandProperty IPAddress); if($ip){$ip}else{'your-pc-ip'}"`) do set "LAN_IP=%%i"
    echo   Network: http://%LAN_IP%:%PORT%
    echo   WARNING: Server is exposed to your local network without authentication.
    echo   If blocked, allow inbound TCP %PORT% in Windows Firewall.
)
echo.
echo   Press Ctrl+C to stop.
echo.

set "PYTHONPATH=%~dp0server\src"
cd /d "%~dp0server"
"%PY_EXE%" -m uvicorn asure_flow.main:app --host %HOST% --port %PORT% --ws-max-size 1048576
pause
exit /b %errorlevel%
