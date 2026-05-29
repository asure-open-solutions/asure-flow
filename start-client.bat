@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo   AsureFlow - Client
echo   ===================
echo.

where node >nul 2>nul
if errorlevel 1 (
    echo   ERROR: Node.js is not installed or not on PATH.
    echo   Install Node.js from https://nodejs.org/
    echo.
    pause
    exit /b 1
)
where npm >nul 2>nul
if errorlevel 1 (
    echo   ERROR: npm is not installed or not on PATH.
    pause
    exit /b 1
)

if not "%~1"=="" (
    set "ASUREFLOW_SERVER=%~1"
) else if not defined ASUREFLOW_SERVER (
    set "ASUREFLOW_SERVER=http://localhost:8000"
)

if not exist "client\node_modules" (
    echo   [setup] Installing client dependencies...
    cd /d "%~dp0client"
    call npm install
    if errorlevel 1 (
        echo   ERROR: Failed to install client dependencies.
        pause
        exit /b 1
    )
    cd /d "%~dp0"
)

echo   Server: %ASUREFLOW_SERVER%
echo   Client: Electron app
echo   Web URL: http://127.0.0.1:5173
echo.
echo   Press Ctrl+C to stop.
echo.

cd /d "%~dp0client"
set "ELECTRON_RUN_AS_NODE="
call npm run dev -- --host 127.0.0.1
pause
exit /b %errorlevel%
