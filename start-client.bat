@echo off
title Asure Flow — Client
cd /d "%~dp0"
echo.
echo === Asure Flow — Client ===
echo.

:: Accept server URL as argument: start-client.bat http://192.168.1.50:8000
if "%~1"=="" (
    if "%ASUREFLOW_SERVER%"=="" (
        set ASUREFLOW_SERVER=http://localhost:8000
    )
) else (
    set ASUREFLOW_SERVER=%~1
)

echo Connecting to server: %ASUREFLOW_SERVER%
echo.

:: Auto-install on first run
if not exist "%~dp0client\node_modules" (
    echo First run — installing client dependencies...
    cd /d "%~dp0client"
    call npm install
    cd /d "%~dp0"
    echo.
)

cd /d "%~dp0client"
call npm run dev
pause
