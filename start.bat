@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo Starting AsureFlow server and client...
start "AsureFlow Server" "%~dp0start-server.bat"

if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do (
        if /i "%%a"=="PORT" set "PORT=%%b"
    )
)
if not defined PORT set "PORT=8000"

echo Waiting for the server health check...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$deadline = (Get-Date).AddMinutes(3); while ((Get-Date) -lt $deadline) { try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:%PORT%/api/health' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; Start-Sleep -Milliseconds 500 }; exit 1"
if errorlevel 1 (
    echo.
    echo ERROR: AsureFlow server did not become ready within 3 minutes.
    echo Check the "AsureFlow Server" window for setup or model errors.
    echo.
    pause
    exit /b 1
)

echo Server is ready. Starting the client...
start "AsureFlow Client" "%~dp0start-client.bat"
exit /b 0
