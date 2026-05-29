@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo Starting AsureFlow server and client...
start "AsureFlow Server" "%~dp0start-server.bat"
timeout /t 5 /nobreak >nul
start "AsureFlow Client" "%~dp0start-client.bat"
exit /b 0
