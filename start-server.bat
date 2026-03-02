@echo off
title Asure Flow — Server
cd /d "%~dp0"
echo.
echo === Asure Flow — Server ===
echo.

:: Check if setup has been done
if not exist "%~dp0server\.venv" (
    echo First run detected — running setup...
    echo.
    powershell -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1"
)

powershell -ExecutionPolicy Bypass -File "%~dp0scripts\start-server.ps1"
pause
