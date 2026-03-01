@echo off
title Asure Flow
echo.
echo === Asure Flow ===
echo.

:: Check if setup has been done
if not exist "server\.venv" (
    echo First run detected — running setup...
    echo.
    powershell -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1"
)

:: Start
powershell -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1"
pause
