@echo off
title AI Assistant
cd /d "%~dp0"

echo Starting AI Assistant...

set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo Creating virtual environment...
    python -m venv ".venv"
)

if not exist "%PY%" (
    echo ERROR: Failed to create venv. Is Python installed?
    pause
    exit /b 1
)

echo Installing dependencies...
"%PY%" -m pip install -q -r requirements.txt

echo Starting server...
set "AI_ASSISTANT_ENABLE_AUDIO=1"
start "AI_ASSISTANT_SERVER" /min cmd /c ""%PY%" main.py"

echo Browser will open automatically when the server is ready...

echo.
echo ========================================
echo   AI Assistant is running!
echo   Close this window to stop the server.
echo ========================================
echo.

pause > nul

echo Stopping server...
taskkill /f /fi "WINDOWTITLE eq AI_ASSISTANT_SERVER" > nul 2>&1
echo Done.
